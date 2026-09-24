from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cachalot.config import DEFAULT_CONFIG
from cachalot.model.generation import (
    GenerationResult,
    generate_messages,
)
from cachalot.model.text_decode_runtime import (
    TextDecodeRuntime,
)

DEFAULT_MODEL_PATH = DEFAULT_CONFIG.resolved_model_path


@dataclass(frozen=True)
class ChatResponse:
    """
    Stable public response object for callers such as:

    - CLI
    - local HTTP server
    - OpenAI-compatible adapter
    - agent harness integrations
    """

    content: str
    message: dict[str, Any] | None
    completion_tokens: tuple[int, ...]
    prompt_tokens: tuple[int, ...]
    stopped_on_eos: bool
    reused_prefix_tokens: int = 0

    @classmethod
    def from_generation(
        cls,
        result: GenerationResult,
    ) -> ChatResponse:
        return cls(
            content=result.completion_text,
            message=result.message,
            completion_tokens=result.completion_tokens,
            prompt_tokens=result.prompt_tokens,
            stopped_on_eos=result.stopped_on_eos,
            reused_prefix_tokens=result.reused_prefix_tokens,
        )


class V41Model:
    """
    Long-lived DeepSeek-V4.1 text runtime.

    This is intentionally the stable API boundary.

    Frontends should depend on V41Model rather than directly on
    TextDecodeRuntime so the execution engine can be optimized later
    without changing CLI/server/harness integrations.
    """

    def __init__(
        self,
        runtime: TextDecodeRuntime,
    ) -> None:
        self.runtime = runtime
        self.model_path = str(
            runtime.model_path
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        max_seq_len: int = DEFAULT_CONFIG.max_seq_len,
        expert_cache_budget_bytes: int = DEFAULT_CONFIG.expert_cache_budget_bytes,
        mlx_cache_limit_bytes: int = DEFAULT_CONFIG.mlx_cache_limit_bytes,
        mlx_wired_limit_bytes: int = DEFAULT_CONFIG.mlx_wired_limit_bytes,
        io_workers: int = DEFAULT_CONFIG.io_workers,
        verbose: bool = False,
        warmup: bool = True,
    ) -> V41Model:
        """
        Load the checkpoint and build the runtime.

        expert_cache_budget_bytes=0 and mlx_wired_limit_bytes=0 mean
        auto-size from this machine's unified memory (see cachalot.config).
        """
        runtime = TextDecodeRuntime(
            str(model_path or DEFAULT_MODEL_PATH),
            max_seq_len=max_seq_len,
            expert_cache_budget_bytes=expert_cache_budget_bytes,
            mlx_cache_limit_bytes=mlx_cache_limit_bytes,
            mlx_wired_limit_bytes=mlx_wired_limit_bytes,
            io_workers=io_workers,
            verbose=verbose,
        )

        if warmup:
            if verbose:
                print("Compiling kernels with a warm-up prefill + decode step...")
            runtime.warmup()

        return cls(runtime)

    @property
    def tokenizer(self):
        return self.runtime.tokenizer

    @property
    def position(self) -> int:
        return self.runtime.position

    @property
    def max_seq_len(self) -> int:
        return self.runtime.max_seq_len

    def reset(self) -> None:
        self.runtime.reset()

    @property
    def prefix_cache(self):
        return self.runtime.prefix_cache

    def set_decode_miss_budget(self, max_misses: int | None) -> None:
        """
        Opt-in approximation for faster decode: per layer, load at most
        `max_misses` non-resident experts (highest router weight first) and
        drop the rest, rescaling the remaining weights. None restores exact
        inference. Changes model output; measure before using.
        """
        self.runtime.expert_store.decode_miss_budget = (
            None if max_misses is None else max(0, int(max_misses))
        )

    def stats(self) -> dict:
        """Expert-store, prefix-cache and MLX memory counters."""
        import mlx.core as mx

        s = self.runtime.expert_store.stats()
        pc = self.runtime.prefix_cache
        return {
            "expert_hits": s.cache_hits,
            "expert_misses": s.cache_misses,
            "expert_hit_rate": s.hit_rate,
            "skipped_experts": self.runtime.expert_store.skipped_experts,
            "predicted_loads": self.runtime.expert_store.predicted_loads,
            "predicted_used": self.runtime.expert_store.predicted_used,
            "decode_miss_budget": self.runtime.expert_store.decode_miss_budget,
            "ssd_bytes_read": s.ssd_bytes_read,
            "expert_reads": getattr(s, "reads", 0),
            "expert_fast_reads": getattr(s, "fast_reads", 0),
            "expert_read_seconds": getattr(s, "read_wall_seconds", 0.0),
            "resident_experts": len(self.runtime.expert_store),
            "resident_bytes": self.runtime.expert_store.current_bytes,
            "prefix_cache_entries": len(pc),
            "prefix_cache_hits": pc.hits,
            "prefix_cache_misses": pc.misses,
            "prefix_cache_reused_tokens": pc.reused_tokens,
            "prefix_cache_bytes": pc.nbytes,
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_cache_bytes": mx.get_cache_memory(),
            "mlx_peak_bytes": mx.get_peak_memory(),
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        thinking_mode: str = "chat",
        reasoning_effort: str | int | None = None,
        seed: int | None = None,
        reset: bool = True,
        verbose: bool | None = None,
        use_prefix_cache: bool = True,
    ) -> ChatResponse:
        result = generate_messages(
            self.runtime,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            seed=seed,
            reset=reset,
            verbose=verbose,
            use_prefix_cache=use_prefix_cache,
        )

        return ChatResponse.from_generation(
            result
        )

    def close(self) -> None:
        self.runtime.close()

    def __enter__(self) -> V41Model:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
