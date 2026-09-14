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

DEFAULT_MODEL_PATH = (
    "/Volumes/X10Pro/Flash4-1/"
    "DeepSeek-V4.1-Flash"
)


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
        max_seq_len: int = 4096,
        mlx_cache_limit_bytes: int = DEFAULT_CONFIG.mlx_cache_limit_bytes,
        io_workers: int = 8,
        verbose: bool = False,
    ) -> V41Model:
        runtime = TextDecodeRuntime(
            str(model_path),
            max_seq_len=max_seq_len,
            mlx_cache_limit_bytes=mlx_cache_limit_bytes,
            io_workers=io_workers,
            verbose=verbose,
        )

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
