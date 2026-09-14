from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import mlx.core as mx

from cachalot.model.text_decode_runtime import (
    TextDecodeRuntime,
)


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[int, ...]
    completion_text: str
    message: dict[str, Any] | None
    stopped_on_eos: bool


def load_official_encoding(
    model_path: str | Path,
) -> ModuleType:
    """
    Load the prompt/protocol implementation shipped with the
    official DeepSeek-V4.1 checkpoint.

    We intentionally use the checkpoint's encoding.py instead
    of maintaining a second prompt-format implementation.
    """
    path = (
        Path(model_path)
        / "encoding"
        / "encoding.py"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Official V4.1 encoding module not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "deepseek_v41_official_encoding",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load official encoding module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def sample_logits(
    logits: mx.array,
    *,
    temperature: float = 1.0,
) -> int:
    """
    Match the released DeepSeek-V4.1 sampling distribution.

    Official PyTorch implementation:

        if temperature == 0:
            argmax(logits)

        probs = softmax(
            logits / max(temperature, 1e-5),
            dtype=float32,
        )

        argmax(
            probs / Exp(1)
        )

    The exponential-race expression is distribution-equivalent
    to Gumbel-max:

        argmax(
            logits / T + Gumbel(0, 1)
        )

    MLX 0.32.2 provides the Gumbel sampler directly.
    """
    if logits.ndim != 1:
        raise ValueError(
            "sample_logits expects 1D vocabulary logits; "
            f"got shape {logits.shape}"
        )

    logits_f32 = logits.astype(
        mx.float32
    )

    if temperature == 0:
        return int(
            mx.argmax(
                logits_f32
            ).item()
        )

    temperature = max(
        float(temperature),
        1e-5,
    )

    noise = mx.random.gumbel(
        shape=logits_f32.shape,
        dtype=mx.float32,
    )

    sampled = mx.argmax(
        logits_f32 / temperature
        + noise
    )

    return int(
        sampled.item()
    )


def encode_messages(
    runtime: TextDecodeRuntime,
    messages: list[dict[str, Any]],
    *,
    thinking_mode: str = "chat",
    reasoning_effort: str | int | None = None,
) -> tuple[str, list[int]]:
    """
    Encode OpenAI-style messages with the official V4.1 protocol.
    """
    encoding = load_official_encoding(
        runtime.model_path
    )

    prompt = encoding.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
    )

    # Match the released inference script exactly:
    #
    #   tokenizer.encode(prompt)
    #
    token_ids = runtime.tokenizer.encode(
        prompt
    )

    return (
        prompt,
        list(token_ids),
    )


def generate_messages(
    runtime: TextDecodeRuntime,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    thinking_mode: str = "chat",
    reasoning_effort: str | int | None = None,
    seed: int | None = None,
    reset: bool = True,
    verbose: bool | None = None,
) -> GenerationResult:
    """
    Text-only DeepSeek-V4.1 message generation.

    Path:

        OpenAI-style messages
        -> official encoding.py
        -> official tokenizer.encode()
        -> sequential prefill
        -> official-equivalent sampling
        -> autoregressive decode
        -> EOS stopping
        -> official completion parser

    The current runtime deliberately performs sequential single-token
    prefill. A vectorized/chunked prefill path is a later optimization.
    """
    if verbose is None:
        verbose = bool(
            getattr(runtime, "verbose", False)
        )

    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens must be non-negative"
        )

    if seed is not None:
        mx.random.seed(
            int(seed)
        )

    if reset:
        runtime.reset()

    encoding = load_official_encoding(
        runtime.model_path
    )

    prompt = encoding.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
    )

    prompt_tokens = list(
        runtime.tokenizer.encode(
            prompt
        )
    )

    if not prompt_tokens:
        raise ValueError(
            "Official prompt encoding produced no tokens"
        )

    if (
        len(prompt_tokens)
        + max_new_tokens
        > runtime.max_seq_len
    ):
        raise ValueError(
            "Prompt + requested generation exceeds runtime "
            f"max_seq_len={runtime.max_seq_len}: "
            f"prompt={len(prompt_tokens)}, "
            f"max_new_tokens={max_new_tokens}"
        )

    if verbose:
        print(
            "[generate] prompt tokens =",
            len(prompt_tokens),
        )

    # --------------------------------------------------------
    # Prefill.
    #
    # Layer-major prompt execution preserves decode-exact state
    # while grouping routed MoE work expert-major across tokens.
    #
    # The returned logits correspond to the LAST prompt token and
    # predict the first completion token.
    # --------------------------------------------------------
    if verbose:
        print(
            "[generate] prefill "
            f"{len(prompt_tokens)} tokens"
        )

    result = runtime.prefill_tokens(
        prompt_tokens
    )

    completion_tokens: list[int] = []
    stopped_on_eos = False

    eos_id = (
        runtime.tokenizer.eos_token_id
    )

    # --------------------------------------------------------
    # Generation.
    # --------------------------------------------------------
    for step in range(
        max_new_tokens
    ):
        next_token = sample_logits(
            result.logits,
            temperature=temperature,
        )

        if verbose:
            print(
                f"[generate] step={step} "
                f"token_id={next_token} "
                f"text={runtime.tokenizer.decode([next_token], skip_special_tokens=False)!r}"
            )

        if next_token == eos_id:
            stopped_on_eos = True
            break

        completion_tokens.append(
            next_token
        )

        # Feed the sampled token back into the model. Its output
        # logits predict the following token.
        result = runtime.decode_token(
            next_token
        )

    completion_text = (
        runtime.tokenizer.decode(
            completion_tokens,
            skip_special_tokens=False,
        )
    )

    parser_text = completion_text

    if stopped_on_eos:
        parser_text += runtime.tokenizer.eos_token

    try:
        message = (
            encoding.parse_message_from_completion_text(
                parser_text,
                thinking_mode=thinking_mode,
            )
        )
    except Exception:
        # Keep raw completion usable even if the protocol parser
        # rejects a truncated generation.
        message = None

    return GenerationResult(
        prompt=prompt,
        prompt_tokens=tuple(
            prompt_tokens
        ),
        completion_tokens=tuple(
            completion_tokens
        ),
        completion_text=(
            completion_text
        ),
        message=message,
        stopped_on_eos=(
            stopped_on_eos
        ),
    )
