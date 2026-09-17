from __future__ import annotations

import importlib.util
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any

import mlx.core as mx

from cachalot.model.text_decode_runtime import (
    DecodeResult,
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
    reused_prefix_tokens: int = 0


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
    top_p: float = 1.0,
    top_k: int = 0,
    recent: list[int] | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    no_repeat_ngram_size: int = 0,
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

    top_k / top_p are OpenAI-API conveniences layered on top: they
    mask the tail of the distribution before the Gumbel draw and leave
    the official path untouched when unset (top_p >= 1, top_k <= 0).

    The penalties are the same kind of addition, and they exist because of a
    measured failure: on the 2-bit expert bank, free-running generation on code
    prompts falls into a repeating loop -- 828 tokens of period 2 in one
    measured reply -- that the teacher-forced quality gate cannot see at all
    (benchmarks/repetition_quality.py).

    `frequency_penalty` subtracts `penalty * count` and is the one that breaks
    such a loop: a classic repetition penalty fires once per *unique* token, so
    a token the model is confident about survives it no matter how many times it
    has already been emitted, while a frequency penalty grows without bound.
    `presence_penalty` subtracts a flat amount for any token already seen.
    `no_repeat_ngram_size` is the hard guarantee: it bans every token that would
    complete an n-gram already present. Use it sparingly on code, where
    `for (size_t i = 0;` legitimately repeats.

    `recent` is the window the counts are taken over -- the caller decides how
    far back to look, so a long reply does not accumulate penalty until it
    cannot use a common word at all.

    All three are off by default and the official path is then bit-for-bit
    unchanged.
    """
    if logits.ndim != 1:
        raise ValueError(
            "sample_logits expects 1D vocabulary logits; "
            f"got shape {logits.shape}"
        )

    logits_f32 = logits.astype(
        mx.float32
    )

    # Penalties apply to the raw logits, before temperature and before the
    # greedy shortcut. Applying them after the division would make their
    # strength depend on the temperature, and applying them after the
    # temperature-0 return would leave greedy decoding -- the setting most
    # prone to looping -- with no protection at all.
    if recent and (frequency_penalty or presence_penalty):
        counts: dict[int, int] = {}
        for token in recent:
            counts[token] = counts.get(token, 0) + 1
        ids = mx.array(list(counts.keys()), dtype=mx.int32)
        occurrences = mx.array(list(counts.values()), dtype=mx.float32)
        adjustment = (
            occurrences * float(frequency_penalty)
            + float(presence_penalty)
        )
        logits_f32 = logits_f32.at[ids].add(-adjustment)

    if recent and no_repeat_ngram_size and no_repeat_ngram_size > 1:
        size = int(no_repeat_ngram_size)
        prefix = tuple(recent[-(size - 1):])
        if len(prefix) == size - 1:
            banned = {
                recent[i + size - 1]
                for i in range(len(recent) - size + 1)
                if tuple(recent[i : i + size - 1]) == prefix
            }
            if banned:
                logits_f32 = logits_f32.at[
                    mx.array(sorted(banned), dtype=mx.int32)
                ].add(-mx.inf)

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

    scaled = logits_f32 / temperature

    if top_k and top_k > 0 and top_k < scaled.shape[0]:
        kth = mx.sort(scaled)[-int(top_k)]
        scaled = mx.where(
            scaled < kth,
            mx.array(-mx.inf, dtype=mx.float32),
            scaled,
        )

    if top_p is not None and 0.0 < top_p < 1.0:
        order = mx.argsort(-scaled)
        sorted_logits = scaled[order]
        probs = mx.softmax(sorted_logits)
        cumulative = mx.cumsum(probs)
        # Keep the smallest set whose mass reaches top_p (always keep the top token).
        keep_sorted = (cumulative - probs) < top_p
        keep = mx.zeros_like(keep_sorted)
        keep = keep.at[order].add(keep_sorted)
        scaled = mx.where(
            keep,
            scaled,
            mx.array(-mx.inf, dtype=mx.float32),
        )

    noise = mx.random.gumbel(
        shape=scaled.shape,
        dtype=mx.float32,
    )

    sampled = mx.argmax(
        scaled
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


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    # Repetition controls; all off by default. See sample_logits.
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    penalty_window: int = 256


@dataclass(frozen=True)
class GenerationEvent:
    """
    One step of streaming generation.

    kind:
        "prefill"  prompt processed; `prompt_tokens`, `reused_prefix_tokens`
        "token"    one sampled token appended; `token`
        "done"     generation finished; `finish_reason` in {"stop", "length", "cancel"}
    """

    kind: str
    token: int | None = None
    prompt_tokens: int = 0
    reused_prefix_tokens: int = 0
    finish_reason: str | None = None
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0


def prepare_prompt(
    runtime: TextDecodeRuntime,
    prompt_tokens: list[int],
    *,
    reset: bool = True,
    use_prefix_cache: bool = True,
    verbose: bool = False,
) -> tuple[DecodeResult, int]:
    """
    Bring the runtime to the state "prompt consumed" and return the logits
    predicting the first completion token plus the number of prefix tokens
    reused from the prefix cache.
    """
    reused = 0
    snap = None

    if reset:
        snap = (
            runtime.prefix_cache.find(tuple(prompt_tokens))
            if use_prefix_cache
            else None
        )

        if snap is not None and (
            len(snap.tokens) < len(prompt_tokens)
            or snap.logits is not None
        ):
            runtime.restore(snap)
            reused = len(snap.tokens)
            suffix = prompt_tokens[reused:]
        else:
            runtime.reset()
            suffix = prompt_tokens
            snap = None
    else:
        suffix = prompt_tokens

    if verbose:
        print(
            "[generate] prefill "
            f"{len(suffix)} tokens "
            f"(reused {reused} from prefix cache)"
        )

    if suffix:
        result = runtime.prefill_tokens(suffix)
    else:
        result = DecodeResult(
            logits=snap.logits,
            hidden=snap.logits,
            routes=(),
            position=runtime.position - 1,
        )

    if use_prefix_cache:
        runtime.prefix_cache.add(
            runtime.snapshot(logits=result.logits)
        )

    return result, reused


def stream_tokens(
    runtime: TextDecodeRuntime,
    prompt_tokens: list[int],
    params: SamplingParams,
    *,
    reset: bool = True,
    use_prefix_cache: bool = True,
    cancel: threading.Event | None = None,
    verbose: bool = False,
) -> Iterator[GenerationEvent]:
    """
    Core autoregressive loop shared by the chat API, the completions API
    and the CLI. Yields one GenerationEvent per step.
    """
    if params.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    if not prompt_tokens:
        raise ValueError("prompt must contain at least one token")

    if len(prompt_tokens) + params.max_new_tokens > runtime.max_seq_len:
        raise ValueError(
            "Prompt + requested generation exceeds runtime "
            f"max_seq_len={runtime.max_seq_len}: "
            f"prompt={len(prompt_tokens)}, "
            f"max_new_tokens={params.max_new_tokens}"
        )

    if params.seed is not None:
        mx.random.seed(int(params.seed))

    t0 = perf_counter()
    result, reused = prepare_prompt(
        runtime,
        prompt_tokens,
        reset=reset,
        use_prefix_cache=use_prefix_cache,
        verbose=verbose,
    )
    prefill_seconds = perf_counter() - t0

    yield GenerationEvent(
        kind="prefill",
        prompt_tokens=len(prompt_tokens),
        reused_prefix_tokens=reused,
        prefill_seconds=prefill_seconds,
    )

    eos_id = runtime.tokenizer.eos_token_id
    stop_ids = set(params.stop_token_ids) | {eos_id}

    finish_reason = "length"
    generated = 0
    t1 = perf_counter()

    # Penalties count over the reply only, not the prompt: penalizing what the
    # user wrote would push the model away from the subject it was asked about.
    history: list[int] = []
    penalties_on = bool(
        params.frequency_penalty
        or params.presence_penalty
        or params.no_repeat_ngram_size
    )
    window = max(1, int(params.penalty_window))

    for _ in range(params.max_new_tokens):
        if cancel is not None and cancel.is_set():
            finish_reason = "cancel"
            break

        next_token = sample_logits(
            result.logits,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            recent=history[-window:] if penalties_on else None,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            no_repeat_ngram_size=params.no_repeat_ngram_size,
        )

        if next_token in stop_ids:
            finish_reason = "stop"
            break

        generated += 1
        if penalties_on:
            history.append(next_token)
        yield GenerationEvent(kind="token", token=next_token)

        # Feed the sampled token back; its logits predict the next one.
        result = runtime.decode_token(next_token)

    if use_prefix_cache and generated:
        # State now covers prompt + reply; the next turn usually extends it.
        runtime.prefix_cache.add(
            runtime.snapshot(logits=result.logits)
        )

    yield GenerationEvent(
        kind="done",
        finish_reason=finish_reason,
        prompt_tokens=len(prompt_tokens),
        reused_prefix_tokens=reused,
        prefill_seconds=prefill_seconds,
        decode_seconds=perf_counter() - t1,
    )


def generate_messages(
    runtime: TextDecodeRuntime,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    thinking_mode: str = "chat",
    reasoning_effort: str | int | None = None,
    seed: int | None = None,
    reset: bool = True,
    verbose: bool | None = None,
    use_prefix_cache: bool = True,
    cancel: threading.Event | None = None,
) -> GenerationResult:
    """
    Text-only DeepSeek-V4.1 message generation.

    Path:

        OpenAI-style messages
        -> official encoding.py
        -> official tokenizer.encode()
        -> prefix-cache-aware prefill
        -> official-equivalent sampling
        -> autoregressive decode
        -> EOS stopping
        -> official completion parser

    Multi-turn requests reuse the longest cached prefix snapshot and
    prefill only the new suffix.
    """
    if verbose is None:
        verbose = bool(getattr(runtime, "verbose", False))

    encoding = load_official_encoding(runtime.model_path)

    prompt = encoding.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
    )

    prompt_tokens = list(runtime.tokenizer.encode(prompt))

    if not prompt_tokens:
        raise ValueError("Official prompt encoding produced no tokens")

    if verbose:
        print("[generate] prompt tokens =", len(prompt_tokens))

    params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )

    completion_tokens: list[int] = []
    finish_reason = "length"
    reused = 0

    for event in stream_tokens(
        runtime,
        prompt_tokens,
        params,
        reset=reset,
        use_prefix_cache=use_prefix_cache,
        cancel=cancel,
        verbose=verbose,
    ):
        if event.kind == "token":
            completion_tokens.append(event.token)
            if verbose:
                print(
                    f"[generate] step={len(completion_tokens) - 1} "
                    f"token_id={event.token} "
                    f"text={runtime.tokenizer.decode([event.token], skip_special_tokens=False)!r}"
                )
        elif event.kind == "done":
            finish_reason = event.finish_reason or "length"
            reused = event.reused_prefix_tokens

    stopped_on_eos = finish_reason == "stop"

    completion_text = runtime.tokenizer.decode(
        completion_tokens,
        skip_special_tokens=False,
    )

    message = parse_completion(
        encoding,
        completion_text,
        thinking_mode=thinking_mode,
        stopped_on_eos=stopped_on_eos,
        eos_token=runtime.tokenizer.eos_token,
    )

    return GenerationResult(
        prompt=prompt,
        prompt_tokens=tuple(prompt_tokens),
        completion_tokens=tuple(completion_tokens),
        completion_text=completion_text,
        message=message,
        stopped_on_eos=stopped_on_eos,
        reused_prefix_tokens=reused,
    )


def parse_completion(
    encoding: ModuleType,
    completion_text: str,
    *,
    thinking_mode: str,
    stopped_on_eos: bool,
    eos_token: str,
) -> dict[str, Any] | None:
    """
    Run the official completion parser; return None when the text is
    malformed (truncated generation) so callers can fall back to raw text.
    """
    parser_text = completion_text
    if stopped_on_eos:
        parser_text += eos_token
    try:
        return encoding.parse_message_from_completion_text(
            parser_text,
            thinking_mode=thinking_mode,
        )
    except Exception:
        return None
