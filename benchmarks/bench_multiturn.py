from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from cachalot.config import default_model_path
from cachalot.model.text_decode_runtime import TextDecodeRuntime

MODEL_PATH = default_model_path()

PROMPT_TOKENS = 256
DECODE_TOKENS = 8

MAX_SEQ_LEN = 4096
IO_WORKERS = 8
EXPERT_CACHE_BUDGET_BYTES = 40 * 1024**3


@dataclass
class StatsSnapshot:
    cache_hits: int
    cache_misses: int
    ssd_bytes_read: int
    ssd_read_seconds: float
    promotion_seconds: float


def snapshot(runtime: TextDecodeRuntime) -> StatsSnapshot:
    stats = runtime.expert_store.stats()

    return StatsSnapshot(
        cache_hits=stats.cache_hits,
        cache_misses=stats.cache_misses,
        ssd_bytes_read=stats.ssd_bytes_read,
        ssd_read_seconds=stats.ssd_read_seconds,
        promotion_seconds=stats.promotion_seconds,
    )


def delta(
    before: StatsSnapshot,
    after: StatsSnapshot,
) -> StatsSnapshot:
    return StatsSnapshot(
        cache_hits=after.cache_hits - before.cache_hits,
        cache_misses=after.cache_misses - before.cache_misses,
        ssd_bytes_read=after.ssd_bytes_read - before.ssd_bytes_read,
        ssd_read_seconds=after.ssd_read_seconds - before.ssd_read_seconds,
        promotion_seconds=after.promotion_seconds - before.promotion_seconds,
    )


def common_prefix_len(
    a: list[int],
    b: list[int],
) -> int:
    count = 0

    for x, y in zip(a, b, strict=False):
        if x != y:
            break

        count += 1

    return count


def resident_layer_counts(
    runtime: TextDecodeRuntime,
) -> list[int]:
    counts = [0] * 40

    with runtime.expert_store._lock:
        for layer, _expert in runtime.expert_store._items.keys():
            counts[layer] += 1

    return counts


def print_prefill_result(
    title: str,
    *,
    seconds: float,
    token_count: int,
    stats: StatsSnapshot,
    runtime: TextDecodeRuntime,
) -> None:
    requests = stats.cache_hits + stats.cache_misses

    hit_rate = (
        stats.cache_hits / requests
        if requests
        else 0.0
    )

    layer_counts = resident_layer_counts(runtime)

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print("seconds          :", seconds)
    print("tok_per_s        :", token_count / seconds)
    print("requests         :", requests)
    print("hits             :", stats.cache_hits)
    print("misses           :", stats.cache_misses)
    print("hit_rate         :", hit_rate)
    print(
        "ssd_read_gib     :",
        stats.ssd_bytes_read / 1024**3,
    )
    print(
        "ssd_seconds_sum  :",
        stats.ssd_read_seconds,
    )
    print(
        "promotion_seconds:",
        stats.promotion_seconds,
    )
    print(
        "resident         :",
        len(runtime.expert_store),
    )
    print(
        "resident_bytes   :",
        runtime.expert_store.current_bytes,
    )
    print(
        "layer_min        :",
        min(layer_counts),
    )
    print(
        "layer_max        :",
        max(layer_counts),
    )


def run_decode(
    runtime: TextDecodeRuntime,
    title: str,
    first_token: int,
) -> list[int]:
    generated = []

    token_id = int(first_token)

    before = snapshot(runtime)

    t0 = perf_counter()

    for _ in range(DECODE_TOKENS):
        result = runtime.decode_token(token_id)

        token_id = int(
            result.logits.argmax().item()
        )

        generated.append(token_id)

    seconds = perf_counter() - t0

    after = snapshot(runtime)

    stats = delta(
        before,
        after,
    )

    requests = stats.cache_hits + stats.cache_misses

    hit_rate = (
        stats.cache_hits / requests
        if requests
        else 0.0
    )

    layer_counts = resident_layer_counts(runtime)

    print()
    print(f"{title} decode:")

    print("  seconds          :", seconds)
    print(
        "  sec_per_token    :",
        seconds / DECODE_TOKENS,
    )
    print("  requests         :", requests)
    print("  hits             :", stats.cache_hits)
    print("  misses           :", stats.cache_misses)
    print("  hit_rate         :", hit_rate)
    print(
        "  ssd_read_gib     :",
        stats.ssd_bytes_read / 1024**3,
    )
    print(
        "  skewed_layers    :",
        sum(
            count != min(layer_counts)
            for count in layer_counts
        ),
    )
    print(
        "  layer_min        :",
        min(layer_counts),
    )
    print(
        "  layer_max        :",
        max(layer_counts),
    )
    print(
        "  generated_ids    :",
        generated,
    )

    return generated


def make_prompt(
    tokenizer,
    text: str,
) -> list[int]:
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if not ids:
        raise RuntimeError(
            "Prompt tokenized to zero tokens"
        )

    while len(ids) < PROMPT_TOKENS:
        ids.extend(ids)

    return ids[:PROMPT_TOKENS]


def main() -> None:
    with TextDecodeRuntime(
        MODEL_PATH,
        max_seq_len=MAX_SEQ_LEN,
        expert_cache_budget_bytes=EXPERT_CACHE_BUDGET_BYTES,
        io_workers=IO_WORKERS,
        verbose=False,
    ) as runtime:

        tokenizer = runtime.tokenizer

        prompt_a = make_prompt(
            tokenizer,
            """
You are reviewing a Python runtime for a large mixture-of-experts
language model running on Apple Silicon. Inspect the expert cache,
prefetch scheduling, SSD access, resident-memory policy, and MLX
execution behavior. Identify correctness issues before suggesting
optimizations. Focus on deterministic behavior and concrete evidence.
""",
        )

        prompt_b = make_prompt(
            tokenizer,
            """
You are debugging a TypeScript backend service with GraphQL, MongoDB,
message queues, authentication, and asynchronous jobs. Trace a
production bug carefully, identify the actual failure path, and propose
the smallest safe patch. Preserve API behavior and provide tests for
the regression.
""",
        )

        prompt_c = make_prompt(
            tokenizer,
            """
You are implementing a React and Next.js feature for an analytics
dashboard. Review server-side data loading, client state, rendering,
caching, error handling, and performance. Keep the implementation
maintainable and avoid unnecessary architectural changes.
""",
        )

        print(
            "prompt lengths:",
            len(prompt_a),
            len(prompt_b),
            len(prompt_c),
        )

        print(
            "common prefix A/B:",
            common_prefix_len(
                prompt_a,
                prompt_b,
            ),
        )

        print(
            "common prefix B/C:",
            common_prefix_len(
                prompt_b,
                prompt_c,
            ),
        )

        # TURN 1
        runtime.reset()

        before = snapshot(runtime)
        t0 = perf_counter()

        result_a = runtime.prefill_tokens(
            prompt_a
        )

        seconds = perf_counter() - t0
        after = snapshot(runtime)

        print_prefill_result(
            "TURN 1 — TASK A / COLD",
            seconds=seconds,
            token_count=len(prompt_a),
            stats=delta(before, after),
            runtime=runtime,
        )

        first_token_a = int(
            result_a.logits.argmax().item()
        )

        run_decode(
            runtime,
            "TURN 1",
            first_token_a,
        )

        # TURN 2
        runtime.reset()

        before = snapshot(runtime)
        t0 = perf_counter()

        result_b = runtime.prefill_tokens(
            prompt_b
        )

        seconds = perf_counter() - t0
        after = snapshot(runtime)

        print_prefill_result(
            "TURN 2 — TASK B / AFTER DECODE",
            seconds=seconds,
            token_count=len(prompt_b),
            stats=delta(before, after),
            runtime=runtime,
        )

        first_token_b = int(
            result_b.logits.argmax().item()
        )

        run_decode(
            runtime,
            "TURN 2",
            first_token_b,
        )

        # TURN 3
        runtime.reset()

        before = snapshot(runtime)
        t0 = perf_counter()

        result_c = runtime.prefill_tokens(
            prompt_c
        )

        seconds = perf_counter() - t0
        after = snapshot(runtime)

        print_prefill_result(
            "TURN 3 — TASK C / AFTER DECODE",
            seconds=seconds,
            token_count=len(prompt_c),
            stats=delta(before, after),
            runtime=runtime,
        )

        first_token_c = int(
            result_c.logits.argmax().item()
        )

        run_decode(
            runtime,
            "TURN 3",
            first_token_c,
        )

        # TURN 4
        runtime.reset()

        before = snapshot(runtime)
        t0 = perf_counter()

        runtime.prefill_tokens(
            prompt_a
        )

        seconds = perf_counter() - t0
        after = snapshot(runtime)

        print_prefill_result(
            "TURN 4 — RETURN TO TASK A",
            seconds=seconds,
            token_count=len(prompt_a),
            stats=delta(before, after),
            runtime=runtime,
        )


if __name__ == "__main__":
    main()
