"""Where a decode token's time goes, for either expert bank format.

profile_decode_timeline.py replaces the FP4 expert kernel and therefore cannot
run against the affine 3-bit bank. This probe touches no numerics: it wraps the
resident store's decode entry point from the outside and splits each token's
wall clock into

  expert wait   main thread blocked inside get_many/get, waiting for a slot to
                be filled from SSD (hits return immediately)
  rest          everything else: attention, the MoE kernels, the head, Python

and reports the store's own byte and read-second counters alongside, so the
bytes-bound share of decode is visible directly.

Run it under benchmarks/guarded_run.sh like every other benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, StoreSnapshot  # noqa: E402
from cachalot.cache.resident_store import ResidentExpertStore  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402

wait_seconds = 0.0
wait_calls = 0


def instrument() -> None:
    original_get_many = ResidentExpertStore.get_many
    original_get = ResidentExpertStore.get

    def get_many(self, *args, **kwargs):
        global wait_seconds, wait_calls
        start = perf_counter()
        result = original_get_many(self, *args, **kwargs)
        wait_seconds += perf_counter() - start
        wait_calls += 1
        return result

    def get(self, *args, **kwargs):
        global wait_seconds, wait_calls
        start = perf_counter()
        result = original_get(self, *args, **kwargs)
        wait_seconds += perf_counter() - start
        wait_calls += 1
        return result

    ResidentExpertStore.get_many = get_many
    ResidentExpertStore.get = get


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=64)
    args = parser.parse_args()

    global wait_seconds, wait_calls

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as runtime:
        store = runtime.expert_store
        expert_bytes = store.expert_bytes
        print(
            f"runtime ready (expert budget {runtime.expert_cache_budget_bytes / 2**30:.1f} GiB, "
            f"wired {runtime.mlx_wired_limit_bytes / 2**30:.1f} GiB, expert {expert_bytes / 2**20:.2f} MiB)",
            flush=True,
        )
        encoding = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(runtime, encoding, text, args.prompt_tokens)
        runtime.reset()
        result = runtime.prefill_tokens(ids)
        mx.eval(result.logits)
        token = int(result.logits.argmax().item())

        instrument()
        before = StoreSnapshot.take(runtime)
        wait_seconds = 0.0
        wait_calls = 0
        read_seconds_before = store.ssd_read_seconds
        bytes_before = store.ssd_bytes_read

        start = perf_counter()
        for _ in range(args.decode_tokens):
            step = runtime.decode_token(token)
            token = int(step.logits.argmax().item())
        wall = perf_counter() - start

        delta = before.delta(StoreSnapshot.take(runtime))
        read_seconds = store.ssd_read_seconds - read_seconds_before
        read_bytes = store.ssd_bytes_read - bytes_before
        tokens = args.decode_tokens
        misses = delta.cache_misses
        hits = delta.cache_hits

        print(
            f"\ndecode {tokens} tokens: {wall:.2f} s = {tokens / wall:.2f} tok/s "
            f"({wall / tokens * 1e3:.0f} ms/token)"
        )
        print(
            f"  expert hit rate {hits / max(1, hits + misses):.1%} | "
            f"{misses / tokens:.1f} misses/token | {read_bytes / tokens / 2**20:.0f} MiB read/token"
        )
        print(
            f"  expert wait {wait_seconds:.2f} s = {wait_seconds / tokens * 1e3:6.1f} ms/token "
            f"({wait_seconds / wall:.1%} of decode), {wait_calls / tokens:.1f} calls/token"
        )
        print(
            f"  rest        {wall - wait_seconds:.2f} s = {(wall - wait_seconds) / tokens * 1e3:6.1f} ms/token "
            f"({1 - wait_seconds / wall:.1%} of decode)"
        )
        print(
            f"  reader threads {read_seconds:.2f} worker-s = {read_seconds / max(1, misses) * 1e3:.2f} ms/miss | "
            f"aggregate {read_bytes / max(read_seconds, 1e-9) / 1e9:.2f} GB/s | "
            f"wall-clock {read_bytes / wall / 1e9:.2f} GB/s"
        )
        floor = read_bytes / wall / 1e9
        print(
            f"  bytes floor at 5.2 GB/s single stream: {read_bytes / 5.2e9 / tokens * 1e3:.1f} ms/token; "
            f"at 7.3 GB/s with 8 reads in flight: {read_bytes / 7.3e9 / tokens * 1e3:.1f} ms/token; "
            f"achieved {floor:.2f} GB/s"
        )
        predicted = store.predicted_loads
        print(
            f"  prediction: {predicted} loads, {store.predicted_used} used "
            f"({store.predicted_used / max(1, predicted):.0%} precision), "
            f"{store.predicted_wasted_bytes / max(1, expert_bytes) / tokens:.1f} wasted loads/token"
        )


if __name__ == "__main__":
    main()
