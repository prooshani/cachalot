"""Do mispredicted expert loads get asked for a few tokens later?

Routing prediction reads 42.7 % of every byte this runtime pulls off the SSD
and two thirds of those loads are never consumed by the layer they were aimed
at (HANDOFF sections 9.10, 9.18). When one of them expires, `prefetch_decode`
has already read the whole expert into a transient slot and
`_sweep_inflight_locked` hands that slot straight back: the bytes are in memory
and are then dropped.

Admitting them instead -- as ordinary residents, evicting the LRU tail -- is
only worth building if the dropped expert is asked for again soon. This probe
measures that and nothing else. It touches no numerics and changes no policy:
it wraps the store from the outside, records

  * every prediction the sweep expires unused, with the decode pass it died on
  * every demand read, with the decode pass that issued it

and reports, for each distance in tokens, what share of demand reads were for
an expert a recent prediction had already read and thrown away. That share is
the ceiling on what an admission policy can save; it cannot be beaten by any
implementation of one, and a low number closes the idea for the cost of this
one run.

Run it under benchmarks/guarded_run.sh like every other benchmark.
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections import Counter
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, StoreSnapshot  # noqa: E402
from cachalot.cache.resident_store import ResidentExpertStore  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402

Key = tuple[int, int]

log_lock = threading.Lock()

# (decode pass, key) for every prediction the sweep released unused.
expired_events: list[tuple[int, Key]] = []
# (decode pass, key, seconds) for every read performed by the demand pool.
demand_events: list[tuple[int, Key, float]] = []
# (decode pass, key) for every read performed by the prediction pool.
predict_events: list[tuple[int, Key]] = []


def instrument(store: ResidentExpertStore) -> None:
    original_sweep = ResidentExpertStore._sweep_inflight_locked
    original_read = ResidentExpertStore._read_into

    def sweep(self, keep):
        # Called with the store lock held, so reading _inflight here is safe.
        before = set(self._inflight)
        original_sweep(self, keep)
        released = before - set(self._inflight)
        if released:
            pass_id = self._decode_pass
            with log_lock:
                for key in released:
                    expired_events.append((pass_id, key))

    def read_into(self, entry, slot):
        started = perf_counter()
        result = original_read(self, entry, slot)
        elapsed = perf_counter() - started
        name = threading.current_thread().name
        key = (entry.layer, entry.expert)
        pass_id = self._decode_pass
        with log_lock:
            if name.startswith("expert-load"):
                demand_events.append((pass_id, key, elapsed))
            elif name.startswith("expert-predict"):
                predict_events.append((pass_id, key))
        return result

    ResidentExpertStore._sweep_inflight_locked = sweep
    ResidentExpertStore._read_into = read_into


def report(tokens: int) -> None:
    with log_lock:
        expired = list(expired_events)
        demand = list(demand_events)
        predicted = list(predict_events)

    # Most recent pass on which each key was read speculatively and dropped.
    last_expiry: dict[Key, int] = {}
    expiry_passes: dict[Key, list[int]] = {}
    for pass_id, key in expired:
        expiry_passes.setdefault(key, []).append(pass_id)

    print(
        f"\n  predicted reads {len(predicted)} ({len(predicted) / tokens:.1f}/token), "
        f"expired unused {len(expired)} ({len(expired) / tokens:.1f}/token), "
        f"demand reads {len(demand)} ({len(demand) / tokens:.1f}/token)"
    )
    if not demand or not expired:
        print("  nothing to correlate")
        return

    # For every demand read, the distance in decode passes back to the most
    # recent expiry of the same expert, if there was one.
    distances: list[int] = []
    unmatched = 0
    demand_seconds_matched = 0.0
    demand_seconds = 0.0
    for pass_id, key, elapsed in demand:
        demand_seconds += elapsed
        prior = [p for p in expiry_passes.get(key, ()) if p <= pass_id]
        if not prior:
            unmatched += 1
            continue
        distances.append(pass_id - max(prior))
        demand_seconds_matched += elapsed

    hist = Counter(distances)
    print(
        f"  demand reads whose expert a prediction had already read and dropped: "
        f"{len(distances)} of {len(demand)} ({len(distances) / len(demand):.1%})"
    )
    print("  distance from the drop to the demand read, in decode passes:")
    cumulative = 0
    for window in (0, 1, 2, 4, 8, 16, 32, 64):
        within = sum(count for distance, count in hist.items() if distance <= window)
        print(
            f"    within {window:3d} tokens: {within:6d} reads "
            f"({within / len(demand):.1%} of all demand reads, "
            f"{within / max(1, len(expired)):.1%} of drops)"
        )
        cumulative = within
    far = len(distances) - cumulative
    if far:
        print(f"    beyond 64 tokens: {far} reads")
    # The other question the same two logs answer: how much speculative
    # reading is a *re-read* of an expert a recent prediction already read and
    # dropped? The router barely moves between adjacent tokens, so a wrong
    # prediction tends to be made again, and the expert is read again.
    repeats: list[int] = []
    fresh = 0
    for pass_id, key in predicted:
        prior = [p for p in expiry_passes.get(key, ()) if p < pass_id]
        if not prior:
            fresh += 1
            continue
        repeats.append(pass_id - max(prior))
    rhist = Counter(repeats)
    print(
        f"\n  speculative reads that re-read an expert a prediction had already "
        f"dropped: {len(repeats)} of {len(predicted)} ({len(repeats) / len(predicted):.1%})"
    )
    print("  distance from the drop to the re-read, in decode passes:")
    for window in (1, 2, 4, 8, 16, 32, 64):
        within = sum(count for distance, count in rhist.items() if distance <= window)
        print(
            f"    within {window:3d} tokens: {within:6d} reads "
            f"({within / len(predicted):.1%} of speculative reads, "
            f"{within / tokens:.1f}/token)"
        )
    # What a blocklist would actually do. A speculative read is blocked when
    # the same expert was read speculatively and dropped within the last N
    # decode passes. The block is a win when that read would have expired
    # unused again, and a loss when this time the prediction was right -- the
    # expert is then read on demand at its layer instead of early.
    expired_keys = {(pass_id, key) for pass_id, key in expired}
    print("\n  what a drop-blocklist would have done, by lifetime:")
    print(f"    {'TTL':>6s} {'blocked':>9s} {'would waste':>12s} {'would be used':>14s}")
    for window in (1, 2, 4, 8, 16, 32, 64):
        blocked_waste = 0
        blocked_used = 0
        for pass_id, key in predicted:
            prior = [p for p in expiry_passes.get(key, ()) if p < pass_id]
            if not prior or pass_id - max(prior) > window:
                continue
            # Unused predictions are released by the sweep in the pass that
            # issued them or the one after it (the token boundary).
            if (pass_id, key) in expired_keys or (pass_id + 1, key) in expired_keys:
                blocked_waste += 1
            else:
                blocked_used += 1
        total = blocked_waste + blocked_used
        print(
            f"    {window:6d} {total / tokens:8.1f}/tok "
            f"{blocked_waste / tokens:11.1f}/tok {blocked_used / tokens:13.1f}/tok"
        )
    print(
        f"  demand read time that could not have been saved: "
        f"{(demand_seconds - demand_seconds_matched) * 1e3 / tokens:.1f} ms/token of "
        f"{demand_seconds * 1e3 / tokens:.1f} ms/token of demand reading (worker time, not wall clock)"
    )
    _ = last_expiry, unmatched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=96)
    args = parser.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as runtime:
        store = runtime.expert_store
        print(
            f"runtime ready (expert budget {runtime.expert_cache_budget_bytes / 2**30:.1f} GiB, "
            f"wired {runtime.mlx_wired_limit_bytes / 2**30:.1f} GiB, "
            f"expert {store.expert_bytes / 2**20:.2f} MiB)",
            flush=True,
        )
        encoding = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(runtime, encoding, text, args.prompt_tokens)
        runtime.reset()
        result = runtime.prefill_tokens(ids)
        mx.eval(result.logits)
        token = int(result.logits.argmax().item())

        instrument(store)
        before = StoreSnapshot.take(runtime)
        start = perf_counter()
        for _ in range(args.decode_tokens):
            step = runtime.decode_token(token)
            token = int(step.logits.argmax().item())
        wall = perf_counter() - start
        delta = before.delta(StoreSnapshot.take(runtime))

        tokens = args.decode_tokens
        hits, misses = delta.cache_hits, delta.cache_misses
        print(
            f"\ndecode {tokens} tokens: {wall:.2f} s = {tokens / wall:.2f} tok/s "
            f"({wall / tokens * 1e3:.0f} ms/token)"
        )
        print(
            f"  expert hit rate {hits / max(1, hits + misses):.1%} | "
            f"{misses / tokens:.1f} misses/token"
        )
        report(tokens)


if __name__ == "__main__":
    main()
