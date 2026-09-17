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

It also wraps ResidentExpertStore._read_into and buckets every expert read by
the worker thread that performed it, because the store's ssd_read_seconds mixes
demand misses (the "expert-load" pool, on the critical path) with speculative
prediction loads (the "expert-predict" pool, which run while the GPU computes).
A single mean over both cannot say whether a blocking read is slow. The merged
union of the read intervals gives the fraction of decode wall clock during which
at least one read was outstanding, and the ratio of summed read time to that
union gives the mean number of reads in flight while the drive was busy.

Run it under benchmarks/guarded_run.sh like every other benchmark.
"""

from __future__ import annotations

import argparse
import sys
import threading
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

# (bucket, started, finished) for every expert read, appended by the worker
# thread that performed it. "demand" is the critical path, "predict" is not.
read_events: list[tuple[str, float, float]] = []
read_events_lock = threading.Lock()

# (entered, left) for every blocking call into the resident store, so a read
# interval can be classified by whether the main thread was waiting on it.
wait_windows: list[tuple[float, float]] = []


def instrument() -> None:
    original_get_many = ResidentExpertStore.get_many
    original_get = ResidentExpertStore.get

    def get_many(self, *args, **kwargs):
        global wait_seconds, wait_calls
        start = perf_counter()
        result = original_get_many(self, *args, **kwargs)
        finished = perf_counter()
        wait_seconds += finished - start
        wait_calls += 1
        wait_windows.append((start, finished))
        return result

    def get(self, *args, **kwargs):
        global wait_seconds, wait_calls
        start = perf_counter()
        result = original_get(self, *args, **kwargs)
        finished = perf_counter()
        wait_seconds += finished - start
        wait_calls += 1
        wait_windows.append((start, finished))
        return result

    original_read_into = ResidentExpertStore._read_into

    def read_into(self, entry, slot):
        name = threading.current_thread().name
        if name.startswith("expert-predict"):
            bucket = "predict"
        elif name.startswith("expert-load"):
            bucket = "demand"
        else:
            bucket = name.split("_")[0]
        started = perf_counter()
        try:
            return original_read_into(self, entry, slot)
        finally:
            finished = perf_counter()
            with read_events_lock:
                read_events.append((bucket, started, finished))

    ResidentExpertStore.get_many = get_many
    ResidentExpertStore.get = get
    ResidentExpertStore._read_into = read_into


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(fraction * (len(sorted_values) - 1) + 0.5))
    return sorted_values[index]


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Wall-clock seconds covered by at least one of the intervals."""
    if not intervals:
        return 0.0
    total = 0.0
    ordered = sorted(intervals)
    current_start, current_end = ordered[0]
    for started, finished in ordered[1:]:
        if started > current_end:
            total += current_end - current_start
            current_start, current_end = started, finished
        elif finished > current_end:
            current_end = finished
    return total + (current_end - current_start)


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged: list[tuple[float, float]] = []
    for started, finished in sorted(intervals):
        if merged and started <= merged[-1][1]:
            if finished > merged[-1][1]:
                merged[-1] = (merged[-1][0], finished)
        else:
            merged.append((started, finished))
    return merged


def _intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Seconds covered by both merged interval lists."""
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _subtract(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merged interval list a minus merged interval list b."""
    result: list[tuple[float, float]] = []
    j = 0
    for start, end in a:
        cursor = start
        while j < len(b) and b[j][1] <= cursor:
            j += 1
        k = j
        while k < len(b) and b[k][0] < end:
            if b[k][0] > cursor:
                result.append((cursor, min(b[k][0], end)))
            cursor = max(cursor, b[k][1])
            if cursor >= end:
                break
            k += 1
        if cursor < end:
            result.append((cursor, end))
    return result


def report_wait_composition(
    events: list[tuple[str, float, float]],
    windows: list[tuple[float, float]],
    wall: float,
    tokens: int,
) -> None:
    """
    What the main thread was actually waiting for inside the resident store.

    A predicted load that arrives before the layer needs it costs nothing; one
    that is still in flight when the layer asks stops the GPU exactly like an
    unpredicted miss. Splitting the blocked time by which pool held a read
    outstanding says whether the lever is prediction timing or prediction
    coverage, and time blocked with no read outstanding at all is the store's
    own overhead on the critical path.
    """
    if not windows:
        return
    blocked = _merge(windows)
    blocked_s = sum(end - start for start, end in blocked)
    demand = _merge([(s, f) for name, s, f in events if name == "demand"])
    predict = _merge([(s, f) for name, s, f in events if name == "predict"])

    on_demand = _intersect(blocked, demand)
    on_predict_only = _intersect(_subtract(blocked, demand), predict)
    idle = blocked_s - on_demand - on_predict_only

    print("\n  what the blocked time was waiting for")
    print(
        f"    blocked total          {blocked_s:6.2f} s = {blocked_s / tokens * 1e3:6.1f} ms/token "
        f"({blocked_s / wall:5.1%} of decode)"
    )
    print(
        f"    a demand read in flight {on_demand:6.2f} s = {on_demand / tokens * 1e3:6.1f} ms/token "
        f"({on_demand / blocked_s:5.1%} of blocked) -- coverage: the miss was never predicted"
    )
    print(
        f"    only a predicted read   {on_predict_only:6.2f} s = {on_predict_only / tokens * 1e3:6.1f} ms/token "
        f"({on_predict_only / blocked_s:5.1%} of blocked) -- timing: predicted right, issued too late"
    )
    print(
        f"    no read outstanding     {idle:6.2f} s = {idle / tokens * 1e3:6.1f} ms/token "
        f"({idle / blocked_s:5.1%} of blocked) -- store overhead: slots, eviction, locks"
    )


def report_reads(events: list[tuple[str, float, float]], wall: float, tokens: int) -> None:
    if not events:
        print("  reads: none recorded")
        return

    print("\n  expert reads by worker pool (demand = critical path, predict = speculative)")
    for bucket in ("demand", "predict"):
        durations = sorted((finished - started) * 1e3 for name, started, finished in events if name == bucket)
        if not durations:
            print(f"    {bucket:8s} none")
            continue
        total_ms = sum(durations)
        print(
            f"    {bucket:8s} {len(durations):5d} reads, {len(durations) / tokens:5.1f}/token | "
            f"mean {total_ms / len(durations):5.2f} ms, p50 {_percentile(durations, 0.5):5.2f}, "
            f"p90 {_percentile(durations, 0.9):5.2f}, max {durations[-1]:6.2f} | "
            f"{total_ms / 1e3:.2f} worker-s"
        )
    other = {name for name, _s, _f in events} - {"demand", "predict"}
    for bucket in sorted(other):
        durations = sorted((finished - started) * 1e3 for name, started, finished in events if name == bucket)
        print(f"    {bucket:8s} {len(durations):5d} reads, mean {sum(durations) / len(durations):5.2f} ms")

    intervals = [(started, finished) for _name, started, finished in events]
    busy = _union_seconds(intervals)
    summed = sum(finished - started for started, finished in intervals)
    print(
        f"    drive busy {busy:.2f} s of {wall:.2f} s decode ({busy / wall:.1%}); "
        f"{summed / max(busy, 1e-9):.2f} reads in flight while busy, "
        f"{summed / wall:.2f} averaged over the whole decode"
    )
    demand = [(started, finished) for name, started, finished in events if name == "demand"]
    if demand:
        demand_busy = _union_seconds(demand)
        demand_summed = sum(finished - started for started, finished in demand)
        print(
            f"    demand alone: busy {demand_busy:.2f} s ({demand_busy / wall:.1%} of decode), "
            f"{demand_summed / max(demand_busy, 1e-9):.2f} in flight while busy"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument(
        "--switch-interval",
        type=float,
        default=None,
        help="sys.setswitchinterval for the decode loop (default 0.005 s); "
             "a smaller value lets a loader thread reclaim the GIL sooner after its pread",
    )
    args = parser.parse_args()

    if args.switch_interval is not None:
        sys.setswitchinterval(args.switch_interval)
    print(f"GIL switch interval {sys.getswitchinterval() * 1e3:.2f} ms", flush=True)

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

        with read_events_lock:
            read_events.clear()
        wait_windows.clear()

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
        with read_events_lock:
            events = list(read_events)
        report_reads(events, wall, tokens)
        report_wait_composition(events, list(wait_windows), wall, tokens)

        predicted = store.predicted_loads
        print(
            f"  prediction: {predicted} loads, {store.predicted_used} used "
            f"({store.predicted_used / max(1, predicted):.0%} precision), "
            f"{store.predicted_wasted_bytes / max(1, expert_bytes) / tokens:.1f} wasted loads/token"
        )


if __name__ == "__main__":
    main()
