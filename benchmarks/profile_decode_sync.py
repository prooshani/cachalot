"""
How much of a decode token is GPU work, how much is the CPU building the next
graph while the GPU waits, and how much is the main thread blocked inside the
expert store waiting for SSD?

The MoE layer has to bring routing back to the CPU to address the resident
expert store, so every layer ends in mx.eval(route.indices, ...). That drains
the pipeline 40 times per token. Between two drains the GPU has nothing queued
while Python calls tolist(), looks entries up, blocks in the store for whatever
is not resident, and constructs the next batch of MLX ops.

This wraps mx.eval and mx.synchronize with a timer for the duration of one
token, and wraps the resident store's blocking entry points with a second one.
Each interval between two evals therefore splits three ways:

  inside eval     the GPU finishing what was queued
  store-blocked   the main thread inside get()/get_many(), waiting for SSD
  CPU             everything else: tolist, dict lookups, graph construction

Two arms:

  resident  the historical arm. A snapshot is restored and the same token is
            decoded N times, so the demand path never misses.
  stream    a real continuation of N tokens at whatever budget the run was
            given, so it misses like a live session does.

The difference between the two arms' CPU column is HANDOFF section 9.23's
"rest above the all-resident floor while streaming" -- about 33 ms that no
mechanism accounts for. If it lands in the store-blocked column it is the
already-priced miss cost; if it lands in CPU it is the store's admission path
(slot acquisition, eviction, the LRU, slot_views); if it lands inside eval it
is the GPU waiting on memory the SSD DMA is also using.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, StoreSnapshot  # noqa: E402
from cachalot.cache.resident_store import ResidentExpertStore  # noqa: E402
from cachalot.storage.engram_reader import EngramRowReader  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402

_real_eval = mx.eval
_real_sync = mx.synchronize
STATS = {"eval_n": 0, "eval_t": 0.0}
SITES: dict[str, list[float]] = {}
# "t" is when the last eval returned; "blocked" is the store-blocked total at
# that moment, so the next eval can charge the interval's blocking to the gap.
LAST = {"t": 0.0, "blocked": 0.0, "engram": 0.0}
BLOCKED = {"t": 0.0, "calls": 0}
# Engram rows are read with synchronous preads on the decode thread, so they
# land in the gap between two evals and look like CPU time. They are not.
ENGRAM = {"t": 0.0, "calls": 0, "rows": 0}


def _site() -> str:
    f = sys._getframe(2)
    return f"{Path(f.f_code.co_filename).name}:{f.f_lineno}"


def _record(name: str, wait: float, gap: float, blocked: float, engram: float) -> None:
    row = SITES.setdefault(name, [0.0, 0.0, 0.0, 0.0, 0])
    row[0] += wait
    row[1] += gap
    row[2] += blocked
    row[3] += engram
    row[4] += 1


def _timed_eval(*args, **kwargs):
    t0 = perf_counter()
    gap = t0 - LAST["t"]
    blocked = BLOCKED["t"] - LAST["blocked"]
    engram = ENGRAM["t"] - LAST["engram"]
    _real_eval(*args, **kwargs)
    t1 = perf_counter()
    STATS["eval_t"] += t1 - t0
    STATS["eval_n"] += 1
    _record(_site(), t1 - t0, gap, blocked, engram)
    LAST["t"] = perf_counter()
    LAST["blocked"] = BLOCKED["t"]
    LAST["engram"] = ENGRAM["t"]


def _timed_sync(*args, **kwargs):
    t0 = perf_counter()
    gap = t0 - LAST["t"]
    blocked = BLOCKED["t"] - LAST["blocked"]
    engram = ENGRAM["t"] - LAST["engram"]
    _real_sync(*args, **kwargs)
    t1 = perf_counter()
    STATS["eval_t"] += t1 - t0
    STATS["eval_n"] += 1
    _record(_site(), t1 - t0, gap, blocked, engram)
    LAST["t"] = perf_counter()
    LAST["blocked"] = BLOCKED["t"]
    LAST["engram"] = ENGRAM["t"]


def instrument_store() -> None:
    """Charge every second the main thread spends inside the store."""
    original_get_many = ResidentExpertStore.get_many
    original_get = ResidentExpertStore.get

    def get_many(self, *args, **kwargs):
        start = perf_counter()
        try:
            return original_get_many(self, *args, **kwargs)
        finally:
            BLOCKED["t"] += perf_counter() - start
            BLOCKED["calls"] += 1

    def get(self, *args, **kwargs):
        start = perf_counter()
        try:
            return original_get(self, *args, **kwargs)
        finally:
            BLOCKED["t"] += perf_counter() - start
            BLOCKED["calls"] += 1

    ResidentExpertStore.get_many = get_many
    ResidentExpertStore.get = get

    # What the decode thread pays for Engram, whoever does the reading. With
    # CACHALOT_DECODE_ENGRAM_PREFETCH the pread runs on a worker thread and
    # the decode thread only waits on the future, so timing read_rows would
    # charge one arm for work the other hides. _apply_engram is the whole of
    # it on the decode thread: the wait (or the read), the dequantisation and
    # the forward's construction.
    original_apply = TextDecodeRuntime._apply_engram

    def apply_engram(self, *args, **kwargs):
        start = perf_counter()
        try:
            return original_apply(self, *args, **kwargs)
        finally:
            ENGRAM["t"] += perf_counter() - start
            ENGRAM["calls"] += 1

    TextDecodeRuntime._apply_engram = apply_engram

    original_read_rows = EngramRowReader.read_rows

    def read_rows(self, layout, row_ids, *args, **kwargs):
        ENGRAM["rows"] += int(getattr(row_ids, "size", len(row_ids)))
        return original_read_rows(self, layout, row_ids, *args, **kwargs)

    EngramRowReader.read_rows = read_rows


def report(label: str, totals, evals, blocks, engrams, counts, sites, n, store_delta,
           blocked_calls, engram_calls, engram_rows) -> None:
    med_total = statistics.median(totals)
    med_eval = statistics.median(evals)
    med_block = statistics.median(blocks)
    med_engram = statistics.median(engrams)
    cpus = [t - e - b - g for t, e, b, g in zip(totals, evals, blocks, engrams, strict=True)]
    med_cpu = statistics.median(cpus)
    i = totals.index(min(totals))
    print(f"\n=== {label} ===")
    print(f"  whole token        min {min(totals):6.1f} ms   median {med_total:6.1f} ms")
    print(f"  inside mx.eval     min {min(evals):6.1f} ms   median {med_eval:6.1f} ms"
          f"   ({med_eval / med_total * 100:.1f}% of the token)")
    print(f"  store-blocked      min {min(blocks):6.1f} ms   median {med_block:6.1f} ms"
          f"   ({med_block / med_total * 100:.1f}%)   {blocked_calls / n:.1f} calls/token")
    print(f"  engram on the CPU  min {min(engrams):6.1f} ms   median {med_engram:6.1f} ms"
          f"   ({med_engram / med_total * 100:.1f}%)   {engram_calls / n:.1f} calls/token,"
          f" {engram_rows / max(1, engram_calls):.0f} rows/call")
    print(f"  CPU, none of those min {min(cpus):6.1f} ms"
          f"   median {med_cpu:6.1f} ms   ({med_cpu / med_total * 100:.1f}%)")
    print(f"  eval/synchronize calls per token: {statistics.median(counts):.0f}")
    if store_delta is not None:
        req = store_delta.cache_hits + store_delta.cache_misses
        print(f"  expert hit rate {store_delta.cache_hits / max(1, req):.1%} | "
              f"{store_delta.cache_misses / n:.1f} misses/token | "
              f"{store_delta.ssd_bytes_read / n / 2**20:.0f} MiB read/token")
    print(f"  fastest token: {totals[i]:.1f} ms total, {evals[i]:.1f} in eval, "
          f"{blocks[i]:.1f} blocked, {engrams[i]:.1f} engram, {counts[i]} evals")
    print("\n  per call site, per token (gap = wall since the previous eval returned; blocked and"
          " engram are the parts of it inside the store and inside a pread):")
    print(f"    {'site':28s} {'calls':>6s} {'in eval':>9s} {'gap':>9s} {'blocked':>9s}"
          f" {'engram':>9s} {'cpu':>9s}")
    rows = sorted(sites.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
    for name, (wait, gap, blocked, engram, calls) in rows:
        print(f"    {name:28s} {calls / n:6.1f} {wait / n * 1e3:8.1f} ms {gap / n * 1e3:8.1f} ms"
              f" {blocked / n * 1e3:8.1f} ms {engram / n * 1e3:8.1f} ms"
              f" {(gap - blocked - engram) / n * 1e3:8.1f} ms")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--mode", choices=("resident", "stream", "both"), default="resident")
    ap.add_argument("--stream-tokens", type=int, default=48,
                    help="tokens of real continuation for the streaming arm")
    args = ap.parse_args()

    instrument_store()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        print(f"runtime ready (expert budget {rt.expert_cache_budget_bytes / 2**30:.1f} GiB, "
              f"wired {rt.mlx_wired_limit_bytes / 2**30:.1f} GiB, "
              f"expert {rt.expert_store.expert_bytes / 2**20:.2f} MiB)", flush=True)
        enc = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        snap = rt.snapshot()
        rt.decode_token(tok)
        rt.restore(snap)

        store = rt.expert_store
        pred0 = (store.predicted_loads, store.predicted_expired, store.ssd_bytes_read)

        if args.mode in ("resident", "both"):
            SITES.clear()
            BLOCKED["t"] = 0.0
            BLOCKED["calls"] = 0
            ENGRAM["t"] = 0.0
            ENGRAM["calls"] = 0
            ENGRAM["rows"] = 0
            totals, evals, blocks, engrams, counts = [], [], [], [], []
            before = StoreSnapshot.take(rt)
            mx.eval = _timed_eval
            mx.synchronize = _timed_sync
            try:
                for _ in range(args.repeats):
                    rt.restore(snap)
                    STATS["eval_n"] = 0
                    STATS["eval_t"] = 0.0
                    b0 = BLOCKED["t"]
                    g0 = ENGRAM["t"]
                    t0 = perf_counter()
                    LAST["t"] = t0
                    LAST["blocked"] = b0
                    LAST["engram"] = g0
                    r = rt.decode_token(tok)
                    mx.eval(r.logits)
                    totals.append((perf_counter() - t0) * 1e3)
                    evals.append(STATS["eval_t"] * 1e3)
                    blocks.append((BLOCKED["t"] - b0) * 1e3)
                    engrams.append((ENGRAM["t"] - g0) * 1e3)
                    counts.append(STATS["eval_n"])
            finally:
                mx.eval = _real_eval
                mx.synchronize = _real_sync
            delta = before.delta(StoreSnapshot.take(rt))

            # An "all-resident" token is all-resident on the demand path only: a
            # mispredicted expert was never demanded, so it is not resident, and
            # the same wrong prediction is read, dropped and read again on every
            # repeat. That is what CACHALOT_PREDICT_SUBMIT=0 removes.
            pred = (
                store.predicted_loads - pred0[0],
                store.predicted_expired - pred0[1],
                store.ssd_bytes_read - pred0[2],
            )
            report(f"all-resident, {args.prompt_tokens}-token context, {args.repeats} repeats of one token",
                   totals, evals, blocks, engrams, counts, dict(SITES), args.repeats, delta,
                   BLOCKED["calls"], ENGRAM["calls"], ENGRAM["rows"])
            print(f"  speculative reads this arm issued: {pred[0] / args.repeats:.1f} loads/token, "
                  f"{pred[1] / args.repeats:.1f} expired/token, "
                  f"{pred[2] / args.repeats / 2**20:.0f} MiB/token")

        if args.mode in ("stream", "both"):
            # A real continuation: every token is a new position with a new
            # working set, so the demand path misses the way a live session's
            # does. Nothing is restored between tokens.
            rt.restore(snap)
            SITES.clear()
            BLOCKED["t"] = 0.0
            BLOCKED["calls"] = 0
            ENGRAM["t"] = 0.0
            ENGRAM["calls"] = 0
            ENGRAM["rows"] = 0
            totals, evals, blocks, engrams, counts = [], [], [], [], []
            before = StoreSnapshot.take(rt)
            token = tok
            mx.eval = _timed_eval
            mx.synchronize = _timed_sync
            try:
                for _ in range(args.stream_tokens):
                    STATS["eval_n"] = 0
                    STATS["eval_t"] = 0.0
                    b0 = BLOCKED["t"]
                    g0 = ENGRAM["t"]
                    t0 = perf_counter()
                    LAST["t"] = t0
                    LAST["blocked"] = b0
                    LAST["engram"] = g0
                    step = rt.decode_token(token)
                    mx.eval(step.logits)
                    totals.append((perf_counter() - t0) * 1e3)
                    evals.append(STATS["eval_t"] * 1e3)
                    blocks.append((BLOCKED["t"] - b0) * 1e3)
                    engrams.append((ENGRAM["t"] - g0) * 1e3)
                    counts.append(STATS["eval_n"])
                    token = int(step.logits.argmax().item())
            finally:
                mx.eval = _real_eval
                mx.synchronize = _real_sync
            delta = before.delta(StoreSnapshot.take(rt))
            n = args.stream_tokens
            report(f"streaming, {args.prompt_tokens}-token context, {n} tokens of continuation",
                   totals, evals, blocks, engrams, counts, dict(SITES), n, delta,
                   BLOCKED["calls"], ENGRAM["calls"], ENGRAM["rows"])
            print(f"  tokens by third (median ms): "
                  f"{statistics.median(totals[:n // 3]):.1f} / "
                  f"{statistics.median(totals[n // 3:2 * n // 3]):.1f} / "
                  f"{statistics.median(totals[2 * n // 3:]):.1f}")


if __name__ == "__main__":
    main()
