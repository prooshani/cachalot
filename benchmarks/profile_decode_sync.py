"""
How much of an all-resident decode token is GPU work and how much is the CPU
building the next graph while the GPU waits?

The MoE layer has to bring routing back to the CPU to address the resident
expert store, so every layer ends in mx.eval(route.indices, ...). That drains
the pipeline 40 times per token. Between two drains the GPU has nothing queued
while Python calls tolist(), looks entries up and constructs the next batch of
MLX ops.

This wraps mx.eval and mx.synchronize with a timer for the duration of one
token. Time inside eval is the GPU finishing what was queued; time outside it
is CPU-side graph construction, during which the GPU is idle. The split says
which side a speed lever has to attack.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402

_real_eval = mx.eval
_real_sync = mx.synchronize
STATS = {"eval_n": 0, "eval_t": 0.0}
SITES: dict[str, list[float]] = {}
LAST = {"t": 0.0}


def _site() -> str:
    f = sys._getframe(2)
    return f"{Path(f.f_code.co_filename).name}:{f.f_lineno}"


def _record(name: str, wait: float, gap: float) -> None:
    row = SITES.setdefault(name, [0.0, 0.0, 0])
    row[0] += wait
    row[1] += gap
    row[2] += 1


def _timed_eval(*args, **kwargs):
    t0 = perf_counter()
    gap = t0 - LAST["t"]
    _real_eval(*args, **kwargs)
    t1 = perf_counter()
    STATS["eval_t"] += t1 - t0
    STATS["eval_n"] += 1
    _record(_site(), t1 - t0, gap)
    LAST["t"] = perf_counter()


def _timed_sync(*args, **kwargs):
    t0 = perf_counter()
    gap = t0 - LAST["t"]
    _real_sync(*args, **kwargs)
    t1 = perf_counter()
    STATS["eval_t"] += t1 - t0
    STATS["eval_n"] += 1
    _record(_site(), t1 - t0, gap)
    LAST["t"] = perf_counter()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=12)
    args = ap.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
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

        totals, evals, counts = [], [], []
        mx.eval = _timed_eval
        mx.synchronize = _timed_sync
        try:
            for _ in range(args.repeats):
                rt.restore(snap)
                STATS["eval_n"] = 0
                STATS["eval_t"] = 0.0
                t0 = perf_counter()
                LAST["t"] = t0
                r = rt.decode_token(tok)
                mx.eval(r.logits)
                total = (perf_counter() - t0) * 1e3
                totals.append(total)
                evals.append(STATS["eval_t"] * 1e3)
                counts.append(STATS["eval_n"])
        finally:
            mx.eval = _real_eval
            mx.synchronize = _real_sync

        # An "all-resident" token is all-resident on the demand path only: a
        # mispredicted expert was never demanded, so it is not resident, and
        # the same wrong prediction is read, dropped and read again on every
        # repeat. That is what CACHALOT_PREDICT_SUBMIT=0 removes.
        pred = (
            store.predicted_loads - pred0[0],
            store.predicted_expired - pred0[1],
            store.ssd_bytes_read - pred0[2],
        )

        i = totals.index(min(totals))
        med_total = statistics.median(totals)
        med_eval = statistics.median(evals)
        print(f"all-resident token, {args.prompt_tokens}-token context, {args.repeats} repeats")
        print(f"  whole token        min {min(totals):6.1f} ms   median {med_total:6.1f} ms")
        print(f"  inside mx.eval     min {min(evals):6.1f} ms   median {med_eval:6.1f} ms"
              f"   ({med_eval / med_total * 100:.1f}% of the token)")
        print(f"  CPU outside eval   min {min(t - e for t, e in zip(totals, evals, strict=True)):6.1f} ms"
              f"   median {med_total - med_eval:6.1f} ms   ({(med_total - med_eval) / med_total * 100:.1f}%)")
        print(f"  eval/synchronize calls per token: {statistics.median(counts):.0f}")
        print(f"  speculative reads this arm issued: {pred[0] / args.repeats:.1f} loads/token, "
              f"{pred[1] / args.repeats:.1f} expired/token, "
              f"{pred[2] / args.repeats / 2**20:.0f} MiB/token")
        print(f"  fastest token: {totals[i]:.1f} ms total, {evals[i]:.1f} ms in eval, {counts[i]} evals")
        print("\n  per call site, per token (gap = CPU time since the previous eval returned):")
        print(f"    {'site':28s} {'calls':>6s} {'in eval':>9s} {'gap before':>11s}")
        rows = sorted(SITES.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
        n = args.repeats
        for name, (wait, gap, calls) in rows:
            print(f"    {name:28s} {calls / n:6.1f} {wait / n * 1e3:8.1f} ms {gap / n * 1e3:10.1f} ms")


if __name__ == "__main__":
    main()
