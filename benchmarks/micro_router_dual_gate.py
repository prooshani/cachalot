"""
Screen: is a decode layer's two router score passes worth fusing into one?

Every MoE decode layer runs route_topk_fused twice over the same input vector:
once with its own gate matrix, once with layer L+1's for the prefetch
predictor (moe_layer_metal.moe_layer_forward). That is two scores kernels of
384 rows each and two top-k kernels per layer, 160 launches per token.

The scores kernel puts one simdgroup on each expert row, so a 384-row launch
occupies 12,288 threads on a GPU that can hold far more. If it is latency
bound rather than bandwidth bound, scoring 768 rows in one launch costs about
what 384 costs and the second gate becomes free.

This is a screen, not a gate: it decides whether to spend the day writing the
fused path, and it says nothing about what the runtime does with it.

    PYTHONPATH=src python benchmarks/micro_router_dual_gate.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cachalot.model.router_fused_metal import (  # noqa: E402
    _scores_kernel,
    _topk_kernel,
    route_topk_fused,
)

HIDDEN = 5120
N_EXPERTS = 384
TOPK = 6


def chained(fn, n=40, repeats=7):
    """Per-call GPU+build time with n launches inside one lazy graph."""
    mx.eval(fn())
    mx.synchronize()
    times, builds = [], []
    for _ in range(repeats):
        t0 = perf_counter()
        outs = [fn() for _ in range(n)]
        t_build = perf_counter() - t0
        mx.eval(*outs)
        mx.synchronize()
        times.append((perf_counter() - t0) / n * 1e3)
        builds.append(t_build / n * 1e3)
    return min(times), statistics.median(times), min(builds)


def main():
    mx.random.seed(0)
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    w_own = mx.random.normal((N_EXPERTS, HIDDEN)).astype(mx.bfloat16) * 0.02
    w_next = mx.random.normal((N_EXPERTS, HIDDEN)).astype(mx.bfloat16) * 0.02
    b_own = mx.random.normal((N_EXPERTS,)).astype(mx.float32) * 0.1
    b_next = mx.random.normal((N_EXPERTS,)).astype(mx.float32) * 0.1
    w_both = mx.concatenate([w_own, w_next], axis=0)
    mx.eval(x, w_own, w_next, b_own, b_next, w_both)

    n_threads = 512  # next power of two at or above 384, as the shipped path picks
    scores_k = _scores_kernel(HIDDEN)
    topk_k = _topk_kernel(n_threads, TOPK)
    n_one = mx.array([N_EXPERTS], dtype=mx.uint32)
    n_two = mx.array([2 * N_EXPERTS], dtype=mx.uint32)
    temp = mx.array([1.0], dtype=mx.float32)
    scale = mx.array([1.5], dtype=mx.float32)
    mx.eval(n_one, n_two, temp, scale)

    def scores(weight, n_arr, rows):
        return scores_k(
            inputs=[x, weight, n_arr, temp],
            template=[("T", weight.dtype)],
            grid=(rows * 32, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows,)],
            output_dtypes=[mx.float32],
        )[0]

    def topk(sc, bias):
        return topk_k(
            inputs=[sc, bias, n_one, scale],
            template=[],
            grid=(n_threads, 1, 1),
            threadgroup=(n_threads, 1, 1),
            output_shapes=[(TOPK,), (TOPK,)],
            output_dtypes=[mx.int32, mx.float32],
        )

    rows = []

    def rec(name, fn, count=40):
        lo, med, build = chained(fn)
        rows.append((name, lo, count))
        print(f"{name:46s} {lo:6.3f} ms (median {med:6.3f}, build {build:5.3f}) "
              f"x{count} = {lo * count:5.1f} ms", flush=True)

    print(f"hidden {HIDDEN}, {N_EXPERTS} experts, top-{TOPK}, gate matrix "
          f"{N_EXPERTS * HIDDEN * 2 / 2**20:.1f} MiB\n")

    rec("shipped: route_topk_fused x2 (own + predictor)",
        lambda: (route_topk_fused(x, w_own, b_own, topk=TOPK).indices,
                 route_topk_fused(x, w_next, b_next, topk=TOPK).indices))
    rec("  one route_topk_fused, for reference",
        lambda: route_topk_fused(x, w_own, b_own, topk=TOPK).indices)
    rec("  scores kernel alone, 384 rows",
        lambda: scores(w_own, n_one, N_EXPERTS))
    rec("  scores kernel alone, 768 rows (stacked gates)",
        lambda: scores(w_both, n_two, 2 * N_EXPERTS))

    def fused():
        sc = scores(w_both, n_two, 2 * N_EXPERTS)
        a = topk(sc[:N_EXPERTS], b_own)
        b = topk(sc[N_EXPERTS:], b_next)
        return a[0], b[0]

    rec("fused: one 768-row scores + two top-k", fused)

    base = rows[0][1]
    best = rows[-1][1]
    print(f"\nper layer: {base:.3f} ms shipped against {best:.3f} ms fused "
          f"({(base - best) / base * 100:+.1f} %)")
    print(f"per token (40 layers): {base * 40:.1f} ms against {best * 40:.1f} ms, "
          f"a saving of {(base - best) * 40:.1f} ms")


if __name__ == "__main__":
    main()
