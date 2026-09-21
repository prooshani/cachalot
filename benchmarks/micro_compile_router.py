"""
Screen: what is left in the router pass once its constants are memoised?

A decoded token runs `route_topk_fused` eighty times -- once for each layer's
own routing and once more for the predictor's look at layer L+1 -- and each
call builds two Metal kernel launches. Section 6.3.1 priced the *GPU* side of
that at 0.025 ms per call and closed fusing the two passes into one at 0.6 ms
per token, but it never priced the CPU side, which is what section 9.15 is
about.

Three arms, all bit-identical:

  shipped       route_topk_fused as it stands, constants memoised
  compiled      the same function under mx.compile, one trace reused for every
                layer because the gate weights enter as arrays
  compiled x2   the layer's own pass and the predictor's in one traced graph,
                which is what a layer actually issues

Construction is timed with no eval in the loop; chained runs many launches
inside one eval.

    PYTHONPATH=src python benchmarks/micro_compile_router.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cachalot.model.router_fused_metal import route_topk_fused  # noqa: E402

LAYERS = 40
N_EXPERTS = 384
HIDDEN = 5120
TOPK = 6


def build_time(fn, n=200):
    outs = fn()
    mx.eval(outs)
    mx.synchronize()
    t0 = perf_counter()
    kept = [fn() for _ in range(n)]
    build = (perf_counter() - t0) / n * 1e3
    mx.eval(kept)
    mx.synchronize()
    return build


def chained(fn, n=40, repeats=7):
    mx.eval(fn())
    mx.synchronize()
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        outs = [fn() for _ in range(n)]
        mx.eval(outs)
        mx.synchronize()
        times.append((perf_counter() - t0) / n * 1e3)
    return min(times), statistics.median(times)


def main():
    mx.random.seed(0)
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    w = (mx.random.normal((N_EXPERTS, HIDDEN)) * 0.02).astype(mx.bfloat16)
    b = mx.random.normal((N_EXPERTS,)).astype(mx.float32)
    w_next = (mx.random.normal((N_EXPERTS, HIDDEN)) * 0.02).astype(mx.bfloat16)
    b_next = mx.random.normal((N_EXPERTS,)).astype(mx.float32)
    mx.eval(x, w, b, w_next, b_next)

    def shipped():
        r = route_topk_fused(x, w, b, topk=TOPK)
        return [r.indices, r.weights]

    def shipped_pair():
        r = route_topk_fused(x, w, b, topk=TOPK)
        p = route_topk_fused(x, w_next, b_next, topk=TOPK)
        return [r.indices, r.weights, p.indices]

    def core(xin, wt, bs):
        r = route_topk_fused(xin, wt, bs, topk=TOPK)
        return [r.indices, r.weights]

    def core_pair(xin, wt, bs, wt2, bs2):
        r = route_topk_fused(xin, wt, bs, topk=TOPK)
        p = route_topk_fused(xin, wt2, bs2, topk=TOPK)
        return [r.indices, r.weights, p.indices]

    c_one = mx.compile(core)
    c_pair = mx.compile(core_pair)

    def compiled():
        return c_one(x, w, b)

    def compiled_pair():
        return c_pair(x, w, b, w_next, b_next)

    ref, ref_pair = shipped(), shipped_pair()
    got, got_pair = compiled(), compiled_pair()
    mx.eval(ref, ref_pair, got, got_pair)
    print("compiled == shipped:      ",
          all(bool(mx.array_equal(a, c)) for a, c in zip(ref, got, strict=True)))
    print("compiled pair == shipped: ",
          all(bool(mx.array_equal(a, c)) for a, c in zip(ref_pair, got_pair, strict=True)))
    print()

    print("graph construction only (CPU side):")
    for name, fn, per_token in (("shipped, one pass", shipped, LAYERS),
                                ("mx.compile, one pass", compiled, LAYERS),
                                ("shipped, layer+predictor", shipped_pair, LAYERS),
                                ("mx.compile, layer+predictor", compiled_pair, LAYERS)):
        t = build_time(fn)
        print(f"  {name:30s} {t:7.4f} ms -> {t * per_token:6.2f} ms/token")

    print("\nchained, 40 launches in one graph (CPU + GPU):")
    for name, fn in (("shipped, layer+predictor", shipped_pair),
                     ("mx.compile, layer+predictor", compiled_pair)):
        lo, med = chained(fn)
        print(f"  {name:30s} min {lo:7.4f} ms  median {med:7.4f} ms -> {lo * LAYERS:6.2f} ms/token")


if __name__ == "__main__":
    main()
