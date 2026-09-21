"""
Screen: what does mx.compile take off the CPU cost of the *fused Metal* glue?

`benchmarks/micro_compile_hc.py` already asked this question, but it asked it of
`hyper_connection_mlx`, and the runtime has been on the single-launch Metal
kernels in `decode_fused_metal` since before that script was written. Section
6.3.1 withdrew a number for exactly that reason, so this screen re-asks it on
the path the runtime takes.

Per decoded token the runtime issues the glue 240 times: `hc_mixes_1d`,
`hc_pre_norm_1d` and `hc_post_1d`, twice each (attention and FFN) on each of
forty layers. Each of those calls is one GPU dispatch, so there is nothing left
to fuse on the GPU side; what is left is the Python that builds the call. Every
call allocates two or three scalar `mx.array` parameter buffers, assembles an
input list and reshapes the output.

Three arms, all producing identical bits:

  fused          the runtime's path as it ships
  cached consts  the same, with the scalar parameter arrays memoised per shape
  mx.compile     the traced graph, which pays construction once

Construction is timed with no eval inside the loop (the CPU side, which is what
section 9.15 is about); chained timing runs many launches inside one eval.

    PYTHONPATH=src python benchmarks/micro_compile_hc_fused.py
"""
from __future__ import annotations

import statistics
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cachalot.model.decode_fused_metal import (  # noqa: E402
    _hc_mixes_kernel,
    _hc_post_kernel,
    _hc_pre_norm_kernel,
    hc_mixes_1d,
    hc_post_1d,
    hc_pre_norm_1d,
)

LAYERS = 40
SUBLAYERS = 2 * LAYERS          # attention and FFN on every layer
HIDDEN = 5120
HC = 4
SINKHORN = 20
NORM_EPS = 1e-20
HC_EPS = 1e-6


# ---------------------------------------------------------------------------
# arm 2: the same kernels with their scalar parameter arrays memoised
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _u32(value: int) -> mx.array:
    a = mx.array([value], dtype=mx.uint32)
    mx.eval(a)
    return a


@lru_cache(maxsize=64)
def _f32(value: float) -> mx.array:
    a = mx.array([value], dtype=mx.float32)
    mx.eval(a)
    return a


def hc_mixes_cached(x, hc_fn, hc_scale, hc_base, *, norm_eps, hc_mult, sinkhorn_iters, hc_eps):
    n = x.size
    return _hc_mixes_kernel(hc_mult, sinkhorn_iters)(
        inputs=[x.reshape(-1), hc_fn, hc_scale, hc_base, _u32(n), _f32(norm_eps), _f32(hc_eps)],
        template=[],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(hc_mult,), (hc_mult,), (hc_mult, hc_mult)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


def hc_pre_norm_cached(x, pre_mix, weight, *, eps):
    hc, n = x.shape
    return _hc_pre_norm_kernel(hc, True)(
        inputs=[x, pre_mix, weight, _u32(n), _f32(eps)],
        template=[("T", x.dtype)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
    )[0]


def hc_post_cached(y, residual, post, comb):
    hc, n = residual.shape
    return _hc_post_kernel(hc)(
        inputs=[y, residual, post, comb.reshape(-1), _u32(n)],
        template=[("T", residual.dtype)],
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(hc * n,)],
        output_dtypes=[residual.dtype],
    )[0].reshape(hc, n)


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
    x = mx.random.normal((HC, HIDDEN)).astype(mx.bfloat16)
    hc_fn = (mx.random.normal(((2 + HC) * HC, HC * HIDDEN)) * 0.01).astype(mx.bfloat16)
    hc_scale = mx.array([0.1, 0.1, 0.1], dtype=mx.float32)
    hc_base = mx.random.normal(((2 + HC) * HC,)).astype(mx.float32)
    norm_w = mx.ones((HIDDEN,)).astype(mx.bfloat16)
    pre_mix = mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.float32)
    y = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    mx.eval(x, hc_fn, hc_scale, hc_base, norm_w, pre_mix, y)

    # the two coefficient vectors a real hc_post is handed
    pre0, post0, comb0 = hc_mixes_1d(x, hc_fn, hc_scale, hc_base, norm_eps=NORM_EPS,
                                     hc_mult=HC, sinkhorn_iters=SINKHORN, hc_eps=HC_EPS)
    mx.eval(pre0, post0, comb0)

    # ---------------- arm 1: the shipped fused path ----------------
    def shipped_pre():
        pre, post, comb = hc_mixes_1d(x, hc_fn, hc_scale, hc_base, norm_eps=NORM_EPS,
                                      hc_mult=HC, sinkhorn_iters=SINKHORN, hc_eps=HC_EPS)
        inp = hc_pre_norm_1d(x, pre_mix, norm_w, eps=NORM_EPS)
        return [inp, pre, post, comb]

    def shipped_post():
        return [hc_post_1d(y, x, post0, comb0)]

    # ---------------- arm 2: memoised scalar parameter arrays ----------------
    def cached_pre():
        pre, post, comb = hc_mixes_cached(x, hc_fn, hc_scale, hc_base, norm_eps=NORM_EPS,
                                          hc_mult=HC, sinkhorn_iters=SINKHORN, hc_eps=HC_EPS)
        inp = hc_pre_norm_cached(x, pre_mix, norm_w, eps=NORM_EPS)
        return [inp, pre, post, comb]

    def cached_post():
        return [hc_post_cached(y, x, post0, comb0)]

    # ---------------- arm 3: mx.compile ----------------
    def glue_pre(xin, fn_w, scale, base, mix, w):
        pre, post, comb = hc_mixes_1d(xin, fn_w, scale, base, norm_eps=NORM_EPS,
                                      hc_mult=HC, sinkhorn_iters=SINKHORN, hc_eps=HC_EPS)
        inp = hc_pre_norm_1d(xin, mix, w, eps=NORM_EPS)
        return [inp, pre, post, comb]

    def glue_post(yin, res, post, comb):
        return [hc_post_1d(yin, res, post, comb)]

    c_pre = mx.compile(glue_pre)
    c_post = mx.compile(glue_post)

    def compiled_pre():
        return c_pre(x, hc_fn, hc_scale, hc_base, pre_mix, norm_w)

    def compiled_post():
        return c_post(y, x, post0, comb0)

    # ---------------- parity ----------------
    ref_pre, ref_post = shipped_pre(), shipped_post()
    for label, arm_pre, arm_post in (("cached consts", cached_pre, cached_post),
                                     ("mx.compile", compiled_pre, compiled_post)):
        got_pre, got_post = arm_pre(), arm_post()
        mx.eval(ref_pre, ref_post, got_pre, got_post)
        ok_pre = all(bool(mx.array_equal(a, b)) for a, b in zip(ref_pre, got_pre, strict=True))
        ok_post = bool(mx.array_equal(ref_post[0], got_post[0]))
        print(f"{label:14s} bit-identical: mixes+pre_norm {ok_pre}  hc_post {ok_post}")
    print()

    arms = (
        ("fused, as shipped", shipped_pre, shipped_post),
        ("cached consts", cached_pre, cached_post),
        ("mx.compile", compiled_pre, compiled_post),
    )

    print(f"graph construction only, per sublayer (CPU side), x{SUBLAYERS} per token:")
    base = None
    for name, arm_pre, arm_post in arms:
        b = build_time(arm_pre) + build_time(arm_post)
        if base is None:
            base = b
        print(f"  {name:20s} {b:7.4f} ms -> {b * SUBLAYERS:6.2f} ms/token"
              f"   ({(base - b) * SUBLAYERS:+6.2f} ms/token)")

    print(f"\nchained, 40 launches in one graph (CPU + GPU), x{SUBLAYERS} per token:")
    for name, arm_pre, arm_post in arms:
        lo_a, med_a = chained(arm_pre)
        lo_b, med_b = chained(arm_post)
        print(f"  {name:20s} min {lo_a + lo_b:7.4f} ms  median {med_a + med_b:7.4f} ms"
              f" -> {(lo_a + lo_b) * SUBLAYERS:6.2f} ms/token")


if __name__ == "__main__":
    main()
