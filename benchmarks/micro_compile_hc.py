"""Does mx.compile shrink the hyper-connection glue (hc_mixes / hc_pre+rms_norm / hc_post)?"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cachalot.model.hyper_connection_mlx import hc_mixes, hc_post, hc_pre  # noqa: E402
from cachalot.model.norm_rope_mlx import rms_norm  # noqa: E402


def timeit(fn, n=60):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    mx.random.seed(0)
    x = mx.random.normal((4, 5120)).astype(mx.bfloat16)
    hc_fn = mx.random.normal((24, 4 * 5120)).astype(mx.bfloat16) * 0.01
    hc_scale = mx.array([0.1, 0.1, 0.1])
    hc_base = mx.random.normal((24,))
    norm_w = mx.ones((5120,)).astype(mx.bfloat16)
    pre_mix = mx.array([1.0, 0, 0, 0])
    y = mx.random.normal((5120,)).astype(mx.bfloat16)
    post = mx.ones((4,))
    comb = mx.ones((4, 4)) / 4
    mx.eval(x, hc_fn, hc_scale, hc_base, norm_w, pre_mix, y, post, comb)

    mixes = partial(hc_mixes, norm_eps=1e-20, hc_mult=4, sinkhorn_iters=20, hc_eps=1e-6)

    def pre_norm(x, pre_mix, w):
        return rms_norm(hc_pre(x, pre_mix), w, eps=1e-20)

    c_mixes = mx.compile(mixes)
    c_pre_norm = mx.compile(pre_norm)
    c_post = mx.compile(hc_post)

    a = mixes(x, hc_fn, hc_scale, hc_base)
    b = c_mixes(x, hc_fn, hc_scale, hc_base)
    mx.eval(*a, *b)
    print("hc_mixes compiled == eager:", all(bool(mx.array_equal(u, v)) for u, v in zip(a, b, strict=True)))
    print("pre_norm compiled == eager:", bool(mx.array_equal(pre_norm(x, pre_mix, norm_w), c_pre_norm(x, pre_mix, norm_w))))
    print("hc_post compiled == eager:", bool(mx.array_equal(hc_post(y, x, post, comb), c_post(y, x, post, comb))))

    print(f"hc_mixes   eager {timeit(lambda: mixes(x, hc_fn, hc_scale, hc_base)[2]):.3f} ms  compiled {timeit(lambda: c_mixes(x, hc_fn, hc_scale, hc_base)[2]):.3f} ms")
    print(f"pre+norm   eager {timeit(lambda: pre_norm(x, pre_mix, norm_w)):.3f} ms  compiled {timeit(lambda: c_pre_norm(x, pre_mix, norm_w)):.3f} ms")
    print(f"hc_post    eager {timeit(lambda: hc_post(y, x, post, comb)):.3f} ms  compiled {timeit(lambda: c_post(y, x, post, comb)):.3f} ms")

    # whole glue chain per sub-layer in one compiled function
    def glue(x, hc_fn, hc_scale, hc_base, pre_mix, w):
        pre, post, comb = mixes(x, hc_fn, hc_scale, hc_base)
        return rms_norm(hc_pre(x, pre_mix), w, eps=1e-20), pre, post, comb

    c_glue = mx.compile(glue)
    print(f"glue chain eager {timeit(lambda: glue(x, hc_fn, hc_scale, hc_base, pre_mix, norm_w)[0]):.3f} ms  compiled {timeit(lambda: c_glue(x, hc_fn, hc_scale, hc_base, pre_mix, norm_w)[0]):.3f} ms")


if __name__ == "__main__":
    main()
