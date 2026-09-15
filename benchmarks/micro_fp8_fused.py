"""Fused FP8 quantize+GEMV vs unfused chain on real layer weights: exactness and speed."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.fp8_fused_metal import fp8_linear_fused  # noqa: E402
from cachalot.model.fp8_linear_metal import fp8_linear  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402


def t(fn, n=40):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        kw = rt._common_block_kwargs(5, compressed=True)
        mx.random.seed(0)
        cases = [
            ("wq_a  [1280,5120]", kw["wq_a"], kw["wq_a_scales"], 5120, 0.7),
            ("wq_b  [32768,1280]", kw["wq_b"], kw["wq_b_scales"], 1280, 0.4),
            ("wkv   [?,5120]", kw["wkv"], kw["wkv_scales"], 5120, 0.7),
            ("wo_b  [5120,8192]", kw["wo_b"], kw["wo_b_scales"], 8192, 0.05),
            ("shared_w1 [2304,5120]", kw["shared_w1"], kw["shared_w1_scales"], 5120, 0.7),
        ]
        total_unfused = total_fused = 0.0
        for name, w, s, k, std in cases:
            x = (mx.random.normal((k,)) * std).astype(mx.bfloat16)
            mx.eval(x)
            a = fp8_linear(x, w, s)
            b = fp8_linear_fused(x, w, s)
            mx.eval(a, b)
            tu, tf = t(lambda x=x, w=w, s=s: fp8_linear(x, w, s)), t(lambda x=x, w=w, s=s: fp8_linear_fused(x, w, s))
            total_unfused += tu
            total_fused += tf
            print(f"{name:24s} shape {tuple(w.shape)} | bit-identical {bool(mx.array_equal(a, b))} "
                  f"max|diff| {float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max()):.2e} | unfused {tu:.3f} ms fused {tf:.3f} ms")
        print(f"sum over 5 linears: unfused {total_unfused:.3f} ms, fused {total_fused:.3f} ms")


if __name__ == "__main__":
    main()
