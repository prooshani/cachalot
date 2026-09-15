"""Vectorized FP8 GEMV (pre-decoded activation) vs the scalar kernel on real layer weights: accuracy and GPU time."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.fp8_fused_metal import (  # noqa: E402
    fp8_gemv_decoded,
    quantize_activation_fp8_fused,
)
from cachalot.model.fp8_gemv_metal import fp8_gemv_quantized  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402


def chained(fn, n=40):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    outs = [fn() for _ in range(n)]
    mx.eval(*outs)
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        kw = rt._common_block_kwargs(5, compressed=True)
        mx.random.seed(0)
        for wname, sname in (("wq_a", "wq_a_scales"), ("wq_b", "wq_b_scales"), ("wkv", "wkv_scales"), ("wo_b", "wo_b_scales"),
                             ("shared_w1", "shared_w1_scales"), ("shared_w2", "shared_w2_scales")):
            w, sc = kw[wname], kw[sname]
            n, k = w.shape
            x = (mx.random.normal((k,)) * 0.7).astype(mx.bfloat16)
            q, scales, deq = quantize_activation_fp8_fused(x)
            ref = fp8_gemv_quantized(q, scales, w, sc, out_features=n, in_features=k)
            out = fp8_gemv_decoded(deq, scales, w, sc)
            mx.eval(ref, out)
            rel = float(mx.abs(ref - out).max() / (mx.abs(ref).max() + 1e-30))
            same_bf16 = bool(mx.array_equal(ref.astype(mx.bfloat16), out.astype(mx.bfloat16)))
            t_old = chained(lambda q=q, scales=scales, w=w, sc=sc, n=n, k=k: fp8_gemv_quantized(q, scales, w, sc, out_features=n, in_features=k))
            t_new = chained(lambda deq=deq, scales=scales, w=w, sc=sc: fp8_gemv_decoded(deq, scales, w, sc))
            gbs = w.nbytes / (t_new * 1e-3) / 1e9
            print(f"{wname:10s} [{n:6d},{k:5d}] {w.nbytes / 1e6:5.1f} MB | old {t_old:6.3f} ms  new {t_new:6.3f} ms ({gbs:4.0f} GB/s) | "
                  f"max rel diff {rel:.1e} bf16-identical {same_bf16}")


if __name__ == "__main__":
    main()
