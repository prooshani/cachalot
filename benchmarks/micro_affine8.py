"""Exactness and speed of the FP4 -> affine-8bit repack + quantized_matmul vs dense bf16 dequant + GEMM."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, load_expert_standalone  # noqa: E402
from cachalot.model.fp4_affine8_metal import fp4_affine8_matmul, fp4_to_affine8  # noqa: E402
from cachalot.model.fp4_dequant_metal import dequantize_fp4_dense  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader  # noqa: E402


def t(fn, n=30):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    index = build_expert_index(MODEL_PATH)
    for key in ((9, 42), (0, 0), (39, 383)):
        e = load_expert_standalone(ExpertReader(), index[key])
        for name, pk, sc, n, k in (("w1", e.w1_weight, e.w1_scale, 2304, 5120), ("w2", e.w2_weight, e.w2_scale, 5120, 2304)):
            dense = dequantize_fp4_dense(pk, sc, n, k, dtype=mx.float32)
            w_q, s, b = fp4_to_affine8(pk, sc, n, k)
            deq = mx.dequantize(w_q, s, b, group_size=32, bits=8).astype(mx.float32)
            mx.eval(dense, deq)
            print(f"expert {key} {name}: repack exact = {bool(mx.array_equal(dense, deq))}  (max|diff| {float(mx.abs(dense - deq).max()):.2e})")
    e = load_expert_standalone(ExpertReader(), index[(9, 42)])
    for M in (4, 13, 40, 128):
        x = mx.random.normal((M, 5120)).astype(mx.bfloat16)
        mx.eval(x)

        def dense_path(x=x):
            w = dequantize_fp4_dense(e.w1_weight, e.w1_scale, 2304, 5120, dtype=mx.bfloat16)
            return x @ w.T

        def affine_path(x=x):
            return fp4_affine8_matmul(x, e.w1_weight, e.w1_scale, out_features=2304, in_features=5120)

        a, bb = dense_path(), affine_path()
        mx.eval(a, bb)
        print(f"M={M:3d}: dense dequant+GEMM {t(dense_path):.3f} ms | affine8 repack+qmm {t(affine_path):.3f} ms | "
              f"max|diff| {float(mx.abs(a.astype(mx.float32) - bb.astype(mx.float32)).max()):.2e}")


if __name__ == "__main__":
    main()
