"""Cost breakdown of routed_expert_forward_batched for M tokens: dequant vs matmul vs quantized_matmul alternative."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, load_expert_standalone  # noqa: E402
from cachalot.model.fp4_dequant_metal import dequantize_fp4_dense  # noqa: E402
from cachalot.model.moe_prefill_batched import routed_expert_forward_batched  # noqa: E402
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
    e = load_expert_standalone(ExpertReader(), index[(9, 42)])
    for M in (4, 13, 40):
        x = mx.random.normal((M, 5120)).astype(mx.bfloat16)
        mx.eval(x)
        w1 = dequantize_fp4_dense(e.w1_weight, e.w1_scale, 2304, 5120, dtype=mx.bfloat16)
        mx.eval(w1)
        d = t(lambda: dequantize_fp4_dense(e.w1_weight, e.w1_scale, 2304, 5120, dtype=mx.bfloat16))
        mm = t(lambda x=x, w1=w1: x @ w1.T)
        full = t(lambda x=x, M=M: routed_expert_forward_batched(x, w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight,
                                                                w2_scales=e.w2_scale, w3_packed=e.w3_weight, w3_scales=e.w3_scale,
                                                                weights=mx.ones((M,))))
        print(f"M={M:2d}: dequant one matrix {d:.3f} ms | bf16 matmul [M,5120]x[5120,2304] {mm:.3f} ms | full expert forward {full:.3f} ms")

    # alternative: exact affine-8bit representation for mx.quantized_matmul
    packed = np.array(e.w1_weight).reshape(2304, 2560)
    scales_e = np.array(e.w1_scale).reshape(2304, 160).astype(np.int32) - 127
    lo = packed & 0x0F
    hi = packed >> 4
    nib = np.stack([lo, hi], axis=-1).reshape(2304, 5120)
    table2 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int32)  # 2 * E2M1 value
    q = (table2[nib] + 12).astype(np.uint8)                       # 0..24
    e_rep = np.repeat(scales_e, 32, axis=1)
    scale_f = (0.5 * np.exp2(scales_e)).astype(np.float32)        # per group of 32
    bias_f = (-6.0 * np.exp2(scales_e)).astype(np.float32)
    w_q = mx.array(q.view(np.uint32).reshape(2304, 5120 // 4))     # 4 x 8-bit per uint32
    sc = mx.array(scale_f).astype(mx.bfloat16)
    bi = mx.array(bias_f).astype(mx.bfloat16)
    ref = np.array(dequantize_fp4_dense(e.w1_weight, e.w1_scale, 2304, 5120))
    deq = mx.dequantize(w_q, sc, bi, group_size=32, bits=8)
    mx.eval(deq)
    print("affine-8bit exact:", float(np.abs(np.array(deq.astype(mx.float32)) - ref).max()))
    for M in (4, 13, 40):
        x = mx.random.normal((M, 5120)).astype(mx.bfloat16)
        mx.eval(x)
        qm = t(lambda x=x: mx.quantized_matmul(x, w_q, sc, bi, transpose=True, group_size=32, bits=8))
        ref_out = x @ mx.array(ref).astype(mx.bfloat16).T  # noqa: B023
        out = mx.quantized_matmul(x, w_q, sc, bi, transpose=True, group_size=32, bits=8)
        mx.eval(ref_out, out)
        print(f"M={M:2d}: quantized_matmul 8-bit {qm:.3f} ms | max|diff| vs bf16 matmul {float(mx.abs(out.astype(mx.float32) - ref_out.astype(mx.float32)).max()):.3e}")
    _ = e_rep


if __name__ == "__main__":
    main()
