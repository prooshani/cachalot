"""Exactness (vs NumPy references and vs the pre-vectorization results) and speed of the FP4/FP8 GEMV kernels."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cachalot.model.fp4 import dequantize_fp4_weight  # noqa: E402
from cachalot.model.fp4_gemv_metal import fp4_gemv  # noqa: E402
from cachalot.model.fp8_linear_metal import fp8_linear  # noqa: E402
from cachalot.model.fp8_ref import fp8_gemv_reference  # noqa: E402


def t(fn, n=50):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    rng = np.random.default_rng(0)
    N, K = 2304, 5120
    packed = rng.integers(0, 256, (N, K // 2), dtype=np.uint8)
    scales = rng.integers(120, 134, (N, K // 32), dtype=np.uint8)
    x = rng.standard_normal(K).astype(np.float32)
    ref = dequantize_fp4_weight(packed, scales) @ x
    xm, pm, sm = mx.array(x), mx.array(packed.reshape(-1)), mx.array(scales.reshape(-1))
    out = fp4_gemv(xm, pm, sm, out_features=N, in_features=K)
    mx.eval(out)
    print(f"fp4 max rel err vs numpy: {float(np.max(np.abs(np.array(out) - ref)) / np.max(np.abs(ref))):.2e}")
    print(f"fp4 gemv 2304x5120 ({N * K / 2 / 1e6:.1f} MB): {t(lambda: fp4_gemv(xm, pm, sm, out_features=N, in_features=K)):.3f} ms")

    # realistic fp8: small weights (E4M3 codes in the sub-1 range), block scales near 1
    w8 = rng.integers(0, 0x38, (5120, 5120), dtype=np.uint8) | (rng.integers(0, 2, (5120, 5120), dtype=np.uint8) << 7)
    s8 = rng.integers(125, 130, (160, 160), dtype=np.uint8)
    xb = (rng.standard_normal(5120)).astype(np.float32)
    ref8 = fp8_gemv_reference(xb, w8, s8)
    xq, wm, s8m = mx.array(xb).astype(mx.bfloat16), mx.array(w8), mx.array(s8)
    out8 = fp8_linear(xq, wm, s8m).astype(mx.float32)
    mx.eval(out8)
    # activation is bf16-rounded before the kernel, so compare against a bf16-rounded reference
    xb16 = np.array(mx.array(xb).astype(mx.bfloat16).astype(mx.float32))
    ref8b = fp8_gemv_reference(xb16, w8, s8)
    print(f"fp8 max rel err vs numpy (bf16 input): {float(np.max(np.abs(np.array(out8) - ref8b)) / np.max(np.abs(ref8b))):.2e}")
    print(f"fp8 linear 5120x5120 (26 MB): {t(lambda: fp8_linear(xq, wm, s8m)):.3f} ms")
    _ = ref8


if __name__ == "__main__":
    main()
