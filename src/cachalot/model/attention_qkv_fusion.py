from __future__ import annotations

import mlx.core as mx

from cachalot.model.fp8_linear_metal import (
    fp8_linear_quantized,
    quantize_fp8_activation,
)

# wq_a and wkv both read x, the layer's hidden state, and neither depends on
# the other's output -- same shape as shared_expert_metal.py's w1/w3 fusion
# (HANDOFF section 9.27). Concatenated into one [1792, 5120] GEMV instead of
# two ([1280,5120] then [512,5120]), one activation quantization instead of
# two. HANDOFF section 7.1.10 named these the two worst-throughput shapes in
# the FP8 GEMV family (171 and 87 GB/s, against 383 for the largest) and
# pinned it on occupancy -- 512 and 1280 output rows launch too few
# simdgroups to fill the GPU; section 9.34 confirms the fused kernel reaches
# 221 GB/s against a 237 GB/s mx.sum ceiling on the same bytes, recovering
# 0.35 ms/token, bit-identical (benchmarks/micro_qkv_fusion_roofline.py).
#
# Same identity-keyed cache pattern as _fused_w13: the resident weight
# arrays never move once loaded, so id(wq_a) recurs on every call for a
# given layer.
_fused_wqkv_cache: dict[int, tuple[mx.array, mx.array, mx.array, mx.array]] = {}


def _fused_wqkv(
    wq_a: mx.array,
    wq_a_scales: mx.array,
    wkv: mx.array,
    wkv_scales: mx.array,
) -> tuple[mx.array, mx.array]:
    key = id(wq_a)
    cached = _fused_wqkv_cache.get(key)
    if cached is not None and cached[0] is wq_a and cached[1] is wkv:
        return cached[2], cached[3]

    wqkv = mx.concatenate([wq_a, wkv], axis=0)
    wqkv_scales = mx.concatenate([wq_a_scales, wkv_scales], axis=0)
    mx.eval(wqkv, wqkv_scales)

    _fused_wqkv_cache[key] = (wq_a, wkv, wqkv, wqkv_scales)
    return wqkv, wqkv_scales


def fused_qr_kv_linear(
    x: mx.array,
    wq_a: mx.array,
    wq_a_scales: mx.array,
    wkv: mx.array,
    wkv_scales: mx.array,
) -> tuple[mx.array, mx.array]:
    """
    Equivalent to:

        qr        = fp8_linear(x, wq_a, wq_a_scales)
        window_kv = fp8_linear(x, wkv, wkv_scales)

    issued as one GEMV. Bit-identical to the two-call form by construction:
    same activation quantization, same per-row block arithmetic, just more
    rows in one launch.
    """
    wqkv, wqkv_scales = _fused_wqkv(wq_a, wq_a_scales, wkv, wkv_scales)
    qx = quantize_fp8_activation(x)
    out = fp8_linear_quantized(qx, wqkv, wqkv_scales)

    rows = wq_a.shape[0]
    return out[:rows], out[rows:]
