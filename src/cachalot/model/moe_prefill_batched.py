"""
Batched building blocks for layer-major prefill.

Numerics follow the decode kernels element for element (FP4/FP8 values and
block scales are exact in fp32; activations quantized with the official
act_quant semantics); only the fp32 accumulation order inside the matmuls
differs from the single-vector Metal GEMVs.
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from cachalot.model.fp4_affine8_metal import fp4_affine8_matmul

HIDDEN = 5120
INTERMEDIATE = 2304
FP8_BLOCK = 32
FP8_MAX = 448.0
FP8_MIN_AMAX = 1e-4
# FP8 linears in prefill multiply dequantized E4M3 values (3 mantissa bits) by
# power-of-two block scales; every operand is exact in bf16, and a bf16 GEMM
# accumulates in fp32, so bf16 operands give the same products as the fp32
# path at half the bytes and several times the matrix throughput.
PREFILL_BF16_GEMM = os.environ.get("CACHALOT_PREFILL_BF16_GEMM", "1") != "0"


def routed_expert_forward_batched(
    x: mx.array,
    *,
    w1_packed: mx.array,
    w1_scales: mx.array,
    w2_packed: mx.array,
    w2_scales: mx.array,
    w3_packed: mx.array,
    w3_scales: mx.array,
    weights: mx.array,
    swiglu_limit: float = 10.0,
) -> mx.array:
    """
    x: [M, HIDDEN] (bf16 or fp32), weights: [M] router weights.
    Returns fp32 [M, HIDDEN] = W2 (silu(min(W1 x, L)) * clip(W3 x, -L, L) * w).
    """
    # Official expert linears return bf16 (fp32 accumulation inside the
    # GEMM), then the gating math runs in fp32. The FP4 weights are repacked
    # exactly into MLX's affine 8-bit layout and multiplied with
    # mx.quantized_matmul, avoiding a dense dequantization round trip.
    xb = x.astype(mx.bfloat16)
    gate = fp4_affine8_matmul(xb, w1_packed, w1_scales, out_features=INTERMEDIATE, in_features=HIDDEN).astype(mx.float32)
    up = fp4_affine8_matmul(xb, w3_packed, w3_scales, out_features=INTERMEDIATE, in_features=HIDDEN).astype(mx.float32)
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, mx.array(swiglu_limit, dtype=mx.float32))
    hidden = nn.silu(gate) * up
    hidden = hidden * weights.astype(mx.float32)[:, None]
    return fp4_affine8_matmul(
        hidden.astype(mx.bfloat16), w2_packed, w2_scales, out_features=HIDDEN, in_features=INTERMEDIATE
    ).astype(mx.float32)


def quantize_activation_fp8_rows(x: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
    """
    Official act_quant per 32-block, applied row-wise, returned *dequantized*
    (exact: E4M3 value x power-of-two scale, representable in bf16 and fp32).
    """
    xf = x.astype(mx.float32)
    m, k = xf.shape
    blocks = xf.reshape(m, k // FP8_BLOCK, FP8_BLOCK)
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=-1), mx.array(FP8_MIN_AMAX, dtype=mx.float32))
    scales = mx.power(mx.array(2.0, dtype=mx.float32), mx.ceil(mx.log2(amax / FP8_MAX)))
    normalized = mx.clip(blocks / scales[..., None], -FP8_MAX, FP8_MAX)
    q = mx.from_fp8(mx.to_fp8(normalized), dtype=mx.float32)
    return (q * scales[..., None]).reshape(m, k).astype(dtype)


def dequantize_fp8_weight(weight: mx.array, weight_scales: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
    """weight uint8 E4M3 [N, K]; weight_scales uint8 E8M0 [ceil(N/32), K/32] -> [N, K] (exact in bf16 or fp32)."""
    n, k = weight.shape
    values = mx.from_fp8(weight, dtype=dtype)
    scales = mx.power(mx.array(2.0, dtype=dtype), (weight_scales.astype(mx.int32) - 127).astype(dtype))
    sn, sk = weight_scales.shape
    values = values.reshape(sn, n // sn, sk, k // sk) * scales[:, None, :, None]
    return values.reshape(n, k)


def gemm_dtype() -> mx.Dtype:
    return mx.bfloat16 if PREFILL_BF16_GEMM else mx.float32


def fp8_linear_rows(x: mx.array, weight: mx.array, weight_scales: mx.array) -> mx.array:
    """x [T, K] -> bf16 [T, N] with official FP8 GEMM semantics (fp32 accumulation of exact products)."""
    dt = gemm_dtype()
    qx = quantize_activation_fp8_rows(x, dt)
    return (qx @ dequantize_fp8_weight(weight, weight_scales, dt).T).astype(mx.bfloat16)


def shared_expert_forward_batched(
    x: mx.array,
    *,
    w1: mx.array,
    w1_scales: mx.array,
    w2: mx.array,
    w2_scales: mx.array,
    w3: mx.array,
    w3_scales: mx.array,
    swiglu_limit: float = 10.0,
) -> mx.array:
    """x: [M, HIDDEN] bf16. Returns bf16 [M, HIDDEN] with the decode path's casts."""
    original_dtype = x.dtype
    dt = gemm_dtype()
    qx = quantize_activation_fp8_rows(x, dt)
    gate = qx @ dequantize_fp8_weight(w1, w1_scales, dt).T
    up = qx @ dequantize_fp8_weight(w3, w3_scales, dt).T
    # decode path: fp8_linear_quantized(...).astype(bf16).astype(fp32)
    gate = gate.astype(mx.bfloat16).astype(mx.float32)
    up = up.astype(mx.bfloat16).astype(mx.float32)
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, swiglu_limit)
    hidden = (nn.silu(gate) * up).astype(original_dtype)
    qh = quantize_activation_fp8_rows(hidden, dt)
    return (qh @ dequantize_fp8_weight(w2, w2_scales, dt).T).astype(mx.bfloat16)


def route_topk_rows(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    *,
    topk: int = 6,
    gate_temp: float = 1.0,
    route_scale: float = 1.5,
    norm_topk_prob: bool = True,
):
    """Row-wise version of router_mlx.route_topk with identical per-row math."""
    xf = x.astype(mx.float32)
    wf = weight.astype(mx.float32)
    bf = bias.astype(mx.float32)
    scores = (xf @ wf.T) / gate_temp
    scores = mx.sqrt(nn.softplus(scores))
    order = mx.argsort(scores + bf, axis=-1)
    indices = order[:, -topk:]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if norm_topk_prob and topk > 1:
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    weights = weights * route_scale
    return indices, weights, scores
