"""
Batched building blocks for layer-major prefill.

Numerics follow the decode kernels element for element (FP4/FP8 values and
block scales are exact in fp32; activations quantized with the official
act_quant semantics); only the fp32 accumulation order inside the matmuls
differs from the single-vector Metal GEMVs.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from cachalot.model.fp4_dequant_metal import dequantize_fp4_dense

HIDDEN = 5120
INTERMEDIATE = 2304
FP8_BLOCK = 32
FP8_MAX = 448.0
FP8_MIN_AMAX = 1e-4


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
    xf = x.astype(mx.float32)
    w1 = dequantize_fp4_dense(w1_packed, w1_scales, INTERMEDIATE, HIDDEN)
    w3 = dequantize_fp4_dense(w3_packed, w3_scales, INTERMEDIATE, HIDDEN)
    gate = xf @ w1.T
    up = xf @ w3.T
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, mx.array(swiglu_limit, dtype=mx.float32))
    hidden = nn.silu(gate) * up
    hidden = hidden * weights.astype(mx.float32)[:, None]
    w2 = dequantize_fp4_dense(w2_packed, w2_scales, HIDDEN, INTERMEDIATE)
    return hidden @ w2.T


def quantize_activation_fp8_rows(x: mx.array) -> mx.array:
    """
    Official act_quant per 32-block, applied row-wise, returned *dequantized*
    in fp32 (exact: E4M3 value x power-of-two scale).
    """
    xf = x.astype(mx.float32)
    m, k = xf.shape
    blocks = xf.reshape(m, k // FP8_BLOCK, FP8_BLOCK)
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=-1), mx.array(FP8_MIN_AMAX, dtype=mx.float32))
    scales = mx.power(mx.array(2.0, dtype=mx.float32), mx.ceil(mx.log2(amax / FP8_MAX)))
    normalized = mx.clip(blocks / scales[..., None], -FP8_MAX, FP8_MAX)
    q = mx.from_fp8(mx.to_fp8(normalized), dtype=mx.float32)
    return (q * scales[..., None]).reshape(m, k)


def dequantize_fp8_weight(weight: mx.array, weight_scales: mx.array) -> mx.array:
    """weight uint8 E4M3 [N, K]; weight_scales uint8 E8M0 [ceil(N/32), K/32] -> fp32 [N, K]."""
    n, k = weight.shape
    values = mx.from_fp8(weight, dtype=mx.float32)
    scales = mx.power(mx.array(2.0, dtype=mx.float32), weight_scales.astype(mx.int32) - 127)
    sn, sk = weight_scales.shape
    values = values.reshape(sn, n // sn, sk, k // sk) * scales[:, None, :, None]
    return values.reshape(n, k)


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
    qx = quantize_activation_fp8_rows(x)
    gate = qx @ dequantize_fp8_weight(w1, w1_scales).T
    up = qx @ dequantize_fp8_weight(w3, w3_scales).T
    # decode path: fp8_linear_quantized(...).astype(bf16).astype(fp32)
    gate = gate.astype(mx.bfloat16).astype(mx.float32)
    up = up.astype(mx.bfloat16).astype(mx.float32)
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, swiglu_limit)
    hidden = (nn.silu(gate) * up).astype(original_dtype)
    qh = quantize_activation_fp8_rows(hidden)
    return (qh @ dequantize_fp8_weight(w2, w2_scales).T).astype(mx.bfloat16)


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
