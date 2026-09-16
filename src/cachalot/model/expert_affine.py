"""
Routed-expert math for an affine-quantized expert bank (storage.index.ExpertFormat
kind "affine"), e.g. oMLX's oQ3e conversion: 3-bit weights, group 64, fp16
scales and biases.

Slot bytes are viewed in place (no copy) as MLX quantized matrices and
multiplied with mx.quantized_matmul. Numerics follow the official routed
expert: bf16 activations in, bf16 linear outputs, fp32 gating, router weight
applied before the down projection. With bf16 activations and fp16 scales MLX
returns fp32 products (measured the most accurate combination).
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from cachalot.storage.index import ExpertFormat

_DTYPES = {"uint32": mx.uint32, "float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32, "uint8": mx.uint8}


def affine_views(arrays: dict[str, mx.array], fmt: ExpertFormat, proj: str) -> tuple[mx.array, mx.array, mx.array]:
    """(weight, scales, biases) views of one projection's slot bytes."""
    out = []
    for field in ("weight", "scales", "biases"):
        short = f"{proj}.{field}"
        out.append(arrays[short].view(_DTYPES[fmt.dtypes[short]]).reshape(fmt.shapes[short]))
    return out[0], out[1], out[2]


def _qmm(x: mx.array, arrays: dict[str, mx.array], fmt: ExpertFormat, proj: str) -> mx.array:
    w, s, b = affine_views(arrays, fmt, proj)
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=fmt.group_size, bits=fmt.bits)


def affine_expert_forward(
    x: mx.array,
    arrays: dict[str, mx.array],
    fmt: ExpertFormat,
    weight: float,
    swiglu_limit: float = 10.0,
) -> mx.array:
    """x: [HIDDEN] -> fp32 [HIDDEN] = W2 (silu(min(W1 x, L)) * clip(W3 x, -L, L) * weight)."""
    xb = x.astype(mx.bfloat16)
    gate = _qmm(xb, arrays, fmt, "w1").astype(mx.bfloat16).astype(mx.float32)
    up = _qmm(xb, arrays, fmt, "w3").astype(mx.bfloat16).astype(mx.float32)
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, mx.array(swiglu_limit, dtype=mx.float32))
    hidden = (nn.silu(gate) * up * weight).astype(mx.bfloat16)
    return _qmm(hidden, arrays, fmt, "w2").astype(mx.bfloat16).astype(mx.float32)


def affine_expert_forward_batched(
    x: mx.array,
    arrays: dict[str, mx.array],
    fmt: ExpertFormat,
    weights: mx.array,
    swiglu_limit: float = 10.0,
) -> mx.array:
    """x: [M, HIDDEN], weights: [M] -> fp32 [M, HIDDEN]; same contract as routed_expert_forward_batched."""
    xb = x.astype(mx.bfloat16)
    gate = _qmm(xb, arrays, fmt, "w1").astype(mx.bfloat16).astype(mx.float32)
    up = _qmm(xb, arrays, fmt, "w3").astype(mx.bfloat16).astype(mx.float32)
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, mx.array(swiglu_limit, dtype=mx.float32))
    hidden = (nn.silu(gate) * up * weights.astype(mx.float32)[:, None]).astype(mx.bfloat16)
    return _qmm(hidden, arrays, fmt, "w2").astype(mx.bfloat16).astype(mx.float32)
