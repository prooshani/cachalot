from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from cachalot.model.fp8_linear_metal import (
    fp8_linear_quantized,
    quantize_fp8_activation,
)


def shared_expert_forward(
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
    """
    DeepSeek shared FP8 expert for single-token decode.

    Official semantics:

        gate = w1(x).float()
        up   = w3(x).float()

        up   = clamp(up, -limit, +limit)
        gate = clamp(gate, max=limit)

        h = silu(gate) * up

        return w2(h.to(original_dtype))

    w1 and w3 reuse one activation quantization.
    """

    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D for decode, got {x.shape}"
        )

    original_dtype = x.dtype

    # w1 and w3 consume the same activation.
    qx = quantize_fp8_activation(x)

    gate = fp8_linear_quantized(
        qx,
        w1,
        w1_scales,
    ).astype(mx.float32)

    up = fp8_linear_quantized(
        qx,
        w3,
        w3_scales,
    ).astype(mx.float32)

    if swiglu_limit > 0:
        up = mx.clip(
            up,
            -swiglu_limit,
            swiglu_limit,
        )

        gate = mx.minimum(
            gate,
            swiglu_limit,
        )

    hidden = (
        nn.silu(gate)
        * up
    )

    # Match:
    #
    #     self.w2(x.to(dtype))
    #
    hidden = hidden.astype(
        original_dtype
    )

    qhidden = quantize_fp8_activation(
        hidden
    )

    return fp8_linear_quantized(
        qhidden,
        w2,
        w2_scales,
    )
