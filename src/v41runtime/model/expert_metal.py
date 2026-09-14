from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from v41runtime.model.fp4_gemv_metal import fp4_gemv


def routed_expert_forward(
    x: mx.array,
    *,
    w1_packed: mx.array,
    w1_scales: mx.array,
    w2_packed: mx.array,
    w2_scales: mx.array,
    w3_packed: mx.array,
    w3_scales: mx.array,
    weight: mx.array | float | None = None,
    swiglu_limit: float = 10.0,
) -> mx.array:
    gate = fp4_gemv(
        x,
        w1_packed,
        w1_scales,
        out_features=2304,
        in_features=5120,
    )

    up = fp4_gemv(
        x,
        w3_packed,
        w3_scales,
        out_features=2304,
        in_features=5120,
    )

    if swiglu_limit > 0:
        # Matches official DeepSeek implementation:
        # up   -> clamp on both sides
        # gate -> clamp only from above
        up = mx.clip(
            up,
            -swiglu_limit,
            swiglu_limit,
        )

        gate = mx.minimum(
            gate,
            mx.array(swiglu_limit, dtype=gate.dtype),
        )

    hidden = nn.silu(gate) * up

    # DeepSeek applies the router weight before w2.
    if weight is not None:
        hidden = hidden * weight

    return fp4_gemv(
        hidden,
        w2_packed,
        w2_scales,
        out_features=5120,
        in_features=2304,
    )
