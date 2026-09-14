from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from v41runtime.model.fp8_act_mlx import (
    quantize_activation_fp8_mlx,
)
from v41runtime.model.fp8_gemv_metal import (
    fp8_gemv_quantized,
)


@dataclass(frozen=True)
class QuantizedActivation:
    values: mx.array
    scales: mx.array
    in_features: int


def quantize_fp8_activation(
    x: mx.array,
) -> QuantizedActivation:
    if x.ndim != 1:
        raise ValueError(
            f"Decode FP8 activation must be 1D, "
            f"got {x.shape}"
        )

    values, scales = (
        quantize_activation_fp8_mlx(x)
    )

    return QuantizedActivation(
        values=values,
        scales=scales,
        in_features=x.size,
    )


def fp8_linear_quantized(
    x: QuantizedActivation,
    weight: mx.array,
    weight_scales: mx.array,
) -> mx.array:
    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2D, got {weight.shape}"
        )

    out_features, in_features = weight.shape

    if in_features != x.in_features:
        raise ValueError(
            f"weight expects {in_features} inputs, "
            f"quantized activation has {x.in_features}"
        )

    return fp8_gemv_quantized(
        x.values,
        x.scales,
        weight,
        weight_scales,
        out_features=out_features,
        in_features=in_features,
    ).astype(mx.bfloat16)


def fp8_linear(
    x: mx.array,
    weight: mx.array,
    weight_scales: mx.array,
) -> mx.array:
    """
    Convenience path when the activation is used by only one
    FP8 projection.
    """
    qx = quantize_fp8_activation(x)

    return fp8_linear_quantized(
        qx,
        weight,
        weight_scales,
    )
