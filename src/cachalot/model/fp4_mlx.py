from __future__ import annotations

import mlx.core as mx

FP4_E2M1_TABLE = mx.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=mx.float32,
)


def decode_e8m0(raw: mx.array) -> mx.array:
    exponent = raw.astype(mx.int32) - 127
    return mx.power(
        mx.array(2.0, dtype=mx.float32),
        exponent,
    )


def unpack_fp4_e2m1(packed: mx.array) -> mx.array:
    packed = packed.astype(mx.uint8)

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    low_values = FP4_E2M1_TABLE[low.astype(mx.int32)]
    high_values = FP4_E2M1_TABLE[high.astype(mx.int32)]

    stacked = mx.stack(
        [low_values, high_values],
        axis=-1,
    )

    shape = packed.shape[:-1] + (packed.shape[-1] * 2,)

    return stacked.reshape(shape)


def dequantize_fp4_weight(
    packed: mx.array,
    scales: mx.array,
    block_size: int = 32,
) -> mx.array:
    unpacked = unpack_fp4_e2m1(packed)

    n, k = unpacked.shape

    if k % block_size != 0:
        raise ValueError(
            f"K={k} must be divisible by block_size={block_size}"
        )

    expected_scale_shape = (
        n,
        k // block_size,
    )

    if scales.shape != expected_scale_shape:
        raise ValueError(
            f"Scale shape {scales.shape} does not match "
            f"expected {expected_scale_shape}"
        )

    decoded_scales = decode_e8m0(scales)

    return (
        unpacked.reshape(n, -1, block_size)
        * decoded_scales[..., None]
    ).reshape(n, k)
