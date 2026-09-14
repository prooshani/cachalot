from __future__ import annotations

import mlx.core as mx


FP8_BLOCK_SIZE = 32
FP8_MAX = 448.0
FP8_MIN_AMAX = 1e-4


def quantize_activation_fp8_mlx(
    x: mx.array,
    block_size: int = FP8_BLOCK_SIZE,
) -> tuple[mx.array, mx.array]:
    """
    DeepSeek V4.1 activation quantization for FP8 linear layers.

    Returns:
        q:
            uint8 E4M3 bytes, same logical shape as x.

        scales:
            float32 power-of-two scales, one per 32 values.

    Semantics match inference/kernel.py:

        amax = max(abs(block), 1e-4)
        scale = 2^ceil(log2(amax / 448))
        q = E4M3(clamp(x / scale, -448, 448))
    """
    if x.ndim != 1:
        raise ValueError(
            f"Expected 1D activation, got shape {x.shape}"
        )

    if x.size % block_size != 0:
        raise ValueError(
            f"x.size={x.size} must be divisible "
            f"by block_size={block_size}"
        )

    xf = x.astype(mx.float32)

    blocks = xf.reshape(
        (-1, block_size)
    )

    amax = mx.max(
        mx.abs(blocks),
        axis=1,
    )

    amax = mx.maximum(
        amax,
        mx.array(
            FP8_MIN_AMAX,
            dtype=mx.float32,
        ),
    )

    exponent = mx.ceil(
        mx.log2(
            amax / FP8_MAX
        )
    )

    scales = mx.power(
        mx.array(2.0, dtype=mx.float32),
        exponent,
    )

    normalized = mx.clip(
        blocks / scales[:, None],
        -FP8_MAX,
        FP8_MAX,
    )

    q = mx.to_fp8(
        normalized
    ).reshape(x.shape)

    return q, scales


def fp8_roundtrip_activation_mlx(
    x: mx.array,
    block_size: int = FP8_BLOCK_SIZE,
) -> mx.array:
    """
    Equivalent to DeepSeek:

        act_quant(
            x,
            block_size=32,
            scale_fmt="ue8m0",
            scale_dtype=E8M0,
            inplace=True,
        )

    The official kernel:
        1. computes power-of-two scale per block,
        2. quantizes to E4M3,
        3. immediately dequantizes,
        4. writes the rounded result back in x's dtype.

    The scale tensor itself is discarded by Attention._window_kv().
    """
    dtype = x.dtype

    q, scales = quantize_activation_fp8_mlx(
        x,
        block_size=block_size,
    )

    dequant = (
        mx.from_fp8(q)
        .reshape((-1, block_size))
        .astype(mx.float32)
        * scales[:, None]
    )

    return dequant.reshape(x.shape).astype(dtype)
