from __future__ import annotations

import mlx.core as mx


FP4_MAX = 6.0

# Official kernel.py minima:
#
# E8M0:
#   amax = max(amax, 6 * 2^-126)
#
# E4M3:
#   amax = max(amax, 6 * 2^-9)
#
FP4_E8M0_MIN_AMAX = FP4_MAX * (2.0 ** -126)
FP4_E4M3_MIN_AMAX = FP4_MAX * (2.0 ** -9)


def _round_fp4_e2m1(
    x: mx.array,
) -> mx.array:
    """
    Round float32 values already clamped to [-6, 6]
    to FP4 E2M1FN, returning the dequantized float32 value.

    Representable magnitudes:

        0, 0.5, 1, 1.5, 2, 3, 4, 6

    Midpoint behavior matches ml_dtypes.float4_e2m1fn /
    the official FP4 cast: round-to-nearest-even.

        0.25 -> 0
        0.75 -> 1
        1.25 -> 1
        1.75 -> 2
        2.50 -> 2
        3.50 -> 4
        5.00 -> 4

    Signed zero is preserved by using x * 0.0 for the
    zero bucket rather than constructing a fresh +0.
    """
    xf = x.astype(mx.float32)
    ax = mx.abs(xf)

    # Piecewise nearest-E2M1 with ties-to-even.
    #
    # Pay attention to <= versus < at each midpoint:
    # that is what implements the alternating tie direction.
    mag = mx.where(
        ax <= 0.25,
        0.0,
        mx.where(
            ax < 0.75,
            0.5,
            mx.where(
                ax <= 1.25,
                1.0,
                mx.where(
                    ax < 1.75,
                    1.5,
                    mx.where(
                        ax <= 2.5,
                        2.0,
                        mx.where(
                            ax < 3.5,
                            3.0,
                            mx.where(
                                ax <= 5.0,
                                4.0,
                                6.0,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(mx.float32)

    signed_nonzero = mx.where(
        xf < 0.0,
        -mag,
        mag,
    )

    # Preserve -0 when FP4 rounds a negative tiny value,
    # and also preserve an original -0 input.
    signed_zero = xf * mx.array(
        0.0,
        dtype=mx.float32,
    )

    return mx.where(
        mag == 0.0,
        signed_zero,
        signed_nonzero,
    )


def fp4_act_roundtrip_mlx(
    x: mx.array,
    *,
    block_size: int = 32,
    scale_dtype: str = "e8m0",
) -> mx.array:
    """
    Equivalent to official:

        fp4_act_quant(
            x,
            block_size,
            inplace=True,
            scale_dtype=...
        )

    This function implements only the inplace path used by
    DeepSeek V4.1 Flash attention.

    scale_dtype:

        "e8m0"
            Indexer Q/K:
                block_size = 32
                E8M0 power-of-two scales.

        "e4m3"
            Compressed KV:
                block_size = 16
                E4M3FN scales.

    Official operation per block:

        scale = quantized_scale(amax / 6)
        q     = FP4_E2M1(clamp(x / scale, -6, 6))
        y     = q * scale

    y is written back in x.dtype.
    """
    if x.ndim < 1:
        raise ValueError(
            f"x must have at least one dimension, got {x.shape}"
        )

    n = x.shape[-1]

    if n % block_size != 0:
        raise ValueError(
            f"last dimension {n} must be divisible "
            f"by block_size={block_size}"
        )

    if scale_dtype not in {
        "e8m0",
        "e4m3",
    }:
        raise ValueError(
            f"Unsupported scale_dtype={scale_dtype!r}; "
            "expected 'e8m0' or 'e4m3'"
        )

    dtype = x.dtype

    xf = x.astype(mx.float32)

    blocks = xf.reshape(
        (-1, block_size)
    )

    amax = mx.max(
        mx.abs(blocks),
        axis=1,
    )

    if scale_dtype == "e8m0":
        #
        # Official:
        #
        #   amax = max(amax, 6 * 2^-126)
        #   scale = 2^ceil(log2(amax / 6))
        #
        amax = mx.maximum(
            amax,
            mx.array(
                FP4_E8M0_MIN_AMAX,
                dtype=mx.float32,
            ),
        )

        exponent = mx.ceil(
            mx.log2(
                amax / FP4_MAX
            )
        )

        scales = mx.power(
            mx.array(
                2.0,
                dtype=mx.float32,
            ),
            exponent,
        )

    else:
        #
        # Official:
        #
        #   amax = max(amax, 6 * 2^-9)
        #   scale = E4M3(amax / 6)
        #
        # mx.to_fp8/from_fp8 is E4M3FN and is already
        # validated elsewhere in this runtime.
        #
        amax = mx.maximum(
            amax,
            mx.array(
                FP4_E4M3_MIN_AMAX,
                dtype=mx.float32,
            ),
        )

        raw_scale = (
            amax / FP4_MAX
        ).astype(mx.float32)

        scales = mx.from_fp8(
            mx.to_fp8(
                raw_scale
            )
        ).astype(mx.float32)

    normalized = mx.clip(
        blocks / scales[:, None],
        -FP4_MAX,
        FP4_MAX,
    )

    q_dequant = _round_fp4_e2m1(
        normalized
    )

    y = (
        q_dequant
        * scales[:, None]
    )

    return (
        y.reshape(x.shape)
        .astype(dtype)
    )
