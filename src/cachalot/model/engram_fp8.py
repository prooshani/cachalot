from __future__ import annotations

import mlx.core as mx

ENGRAM_HEAD_DIM = 256
ENGRAM_BLOCK_SIZE = 32
ENGRAM_SCALE_COUNT = ENGRAM_HEAD_DIM // ENGRAM_BLOCK_SIZE

ENGRAM_WEIGHT_ROW_BYTES = ENGRAM_HEAD_DIM
ENGRAM_SCALE_ROW_BYTES = ENGRAM_SCALE_COUNT
ENGRAM_ROW_BYTES = ENGRAM_WEIGHT_ROW_BYTES + ENGRAM_SCALE_ROW_BYTES


def decode_e8m0(
    raw: mx.array,
) -> mx.array:
    exponent = raw.astype(mx.int32) - 127

    return mx.power(
        mx.array(2.0, dtype=mx.float32),
        exponent,
    )


def dequantize_engram_rows(
    weights_fp8: mx.array,
    scales_e8m0: mx.array,
) -> mx.array:
    """
    weights_fp8:
        uint8 array shaped [N, 256], containing raw E4M3 bytes.

    scales_e8m0:
        uint8 array shaped [N, 8], one scale per block of 32 values.

    Returns:
        float32 array shaped [N, 256].
    """

    if weights_fp8.ndim != 2:
        raise ValueError(
            f"weights_fp8 must be rank 2, got {weights_fp8.ndim}"
        )

    if scales_e8m0.ndim != 2:
        raise ValueError(
            f"scales_e8m0 must be rank 2, got {scales_e8m0.ndim}"
        )

    if weights_fp8.shape[1] != ENGRAM_HEAD_DIM:
        raise ValueError(
            f"expected weight width {ENGRAM_HEAD_DIM}, "
            f"got {weights_fp8.shape[1]}"
        )

    if scales_e8m0.shape != (
        weights_fp8.shape[0],
        ENGRAM_SCALE_COUNT,
    ):
        raise ValueError(
            f"scale shape {scales_e8m0.shape} does not match "
            f"expected {(weights_fp8.shape[0], ENGRAM_SCALE_COUNT)}"
        )

    # DeepSeek stores E4M3 values as raw bytes.
    values = mx.from_fp8(
        weights_fp8.astype(mx.uint8),
        dtype=mx.float32,
    )

    scales = decode_e8m0(
        scales_e8m0
    )

    return (
        values.reshape(
            weights_fp8.shape[0],
            ENGRAM_SCALE_COUNT,
            ENGRAM_BLOCK_SIZE,
        )
        * scales[..., None]
    ).reshape(
        weights_fp8.shape[0],
        ENGRAM_HEAD_DIM,
    )
