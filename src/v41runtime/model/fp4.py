from __future__ import annotations

import numpy as np


FP4_E2M1_TABLE = np.array(
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
    dtype=np.float32,
)


def decode_e8m0(raw: np.ndarray) -> np.ndarray:
    """
    Decode unsigned E8M0 exponent-only values to float32.

    Normal E8M0 value:
        2 ** (raw - 127)

    The DeepSeek checkpoint uses these values as positive block scales.
    """
    raw = np.asarray(raw, dtype=np.uint8)

    return np.exp2(
        raw.astype(np.int16) - 127
    ).astype(np.float32)


def unpack_fp4_e2m1(packed: np.ndarray) -> np.ndarray:
    """
    Unpack DeepSeek float4_e2m1fn_x2 bytes.

    Low nibble is the first logical FP4 value,
    high nibble is the second.
    """
    packed = np.asarray(packed, dtype=np.uint8)

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    out = np.empty(
        packed.shape[:-1] + (packed.shape[-1] * 2,),
        dtype=np.float32,
    )

    out[..., 0::2] = FP4_E2M1_TABLE[low]
    out[..., 1::2] = FP4_E2M1_TABLE[high]

    return out


def dequantize_fp4_weight(
    packed: np.ndarray,
    scales: np.ndarray,
    block_size: int = 32,
) -> np.ndarray:
    """
    Decode packed FP4 expert weights using per-32 E8M0 scales.

    packed:
        [N, K // 2] uint8

    scales:
        [N, K // 32] uint8 containing E8M0

    returns:
        [N, K] float32
    """
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
