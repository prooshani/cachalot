from __future__ import annotations

import math

import ml_dtypes
import numpy as np

FP8_BLOCK_SIZE = 32
FP8_MAX = 448.0
FP8_MIN_AMAX = 1e-4


def decode_e8m0(
    raw: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(
        raw,
        dtype=np.uint8,
    )

    # E8M0 represents powers of two.
    return np.exp2(
        raw.astype(np.int16) - 127
    ).astype(np.float32)


def quantize_activation_fp8(
    x: np.ndarray,
    block_size: int = FP8_BLOCK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact semantic reference for DeepSeek act_quant():

      amax = max(abs(block), 1e-4)
      scale = 2^ceil(log2(amax / 448))
      q = FP8_E4M3FN(clamp(x / scale, -448, 448))

    Returns:
      q      : float8_e4m3fn values
      scales : float32 power-of-two scales
    """
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    if x.ndim != 1:
        raise ValueError(
            f"Expected 1D activation, got {x.shape}"
        )

    if x.size % block_size != 0:
        raise ValueError(
            f"K={x.size} must be divisible "
            f"by block_size={block_size}"
        )

    blocks = x.reshape(
        -1,
        block_size,
    )

    amax = np.maximum(
        np.max(
            np.abs(blocks),
            axis=1,
        ),
        FP8_MIN_AMAX,
    )

    raw_scale = amax / FP8_MAX

    scales = np.exp2(
        np.ceil(
            np.log2(raw_scale)
        )
    ).astype(np.float32)

    normalized = np.clip(
        blocks / scales[:, None],
        -FP8_MAX,
        FP8_MAX,
    )

    q = normalized.astype(
        ml_dtypes.float8_e4m3fn
    )

    return (
        q.reshape(-1),
        scales,
    )


def fp8_gemv_reference(
    x: np.ndarray,
    weight_raw: np.ndarray,
    scale_raw: np.ndarray,
    *,
    block_size: int = FP8_BLOCK_SIZE,
) -> np.ndarray:
    """
    CPU reference for official DeepSeek FP8 linear:

        y = fp8_gemm(
            quantize_fp8(x),
            activation_scales,
            weight_fp8,
            weight_e8m0_scales,
        )

    For M=1 decode.

    weight_raw:
        uint8 [N, K], raw E4M3 checkpoint bytes

    scale_raw:
        uint8 [ceil(N/32), K/32], raw E8M0 bytes

    Output:
        float32 [N]

    The official kernel accumulates in FP32 and emits BF16.
    This function intentionally retains FP32 so it can serve
    as a high-precision correctness oracle.
    """
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    weight_raw = np.asarray(
        weight_raw,
        dtype=np.uint8,
    )

    scale_raw = np.asarray(
        scale_raw,
        dtype=np.uint8,
    )

    if weight_raw.ndim != 2:
        raise ValueError(
            f"weight_raw must be 2D, "
            f"got {weight_raw.shape}"
        )

    n, k = weight_raw.shape

    if x.shape != (k,):
        raise ValueError(
            f"x shape {x.shape} != ({k},)"
        )

    if k % block_size != 0:
        raise ValueError(
            f"K={k} is not divisible by {block_size}"
        )

    expected_scale_shape = (
        math.ceil(n / block_size),
        k // block_size,
    )

    if scale_raw.shape != expected_scale_shape:
        raise ValueError(
            f"scale shape {scale_raw.shape} "
            f"!= {expected_scale_shape}"
        )

    qx, activation_scales = (
        quantize_activation_fp8(
            x,
            block_size,
        )
    )

    qx_f32 = qx.astype(
        np.float32
    ).reshape(
        -1,
        block_size,
    )

    weight_fp8 = weight_raw.view(
        ml_dtypes.float8_e4m3fn
    )

    weight_scales = decode_e8m0(
        scale_raw
    )

    output = np.zeros(
        n,
        dtype=np.float32,
    )

    n_k_blocks = k // block_size

    # Match the official kernel's scale correction:
    #
    #   sum_kblock(
    #       dot(qx_block, qw_block)
    #       * act_scale
    #       * weight_scale[out//32, kblock]
    #   )
    #
    for kb in range(n_k_blocks):
        start = kb * block_size
        end = start + block_size

        w_block = (
            weight_fp8[:, start:end]
            .astype(np.float32)
        )

        partial = (
            w_block
            @ qx_f32[kb]
        )

        output += (
            partial
            * activation_scales[kb]
            * weight_scales[
                np.arange(n) // block_size,
                kb,
            ]
        )

    return output
