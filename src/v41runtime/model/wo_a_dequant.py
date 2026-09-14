from __future__ import annotations

import ml_dtypes
import mlx.core as mx
import numpy as np

from v41runtime.model.fp8_ref import decode_e8m0


def dequantize_wo_a(
    weight_raw: mx.array,
    scale_raw: mx.array,
) -> mx.array:
    """
    Reproduce inference/convert.py handling of wo_a:

        weight = (
            weight
            .unflatten(0, (-1, out_block_size))
            .unflatten(-1, (-1, in_block_size))
            .float()
            * scale[:, None, :, None].float()
        )
        weight = (
            weight
            .flatten(2, 3)
            .flatten(0, 1)
            .bfloat16()
        )

    Released HF checkpoint:
        weight_raw : uint8 E4M3 bytes [N, K]
        scale_raw  : uint8 E8M0 bytes [N/32, K/32]

    Returns:
        BF16 [N, K]
    """
    if weight_raw.ndim != 2:
        raise ValueError(
            f"weight_raw must be 2D, got {weight_raw.shape}"
        )

    if scale_raw.ndim != 2:
        raise ValueError(
            f"scale_raw must be 2D, got {scale_raw.shape}"
        )

    n, k = weight_raw.shape
    sn, sk = scale_raw.shape

    if n % sn != 0 or k % sk != 0:
        raise ValueError(
            f"Incompatible weight/scale shapes: "
            f"{weight_raw.shape}, {scale_raw.shape}"
        )

    out_block = n // sn
    in_block = k // sk

    if (out_block, in_block) not in (
        (32, 32),
        (128, 128),
    ):
        raise ValueError(
            f"Unsupported wo_a block size "
            f"{(out_block, in_block)}"
        )

    # wo_a conversion happens once at model load time,
    # so using NumPy here avoids keeping a giant FP32 MLX
    # intermediate in the inference graph.
    w_raw_np = np.asarray(
        weight_raw,
        dtype=np.uint8,
    )

    s_raw_np = np.asarray(
        scale_raw,
        dtype=np.uint8,
    )

    weight_fp8 = w_raw_np.view(
        ml_dtypes.float8_e4m3fn
    )

    scales = decode_e8m0(
        s_raw_np
    )

    # Process one output block at a time to avoid constructing
    # the entire 8192x4096 matrix in FP32 simultaneously.
    chunks = []

    for block_idx in range(sn):
        row_start = (
            block_idx * out_block
        )

        row_end = (
            row_start + out_block
        )

        # [out_block, K]
        block = (
            weight_fp8[
                row_start:row_end
            ]
            .astype(np.float32)
            .reshape(
                out_block,
                sk,
                in_block,
            )
        )

        block *= scales[
            block_idx
        ][None, :, None]

        block = block.reshape(
            out_block,
            k,
        )

        # Match official conversion's final BF16 storage.
        chunks.append(
            mx.array(block)
            .astype(mx.bfloat16)
        )

    result = mx.concatenate(
        chunks,
        axis=0,
    )

    mx.eval(result)

    return result
