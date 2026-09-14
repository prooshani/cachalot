from __future__ import annotations

import mlx.core as mx
import numpy as np

from cachalot.model.engram_fp8 import dequantize_engram_rows
from cachalot.storage.engram_reader import (
    EngramRowReader,
    EngramTableLayout,
)


def load_engram_rows(
    reader: EngramRowReader,
    layout: EngramTableLayout,
    row_ids: np.ndarray,
) -> mx.array:
    rows = reader.read_rows(
        layout,
        row_ids,
    )

    count = rows.count

    if count == 0:
        return mx.zeros(
            (0, layout.head_dim),
            dtype=mx.float32,
        )

    weights_np = np.frombuffer(
        rows.weights,
        dtype=np.uint8,
    ).reshape(
        count,
        layout.head_dim,
    )

    scales_np = np.frombuffer(
        rows.scales,
        dtype=np.uint8,
    ).reshape(
        count,
        layout.scale_count,
    )

    weights_mx = mx.array(weights_np)
    scales_mx = mx.array(scales_np)

    result = dequantize_engram_rows(
        weights_mx,
        scales_mx,
    )

    mx.eval(result)

    return result
