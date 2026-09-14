from __future__ import annotations

from pathlib import Path

from v41runtime.storage.engram_reader import EngramTableLayout
from v41runtime.storage.index import read_safetensors_header


def build_engram_table_layout(
    shard: str | Path,
    layer: int,
) -> EngramTableLayout:
    shard = Path(shard)

    header, data_start = read_safetensors_header(
        shard
    )

    weight_name = (
        f"layers.{layer}.engram.embed.weight"
    )

    scale_name = (
        f"layers.{layer}.engram.embed.scale"
    )

    try:
        weight = header[weight_name]
    except KeyError as exc:
        raise KeyError(
            f"{weight_name!r} not found in {shard}"
        ) from exc

    try:
        scale = header[scale_name]
    except KeyError as exc:
        raise KeyError(
            f"{scale_name!r} not found in {shard}"
        ) from exc

    weight_shape = tuple(
        int(x)
        for x in weight["shape"]
    )

    scale_shape = tuple(
        int(x)
        for x in scale["shape"]
    )

    if len(weight_shape) != 2:
        raise ValueError(
            f"Unexpected Engram weight shape: "
            f"{weight_shape}"
        )

    if len(scale_shape) != 2:
        raise ValueError(
            f"Unexpected Engram scale shape: "
            f"{scale_shape}"
        )

    rows, head_dim = weight_shape

    if scale_shape[0] != rows:
        raise ValueError(
            "Engram weight/scale row counts differ: "
            f"{rows} vs {scale_shape[0]}"
        )

    expected_scale_count = head_dim // 32

    if scale_shape[1] != expected_scale_count:
        raise ValueError(
            f"Expected {expected_scale_count} scales per row, "
            f"got {scale_shape[1]}"
        )

    if weight["dtype"] != "F8_E4M3":
        raise ValueError(
            f"Unexpected Engram weight dtype: "
            f"{weight['dtype']}"
        )

    if scale["dtype"] != "F8_E8M0":
        raise ValueError(
            f"Unexpected Engram scale dtype: "
            f"{scale['dtype']}"
        )

    weight_start, weight_end = (
        int(x)
        for x in weight["data_offsets"]
    )

    scale_start, scale_end = (
        int(x)
        for x in scale["data_offsets"]
    )

    expected_weight_bytes = (
        rows * head_dim
    )

    expected_scale_bytes = (
        rows * expected_scale_count
    )

    if (
        weight_end - weight_start
        != expected_weight_bytes
    ):
        raise ValueError(
            "Engram weight byte size does not "
            "match its shape"
        )

    if (
        scale_end - scale_start
        != expected_scale_bytes
    ):
        raise ValueError(
            "Engram scale byte size does not "
            "match its shape"
        )

    return EngramTableLayout(
        layer=layer,
        shard=shard,
        rows=rows,
        head_dim=head_dim,
        data_start=data_start,
        weight_relative_start=weight_start,
        scale_relative_start=scale_start,
    )
