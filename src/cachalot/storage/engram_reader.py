from __future__ import annotations

import mmap
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np


@dataclass(frozen=True)
class EngramTableLayout:
    layer: int
    shard: Path

    rows: int
    head_dim: int

    data_start: int

    weight_relative_start: int
    scale_relative_start: int

    @property
    def scale_count(self) -> int:
        return self.head_dim // 32

    @property
    def weight_row_bytes(self) -> int:
        return self.head_dim

    @property
    def scale_row_bytes(self) -> int:
        return self.scale_count

    @property
    def row_payload_bytes(self) -> int:
        return (
            self.weight_row_bytes
            + self.scale_row_bytes
        )

    @property
    def weight_start(self) -> int:
        return (
            self.data_start
            + self.weight_relative_start
        )

    @property
    def scale_start(self) -> int:
        return (
            self.data_start
            + self.scale_relative_start
        )


@dataclass(frozen=True)
class EngramRows:
    ids: np.ndarray

    weights: bytes
    scales: bytes

    head_dim: int
    scale_count: int

    @property
    def count(self) -> int:
        return int(self.ids.size)

    @property
    def payload_bytes(self) -> int:
        return len(self.weights) + len(self.scales)


@dataclass
class _MappedShard:
    fd: int
    mapping: mmap.mmap


class EngramRowReader:
    def __init__(self) -> None:
        self._shards: dict[Path, _MappedShard] = {}
        self._lock = RLock()

    def _mapping(
        self,
        path: Path,
    ) -> mmap.mmap:
        with self._lock:
            mapped = self._shards.get(path)

            if mapped is None:
                fd = os.open(
                    path,
                    os.O_RDONLY,
                )

                try:
                    mapping = mmap.mmap(
                        fd,
                        length=0,
                        access=mmap.ACCESS_READ,
                    )
                except Exception:
                    os.close(fd)
                    raise

                mapped = _MappedShard(
                    fd=fd,
                    mapping=mapping,
                )

                self._shards[path] = mapped

            return mapped.mapping

    def read_rows(
        self,
        layout: EngramTableLayout,
        row_ids: np.ndarray,
    ) -> EngramRows:
        ids = np.asarray(
            row_ids,
            dtype=np.int64,
        ).reshape(-1)

        if ids.size == 0:
            return EngramRows(
                ids=ids,
                weights=b"",
                scales=b"",
                head_dim=layout.head_dim,
                scale_count=layout.scale_count,
            )

        if np.any(ids < 0) or np.any(ids >= layout.rows):
            raise IndexError(
                f"Engram row outside [0, {layout.rows})"
            )

        mapping = self._mapping(
            layout.shard
        )

        weights = bytearray(
            ids.size * layout.weight_row_bytes
        )

        scales = bytearray(
            ids.size * layout.scale_row_bytes
        )

        weight_out = memoryview(weights)
        scale_out = memoryview(scales)

        for i, row_id in enumerate(ids):
            row = int(row_id)

            weight_offset = (
                layout.weight_start
                + row * layout.weight_row_bytes
            )

            scale_offset = (
                layout.scale_start
                + row * layout.scale_row_bytes
            )

            w0 = i * layout.weight_row_bytes
            w1 = w0 + layout.weight_row_bytes

            s0 = i * layout.scale_row_bytes
            s1 = s0 + layout.scale_row_bytes

            weight_out[w0:w1] = mapping[
                weight_offset:
                weight_offset + layout.weight_row_bytes
            ]

            scale_out[s0:s1] = mapping[
                scale_offset:
                scale_offset + layout.scale_row_bytes
            ]

        return EngramRows(
            ids=ids,
            weights=bytes(weights),
            scales=bytes(scales),
            head_dim=layout.head_dim,
            scale_count=layout.scale_count,
        )

    def close(self) -> None:
        with self._lock:
            for mapped in self._shards.values():
                mapped.mapping.close()
                os.close(mapped.fd)

            self._shards.clear()

    def __enter__(self) -> EngramRowReader:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
