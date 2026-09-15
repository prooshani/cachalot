from __future__ import annotations

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


class EngramRowReader:
    """
    Random-access reader for Engram embedding rows.

    Rows are 256 B of E4M3 plus 8 B of E8M0 scales scattered across a
    multi-GB table. Each row is fetched with pread() on a plain descriptor
    (no mmap): a prefill touches ~12k rows per Engram layer, and faulting
    them through a shared mapping from several threads took 9-18 s, while
    parallel preads complete in well under a second on NVMe.
    """

    WORKERS = 16

    def __init__(self) -> None:
        self._fds: dict[Path, int] = {}
        self._lock = RLock()
        self._pool = None

    def _fd(self, path: Path) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                self._fds[path] = fd
            return fd

    def _executor(self):
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(self.WORKERS, thread_name_prefix="engram-rows")
        return self._pool

    def read_rows(
        self,
        layout: EngramTableLayout,
        row_ids: np.ndarray,
    ) -> EngramRows:
        ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)

        if ids.size == 0:
            return EngramRows(
                ids=ids, weights=b"", scales=b"", head_dim=layout.head_dim, scale_count=layout.scale_count,
            )

        if np.any(ids < 0) or np.any(ids >= layout.rows):
            raise IndexError(f"Engram row outside [0, {layout.rows})")

        fd = self._fd(layout.shard)
        wrb, srb = layout.weight_row_bytes, layout.scale_row_bytes
        weights = bytearray(ids.size * wrb)
        scales = bytearray(ids.size * srb)
        weight_out = memoryview(weights)
        scale_out = memoryview(scales)

        def fetch(lo: int, hi: int) -> None:
            for i in range(lo, hi):
                row = int(ids[i])
                os.preadv(fd, [weight_out[i * wrb : (i + 1) * wrb]], layout.weight_start + row * wrb)
                os.preadv(fd, [scale_out[i * srb : (i + 1) * srb]], layout.scale_start + row * srb)

        n = int(ids.size)
        if n >= 64:
            step = (n + self.WORKERS - 1) // self.WORKERS
            list(self._executor().map(lambda lo: fetch(lo, min(lo + step, n)), range(0, n, step)))
        else:
            fetch(0, n)

        return EngramRows(
            ids=ids, weights=bytes(weights), scales=bytes(scales), head_dim=layout.head_dim, scale_count=layout.scale_count,
        )

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()
            if self._pool is not None:
                self._pool.shutdown(wait=False)
                self._pool = None

    def __enter__(self) -> EngramRowReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
