from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from cachalot.storage.index import ExpertEntry, merge_contiguous_ranges


@dataclass(frozen=True)
class ReadChunk:
    shard: Path
    start: int
    data: bytes


class ExpertReader:
    def __init__(self) -> None:
        self._fds: dict[Path, int] = {}
        self._lock = RLock()

    def _fd(self, path: Path) -> int:
        with self._lock:
            fd = self._fds.get(path)

            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                self._fds[path] = fd

            return fd

    def read_expert(self, entry: ExpertEntry) -> tuple[ReadChunk, ...]:
        chunks: list[ReadChunk] = []

        for read_range in merge_contiguous_ranges(entry):
            fd = self._fd(read_range.shard)

            data = os.pread(
                fd,
                read_range.size,
                read_range.start,
            )

            if len(data) != read_range.size:
                raise OSError(
                    f"Short read from {read_range.shard}: "
                    f"expected {read_range.size}, got {len(data)}"
                )

            chunks.append(
                ReadChunk(
                    shard=read_range.shard,
                    start=read_range.start,
                    data=data,
                )
            )

        return tuple(chunks)

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)

            self._fds.clear()

    def __enter__(self) -> ExpertReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
