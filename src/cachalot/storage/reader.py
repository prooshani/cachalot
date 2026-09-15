from __future__ import annotations

import fcntl
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
    """
    Positional reads of routed-expert byte ranges from safetensors shards.

    bypass_page_cache=True sets F_NOCACHE on the shard descriptors so the
    multi-hundred-GB expert stream does not evict the rest of the system
    (including MLX-resident weights) from memory. Expert reads are almost
    never page-cache hits anyway: a token's misses are experts not seen
    for many tokens, and prefill streams far more than the cache can hold.
    """

    def __init__(
        self,
        bypass_page_cache: bool | None = None,
        mirror_path: str | Path | None = None,
        mirror_fraction: float | None = None,
    ) -> None:
        self._fds: dict[Path, int] = {}
        self._lock = RLock()
        if bypass_page_cache is None:
            bypass_page_cache = os.environ.get("CACHALOT_PAGE_CACHE", "0") != "1"
        self.bypass_page_cache = bool(bypass_page_cache)
        # Optional second, identical copy of the checkpoint on another drive
        # (CACHALOT_MIRROR_PATH): the tail `mirror_fraction` of every expert
        # read comes from it concurrently, so a single expert lands sooner
        # and the two drives' bandwidths add up. 10 % suits a 1 GB/s USB
        # drive next to a 5.5 GB/s internal one; use bandwidth_b / total.
        if mirror_path is None:
            mirror_path = os.environ.get("CACHALOT_MIRROR_PATH") or None
        if mirror_fraction is None:
            mirror_fraction = float(os.environ.get("CACHALOT_MIRROR_FRACTION", "0.10"))
        self.mirror_path = Path(mirror_path) if mirror_path else None
        self.mirror_fraction = min(max(float(mirror_fraction), 0.0), 0.9)
        self._mirror_pool = None
        if self.mirror_path is not None and not self.mirror_path.is_dir():
            raise FileNotFoundError(f"CACHALOT_MIRROR_PATH {self.mirror_path} is not a directory")

    def _mirror_executor(self):
        if self._mirror_pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._mirror_pool = ThreadPoolExecutor(8, thread_name_prefix="expert-mirror")
        return self._mirror_pool

    @staticmethod
    def _split_buffers(buffers: list[memoryview], cut: int) -> tuple[list[memoryview], list[memoryview]]:
        """Split a list of contiguous destination views at byte offset `cut`."""
        head: list[memoryview] = []
        tail: list[memoryview] = []
        seen = 0
        for buf in buffers:
            n = buf.nbytes
            if seen + n <= cut:
                head.append(buf)
            elif seen >= cut:
                tail.append(buf)
            else:
                k = cut - seen
                head.append(buf[:k])
                tail.append(buf[k:])
            seen += n
        return head, tail

    def _fd(self, path: Path) -> int:
        with self._lock:
            fd = self._fds.get(path)

            if fd is None:
                fd = os.open(path, os.O_RDONLY)

                if self.bypass_page_cache:
                    try:
                        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                        fcntl.fcntl(fd, fcntl.F_RDAHEAD, 0)
                    except OSError:
                        pass

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

    def read_expert_into(
        self,
        entry: ExpertEntry,
        views: dict[str, memoryview | bytearray | object],
    ) -> int:
        """
        Read an expert directly into caller-provided writable buffers,
        one per tensor short name ("w1.weight", ...). Contiguous tensors
        are gathered with a single preadv. Returns bytes read.
        """
        total = 0

        for read_range in merge_contiguous_ranges(entry):
            fd = self._fd(read_range.shard)
            buffers = []

            for tensor in read_range.tensors:
                short = ".".join(tensor.name.rsplit(".", 2)[-2:])
                target = memoryview(views[short]).cast("B")

                if target.nbytes != tensor.size:
                    raise ValueError(
                        f"slot buffer for {short} has {target.nbytes} bytes, "
                        f"tensor {tensor.name} has {tensor.size}"
                    )

                buffers.append(target)

            if self.mirror_path is not None and self.mirror_fraction > 0:
                cut = int(read_range.size * (1.0 - self.mirror_fraction)) // 4096 * 4096
                head, tail = self._split_buffers(buffers, cut)
                if head and tail:
                    mirror_fd = self._fd(self.mirror_path / read_range.shard.name)
                    tail_future = self._mirror_executor().submit(
                        os.preadv, mirror_fd, tail, read_range.start + cut
                    )
                    got = os.preadv(fd, head, read_range.start)
                    got += tail_future.result()
                else:
                    got = os.preadv(fd, buffers, read_range.start)
            else:
                got = os.preadv(fd, buffers, read_range.start)

            if got != read_range.size:
                raise OSError(
                    f"Short read from {read_range.shard}: "
                    f"expected {read_range.size}, got {got}"
                )

            total += got

        return total

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)

            self._fds.clear()
            if self._mirror_pool is not None:
                self._mirror_pool.shutdown(wait=False)
                self._mirror_pool = None

    def __enter__(self) -> ExpertReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
