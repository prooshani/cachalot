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

    readahead controls F_RDAHEAD independently of that. An expert is up to
    nine scattered pieces inside multi-GB shards, and the next expert to be
    read is somewhere else entirely, so pages the kernel reads ahead of a
    piece belong to experts nobody asked for: they consume drive bandwidth
    that decode is short of and evict pages that would otherwise have been
    hits. It defaults to off whenever the page cache is bypassed, which is
    what this reader has always done, and on otherwise, which is what the
    shipped CACHALOT_PAGE_CACHE=1 configuration has always done.
    CACHALOT_RDAHEAD=0 turns it off with the page cache still enabled.
    """

    def __init__(
        self,
        bypass_page_cache: bool | None = None,
        mirror_path: str | Path | None = None,
        mirror_fraction: float | None = None,
        readahead: bool | None = None,
    ) -> None:
        self._fds: dict[Path, int] = {}
        self._lock = RLock()
        if bypass_page_cache is None:
            bypass_page_cache = os.environ.get("CACHALOT_PAGE_CACHE", "0") != "1"
        self.bypass_page_cache = bool(bypass_page_cache)
        if readahead is None:
            env = os.environ.get("CACHALOT_RDAHEAD")
            readahead = (env != "0") if env is not None else not self.bypass_page_cache
        self.readahead = bool(readahead)
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
        self._piece_pool = None
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
                    except OSError:
                        pass

                if not self.readahead:
                    try:
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

        ranges = merge_contiguous_ranges(entry)
        if len(ranges) > 1 and (self.mirror_path is None or self.mirror_fraction <= 0):
            # Stacked banks (storage.index.build_stacked_expert_index) split an
            # expert into up to nine pieces in different tensors. Serial preads
            # are latency-bound (9 pieces of 15.5 MB: 4.2 ms vs 3.3 ms for one
            # 18.8 MB FP4 read); issued concurrently they take 2.8 ms with the
            # same aggregate bandwidth (measured 2026-09-16, internal SSD).
            return self._read_pieces_concurrently(ranges, views)

        for read_range in ranges:
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
                mirror_file = self.mirror_path / read_range.shard.name
                if head and tail and not mirror_file.exists():
                    # a different expert bank (CACHALOT_EXPERT_BANK) has no mirror copy
                    print(f"[reader] mirror {self.mirror_path} lacks {read_range.shard.name}; mirror striping off", flush=True)
                    self.mirror_path = None
                    head, tail = buffers, []
                if head and tail:
                    mirror_fd = self._fd(mirror_file)
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

    def _buffers_for(self, read_range, views) -> list[memoryview]:
        buffers = []
        for tensor in read_range.tensors:
            short = ".".join(tensor.name.rsplit(".", 2)[-2:])
            target = memoryview(views[short]).cast("B")
            if target.nbytes != tensor.size:
                raise ValueError(
                    f"slot buffer for {short} has {target.nbytes} bytes, tensor {tensor.name} has {tensor.size}"
                )
            buffers.append(target)
        return buffers

    def _piece_executor(self):
        if self._piece_pool is None:
            from concurrent.futures import ThreadPoolExecutor

            with self._lock:
                if self._piece_pool is None:
                    self._piece_pool = ThreadPoolExecutor(16, thread_name_prefix="expert-pieces")
        return self._piece_pool

    def _read_pieces_concurrently(self, ranges, views) -> int:
        jobs = [(self._fd(r.shard), self._buffers_for(r, views), r.start, r.size, r.shard) for r in ranges]
        pool = self._piece_executor()
        futures = [pool.submit(os.preadv, fd, bufs, start) for fd, bufs, start, _, _ in jobs[1:]]
        fd, bufs, start, _, _ = jobs[0]
        results = [os.preadv(fd, bufs, start)] + [f.result() for f in futures]
        total = 0
        for got, (_, _, _, size, shard) in zip(results, jobs, strict=True):
            if got != size:
                raise OSError(f"Short read from {shard}: expected {size}, got {got}")
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
            if self._piece_pool is not None:
                self._piece_pool.shutdown(wait=False)
                self._piece_pool = None

    def __enter__(self) -> ExpertReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
