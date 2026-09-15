"""
Single-expert SSD read latency: one preadv per contiguous range (current loader)
vs the same ranges split into N chunks read concurrently. Decode loads only
~1-3 experts per layer, so per-expert parallelism decides the miss cost.
"""
from __future__ import annotations

import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader, merge_contiguous_ranges  # noqa: E402


def read_chunked(reader, entry, buf, chunks, pool):
    """Split every contiguous range into `chunks` pieces and pread them concurrently into buf."""
    jobs = []
    offset = 0
    for rr in merge_contiguous_ranges(entry):
        fd = reader._fd(rr.shard)
        step = (rr.size + chunks - 1) // chunks
        step = (step + 4095) // 4096 * 4096
        for c in range(0, rr.size, step):
            n = min(step, rr.size - c)
            jobs.append((fd, offset + c, rr.start + c, n))
        offset += rr.size

    def one(job):
        fd, dst, src, n = job
        mv = memoryview(buf)[dst:dst + n]
        got = os.preadv(fd, [mv], src)
        assert got == n, (got, n)
        return n

    if chunks == 1 and len(jobs) == len(list(merge_contiguous_ranges(entry))):
        return sum(one(j) for j in jobs)
    return sum(pool.map(one, jobs))


def main():
    index = build_expert_index(MODEL_PATH)
    entries = list(index.values())
    rng = random.Random(7)
    reader = ExpertReader()
    size = sum(t.size for t in entries[0].tensors)
    pool = ThreadPoolExecutor(32)
    print(f"expert bytes {size / 1e6:.1f} MB; contiguous ranges per expert: {len(list(merge_contiguous_ranges(entries[0])))}")
    # destination buffers are allocated once and pre-touched: fresh allocations
    # would add ~1,150 page faults per expert to every read
    all_bufs = [bytearray(size) for _ in range(4)]
    for b in all_bufs:
        memoryview(b)[::4096] = b"\x01" * len(memoryview(b)[::4096])
    for concurrent_experts in (1, 2, 4):
        for chunks in (1, 2, 4, 8, 16):
            times = []
            for _ in range(12):
                picks = [rng.choice(entries) for _ in range(concurrent_experts)]
                bufs = all_bufs[:concurrent_experts]
                t0 = perf_counter()
                if concurrent_experts == 1:
                    read_chunked(reader, picks[0], bufs[0], chunks, pool)
                else:
                    with ThreadPoolExecutor(concurrent_experts) as outer:
                        list(outer.map(lambda pe, chunks=chunks: read_chunked(reader, pe[0], pe[1], chunks, pool), zip(picks, bufs, strict=True)))
                times.append(perf_counter() - t0)
            med = statistics.median(times)
            print(f"experts in flight {concurrent_experts} | chunks per range {chunks:2d} | median {med * 1e3:6.1f} ms "
                  f"| {concurrent_experts * size / med / 1e9:5.2f} GB/s", flush=True)


if __name__ == "__main__":
    main()
