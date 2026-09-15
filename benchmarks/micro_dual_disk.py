"""
Two identical checkpoint copies on two drives: does splitting expert reads
across them lower per-expert latency (decode) or raise aggregate throughput
(prefill)? Strategies: each drive alone, whole experts alternated between
drives, and byte-range striping of every expert at several split ratios.

Usage:
    PYTHONPATH=src python benchmarks/micro_dual_disk.py <path_a> <path_b>
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
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader, merge_contiguous_ranges  # noqa: E402

ALIGN = 4096


def plan(entry_a, entry_b, frac_b: float):
    """(fd-key, dst offset, src offset, length) pieces: first (1-frac_b) of each range from A, rest from B."""
    pieces = []
    dst = 0
    for ra, rb in zip(merge_contiguous_ranges(entry_a), merge_contiguous_ranges(entry_b), strict=True):
        cut = int(ra.size * (1.0 - frac_b)) // ALIGN * ALIGN
        if cut > 0:
            pieces.append(("a", ra.shard, dst, ra.start, cut))
        if ra.size - cut > 0:
            pieces.append(("b", rb.shard, dst + cut, rb.start + cut, ra.size - cut))
        dst += ra.size
    return pieces


def read_planned(readers, pieces, buf, pool):
    def one(p):
        which, shard, dst, src, n = p
        fd = readers[which]._fd(shard)
        got = os.preadv(fd, [memoryview(buf)[dst:dst + n]], src)
        assert got == n
        return n

    if len(pieces) == 1:
        return one(pieces[0])
    return sum(pool.map(one, pieces))


def main():
    path_a, path_b = sys.argv[1], sys.argv[2]
    index_a = build_expert_index(path_a)
    index_b = build_expert_index(path_b)
    keys = list(index_a)
    rng = random.Random(int(perf_counter()))   # fresh experts each run: F_NOCACHE does not evict cached pages
    rng.shuffle(keys)
    readers = {"a": ExpertReader(), "b": ExpertReader()}
    size = sum(t.size for t in index_a[keys[0]].tensors)
    bufs = [bytearray(size) for _ in range(8)]
    for b in bufs:
        memoryview(b)[::ALIGN] = b"\x01" * len(memoryview(b)[::ALIGN])
    pool = ThreadPoolExecutor(16)
    outer = ThreadPoolExecutor(8)
    pos = [0]

    def next_keys(n):
        out = keys[pos[0]:pos[0] + n]
        pos[0] += n
        return out

    strategies = [("A only (internal)", lambda k, i: plan(index_a[k], index_b[k], 0.0)),
                  ("B only (X10Pro)", lambda k, i: plan(index_a[k], index_b[k], 1.0)),
                  ("alternate whole experts A/B", lambda k, i: plan(index_a[k], index_b[k], 1.0 if i % 2 else 0.0)),
                  ("6 of 7 experts on A, 1 on B", lambda k, i: plan(index_a[k], index_b[k], 1.0 if i % 7 == 6 else 0.0))]
    for frac in (0.10, 0.15, 0.20, 0.25, 0.35):
        strategies.append((f"stripe every expert, {frac:.0%} from B", lambda k, i, f=frac: plan(index_a[k], index_b[k], f)))

    print(f"expert {size / 1e6:.1f} MB | A = {path_a} | B = {path_b}")
    print(f"{'strategy':38s} | single expert p50 / p90 ms | 8 concurrent: ms per expert, aggregate GB/s")
    for name, planner in strategies:
        # decode case: one expert at a time
        ts = []
        for i, k in enumerate(next_keys(40)):
            t0 = perf_counter()
            read_planned(readers, planner(k, i), bufs[0], pool)
            ts.append((perf_counter() - t0) * 1e3)
        ts.sort()
        p50, p90 = ts[len(ts) // 2], ts[int(len(ts) * 0.9)]
        # prefill case: 8 experts in flight, 5 rounds
        agg = []
        for _r in range(5):
            ks = next_keys(8)
            t0 = perf_counter()
            list(outer.map(lambda ik, planner=planner: read_planned(readers, planner(ik[1], ik[0]), bufs[ik[0]], pool), enumerate(ks)))
            agg.append(perf_counter() - t0)
        per = statistics.median(agg) / 8 * 1e3
        print(f"{name:38s} | {p50:6.2f} / {p90:6.2f}          | {per:6.2f} ms, {size / per / 1e6:5.2f} GB/s", flush=True)


def check_mirror_reader(path_a: str, path_b: str) -> None:
    """ExpertReader with a mirror must produce the same bytes as a plain read."""
    from cachalot.cache.resident_store import tensor_sizes_from_entry

    index = build_expert_index(path_a)
    keys = list(index)[:3] + list(index)[-3:]
    sizes = tensor_sizes_from_entry(index[keys[0]])
    plain = ExpertReader()
    mirrored = ExpertReader(mirror_path=path_b, mirror_fraction=0.1)
    for k in keys:
        views = {n: bytearray(sz) for n, sz in sizes.items()}
        views2 = {n: bytearray(sz) for n, sz in sizes.items()}
        plain.read_expert_into(index[k], views)
        mirrored.read_expert_into(index[k], views2)
        assert all(bytes(views[n]) == bytes(views2[n]) for n in sizes), k
    print(f"mirror reader bytes identical to plain reader on {len(keys)} experts")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "--check":
        check_mirror_reader(sys.argv[1], sys.argv[2])
    else:
        main()
