"""Single-expert read latency by destination: pre-touched bytearray vs. wired MLX slot memory (as the runtime uses)."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.cache.resident_store import tensor_sizes_from_entry  # noqa: E402
from cachalot.cache.slots import ExpertSlotPool  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader  # noqa: E402


def timed(fn, entries, n=60):
    ts = []
    for e in entries[:n]:
        t0 = perf_counter()
        fn(e)
        ts.append((perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[int(len(ts) * 0.9)]


def main():
    index = build_expert_index(MODEL_PATH)
    entries = list(index.values())
    rng = random.Random(11)
    rng.shuffle(entries)
    reader = ExpertReader()
    sizes = tensor_sizes_from_entry(entries[0])
    names = list(sizes)
    total = sum(sizes.values())
    buf = bytearray(total)
    memoryview(buf)[::4096] = b"\x01" * len(memoryview(buf)[::4096])
    offsets = {}
    off = 0
    for n in names:
        offsets[n] = off
        off += sizes[n]
    views_ba = {n: memoryview(buf)[offsets[n]:offsets[n] + sizes[n]] for n in names}
    p50, p90 = timed(lambda e: reader.read_expert_into(e, views_ba), entries[0:60])
    print(f"bytearray destination:            p50 {p50:.2f} ms  p90 {p90:.2f} ms")

    pool = ExpertSlotPool(sizes, 4)
    slot = pool.acquire()
    p50, p90 = timed(lambda e: reader.read_expert_into(e, slot.views), entries[60:120])
    print(f"MLX slot, no wired limit:         p50 {p50:.2f} ms  p90 {p90:.2f} ms")

    mx.set_wired_limit(8 * 2**30)
    pool2 = ExpertSlotPool(sizes, 4)
    slot2 = pool2.acquire()
    p50, p90 = timed(lambda e: reader.read_expert_into(e, slot2.views), entries[120:180])
    print(f"MLX slot, wired limit set:        p50 {p50:.2f} ms  p90 {p90:.2f} ms")
    # a slot the GPU has just read (as in decode: the previous occupant was consumed by kernels)
    arrs = list(slot2.arrays.values())
    mx.eval(*(a.astype(mx.float32).sum() for a in arrs))
    p50, p90 = timed(lambda e: reader.read_expert_into(e, slot2.views), entries[180:240])
    print(f"MLX slot after GPU read of it:    p50 {p50:.2f} ms  p90 {p90:.2f} ms")


if __name__ == "__main__":
    main()
