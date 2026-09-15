"""Does GPU eval on the main thread starve loader threads (GIL)? SSD throughput with and without concurrent evals."""
from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader  # noqa: E402

EXPERT = 18_800_640


def main():
    index = build_expert_index(MODEL_PATH)
    reader = ExpertReader()
    entries = [index[(la, e)] for la in range(20, 30) for e in range(0, 384, 3)]  # ~1280 experts, 24 GB
    bufs = [np.empty(EXPERT, dtype=np.uint8) for _ in range(16)]
    views_for = lambda b: {  # noqa: E731
        "w1.scale": memoryview(b)[0:368640], "w2.scale": memoryview(b)[368640:737280], "w3.scale": memoryview(b)[737280:1105920],
        "w1.weight": memoryview(b)[1105920:7004160], "w2.weight": memoryview(b)[7004160:12902400], "w3.weight": memoryview(b)[12902400:18800640],
    }
    views = [views_for(b) for b in bufs]
    stop = threading.Event()

    def stream(n_experts, base):
        with ThreadPoolExecutor(8) as pool:
            t0 = perf_counter()
            list(pool.map(lambda i: reader.read_expert_into(entries[base + i], views[i % 16]), range(n_experts)))
            return n_experts * EXPERT / (perf_counter() - t0) / 1e9

    def gpu_load():
        w = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
        mx.eval(w)
        n = 0
        while not stop.is_set():
            mx.eval(mx.matmul(w, w))   # ~10 ms of GPU work per eval
            n += 1
        return n

    print(f"SSD alone: {stream(400, 0):.2f} GB/s", flush=True)
    t = threading.Thread(target=gpu_load)
    stop.clear()
    t.start()
    time.sleep(0.5)
    print(f"SSD with GPU evals on another thread: {stream(400, 400):.2f} GB/s", flush=True)
    stop.set()
    t.join()
    # main-thread evals interleaved like prefill: eval between submitting reads
    with ThreadPoolExecutor(8) as pool:
        w = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
        mx.eval(w)
        t0 = perf_counter()
        base = 800
        futs = [pool.submit(reader.read_expert_into, entries[base + i], views[i % 16]) for i in range(8)]
        done = 0
        gpu = 0.0
        for i in range(8, 400):
            futs[i % 8].result()
            done += 1
            futs[i % 8] = pool.submit(reader.read_expert_into, entries[base + i], views[i % 16])
            if i % 8 == 0:
                g0 = perf_counter()
                mx.eval(mx.matmul(w, w))
                gpu += perf_counter() - g0
        for f in futs:
            f.result()
        el = perf_counter() - t0
        print(f"prefill-like loop: {400 * EXPERT / el / 1e9:.2f} GB/s effective; wall {el:.2f}s, of which main-thread eval {gpu:.2f}s "
              f"(SSD floor {400 * EXPERT / 5.6e9:.2f}s)", flush=True)
    _ = os


if __name__ == "__main__":
    main()
