"""
Micro-benchmark: is the FP4 GEMV slower on freshly promoted expert buffers
than on buffers that have been used before? Also: promotion on main thread
vs worker thread, and per-expert kernel time.

No trunk load; uses the expert index/reader directly.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    MODEL_PATH,  # noqa: E402
    load_expert_standalone,  # noqa: E402
)
from cachalot.model.expert_metal import routed_expert_forward  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader  # noqa: E402

mx.set_cache_limit(2 * 1024**3)


def forward(x, experts):
    outs = [routed_expert_forward(x, w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight,
                                  w2_scales=e.w2_scale, w3_packed=e.w3_weight, w3_scales=e.w3_scale, weight=0.3)
            for e in experts]
    y = mx.sum(mx.stack(outs), axis=0)
    mx.eval(y)
    return y


def timed(label, fn, n=1):
    mx.synchronize()
    t0 = perf_counter()
    for _ in range(n):
        fn()
    mx.synchronize()
    dt = (perf_counter() - t0) / n
    print(f"{label:60s} {dt * 1e3:8.2f} ms", flush=True)
    return dt


def main():
    index = build_expert_index(MODEL_PATH)
    reader = ExpertReader()
    x = mx.random.normal((5120,)).astype(mx.bfloat16)
    mx.eval(x)

    def load(entry):
        return load_expert_standalone(reader, entry)

    entries_a = [index[(10, i)] for i in range(6)]
    entries_b = [index[(11, i)] for i in range(6)]
    entries_c = [index[(12, i)] for i in range(6)]

    # 1. promote on main thread, then run kernel twice
    t0 = perf_counter()
    ea = [load(e) for e in entries_a]
    print(f"promote 6 on main: {(perf_counter() - t0) * 1e3:.0f} ms")
    timed("6-expert forward, fresh buffers (main-thread promoted)", lambda: forward(x, ea))
    timed("6-expert forward, same buffers again", lambda: forward(x, ea))
    timed("6-expert forward, same buffers x10 avg", lambda: forward(x, ea), n=10)

    # 2. promote in worker threads (like get_many)
    with ThreadPoolExecutor(6) as pool:
        t0 = perf_counter()
        eb = list(pool.map(load, entries_b))
        print(f"promote 6 in workers: {(perf_counter() - t0) * 1e3:.0f} ms")
        timed("6-expert forward, fresh buffers (worker promoted)", lambda: forward(x, eb))
        timed("6-expert forward, same buffers again", lambda: forward(x, eb))

        # 3. interleave: main computes on eb while workers promote ec concurrently
        fut = [pool.submit(load, e) for e in entries_c]
        timed("6-expert forward on resident set WHILE workers promote", lambda: forward(x, eb))
        ec = [f.result() for f in fut]
        timed("6-expert forward, fresh (promoted concurrently)", lambda: forward(x, ec))
        timed("6-expert forward, same again", lambda: forward(x, ec))

    # 4. single expert kernel cost and eval granularity
    timed("1-expert forward (resident)", lambda: forward(x, ea[:1]), n=10)
    print("mlx active GiB", mx.get_active_memory() / 2**30, "cache GiB", mx.get_cache_memory() / 2**30)


if __name__ == "__main__":
    main()
