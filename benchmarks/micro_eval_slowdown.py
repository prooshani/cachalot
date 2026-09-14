"""
Reproduce the decode pattern without the trunk: 40 "layers", each loading a
few experts from SSD in worker threads, then running the expert kernel and
evaluating. Which ingredient makes the per-layer eval slow?

variants:
  resident       : kernel on already-resident experts, no I/O           (baseline)
  io-only        : SSD reads in workers (bytes discarded) + kernel on resident experts
  promote-keep   : SSD read + mx.array promotion, new experts used, nothing evicted
  promote-evict  : same + evict (del) as many old experts as loaded
  pagecache      : like promote-evict but re-reading the same files (page cache, no SSD)
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
LAYERS = 40
PER_LAYER = 3


def main():
    index = build_expert_index(MODEL_PATH)
    reader = ExpertReader()
    pool = ThreadPoolExecutor(8)
    x = mx.random.normal((5120,)).astype(mx.bfloat16)
    mx.eval(x)

    def load(entry):
        return load_expert_standalone(reader, entry)

    def read_only(entry):
        return sum(len(c.data) for c in reader.read_expert(entry))

    # resident base set: 6 experts per layer from layer 5 (like a warm cache)
    base = {la: [load(index[(5, la * 6 + i)]) for i in range(6)] for la in range(LAYERS)}
    fresh_ids = iter([(lay, e) for lay in range(6, 40) for e in range(384)])

    def kernel(experts):
        outs = [routed_expert_forward(x, w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight,
                                      w2_scales=e.w2_scale, w3_packed=e.w3_weight, w3_scales=e.w3_scale, weight=0.3)
                for e in experts]
        return mx.sum(mx.stack(outs), axis=0)

    def run(variant):
        evals = 0.0
        loads = 0.0
        t_all = perf_counter()
        graveyard = []
        for la in range(LAYERS):
            experts = list(base[la])
            t0 = perf_counter()
            if variant == "io-only":
                list(pool.map(read_only, [index[next(fresh_ids)] for _ in range(PER_LAYER)]))
            elif variant in ("promote-keep", "promote-evict"):
                new = list(pool.map(load, [index[next(fresh_ids)] for _ in range(PER_LAYER)]))
                experts = experts[PER_LAYER:] + new
                if variant == "promote-evict":
                    base[la] = experts  # old ones dropped -> freed now
                else:
                    graveyard.extend(base[la][:PER_LAYER])
                    base[la] = experts
            elif variant == "pagecache":
                ents = [index[(3, la * 3 + i)] for i in range(PER_LAYER)]  # same files every run -> cached
                new = list(pool.map(load, ents))
                experts = experts[PER_LAYER:] + new
                base[la] = experts
            loads += perf_counter() - t0
            y = kernel(experts)
            t0 = perf_counter()
            mx.eval(y)
            evals += perf_counter() - t0
        print(f"{variant:14s}: total {perf_counter() - t_all:5.2f}s | loads {loads:5.2f}s | eval sum {evals * 1e3:6.1f} ms "
              f"({evals / LAYERS * 1e3:4.2f} ms/layer) | mlx active {mx.get_active_memory() / 2**30:.1f} GiB cache {mx.get_cache_memory() / 2**30:.2f}",
              flush=True)

    for v in ("resident", "resident", "io-only", "resident", "promote-keep", "resident", "promote-evict", "resident", "pagecache", "pagecache", "resident"):
        run(v)


if __name__ == "__main__":
    main()
