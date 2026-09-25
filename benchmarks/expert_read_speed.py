"""
How fast do routed experts arrive from the drive, one at a time or several at once (HANDOFF 18.1)?

Times `ExpertReader.read_expert_into` on random experts, each read once only. Use a different SEED per run:
experts read by an earlier run are page-cache hits even with F_NOCACHE (that is how a first sweep on
2026-09-25 showed a false 8.8 GiB/s). A `--splits` sweep (each piece cut into preads of at most S MiB)
was measured the same day and found no gain at any queue depth; the option is gone, the result is in 18.1.

    SEED=11 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/expert_read_speed.py MODEL [--n 120] [--concurrent 1]
--concurrent K reads K experts at once per sample (a layer with several misses, or prediction in flight).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cachalot.storage.reader import ExpertReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--concurrent", type=int, default=1)
    ap.add_argument("--page-cache", action="store_true", help="read through the page cache as serve-minimax.sh does")
    ap.add_argument("--readahead", type=int, default=None, help="1/0: F_RDAHEAD on or off (default: the reader's)")
    args = ap.parse_args()
    root = Path(args.model)
    mt = json.loads((root / "config.json").read_text()).get("model_type", "")
    if mt.startswith("minimax"):
        from cachalot.minimax.experts import build_minimax_expert_index as build
    else:
        from cachalot.glm.experts import build_glm_expert_index as build
    from cachalot.glm.experts import tensor_sizes

    fmt, index = build(root)
    sizes = tensor_sizes(fmt)
    keys = list(index)
    random.seed(int(__import__('os').environ.get('SEED', '7')))
    random.shuffle(keys)
    rd = ExpertReader(bypass_page_cache=not args.page_cache,
                      readahead=None if args.readahead is None else bool(args.readahead))
    print(f"page cache {'on' if args.page_cache else 'off'}, readahead {rd.readahead}")
    bufs = [{k: bytearray(v) for k, v in sizes.items()} for _ in range(args.concurrent)]
    pool = ThreadPoolExecutor(args.concurrent)
    times = []
    for i in range(args.n):
        batch = keys[i * args.concurrent:(i + 1) * args.concurrent]
        t0 = time.perf_counter()
        list(pool.map(lambda kb: rd.read_expert_into(index[kb[0]], kb[1]), zip(batch, bufs)))
        times.append(time.perf_counter() - t0)
    mib = sum(sizes.values()) / 2**20 * args.concurrent
    t = np.array(times[2:]) * 1000
    print(f"concurrent {args.concurrent}  median {np.median(t):6.2f} ms  p90 {np.quantile(t, 0.9):6.2f} ms  "
          f"{mib / 1024 / (np.median(t) / 1000):4.2f} GiB/s", flush=True)


if __name__ == "__main__":
    main()
