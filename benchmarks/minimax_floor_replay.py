"""
MiniMax-M3 decode floor by replay (HANDOFF 18.2): the same decode token at the same position, again and again.

`minimax_decode_floor.py` feeds one token repeatedly, so the context grows and the routing keeps drifting
(hundreds of misses in its "all-hit" half). Here every KV cache is rewound by one position after each step,
so each step is the identical computation: after the first, every routed expert is a hit and what is left is
the GPU work plus the host's per-layer work. Prints the median and minimum over STEPS, then a cProfile.

CONTEXT=N prefills N tokens of FILLER_FILE first (default 64), so the floor can be read at a long context.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    env CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=80 MLX_METAL_FAST_SYNCH=1 PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/minimax_floor_replay.py /Users/hamedprooshani/MiniMax-M3-MLX-3bit
"""
import cProfile
import os
import pstats
import statistics
import sys
import time

import mlx.core as mx

from cachalot.minimax.model import MiniMaxModel

MODEL = sys.argv[1]
STEPS = int(os.environ.get("STEPS", "30"))
PROFILE = int(os.environ.get("PROFILE", "10"))
CONTEXT = int(os.environ.get("CONTEXT", "64"))

m = MiniMaxModel(MODEL, expert_budget_gib=float(os.environ.get("GLM_BUDGET_GIB", "52")), heartbeat_seconds=0,
                 verbose=False)
text = open(os.environ.get("FILLER_FILE", "docs/HANDOFF.md")).read()
tokens = m.tokenizer.encode(text, add_special_tokens=False)
while len(tokens) < CONTEXT + 1:
    tokens += tokens
cache = m.new_cache()
mx.eval(m.prefill(tokens[:CONTEXT], cache) if hasattr(m, "prefill") else m._forward(tokens[:CONTEXT], cache))
m.store.release_prefill()
tok = tokens[CONTEXT]


def step():
    out = m._forward([tok], cache)
    mx.eval(out)
    for c in cache:
        c.offset -= 1
    return out


ref = step()
times, misses = [], 0
for _ in range(STEPS):
    s0 = m.store.stats().cache_misses
    t0 = time.perf_counter()
    out = step()
    times.append(time.perf_counter() - t0)
    misses += m.store.stats().cache_misses - s0
same = bool(mx.array_equal(out, ref).item())
print(f"FLOOR context={CONTEXT} steps={STEPS} median_ms={1000 * statistics.median(times):.1f} "
      f"min_ms={1000 * min(times):.1f} misses={misses} logits_repeat_identical={same} "
      f"argmax={int(mx.argmax(out).item())} logit_sum={float(out.astype(mx.float32).sum().item()):.3f}", flush=True)

if os.environ.get("LOGITS_OUT"):
    import numpy as np

    np.save(os.environ["LOGITS_OUT"], np.array(out.astype(mx.float32)))

if PROFILE:
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for _ in range(PROFILE):
        step()
    pr.disable()
    print(f"PROFILE wall_ms_per_token={1000 * (time.perf_counter() - t0) / PROFILE:.1f}")
    pstats.Stats(pr).sort_stats("tottime").print_stats(15)
m.close()
