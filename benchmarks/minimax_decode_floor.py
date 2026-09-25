"""
MiniMax-M3 (or GLM) decode floor: ms per token when every routed expert is already resident (HANDOFF 18.1).

Prefills a short prompt, then feeds the same token again and again: after a few steps its routing stops
changing, every expert is a hit, and what is left is compute plus the host's work per token. Prints the
floor, then a cProfile of FLOOR_PROFILE tokens (host time by function) and a Metal-free host estimate.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    env CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=80 MLX_METAL_FAST_SYNCH=1 PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/minimax_decode_floor.py /Users/hamedprooshani/MiniMax-M3-MLX-3bit
"""
import cProfile
import json
import os
import pstats
import sys
import time

import mlx.core as mx

MODEL = sys.argv[1]
STEPS = int(os.environ.get("FLOOR_STEPS", "40"))
PROFILE = int(os.environ.get("FLOOR_PROFILE", "10"))
if json.load(open(os.path.join(MODEL, "config.json"))).get("model_type", "").startswith("minimax"):
    from cachalot.minimax.model import MiniMaxModel as Model
else:
    from cachalot.glm.model import GlmModel as Model

m = Model(MODEL, expert_budget_gib=float(os.environ.get("GLM_BUDGET_GIB", "52")), heartbeat_seconds=0)
cache = m.new_cache()
prompt = m.tokenizer.encode("The quick brown fox jumps over the lazy dog.", add_special_tokens=False)
mx.eval(m._forward(prompt, cache))
tok = prompt[-1]


def step():
    out = m._forward([tok], cache)
    mx.eval(out)


times = []
for i in range(STEPS):
    s0 = m.store.stats()
    t0 = time.perf_counter()
    step()
    dt = time.perf_counter() - t0
    s1 = m.store.stats()
    times.append((dt, s1.cache_misses - s0.cache_misses))
tail = [t for t, miss in times[STEPS // 2:]]
misses = sum(miss for _, miss in times[STEPS // 2:])
print(f"FLOOR steps={STEPS} last_half_ms={1000 * sum(tail) / len(tail):.1f} min_ms={1000 * min(tail):.1f} "
      f"misses_last_half={misses}", flush=True)

if PROFILE:
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for _ in range(PROFILE):
        step()
    pr.disable()
    wall = (time.perf_counter() - t0) / PROFILE
    st = pstats.Stats(pr)
    print(f"PROFILE wall_ms_per_token={1000 * wall:.1f}")
    st.sort_stats("tottime").print_stats(18)
m.close()
