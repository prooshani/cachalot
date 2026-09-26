"""
MiniMax-M3 decode floor by replay (HANDOFF 18.2): the same decode token at the same position, again and again.

`minimax_decode_floor.py` feeds one token repeatedly, so the context grows and the routing keeps drifting
(hundreds of misses in its "all-hit" half). Here every KV cache is rewound by one position after each step,
so each step is the identical computation: after the first, every routed expert is a hit and what is left is
the GPU work plus the host's per-layer work. Prints the median and minimum over STEPS, then a cProfile.

CONTEXT=N prefills N tokens of FILLER_FILE first (default 64), so the floor can be read at a long context.
IDLE_MS=T sleeps T ms inside the store lookup of every IDLE_EVERY-th MoE layer (default every 2nd), the way a
miss makes the GPU and the host wait on a read, with no I/O; the report subtracts the sleeps ("other_ms").

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
# ROLE=1: the Darwin role serve/chat apply (UI_FOCAL); QOS=1: the main thread at QOS_CLASS_USER_INTERACTIVE
if os.environ.get("ROLE") == "1":
    from cachalot.darwin_role import apply_darwin_role

    print("darwin role", apply_darwin_role())
if os.environ.get("QOS") == "1":
    import ctypes

    print("qos", ctypes.CDLL(None).pthread_set_qos_class_self_np(0x21, 0))
IDLE_MS = float(os.environ.get("IDLE_MS", "0"))
IDLE_EVERY = int(os.environ.get("IDLE_EVERY", "2"))
IDLE_SPIN = os.environ.get("IDLE_SPIN", "0") == "1"  # busy-wait instead of sleeping: the host stays on-core
# while spinning, evaluate a small GPU op every IDLE_GPU_MS so the GPU does not idle through the wait
IDLE_GPU_MS = float(os.environ.get("IDLE_GPU_MS", "0"))
_poke = mx.ones((256, 256), dtype=mx.float16)
mx.eval(_poke)
# IDLE_READ=1: instead of sleeping, read a not-yet-read expert from the drive into a scratch buffer nobody
# computes on (the same I/O a miss does, with the GPU's data unchanged)
IDLE_READ = os.environ.get("IDLE_READ", "0") == "1"
if IDLE_READ:
    from cachalot.glm.experts import tensor_sizes

    from concurrent.futures import ThreadPoolExecutor

    _read_pool = ThreadPoolExecutor(1)
    _scratch = {name: bytearray(n) for name, n in tensor_sizes(m.expert_format).items()}
    _walk = iter(sorted(m.expert_index.items(), key=lambda kv: (kv[0][1] * 7919 + kv[0][0]) % 7296))
slept = [0.0]
if IDLE_MS > 0:
    _get_many = m.store.get_many
    calls = [0]

    def get_many_idle(entries, prefetch=None, **kw):
        calls[0] += 1
        if calls[0] % IDLE_EVERY == 0:
            t = time.perf_counter()
            if IDLE_READ and IDLE_GPU_MS > 0:
                fut = _read_pool.submit(m.store.reader.read_expert_into, next(_walk)[1], _scratch)
                while not fut.done():
                    mx.eval(_poke @ _poke)
                    tp = time.perf_counter()
                    while time.perf_counter() - tp < IDLE_GPU_MS / 1000 and not fut.done():
                        pass
            elif IDLE_READ:
                m.store.reader.read_expert_into(next(_walk)[1], _scratch)
            elif IDLE_SPIN:
                last = t
                while time.perf_counter() - t < IDLE_MS / 1000:
                    if IDLE_GPU_MS > 0 and time.perf_counter() - last >= IDLE_GPU_MS / 1000:
                        mx.eval(_poke @ _poke)
                        last = time.perf_counter()
            else:
                time.sleep(IDLE_MS / 1000)
            slept[0] += time.perf_counter() - t
        return _get_many(entries, prefetch=prefetch, **kw)

    m.store.get_many = get_many_idle


def step():
    out = m._forward([tok], cache)
    mx.eval(out)
    for c in cache:
        c.offset -= 1
    return out


ref = step()
times, others, misses = [], [], 0
for _ in range(STEPS):
    s0 = m.store.stats().cache_misses
    slept[0] = 0.0
    t0 = time.perf_counter()
    out = step()
    times.append(time.perf_counter() - t0)
    others.append(times[-1] - slept[0])
    misses += m.store.stats().cache_misses - s0
same = bool(mx.array_equal(out, ref).item())
print(f"FLOOR context={CONTEXT} steps={STEPS} median_ms={1000 * statistics.median(times):.1f} "
      f"min_ms={1000 * min(times):.1f} other_ms={1000 * statistics.median(others):.1f} idle_ms={IDLE_MS} "
      f"idle_every={IDLE_EVERY} spin={int(IDLE_SPIN)} read={int(IDLE_READ)} gpu_poke_ms={IDLE_GPU_MS} misses={misses} logits_repeat_identical={same} "
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
