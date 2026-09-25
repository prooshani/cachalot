"""
MiniMax-M3 (or GLM) agent-style follow-up turns (HANDOFF 18.2): prefill N tokens, decode D, then TURNS times
prefill SHORT new tokens of text (a tool result) and decode D greedy tokens. Per turn: the short prefill's seconds
and misses, then the reply's ms per token and misses per token. Same text for every arm (FILLER_FILE/OFFSET).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    env CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=80 MLX_METAL_FAST_SYNCH=1 PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/minimax_followup_turns.py /Users/hamedprooshani/MiniMax-M3-MLX-3bit
"""
import os
import statistics
import sys
import time


from cachalot.minimax.model import MiniMaxModel

MODEL = sys.argv[1]
N = int(os.environ.get("N", "2048"))
D = int(os.environ.get("D", "48"))
SHORT = [int(v) for v in os.environ.get("SHORT", "30,60,120,30,250,30").split(",")]

m = MiniMaxModel(MODEL, expert_budget_gib=float(os.environ.get("GLM_BUDGET_GIB", "52")), heartbeat_seconds=0,
                 verbose=False)
text = open(os.environ.get("FILLER_FILE", "docs/HANDOFF.md")).read()[int(os.environ.get("FILLER_OFFSET", "0")):]
tokens = m.tokenizer.encode(text, add_special_tokens=False)
cache = m.new_cache()
pos = 0


def prefill(n):
    global pos
    s0, t0 = m.store.stats(), time.perf_counter()
    logits = m.prefill(tokens[pos:pos + n], cache)
    pos += n
    s1 = m.store.stats()
    return logits, time.perf_counter() - t0, s1.cache_misses - s0.cache_misses


def decode(logits, n):
    s0, t0, tok, ids = m.store.stats(), time.perf_counter(), int(logits.argmax().item()), []
    for _ in range(n):
        ids.append(tok)
        out = m._forward([tok], cache)
        tok = int(out.argmax().item())
    s1 = m.store.stats()
    return 1000 * (time.perf_counter() - t0) / n, (s1.cache_misses - s0.cache_misses) / n, ids


logits, dt, miss = prefill(N)
ms, mpt, _ = decode(logits, D)
print(f"TURN 0 prefill={N} {dt:.1f}s decode {ms:.1f} ms/token misses/token {mpt:.1f}", flush=True)
pre_s, dec_ms, all_ids = [], [], []
for i, n in enumerate(SHORT, 1):
    logits, dt, miss = prefill(n)
    ms, mpt, ids = decode(logits, D)
    pre_s.append(dt)
    dec_ms.append(ms)
    all_ids += ids
    print(f"TURN {i} prefill={n} {dt:.2f}s misses={miss} decode {ms:.1f} ms/token misses/token {mpt:.1f} "
          f"ids_head={ids[:6]}", flush=True)
print(f"RESULT short_prefill_total_s={sum(pre_s):.2f} decode_mean_ms={statistics.mean(dec_ms):.1f} "
      f"turns_total_s={sum(pre_s) + sum(dec_ms) * D / 1000:.1f} ids_hash={hash(tuple(all_ids))}", flush=True)
m.close()
