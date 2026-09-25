"""
GLM-5.3-Flash prefill timeline (HANDOFF section 17.1); MiniMax-M3 too (CACHALOT_MODEL_PATH at its checkpoint).

Prefills N tokens of plain text (no chat template, no prefix cache) through `GlmModel.prefill` and prints one
line per chunk: seconds, seconds spent waiting in the expert store, expert misses and GiB read. Ends with one
RESULT line and, with LOGITS_OUT=path.npy, saves the final logits (float32) so two arms on the same text can
be compared for quality (`--compare a.npy b.npy`: max |diff|, argmax agreement, KL of the softmax). NLL_LAST=K scores the last K
tokens teacher-forced (the saved file then holds their log-probs, compared position by position);
DECODE_TOKENS=D decodes D greedy tokens after the prefill and reports tok/s and the expert hit rate;
ROUNDS=R repeats prefill-then-decode R times on one cache, each round on the next N tokens (an agent's turns).

Speed A/Bs need fresh text per arm, or the second arm reads experts the first left in the page cache
(section 15.12): FILLER_FILE (default docs/HANDOFF.md) and FILLER_OFFSET (characters).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    env CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=80 PYTHONPATH=src FILLER_OFFSET=0 \
      ~/venvs/deepseek-v41/bin/python benchmarks/glm_prefill_timeline.py 8192
    ~/venvs/deepseek-v41/bin/python benchmarks/glm_prefill_timeline.py --compare a.npy b.npy
"""
import os
import sys
import time

import numpy as np


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    if a.ndim == 2 and a.shape[0] > 1:  # per-position log-probs (NLL_LAST)
        n = min(a.shape[0], b.shape[0])  # both end at the last token
        a, b = a[-n:].astype(np.float64), b[-n:].astype(np.float64)
        kl = (np.exp(a) * (a - b)).sum(axis=-1)
        agree = (a.argmax(-1) == b.argmax(-1)).mean()
        print(f"COMPARE positions={a.shape[0]} mean_kl={kl.mean():.3e} p99_kl={np.quantile(kl, 0.99):.3e} "
              f"max_kl={kl.max():.3e} top1_agree={agree:.4f}")
        return
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()

    def logsoftmax(x):
        x = x - x.max()
        return x - np.log(np.exp(x).sum())

    la, lb = logsoftmax(a), logsoftmax(b)
    kl = float((np.exp(la) * (la - lb)).sum())
    top_a, top_b = np.argsort(-a)[:5], np.argsort(-b)[:5]
    print(f"COMPARE max_abs={np.abs(a - b).max():.4f} argmax={int(top_a[0])}/{int(top_b[0])} "
          f"top5_overlap={len(set(top_a) & set(top_b))} kl={kl:.3e}")


if sys.argv[1] == "--compare":
    compare(sys.argv[2], sys.argv[3])
    sys.exit(0)

import mlx.core as mx  # noqa: E402

from cachalot.glm.model import GlmModel  # noqa: E402

N = int(sys.argv[1])
NLL_LAST = int(os.environ.get("NLL_LAST", "0"))
nll, logprobs = float("nan"), None
MODEL = os.environ.get("CACHALOT_MODEL_PATH", "/Volumes/X10Pro/models/GLM-5.3-Flash-MLX-4bit-MTP")
filler_file = os.environ.get("FILLER_FILE", "docs/HANDOFF.md")
offset = int(os.environ.get("FILLER_OFFSET", "0"))

import json as _json  # noqa: E402

if _json.load(open(os.path.join(MODEL, "config.json"))).get("model_type", "").startswith("minimax_m3"):
    from cachalot.minimax.model import MiniMaxModel as GlmModel  # same interface (HANDOFF 18)
m = GlmModel(MODEL, expert_budget_gib=float(os.environ.get("GLM_BUDGET_GIB", "52")), heartbeat_seconds=0)
text = open(filler_file).read()[offset:]
tokens = list(m.tokenizer.encode(text, add_special_tokens=False))
if not tokens:
    sys.exit(f"no text in {filler_file} after offset {offset}")
ROUNDS = int(os.environ.get("ROUNDS", "1"))
while len(tokens) < N * ROUNDS:
    tokens += tokens
# TF_DECODE=K: after the last round, teacher-force the next K tokens of text one at a time through the decode
# path (qmv kernels, the decode hooks): their NLL, ms per token, and log-probs (TF_OUT) for --compare (HANDOFF 18.2)
TF_DECODE = int(os.environ.get("TF_DECODE", "0"))
while len(tokens) < N * ROUNDS + TF_DECODE + 1:
    tokens += tokens
tf_tokens = tokens[N * ROUNDS - 1:N * ROUNDS + TF_DECODE]
# each round gets its own N tokens (before 2026-09-25 every round repeated the first N)
tokens = tokens[:N * ROUNDS]

store = m.store
wait = [0.0]


def timed(fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            wait[0] += time.perf_counter() - t
    return wrapper


for name in ("get_many", "get_many_prefill", "release_prefill_layer"):
    if hasattr(store, name):
        setattr(store, name, timed(getattr(store, name)))

# ROUTE_TRACE=path.npy: every decode request as (token, layer, expert), for an offline cache-policy replay
# (benchmarks/minimax_policy_replay.py)
ROUTE_TRACE = os.environ.get("ROUTE_TRACE")
route_log, route_step = [], [0]
if ROUTE_TRACE:
    _get_many = store.get_many

    def traced_get_many(entries, *a, **k):
        for e in entries:
            route_log.append((route_step[0], e.layer, e.expert))
        return _get_many(entries, *a, **k)
    store.get_many = traced_get_many


def vm():
    import re
    import subprocess
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    n = {k: int(v) for k, v in re.findall(r"^(.+?):\s+(\d+)\.", out, re.M)}
    g = lambda k: n.get(k, 0) * 16384 / 2**30  # noqa: E731
    return (f"free {g('Pages free'):.1f} wired {g('Pages wired down'):.1f} "
            f"comp {g('Pages occupied by compressor'):.1f} swapouts {n.get('Swapouts', 0)}")

cache = m.new_cache()
chunk = m.PREFILL_CHUNK
DECODE = int(os.environ.get("DECODE_TOKENS", "0"))
while len(tokens) < N * ROUNDS:
    tokens += tokens
all_tokens = tokens
for rnd in range(ROUNDS):
    # each round prefills the next N tokens of text onto the same cache, then decodes: an agent's turn
    tokens = all_tokens[rnd * N:(rnd + 1) * N]
    rows = []
    t_all = time.perf_counter()
    logits = None
    for start in range(0, N, chunk):
        s0, w0, t0 = store.stats(), wait[0], time.perf_counter()
        if NLL_LAST and start + chunk >= N:
            # last chunk: every position's logits, for a teacher-forced NLL over the last NLL_LAST tokens
            part = tokens[start:start + chunk]
            out = m.model(mx.array(part, dtype=mx.int32)[None], cache=cache)
            full = getattr(out, "logits", out)[0]  # GLM returns an object, MiniMax an array
            k = min(NLL_LAST, len(part) - 1)
            scored = full[-k - 1:-1].astype(mx.float32)  # cast only the scored rows (8k x 200k fp32 is 6.5 GB)
            lp = scored - mx.logsumexp(scored, axis=-1, keepdims=True)
            target = mx.array(part[-k:], dtype=mx.int32)
            nll = float((-mx.take_along_axis(lp, target[:, None], axis=-1)).mean().item())
            logprobs = lp.astype(mx.float16)
            logits = full[-1:].astype(mx.float32)
        else:
            logits = m._forward(tokens[start:start + chunk], cache)
        mx.eval(logits)
        s1, dt = store.stats(), time.perf_counter() - t0
        misses = s1.cache_misses - s0.cache_misses
        gib = (s1.ssd_bytes_read - s0.ssd_bytes_read) / 2**30
        rows.append((dt, wait[0] - w0, misses, gib))
        print(f"chunk {start:>6}-{min(start + chunk, N):>6}  {dt:7.1f} s  store-wait {wait[0] - w0:6.1f} s  "
              f"misses {misses:5d}  read {gib:6.1f} GiB  ({gib / dt:4.1f} GiB/s)  {vm()}", flush=True)
    store.release_prefill()
    total = time.perf_counter() - t_all
    lg = logits.astype(mx.float32)
    if DECODE:
        # greedy decode after the prefill: tok/s and expert hit rate, to see what the prefill left in the cache
        d0, t0, tok, out, w0 = store.stats(), time.perf_counter(), int(lg.argmax().item()), [], wait[0]
        for _ in range(DECODE):
            out.append(tok)
            route_step[0] += 1
            step = m._forward([tok], cache)
            mx.eval(step)
            tok = int(step.argmax().item())
        d1, dt = store.stats(), time.perf_counter() - t0
        hits, misses = d1.cache_hits - d0.cache_hits, d1.cache_misses - d0.cache_misses
        reads, fast = d1.reads - d0.reads, d1.fast_reads - d0.fast_reads
        rwall = d1.read_wall_seconds - d0.read_wall_seconds
        print("DECODE round=%d tokens=%d tok_s=%.2f ms_per_token=%.1f store_wait_ms=%.1f hit_rate=%.3f "
              "misses_per_token=%.1f read_ms_avg=%.2f fast_reads=%d/%d ids_head=%s" % (
                  rnd, DECODE, DECODE / dt, 1000 * dt / DECODE, 1000 * (wait[0] - w0) / DECODE,
                  hits / max(1, hits + misses), misses / DECODE, 1000 * rwall / max(1, reads), fast, reads,
                  out[:12]), flush=True)
        if os.environ.get("DECODE_IDS_OUT"):
            np.save(os.environ["DECODE_IDS_OUT"] + f".r{rnd}.npy", np.array(out))
    if os.environ.get("LOGITS_OUT") and rnd == ROUNDS - 1:
        np.save(os.environ["LOGITS_OUT"], np.array(logprobs) if logprobs is not None else np.array(lg))
    print("RESULT round=%d n=%d chunk=%d filler=%s@%d total=%.1f tok_s=%.1f store_wait=%.1f misses=%d read_gib=%.1f "
          "peak_gib=%.1f nll_last%d=%.5f argmax=%d logit_sum=%.4f" % (
              rnd, N, chunk, os.path.basename(filler_file), offset, total, N / total, sum(r[1] for r in rows),
              sum(r[2] for r in rows), sum(r[3] for r in rows), mx.get_peak_memory() / 2**30, NLL_LAST, nll,
              int(lg.argmax().item()), float(lg.sum().item())), flush=True)
if TF_DECODE:
    d0, w0, t0, lps, nlls = store.stats(), wait[0], time.perf_counter(), [], []
    for i in range(TF_DECODE):
        step = m._forward([tf_tokens[i]], cache).astype(mx.float32)
        lp = step - mx.logsumexp(step, axis=-1, keepdims=True)
        mx.eval(lp)
        lps.append(np.array(lp[0]).astype(np.float16))
        nlls.append(-float(lps[-1][tf_tokens[i + 1]]))
    d1, dt = store.stats(), time.perf_counter() - t0
    hits, misses = d1.cache_hits - d0.cache_hits, d1.cache_misses - d0.cache_misses
    print("TF_DECODE tokens=%d nll=%.5f ms_per_token=%.1f store_wait_ms=%.1f other_ms=%.1f hit_rate=%.3f "
          "misses_per_token=%.1f" % (
              TF_DECODE, float(np.mean(nlls)), 1000 * dt / TF_DECODE, 1000 * (wait[0] - w0) / TF_DECODE,
              1000 * (dt - (wait[0] - w0)) / TF_DECODE, hits / max(1, hits + misses), misses / TF_DECODE), flush=True)
    if os.environ.get("TF_OUT"):
        np.save(os.environ["TF_OUT"], np.stack(lps))
if ROUTE_TRACE:
    np.save(ROUTE_TRACE, np.array(route_log, dtype=np.int32))
m.close()
