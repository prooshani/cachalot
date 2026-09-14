"""
Prefix-cache parity and speed check on the real model.

Turn 1: short user prompt, greedy reply.
Turn 2: same conversation + assistant reply + new user message, run twice:
    (a) with the prefix cache (restore snapshot, prefill only the suffix)
    (b) full prefill from reset
and compare first-token logits, greedy continuations, and wall time.

Also isolates decode compute cost: the same token is decoded twice from the
same state; the second pass has every expert resident.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/validate_prefix_cache.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR, StoreSnapshot, write_json  # noqa: E402
from cachalot.model.generation import (  # noqa: E402
    SamplingParams,
    load_official_encoding,
    prepare_prompt,
    stream_tokens,
)
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402


def greedy(runtime, ids, n, use_prefix_cache):
    out = []
    events = {}
    for ev in stream_tokens(runtime, ids, SamplingParams(max_new_tokens=n, temperature=0.0),
                            use_prefix_cache=use_prefix_cache):
        if ev.kind == "token":
            out.append(ev.token)
        elif ev.kind in ("prefill", "done"):
            events[ev.kind] = ev
    return out, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--decode-tokens", type=int, default=12)
    args = ap.parse_args()

    report = {}
    with TextDecodeRuntime(args.model, max_seq_len=4096, verbose=False) as rt:
        enc = load_official_encoding(args.model)
        tok = rt.tokenizer

        def ids_for(messages):
            return list(tok.encode(enc.encode_messages(messages, thinking_mode="chat", reasoning_effort=None)))

        turn1 = [{"role": "user", "content": "In two sentences, why do sperm whales dive so deep?"}]
        ids1 = ids_for(turn1)
        reply1, ev1 = greedy(rt, ids1, args.decode_tokens, True)
        reply1_text = tok.decode(reply1, skip_special_tokens=False)
        print(f"turn1: prompt {len(ids1)} tokens, prefill {ev1['prefill'].prefill_seconds:.1f}s, "
              f"decode {ev1['done'].decode_seconds / max(len(reply1), 1):.2f}s/token, reply={reply1_text!r}", flush=True)
        report["turn1"] = {"prompt_tokens": len(ids1), "prefill_s": ev1["prefill"].prefill_seconds,
                           "decode_s_per_token": ev1["done"].decode_seconds / max(len(reply1), 1), "reply": reply1_text}

        turn2 = turn1 + [{"role": "assistant", "content": reply1_text},
                         {"role": "user", "content": "Now give one surprising fact about their sleep."}]
        ids2 = ids_for(turn2)
        common = 0
        seq1 = tuple(ids1) + tuple(reply1)
        while common < min(len(seq1), len(ids2)) and seq1[common] == ids2[common]:
            common += 1
        print(f"turn2: prompt {len(ids2)} tokens; common prefix with turn1 prompt+reply = {common}", flush=True)

        # (a) cached path: logits of first completion token
        t0 = perf_counter()
        res_a, reused = prepare_prompt(rt, ids2, reset=True, use_prefix_cache=True)
        ta = perf_counter() - t0
        logits_a = mx.array(res_a.logits)
        mx.eval(logits_a)
        print(f"  cached prefill: reused {reused}, prefilled {len(ids2) - reused}, {ta:.1f}s", flush=True)

        # (b) full path
        t0 = perf_counter()
        res_b, _ = prepare_prompt(rt, ids2, reset=True, use_prefix_cache=False)
        tb = perf_counter() - t0
        logits_b = mx.array(res_b.logits)
        mx.eval(logits_b)
        print(f"  full prefill: {len(ids2)} tokens, {tb:.1f}s", flush=True)

        diff = float(mx.max(mx.abs(logits_a - logits_b)).item())
        top_a, top_b = int(mx.argmax(logits_a).item()), int(mx.argmax(logits_b).item())
        print(f"  logits max|diff| = {diff:.3e}, argmax {top_a} vs {top_b}", flush=True)
        report["turn2"] = {"prompt_tokens": len(ids2), "common_prefix": common, "reused": reused,
                           "cached_prefill_s": ta, "full_prefill_s": tb, "logits_max_abs_diff": diff,
                           "argmax_equal": top_a == top_b}

        # greedy continuation parity (cached vs full)
        cont_a, _ = greedy(rt, ids2, args.decode_tokens, True)   # will reuse the snapshot just added
        cont_b, evb = greedy(rt, ids2, args.decode_tokens, False)
        print(f"  greedy cached: {tok.decode(cont_a)!r}\n  greedy full  : {tok.decode(cont_b)!r}", flush=True)
        report["turn2"]["continuation_equal"] = cont_a == cont_b

        # decode compute isolation: same token twice from the same state
        snap = rt.snapshot(logits=res_b.logits)
        token = top_b
        before = StoreSnapshot.take(rt)
        t0 = perf_counter()
        r1 = rt.decode_token(token)
        mx.eval(r1.logits)
        cold = perf_counter() - t0
        d1 = before.delta(StoreSnapshot.take(rt))
        rt.restore(snap)
        before = StoreSnapshot.take(rt)
        t0 = perf_counter()
        r2 = rt.decode_token(token)
        mx.eval(r2.logits)
        warm = perf_counter() - t0
        d2 = before.delta(StoreSnapshot.take(rt))
        same = bool(mx.array_equal(r1.logits, r2.logits).item())
        print(f"decode one token: cold {cold:.2f}s ({d1.cache_misses} misses), all-resident {warm:.2f}s "
              f"({d2.cache_misses} misses), logits identical={same}", flush=True)
        report["decode_isolation"] = {"cold_s": cold, "cold_misses": d1.cache_misses,
                                      "warm_s": warm, "warm_misses": d2.cache_misses, "logits_identical": same}

    write_json(RESULTS_DIR / "validate_prefix_cache.json", report)
    print("saved", RESULTS_DIR / "validate_prefix_cache.json")


if __name__ == "__main__":
    main()
