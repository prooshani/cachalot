"""
Short follow-up prefill vs. decoding the same tokens one by one.

Chat turn 2+ prefills only the new message (10-30 tokens) after restoring the
prefix-cache snapshot. Measured in the terminal chat: 13 new tokens took
10.5 s (0.8 s/token) while decode runs at 0.3 s/token. This benchmark
isolates that: prefill a 512-token context, snapshot it, then consume N new
tokens either through prefill_tokens (layer-major, per-layer admission quota)
or through N decode_token calls (LRU admission, one-layer-early prefetch),
and report wall time, expert misses and SSD bytes for each path, plus the
logits parity between the two.

Run each mode in its own process so the first mode's expert loads do not
warm the cache for the second:

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    for m in prefill decode; do
      benchmarks/guarded_run.sh --budget-gib 28 --tag short_$m -- env PYTHONPATH=src \
        ~/venvs/deepseek-v41/bin/python benchmarks/short_prefill.py --mode $m --new-tokens 16
    done
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, StoreSnapshot  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402

FOLLOW_UP = ("Now summarise the previous answer in exactly three short bullet points, "
             "then suggest one concrete follow-up experiment that a careful engineer would run next, "
             "and explain briefly why it matters for the overall performance of the system.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prefill", "decode"], required=True)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=args.max_seq_len) as rt:
        print(f"runtime ready (expert budget {rt.expert_cache_budget_bytes / 2**30:.1f} GiB, "
              f"wired {rt.mlx_wired_limit_bytes / 2**30:.1f} GiB)", flush=True)
        enc = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)
        suffix = list(rt.tokenizer.encode(FOLLOW_UP))[: args.new_tokens]
        assert len(suffix) == args.new_tokens, f"follow-up text too short: {len(suffix)} tokens"

        rt.reset()
        t0 = perf_counter()
        res = rt.prefill_tokens(ids)
        mx.eval(res.logits)
        print(f"context prefill {len(ids)} tokens: {perf_counter() - t0:.1f} s", flush=True)
        t0 = perf_counter()
        snap = rt.snapshot(res.logits)
        t_snap = perf_counter() - t0
        t0 = perf_counter()
        rt.restore(snap)
        t_restore = perf_counter() - t0
        t0 = perf_counter()
        found = rt.prefix_cache.find(tuple(ids))
        t_find = perf_counter() - t0
        print(f"snapshot {t_snap:.2f} s | restore {t_restore:.2f} s | prefix_cache.find {t_find:.3f} s (hit={found is not None}) "
              f"| max_seq_len {args.max_seq_len}", flush=True)

        before = StoreSnapshot.take(rt)
        st = rt.expert_store
        pred_before = (st.predicted_loads, st.predicted_used)
        t0 = perf_counter()
        if args.mode == "prefill":
            out = rt.prefill_tokens(suffix)
            logits = out.logits
        else:
            for tok in suffix:
                out = rt.decode_token(tok)
            logits = out.logits
        mx.eval(logits)
        dt = perf_counter() - t0
        d = before.delta(StoreSnapshot.take(rt))
        n = args.new_tokens
        print(f"{args.mode}: {n} new tokens in {dt:.2f} s = {dt / n * 1e3:.0f} ms/token | "
              f"expert misses {d.cache_misses} ({d.cache_misses / n:.1f}/token, SSD floor {d.cache_misses * 3.4e-3:.1f} s) | "
              f"hits {d.cache_hits} | read {d.ssd_bytes_read / 1e9:.1f} GB in {d.ssd_read_seconds:.1f} thread-s | "
              f"predicted loads {st.predicted_loads - pred_before[0]} used {st.predicted_used - pred_before[1]}")
        top = mx.argsort(logits)[-5:].tolist()
        print(f"{args.mode}: top-5 next-token ids {top[::-1]} | position {rt.position} | "
              f"resident experts {len(st)}")
        print("text:", repr(rt.tokenizer.decode(top[::-1])))


if __name__ == "__main__":
    main()
