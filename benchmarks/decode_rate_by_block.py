"""
Does a long reply decode more slowly than a short one, and if so, where?

Four interactive sessions put seven long turns between 7.95 tok/s at a
545-token reply and 5.61 at 1,788 (HANDOFF section 7.2.4), and
`decode_anatomy.py` at 512, 1024 and 2048 prompt tokens says context length is
worth about 4 ms per token across that whole range -- not the 29 % the sessions
showed. So the effect, if it is real, belongs to the *generation* rather than
to the context it started from, and no instrument in this repository decodes
long enough to see it: the anatomy stops at 64 tokens.

This decodes one long continuation in a single pass and reports the rate, the
hit rate, the misses and the bytes per token in blocks, so a curve over the
turn is visible rather than one average. It is the same decode loop as
`decode_throughput.py`, which it otherwise duplicates on purpose -- a rate
averaged over 1,792 tokens is exactly the number this is trying to take apart.

Read the blocks, not the total. A rate that falls monotonically over the turn
is a within-turn effect (the reply walks into experts the cache does not hold);
a flat curve says the live difference is the prompt's content -- every fast
turn in those sessions was prose and every slow one was code.

    benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 3600 --tag rateblk -- env ... \\
        python benchmarks/decode_rate_by_block.py --prompt-tokens 512 --decode-tokens 1792
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

MIB = 1024**2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=1792)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--source", type=int, default=0, help="index into prompt_sources()")
    # 4096 holds a 512-token prompt and a 1792-token reply with room to spare.
    # 8192 doubles the window and compressed-KV allocations for nothing and was
    # enough on its own to push a 36 GiB budget into the compressor.
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=args.max_seq_len) as rt:
        print(f"runtime ready (expert budget {rt.expert_cache_budget_bytes / 2**30:.1f} GiB, "
              f"wired {rt.mlx_wired_limit_bytes / 2**30:.1f} GiB)", flush=True)
        enc = load_official_encoding(MODEL_PATH)
        name, text = prompt_sources()[args.source]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)
        rt.reset()
        res = rt.prefill_tokens(ids)
        mx.eval(res.logits)
        tok = int(res.logits.argmax().item())

        print(f"\nprompt {name}, {len(ids)} tokens; decoding {args.decode_tokens} "
              f"in blocks of {args.block}\n", flush=True)
        print("  block   tokens      context   tok/s   ms/token   hit rate   miss/tok   MiB/tok")

        out = []
        rows = []
        for start in range(0, args.decode_tokens, args.block):
            n = min(args.block, args.decode_tokens - start)
            before = StoreSnapshot.take(rt)
            t0 = perf_counter()
            for _ in range(n):
                r = rt.decode_token(tok)
                tok = int(r.logits.argmax().item())
                out.append(tok)
            dt = perf_counter() - t0
            d = before.delta(StoreSnapshot.take(rt))
            req = d.cache_hits + d.cache_misses
            hit = d.cache_hits / max(1, req)
            ctx = len(ids) + start + n
            rows.append((start, n, ctx, n / dt, dt / n * 1e3, hit,
                         d.cache_misses / n, d.ssd_bytes_read / MIB / n))
            print(f"  {start:5d}   {n:6d}   {ctx:10d}   {n / dt:5.2f}   {dt / n * 1e3:8.1f}   "
                  f"{hit:8.1%}   {d.cache_misses / n:8.1f}   {d.ssd_bytes_read / MIB / n:7.0f}",
                  flush=True)

        first, last = rows[0], rows[-1]
        print(f"\nfirst block {first[3]:.2f} tok/s at {first[2]} context, "
              f"last block {last[3]:.2f} at {last[2]}: "
              f"{(last[3] / first[3] - 1) * 100:+.1f} %")
        print(f"hit rate {first[5]:.1%} -> {last[5]:.1%}, "
              f"bytes {first[7]:.0f} -> {last[7]:.0f} MiB/token")
        st = rt.expert_store
        print(f"prediction: {st.predicted_loads} loads, {st.predicted_used} used "
              f"({st.predicted_used / max(1, st.predicted_loads):.0%} precision)")
        print("\ntail of the continuation:", repr(rt.tokenizer.decode(out[-120:])))


if __name__ == "__main__":
    main()
