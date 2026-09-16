"""Decode tok/s after a prompt: prefill N tokens, then greedy-decode M tokens; report speed, hit rate and the text."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=64)
    args = ap.parse_args()
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        print(f"runtime ready (expert budget {rt.expert_cache_budget_bytes / 2**30:.1f} GiB, "
              f"wired {rt.mlx_wired_limit_bytes / 2**30:.1f} GiB)", flush=True)
        enc = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)
        rt.reset()
        t0 = perf_counter()
        res = rt.prefill_tokens(ids)
        mx.eval(res.logits)
        t_prefill = perf_counter() - t0
        tok = int(res.logits.argmax().item())
        before = StoreSnapshot.take(rt)
        out = []
        t0 = perf_counter()
        for _ in range(args.decode_tokens):
            r = rt.decode_token(tok)
            tok = int(r.logits.argmax().item())
            out.append(tok)
        t_decode = perf_counter() - t0
        d = before.delta(StoreSnapshot.take(rt))
        hits = d.cache_hits / max(1, d.cache_hits + d.cache_misses)
        print(f"prefill {len(ids)} tokens: {t_prefill:.1f} s | decode {args.decode_tokens} tokens: {t_decode:.1f} s = "
              f"{args.decode_tokens / t_decode:.2f} tok/s ({t_decode / args.decode_tokens * 1e3:.0f} ms/token) | "
              f"expert hit rate {hits:.1%} | misses/token {d.cache_misses / args.decode_tokens:.1f}")
        st = rt.expert_store
        print(f"predicted loads {st.predicted_loads} used {st.predicted_used} "
              f"({st.predicted_used / max(1, st.predicted_loads):.0%} precision), "
              f"{st.predicted_used / args.decode_tokens:.1f} misses/token served early, "
              f"wasted {st.predicted_wasted_bytes / 18_800_640 / args.decode_tokens:.1f} loads/token")
        print("text:", repr(rt.tokenizer.decode(out)))


if __name__ == "__main__":
    main()
