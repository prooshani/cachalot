"""
Speed / quality trade-off of the opt-in decode miss budget.

For each budget: teacher-forced pass over the exact greedy continuation
(argmax agreement, mean KL(exact || approx)) and a free-running greedy
generation (tok/s, text, agreement prefix with the exact text).

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/miss_budget_quality.py --tokens 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR, write_json  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

PROMPT = ("Explain, in a short paragraph for a software engineer, why a mixture-of-experts model "
          "streamed from an SSD is bandwidth-bound during decoding and what a cache can and cannot do about it.")


def kl(exact_logits, approx_logits):
    p = mx.softmax(exact_logits.astype(mx.float32))
    logp = mx.log(p + 1e-30)
    logq = mx.log(mx.softmax(approx_logits.astype(mx.float32)) + 1e-30)
    return float(mx.sum(p * (logp - logq)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--budgets", default="16,8,4,2,0")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    report = {}

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = list(rt.tokenizer.encode(enc.encode_messages([{"role": "user", "content": PROMPT}], thinking_mode="chat", reasoning_effort=None)))
        rt.reset()
        rt.prefill_tokens(ids)
        store = rt.expert_store

        def run(budget, teacher=None):
            # Real usage: the prompt was just prefilled (its experts are warm),
            # earlier generations are not. Re-prefill instead of restoring.
            rt.reset()
            r0 = rt.prefill_tokens(ids)
            mx.eval(r0.logits)
            rt.restore(rt.snapshot(logits=r0.logits))
            store.decode_miss_budget = budget
            store.skipped_experts = 0
            misses0 = store.stats().cache_misses
            logits_seq = [r0.logits]
            tokens = []
            tok = int(r0.logits.argmax().item())
            t0 = perf_counter()
            for i in range(args.tokens):
                if teacher is not None:
                    tok = teacher[i]
                tokens.append(tok)
                r = rt.decode_token(tok)
                mx.eval(r.logits)
                logits_seq.append(r.logits)
                tok = int(r.logits.argmax().item())
            dt = perf_counter() - t0
            return tokens, logits_seq, dt, store.stats().cache_misses - misses0, store.skipped_experts

        exact_tokens, exact_logits, dt, misses, _ = run(None)
        exact_text = rt.tokenizer.decode(exact_tokens)
        print(f"exact: {args.tokens / dt:.2f} tok/s, {misses / args.tokens:.0f} misses/token\n  {exact_text!r}", flush=True)
        report["exact"] = {"tok_per_s": args.tokens / dt, "misses_per_token": misses / args.tokens, "text": exact_text}

        for b in budgets:
            # teacher-forced quality
            _, approx_logits, _, _, _ = run(b, teacher=exact_tokens)
            agree = sum(int(a.argmax().item()) == int(e.argmax().item()) for a, e in zip(approx_logits[1:], exact_logits[1:], strict=True))
            kls = [kl(e, a) for e, a in zip(exact_logits[1:], approx_logits[1:], strict=True)]
            # free-running speed + text
            toks, _, dt, misses, skipped = run(b)
            text = rt.tokenizer.decode(toks)
            prefix = 0
            while prefix < len(toks) and toks[prefix] == exact_tokens[prefix]:
                prefix += 1
            print(f"budget {b:2d}: {args.tokens / dt:.2f} tok/s | loaded {misses / args.tokens:.0f} misses/token, skipped {skipped / args.tokens:.0f} experts/token "
                  f"| teacher-forced argmax agreement {agree}/{args.tokens} | mean KL {sum(kls) / len(kls):.4f} max {max(kls):.3f} "
                  f"| greedy identical prefix {prefix}/{args.tokens}\n  {text!r}", flush=True)
            report[str(b)] = {"tok_per_s": args.tokens / dt, "misses_per_token": misses / args.tokens,
                              "skipped_per_token": skipped / args.tokens, "argmax_agreement": agree / args.tokens,
                              "mean_kl": sum(kls) / len(kls), "max_kl": max(kls), "identical_prefix": prefix, "text": text}
        store.decode_miss_budget = None

    write_json(RESULTS_DIR / "miss_budget_quality.json", report)


if __name__ == "__main__":
    main()
