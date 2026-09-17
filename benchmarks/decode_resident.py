"""
What does a decode token cost when nothing has to be read?

Section 9.2 of docs/HANDOFF.md records an inconsistency worth settling before
any kernel work: `decode_anatomy.py` attributes 120 ms per token to "rest" --
everything that is not expert wait -- while HANDOFF-2026-09-16.md states in
passing that an all-resident autoregressive decode costs 68 ms per token. If the
second number still holds on the current bank, roughly 50 ms of every token is
neither arithmetic nor expert wait, and finding it is worth more than any kernel
work. If it does not, 120 ms is simply what the model costs and the lever is
ordinary optimization.

The measurement: prefill a short prompt, greedily decode a few tokens, then
reset and do exactly the same thing again. Greedy decode is deterministic, so
the second pass requests precisely the experts the first pass left resident.
When the second pass reports a hit rate at or near 100 %, its per-token time is
the all-resident cost with no reads in it at all.

Keep the window small. The point is a fully resident pass, and a 64-token decode
touches more experts than a safe budget holds; 16 tokens after a 16-token prompt
touches roughly 1,600 experts, which fits inside 36 GiB with room to spare.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh \
      --budget-gib 36 --max-seconds 1800 --tag resident -- env \
      CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 \
      CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
      benchmarks/decode_resident.py --prompt-tokens 16 --decode-tokens 16 --passes 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR, StoreSnapshot  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=16)
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        print(
            f"runtime ready (expert budget {rt.expert_cache_budget_bytes / 2**30:.1f} GiB, "
            f"wired {rt.mlx_wired_limit_bytes / 2**30:.1f} GiB)",
            flush=True,
        )
        enc = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)

        rows = []
        first_tokens = None

        for index in range(args.passes):
            rt.reset()
            before = StoreSnapshot.take(rt)

            t0 = perf_counter()
            result = rt.prefill_tokens(ids)
            mx.eval(result.logits)
            prefill_seconds = perf_counter() - t0

            token = int(result.logits.argmax().item())
            decoded = []

            t0 = perf_counter()
            for _ in range(args.decode_tokens):
                step = rt.decode_token(token)
                token = int(step.logits.argmax().item())
                decoded.append(token)
            decode_seconds = perf_counter() - t0

            delta = before.delta(StoreSnapshot.take(rt))
            requests = delta.cache_hits + delta.cache_misses
            hit_rate = delta.cache_hits / requests if requests else 0.0

            if first_tokens is None:
                first_tokens = decoded
            elif decoded != first_tokens:
                print(
                    "WARNING: this pass decoded different tokens, so it did not "
                    "request the same experts; the comparison below is invalid",
                    flush=True,
                )

            row = {
                "pass": index + 1,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "ms_per_token": decode_seconds / args.decode_tokens * 1e3,
                "tok_per_s": args.decode_tokens / decode_seconds,
                "hit_rate": hit_rate,
                "misses_per_token": delta.cache_misses / args.decode_tokens,
                "gib_read": delta.ssd_bytes_read / 1024**3,
                "resident_experts": len(rt.expert_store),
            }
            rows.append(row)
            print(
                f"pass {index + 1}: prefill {prefill_seconds:5.1f} s | "
                f"decode {row['ms_per_token']:6.1f} ms/token "
                f"({row['tok_per_s']:.2f} tok/s) | hit {hit_rate:6.1%} | "
                f"{row['misses_per_token']:5.1f} misses/token | "
                f"{row['resident_experts']} resident",
                flush=True,
            )

        resident = [r for r in rows[1:] if r["hit_rate"] >= 0.99]
        print()
        if resident:
            best = min(r["ms_per_token"] for r in resident)
            print(
                f"all-resident decode: {best:.1f} ms/token over "
                f"{len(resident)} pass(es) at >=99 % hit"
            )
            print(
                "Compare against decode_anatomy's 'rest' of 120 ms/token at a "
                "36 GiB budget: anything the all-resident pass does not account "
                "for is neither expert wait nor arithmetic."
            )
        else:
            print(
                "no pass reached a 99 % hit rate: the working set does not fit "
                "this budget, so lower --decode-tokens or raise the budget"
            )

        out_path = Path(args.out) if args.out else RESULTS_DIR / "decode_resident.json"
        out_path.write_text(json.dumps({"args": vars(args), "passes": rows}, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
