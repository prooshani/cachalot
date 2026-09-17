"""
How often does free-running generation fall into a repetition loop?

The quality gate this project has used measures teacher-forced NLL and top-1:
it feeds the model the *correct* prefix at every step. A runaway repetition
loop -- "result.res.res.res.res..." for hundreds of tokens -- is a free-running
failure that teacher forcing structurally cannot produce, so the gate scores a
bank that does it at 44.5 % top-1 and calls it a six-point regression while the
model is, in practice, unusable on the prompt that triggered it.

This measures the failure directly. It replays a real multi-turn conversation
the way the chat CLI does -- same encoding, same sampling, same prefix cache --
and reports, per turn:

  rep3       fraction of trigrams that have already appeared in this reply
  max_run    longest span covered by a k-gram repeating back to back, any k
  collapsed  max_run >= --collapse-tokens, i.e. the model never got out
  distinct   unique tokens / total

A collapse is stochastic: it is a trap with a per-token entry probability, not
a decay with position, which is why the same prompt can be fine once and
unusable the next time. Run several seeds and read the collapse *rate*, not any
single generation.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh \
      --budget-gib 36 --max-seconds 5400 --tag rep-q2g128 -- env \
      CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 \
      CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
      benchmarks/repetition_quality.py --seeds 3 --max-new-tokens 900
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR  # noqa: E402
from cachalot.model.generation import (  # noqa: E402
    SamplingParams,
    load_official_encoding,
    stream_tokens,
)
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

# The conversation that failed on 2026-09-17: two code-generation turns after a
# greeting. The third turn is the one that ran away.
CONVERSATION = [
    "hi",
    "write a python app to improt a csv file and imports the data to  json file.",
    "write a C++ app to import a csv file and read the columns and export a json file",
]


def repetition_rate(tokens: list[int], n: int = 3) -> float:
    """Fraction of n-gram positions whose n-gram has appeared earlier."""
    if len(tokens) < n + 1:
        return 0.0
    seen: set[tuple[int, ...]] = set()
    repeats = 0
    total = 0
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        total += 1
        if gram in seen:
            repeats += 1
        seen.add(gram)
    return repeats / total if total else 0.0


def longest_consecutive_repeat(tokens: list[int], max_period: int = 128) -> tuple[int, int]:
    """Longest span covered by some k-gram repeating back to back.

    Returns (span_in_tokens, period). "res.res.res." is period 2 or 3 repeating;
    a duplicated paragraph is a long period repeating twice. Both are failures,
    and the span is what makes them unusable.
    """
    best_span = 0
    best_period = 0
    n = len(tokens)
    for period in range(1, max_period + 1):
        i = 0
        while i + period < n:
            if tokens[i] != tokens[i + period]:
                i += 1
                continue
            # Extend the back-to-back repeat as far as it holds.
            j = i
            while j + period < n and tokens[j] == tokens[j + period]:
                j += 1
            span = j - i + period
            if span >= 2 * period and span > best_span:
                best_span = span
                best_period = period
            i = j + 1
    return best_span, best_period


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--first-seed", type=int, default=20260917)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--collapse-tokens", type=int, default=24,
                    help="a repeat span at least this long counts as a collapse")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--penalty-window", type=int, default=256)
    ap.add_argument("--save-text", default="", help="directory to write each reply to")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=args.max_seq_len) as rt:
        bank = Path(rt.expert_bank_path).name
        print(
            f"runtime ready (bank {bank}, expert budget "
            f"{rt.expert_cache_budget_bytes / 2**30:.1f} GiB)",
            flush=True,
        )
        encoding = load_official_encoding(MODEL_PATH)

        rows = []
        for seed_index in range(args.seeds):
            seed = args.first_seed + seed_index
            params = SamplingParams(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=seed,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                penalty_window=args.penalty_window,
            )
            rt.reset()
            messages: list[dict] = []

            for turn_index, user_text in enumerate(CONVERSATION, 1):
                messages.append({"role": "user", "content": user_text})
                prompt = encoding.encode_messages(messages, thinking_mode="chat", reasoning_effort=None)
                ids = list(rt.tokenizer.encode(prompt))

                produced: list[int] = []
                t0 = perf_counter()
                finish = None
                for event in stream_tokens(rt, ids, params):
                    if event.kind == "token":
                        produced.append(event.token)
                    elif event.kind == "done":
                        finish = event.finish_reason
                seconds = perf_counter() - t0

                text = rt.tokenizer.decode(produced)
                messages.append({"role": "assistant", "content": text})

                rep3 = repetition_rate(produced, 3)
                span, period = longest_consecutive_repeat(produced)
                distinct = len(set(produced)) / len(produced) if produced else 0.0
                collapsed = span >= args.collapse_tokens

                rows.append(
                    {
                        "bank": bank,
                        "seed": seed,
                        "turn": turn_index,
                        "tokens": len(produced),
                        "finish": finish,
                        "rep3": rep3,
                        "max_run": span,
                        "period": period,
                        "distinct": distinct,
                        "collapsed": collapsed,
                        "tok_per_s": len(produced) / seconds if seconds else 0.0,
                    }
                )
                print(
                    f"seed {seed} turn {turn_index}: {len(produced):4d} tokens | "
                    f"rep3 {rep3:6.1%} | longest repeat {span:4d} (period {period}) | "
                    f"distinct {distinct:5.1%} | {'COLLAPSED' if collapsed else 'ok'} | "
                    f"{len(produced) / seconds:.2f} tok/s",
                    flush=True,
                )

                if args.save_text:
                    out_dir = Path(args.save_text)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / f"{bank}_seed{seed}_turn{turn_index}.txt").write_text(text)

        print(f"\n=== {bank} ===")
        print("| turn | replies | mean rep3 | median longest repeat | collapses |")
        print("|---:|---:|---:|---:|---:|")
        for turn_index in range(1, len(CONVERSATION) + 1):
            group = [r for r in rows if r["turn"] == turn_index]
            if not group:
                continue
            mean_rep = sum(r["rep3"] for r in group) / len(group)
            runs = sorted(r["max_run"] for r in group)
            median_run = runs[len(runs) // 2]
            collapses = sum(1 for r in group if r["collapsed"])
            print(
                f"| {turn_index} | {len(group)} | {mean_rep:.1%} | {median_run} | "
                f"{collapses}/{len(group)} |"
            )
        total_collapses = sum(1 for r in rows if r["collapsed"])
        print(f"\ncollapse rate: {total_collapses}/{len(rows)} replies "
              f"({total_collapses / len(rows):.0%})")

        tag = bank
        if args.frequency_penalty or args.presence_penalty or args.no_repeat_ngram_size:
            tag += (f"_fp{args.frequency_penalty}_pp{args.presence_penalty}"
                    f"_ng{args.no_repeat_ngram_size}")
        out_path = Path(args.out) if args.out else RESULTS_DIR / f"repetition_quality_{tag}.json"
        out_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
