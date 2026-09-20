"""
Split a token_rank_probe run into predictions that are open-ended and
predictions that are not, and report them separately.

**Why this exists.** token_rank_probe.py reports one number per run: how often
the correct token was rank 1. That number mixes two completely different
predictions. Choosing *which* header a file includes, or which module it
imports, is genuinely open-ended and a good model is often not rank 1. Finishing
a word it has already started -- `stdex` -> `cept`, `ioman` -> `ip` -- has
exactly one right answer and a good model should be rank 1 almost always.

Measured on 2026-09-20 across four probe arms, this runtime is *worse* at the
second than at the first, and worse at it than at the average of all positions.
That is backwards, and it is the shape of every artefact the corpus collects:
`#include <stdex>`, `LRUCache` written `LRcache`, `map_._.end()` with a
duplicated `_.`, `std std::string`, `utfutf-8`.

A continuation is a token that starts with an alphanumeric character where the
character before it is also alphanumeric, so the model is in the middle of a
word it committed to. Everything else is `other`.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/continuation_rank.py \
      benchmarks/results/rankprobe-*.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_continuation(prev: str, target: str) -> bool:
    return bool(prev) and bool(target) and prev[-1].isalnum() and target[:1].isalnum()


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "rank1": float("nan"), "logprob": float("nan"), "worst": 0}
    return {
        "n": len(rows),
        "rank1": sum(1 for r in rows if r["rank"] == 0) / len(rows),
        "logprob": sum(r["logprob"] for r in rows) / len(rows),
        "worst": max(r["rank"] for r in rows),
    }


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = data.get("rows", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: no per-position rows to read")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="token_rank_probe --out JSON files")
    ap.add_argument("--worst", type=int, default=0,
                    help="also print this many worst-ranked continuations per run")
    args = ap.parse_args()

    print(f"{'run':<28} {'contin n':>8} {'rank1':>6} {'logprob':>8} {'worst':>6} | "
          f"{'other n':>7} {'rank1':>6} | {'all':>5}")
    for name in args.runs:
        path = Path(name)
        rows = load(path)
        cont = [r for r in rows if is_continuation(r["prev"], r["target"])]
        other = [r for r in rows if not is_continuation(r["prev"], r["target"])]
        c, o = summarise(cont), summarise(other)
        allr = sum(1 for r in rows if r["rank"] == 0) / len(rows)
        print(f"{path.stem:<28} {c['n']:>8} {c['rank1']:>5.0%} {c['logprob']:>8.3f} "
              f"{c['worst']:>6} | {o['n']:>7} {o['rank1']:>5.0%} | {allr:>4.0%}")

        if args.worst:
            for r in sorted(cont, key=lambda r: -r["rank"])[: args.worst]:
                preferred = " ".join(f"{t['token']!r}" for t in r["top"][:3])
                print(f"    rank {r['rank']:>6}  want {r['target']!r:<14} "
                      f"after ...{r['prev'][-28:]!r}")
                print(f"                  preferred: {preferred}")

    print("\nA continuation is a token finishing a word already begun: exactly one")
    print("correct answer. It should be the EASIEST class in the run, well above")
    print("the 'all' column. If it is below, the runtime is losing word-internal")
    print("information and that is a defect, not quantization blur.")


if __name__ == "__main__":
    main()
