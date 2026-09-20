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


def word_fragment(prev: str) -> str:
    """The part of the word already committed to, reading back from the cursor."""
    out = ""
    for ch in reversed(prev):
        if ch.isalnum() or ch == "_":
            out = ch + out
        else:
            break
    return out


def split_first_and_repeat(
    rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Continuations, split by whether the word has been spelled out before.

    This split is the whole point. A *first* continuation still needs the model
    to know the word; a *repeat* only needs it to copy one it emitted a few
    tokens ago, which is the easiest prediction in the run and the one this
    runtime is worst at. Pooling them hides that: 86 % and 51 % average to 59 %,
    which looks like ordinary difficulty and is not.
    """
    seen: set[str] = set()
    first: list[dict] = []
    repeat: list[dict] = []
    other: list[dict] = []
    for row in rows:
        if not is_continuation(row["prev"], row["target"]):
            other.append(row)
            continue
        frag = word_fragment(row["prev"])
        word = frag + row["target"]
        (repeat if (frag in seen or word in seen) else first).append(row)
        seen.add(word)
        seen.add(frag)
    return first, repeat, other


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

    print(f"{'run':<28} {'first n':>7} {'rank1':>5} | {'repeat n':>7} {'rank1':>5} "
          f"{'logprob':>8} {'worst':>6} | {'other n':>7} {'rank1':>5} | {'all':>4}")
    for name in args.runs:
        path = Path(name)
        rows = load(path)
        first, repeat, other = split_first_and_repeat(rows)
        f, c, o = summarise(first), summarise(repeat), summarise(other)
        allr = sum(1 for r in rows if r["rank"] == 0) / len(rows)
        print(f"{path.stem:<28} {f['n']:>7} {f['rank1']:>5.0%} | {c['n']:>7} "
              f"{c['rank1']:>5.0%} {c['logprob']:>8.3f} {c['worst']:>6} | "
              f"{o['n']:>7} {o['rank1']:>5.0%} | {allr:>4.0%}")

        if args.worst:
            for r in sorted(repeat, key=lambda r: -r["rank"])[: args.worst]:
                preferred = " ".join(f"{t['token']!r}" for t in r["top"][:3])
                print(f"    rank {r['rank']:>6}  want {r['target']!r:<14} "
                      f"after ...{r['prev'][-28:]!r}")
                print(f"                  preferred: {preferred}")

    print("\nfirst  = finishing a word begun for the first time; the model must know it.")
    print("repeat = finishing a word the text already spelled out, a copy from a few")
    print("         tokens back. No correct model fails that, so a low number there is")
    print("         a defect by construction and needs no reference arm to interpret.")
    print("Measured 2026-09-20: first 86 %, repeat 51 %, everything else 88 %.")


if __name__ == "__main__":
    main()
