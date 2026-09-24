"""
Decode speed against context length, everything else held fixed (HANDOFF section 15.5).

An agent session decodes at a 13-30k-token context; `chat` sessions at a few thousand. This puts N tokens
of filler (a system message cut from HANDOFF.md) in front of the same fixed task, prefills it the way the
server does (chunked), greedy-decodes --warm tokens to settle the expert cache on the reply's own routing,
then times --measure more. Only the filler length differs between arms, so run one arm per process and
alternate the order:

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    for n in 0 16000 0 16000; do benchmarks/decode_vs_context.sh $n; done

(`decode_vs_context.sh` sets serve.sh's environment.) Reports ms/token, expert hit rate and misses per
token for the timed span, plus the time per token spent outside expert loading (attention and the rest),
which is where a context-length cost would show.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, StoreSnapshot  # noqa: E402
from cachalot.model.api import V41Model  # noqa: E402
from cachalot.model.generation import load_official_encoding, prepare_prompt  # noqa: E402

GiB = 1024**3
REPO = Path(__file__).resolve().parent.parent
TASK = (
    "Write a Python function that parses an ISO-8601 date string without using the datetime module, "
    "validates month and day ranges including leap years, and returns a (year, month, day) tuple. "
    "Then explain the leap-year rule you used."
)


def build(tokenizer, encoding, filler_tokens: int) -> list[int]:
    def ids(system: str | None) -> list[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": TASK}]
        return list(tokenizer.encode(encoding.encode_messages(msgs, thinking_mode="chat", reasoning_effort=None)))

    if filler_tokens <= 0:
        return ids(None)
    text = (REPO / "docs/HANDOFF.md").read_text().replace("<｜", "<|").replace("｜>", "|>")
    base = len(ids(None))
    lo, hi, best = 1, len(text), None
    while lo <= hi:
        mid = (lo + hi) // 2
        t = ids(text[:mid])
        if len(t) - base <= filler_tokens:
            best, lo = t, mid + 1
        else:
            hi = mid - 1
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, required=True, help="filler tokens before the fixed task")
    ap.add_argument("--warm", type=int, default=32)
    ap.add_argument("--measure", type=int, default=96)
    ap.add_argument("--budget-gib", type=float, default=52)
    ap.add_argument("--max-seq-len", type=int, default=65536)
    ap.add_argument("--out", default=None, help="append one JSON line here")
    args = ap.parse_args()

    model = V41Model.from_pretrained(
        MODEL_PATH, max_seq_len=args.max_seq_len, expert_cache_budget_bytes=int(args.budget_gib * GiB)
    )
    rt = model.runtime
    enc = load_official_encoding(MODEL_PATH)
    prompt = build(rt.tokenizer, enc, args.context)
    t0 = perf_counter()
    res, _ = prepare_prompt(rt, prompt, use_prefix_cache=False)
    t_prefill = perf_counter() - t0
    tok = int(res.logits.argmax().item())
    out = []
    for _ in range(args.warm):
        tok = int(rt.decode_token(tok).logits.argmax().item())
        out.append(tok)
    before = StoreSnapshot.take(rt)
    r0 = rt.expert_store.stats()
    t0 = perf_counter()
    for _ in range(args.measure):
        tok = int(rt.decode_token(tok).logits.argmax().item())
        out.append(tok)
    t_decode = perf_counter() - t0
    d = before.delta(StoreSnapshot.take(rt))
    r1 = rt.expert_store.stats()
    reads = getattr(r1, "reads", 0) - getattr(r0, "reads", 0)
    n = args.measure
    row = {
        "context": len(prompt),
        "filler": args.context,
        "prefill_s": round(t_prefill, 1),
        "ms_per_token": round(t_decode / n * 1e3, 1),
        "tok_s": round(n / t_decode, 2),
        "hit_rate": round(d.cache_hits / max(1, d.cache_hits + d.cache_misses), 4),
        "misses_per_token": round(d.cache_misses / n, 2),
        # every expert read in the timed span (demand and predicted), the share fast
        # enough to be page-cache hits, and the mean read time (HANDOFF section 15.7)
        "reads_per_token": round(reads / n, 2),
        "fast_read_pct": round(100 * (r1.fast_reads - r0.fast_reads) / reads, 1) if reads else None,
        "read_ms": round(1e3 * (r1.read_wall_seconds - r0.read_wall_seconds) / reads, 2) if reads else None,
        "time": __import__("time").strftime("%H:%M:%S"),
        "text_tail": rt.tokenizer.decode(out[-40:]),
    }
    print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
