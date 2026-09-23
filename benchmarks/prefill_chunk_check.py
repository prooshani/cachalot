"""Chunked prefill against whole-prompt prefill, on the shipped config.

HANDOFF section 15.2. Arms, each from a reset runtime on the same real-prose
prompt (docs/HANDOFF.md), after one warm-up prefill so expert-cache warmth is
comparable:

    whole     one prefill_tokens call (only where it fits in memory)
    chunk=C   generation.prepare_prompt with PREFILL_CHUNK=C

Reports wall time, peak MLX memory, and the last position's logits against
the first arm: max |diff|, KL(first || arm), and whether the top-1 token and
the top-5 set agree. Chunking continues the sequence exactly the way a
prefix-cache hit does, so the logits should agree to bf16 accumulation noise.

    <serve.sh's env> PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \\
        benchmarks/prefill_chunk_check.py 8192:whole,4096,2048 13504:4096,2048
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

from cachalot.model import generation
from cachalot.model import text_decode_runtime as tdr
from cachalot.model.api import V41Model

GiB = 1 << 30
ROOT = Path(__file__).resolve().parent.parent


def prompt_tokens(tokenizer, n: int) -> list[int]:
    ids = list(tokenizer.encode((ROOT / "docs" / "HANDOFF.md").read_text()))
    while len(ids) < n:
        ids += ids
    return ids[:n]


def run_arm(rt, ids, arm):
    rt.reset()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    if arm == "whole":
        res = rt.prefill_tokens(ids)
    else:
        generation.PREFILL_CHUNK = int(arm)
        res, _ = generation.prepare_prompt(rt, ids, use_prefix_cache=False)
    logits = res.logits.astype(mx.float32).reshape(-1)
    mx.eval(logits)
    return logits, time.perf_counter() - t0, mx.get_peak_memory()


def compare(ref, x):
    lp_ref = ref - mx.logsumexp(ref)
    lp = x - mx.logsumexp(x)
    kl = mx.sum(mx.exp(lp_ref) * (lp_ref - lp)).item()
    top5_ref = set(mx.argsort(-ref)[:5].tolist())
    top5 = set(mx.argsort(-x)[:5].tolist())
    return (
        mx.max(mx.abs(ref - x)).item(),
        kl,
        int(mx.argmax(ref).item()) == int(mx.argmax(x).item()),
        top5_ref == top5,
    )


def main() -> None:
    plans = []
    for spec in sys.argv[1:] or ["8192:whole,4096"]:
        n, arms = spec.split(":")
        plans.append((int(n), arms.split(",")))

    model = V41Model.from_pretrained(
        str(tdr.DEFAULT_CONFIG.resolved_model_path),
        max_seq_len=65536,
        expert_cache_budget_bytes=52 * GiB,
    )
    rt = model.runtime
    run_arm(rt, prompt_tokens(rt.tokenizer, 2048), "2048")  # warm-up

    for n, arms in plans:
        ids = prompt_tokens(rt.tokenizer, n)
        ref = None
        print(f"\n=== {n} tokens")
        for arm in arms:
            try:
                logits, dt, peak = run_arm(rt, ids, arm)
            except Exception as exc:
                print(f"  {arm:>6}: FAILED {str(exc)[:100]}")
                continue
            line = f"  {arm:>6}: {dt:7.1f}s  {n / dt:6.1f} tok/s  peak {peak / GiB:6.2f} GiB"
            if ref is None:
                ref = logits
                line += f"  (reference, top-1 {int(mx.argmax(logits).item())})"
            else:
                mad, kl, top1, top5 = compare(ref, logits)
                line += f"  max|dlogit| {mad:.4f}  KL {kl:.2e}  top1 {'same' if top1 else 'DIFF'}  top5 {'same' if top5 else 'DIFF'}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
