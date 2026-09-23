"""Is chunked prefill as good as whole-prompt prefill? Measured downstream.

HANDOFF section 15.2. prefill_chunk_check.py found the last position's logits
move by KL ~0.03 between whole-prompt and chunked prefill at 8192 tokens,
while two chunk sizes agree with each other to KL 6e-5. One position is one
sample, so this measures what the prefilled state is used for: after
prefilling N tokens of real prose, teacher-force the next M true tokens
through decode_token and score them.

Arms (each from a reset runtime, same prompt):
    seq       decode_token for every prompt token -- the ground truth the
              prefill path is defined against ("persistent state after
              prefill equals calling decode_token for the same tokens")
    whole     one prefill_tokens call
    chunk=C   generation.prepare_prompt with PREFILL_CHUNK=C

Reports per arm: mean NLL of the M continuation tokens, and the mean KL of
each arm's M next-token distributions against the reference arm (seq when
present, else the first).

    <serve.sh's env> PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \\
        benchmarks/prefill_chunk_quality.py 1536:seq,whole,512 --cont 64
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx

from cachalot.model import generation
from cachalot.model import text_decode_runtime as tdr
from cachalot.model.api import V41Model

GiB = 1 << 30
ROOT = Path(__file__).resolve().parent.parent


def text_tokens(tokenizer, n: int) -> list[int]:
    ids = list(tokenizer.encode((ROOT / "docs" / "HANDOFF.md").read_text()))
    while len(ids) < n:
        ids += ids
    return ids[:n]


def logprobs(logits: mx.array) -> mx.array:
    x = logits.astype(mx.float32).reshape(-1)
    return x - mx.logsumexp(x)


def run_arm(rt, prompt, cont, arm):
    rt.reset()
    t0 = time.perf_counter()
    if arm == "seq":
        for tok in prompt:
            res = rt.decode_token(tok)
    elif arm == "whole":
        res = rt.prefill_tokens(prompt)
    else:
        generation.PREFILL_CHUNK = int(arm)
        res, _ = generation.prepare_prompt(rt, prompt, use_prefix_cache=False)
    prefill_s = time.perf_counter() - t0
    dists = []
    for tok in cont:
        lp = logprobs(res.logits)
        mx.eval(lp)
        dists.append(lp)
        res = rt.decode_token(tok)
    nll = -sum(d[t].item() for d, t in zip(dists, cont, strict=True)) / len(cont)
    return dists, nll, prefill_s


def mean_kl(ref, other):
    kls = [mx.sum(mx.exp(a) * (a - b)).item() for a, b in zip(ref, other, strict=True)]
    return sum(kls) / len(kls), max(kls)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", nargs="+", help="N:arm,arm,... with arm in seq, whole, or a chunk size")
    ap.add_argument("--cont", type=int, default=64)
    ap.add_argument("--offset", type=int, default=0, help="start the prompt this many tokens into the text")
    args = ap.parse_args()

    model = V41Model.from_pretrained(
        str(tdr.DEFAULT_CONFIG.resolved_model_path),
        max_seq_len=65536,
        expert_cache_budget_bytes=52 * GiB,
    )
    rt = model.runtime
    for plan in args.plans:
        n_str, arms = plan.split(":")
        n = int(n_str)
        ids = text_tokens(rt.tokenizer, args.offset + n + args.cont)[args.offset :]
        prompt, cont = ids[:n], ids[n : n + args.cont]
        print(f"\n=== prompt {n} tokens, {len(cont)} teacher-forced continuation tokens", flush=True)
        ref = None
        for arm in arms.split(","):
            dists, nll, secs = run_arm(rt, prompt, cont, arm)
            line = f"  {arm:>6}: NLL {nll:.4f}  (prompt consumed in {secs:.1f}s)"
            if ref is None:
                ref = dists
                line += "  [reference]"
            else:
                mkl, xkl = mean_kl(ref, dists)
                line += f"  mean KL vs reference {mkl:.2e}  max {xkl:.2e}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
