"""
Teacher-forced negative log-likelihood of in-distribution text through the
runtime. A healthy model scores low (roughly 1.5-2.5 nats/token on its own
model card); a numerically broken runtime scores several nats higher.

Usage:
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/nll_sanity.py --tokens 160
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prefill", type=int, default=48)
    ap.add_argument("--source", default=None)
    args = ap.parse_args()
    text = Path(args.source).read_text() if args.source else (Path(MODEL_PATH) / "README.md").read_text()
    # skip front matter / badges: start at the introduction paragraph
    start = text.find("We introduce")
    text = text[start:] if start > 0 else text
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        ids = list(rt.tokenizer.encode(text))[: args.prefill + args.tokens + 1]
        rt.reset()
        res = rt.prefill_tokens(ids[: args.prefill])
        nll = []
        top1 = 0
        logits = res.logits
        for i in range(args.prefill, args.prefill + args.tokens):
            target = ids[i]
            logp = logits.astype(mx.float32)
            logp = logp - mx.logsumexp(logp)
            nll.append(-float(logp[target].item()))
            top1 += int(int(logits.argmax().item()) == target)
            logits = rt.decode_token(target).logits
        mean = sum(nll) / len(nll)
        print(f"tokens {len(nll)} | mean NLL {mean:.3f} nats | ppl {math.exp(mean):.2f} | top-1 acc {top1 / len(nll):.1%} | "
              f"worst {max(nll):.2f} at {nll.index(max(nll))}", flush=True)
        print("  sample:", repr(rt.tokenizer.decode(ids[args.prefill:args.prefill + 40])))


if __name__ == "__main__":
    main()
