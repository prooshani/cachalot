"""
A short deterministic fingerprint of what the decode path computes.

Every speed change in this project has to be shown not to have moved the
numerics, and NLL to four decimals is a weak check: it averages over a
vocabulary and hides a changed token. This greedily decodes a fixed prompt and
prints, per step, the argmax token id and two checksums of the fp32 logits.
Two arms that agree here agree bit for bit on the path that produced them.

    PYTHONPATH=src python benchmarks/decode_fingerprint.py --decode-tokens 16

Compare two arms by diffing the output:

    diff <(... CACHALOT_PRELAUNCH_SHARED=0 ... decode_fingerprint.py) \\
         <(... CACHALOT_PRELAUNCH_SHARED=1 ... decode_fingerprint.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=16)
    args = ap.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        for step in range(args.decode_tokens):
            out = rt.decode_token(tok)
            logits = out.logits.astype(mx.float32)
            mx.eval(logits)
            total = float(mx.sum(logits).item())
            top = float(mx.max(logits).item())
            tok = int(logits.argmax().item())
            print(f"step {step:3d} token {tok:7d} logit_sum {total:.6f} logit_max {top:.6f}",
                  flush=True)


if __name__ == "__main__":
    main()
