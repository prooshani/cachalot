"""Live end-to-end smoke test for the wq_a/wkv fusion (HANDOFF section 9.34):
prefill + a few decode tokens on the real shipped 2-bit bank, confirming no
crash and sane (non-NaN, non-garbage) output. Mirrors debug_fused_parity.py's
shape, used the same way before the shared expert's w1/w3 fusion shipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402


def main():
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = list(
            rt.tokenizer.encode(
                enc.encode_messages(
                    [{"role": "user", "content": "Say hello in one short sentence."}],
                    thinking_mode="chat",
                    reasoning_effort=None,
                )
            )
        )
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        print(f"prefill ok, {len(ids)} tokens, first decode token {tok} {rt.tokenizer.decode([tok])!r}")
        out = []
        for _ in range(12):
            r = rt.decode_token(tok)
            tok = int(r.logits.argmax().item())
            out.append(tok)
        text = rt.tokenizer.decode(out)
        print(f"12 more tokens: {out}")
        print(f"decoded: {text!r}")


if __name__ == "__main__":
    main()
