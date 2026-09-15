"""Whole-token A/B: fused decode kernels on vs off (same process). Logits must be bit-identical."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model import decode_fused_metal as dfm  # noqa: E402
from cachalot.model import fp8_linear_metal as flm
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402


def timed_token(rt, snap, tok, n=5):
    best = 1e9
    for _ in range(n):
        rt.restore(snap)
        t0 = perf_counter()
        r = rt.decode_token(tok)
        mx.eval(r.logits)
        best = min(best, perf_counter() - t0)
    return r.logits, best


def main():
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = list(rt.tokenizer.encode(enc.encode_messages(
            [{"role": "user", "content": "Describe the deep scattering layer of the ocean in two sentences."}],
            thinking_mode="chat", reasoning_effort=None)))
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        rt.decode_token(tok)          # make the token's experts resident, warm kernels
        snap = rt.snapshot()

        dfm.FUSED_DECODE, flm.FUSED_FP8 = False, False
        logits_off, t_off = timed_token(rt, snap, tok)
        dfm.FUSED_DECODE, flm.FUSED_FP8 = True, True
        logits_on, t_on = timed_token(rt, snap, tok)
        mx.eval(logits_off, logits_on)
        print(f"all-resident decode token: unfused {t_off * 1e3:.1f} ms | fused {t_on * 1e3:.1f} ms | "
              f"logits bit-identical {bool(mx.array_equal(logits_off, logits_on))} "
              f"max|diff| {float(mx.abs(logits_off - logits_on).max()):.2e}")
        # a few more tokens greedy, both modes, compare token ids
        outs = {}
        for mode in (False, True):
            dfm.FUSED_DECODE, flm.FUSED_FP8 = mode, mode
            rt.restore(snap)
            t = tok
            seq = []
            for _ in range(12):
                r = rt.decode_token(t)
                t = int(r.logits.argmax().item())
                seq.append(t)
            outs[mode] = seq
        print("greedy 12 tokens identical:", outs[False] == outs[True], repr(rt.tokenizer.decode(outs[True])))


if __name__ == "__main__":
    main()
