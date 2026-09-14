"""cProfile one cold decode step and one all-resident decode step of the same token."""
from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, StoreSnapshot  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402


def profile_step(rt, tok, label):
    before = StoreSnapshot.take(rt)
    pr = cProfile.Profile()
    t0 = perf_counter()
    pr.enable()
    res = rt.decode_token(tok)
    mx.eval(res.logits)
    pr.disable()
    wall = perf_counter() - t0
    d = before.delta(StoreSnapshot.take(rt))
    print(f"\n===== {label}: wall {wall:.2f}s, misses {d.cache_misses}", flush=True)
    st = pstats.Stats(pr)
    st.sort_stats("cumulative").print_stats(18)
    st.sort_stats("tottime").print_stats(12)
    return res


def main():
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = list(rt.tokenizer.encode(enc.encode_messages(
            [{"role": "user", "content": "List three facts about the deep ocean."}],
            thinking_mode="chat", reasoning_effort=None)))
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        snap = rt.snapshot()
        profile_step(rt, tok, "cold")
        rt.restore(snap)
        profile_step(rt, tok, "all-resident")


if __name__ == "__main__":
    main()
