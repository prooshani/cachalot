"""
Which Python calls make up the CPU half of an all-resident decode token?

profile_decode_sync.py shows about a third of the token is spent outside
mx.eval, building the next graph while the GPU has nothing queued. This runs
the same all-resident token under cProfile and prints the functions that own
that time. cProfile inflates everything, so read the ranking and the shares,
not the absolute milliseconds.
"""
from __future__ import annotations

import cProfile
import pstats
import sys
from io import StringIO
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
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--rows", type=int, default=30)
    args = ap.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        snap = rt.snapshot()
        rt.decode_token(tok)

        def run():
            for _ in range(args.tokens):
                rt.restore(snap)
                r = rt.decode_token(tok)
                mx.eval(r.logits)

        prof = cProfile.Profile()
        prof.enable()
        run()
        prof.disable()

        buf = StringIO()
        st = pstats.Stats(prof, stream=buf).sort_stats("tottime")
        st.print_stats(args.rows)
        text_out = buf.getvalue()
        print(f"{args.tokens} all-resident tokens, {args.prompt_tokens}-token context, sorted by tottime\n")
        print("\n".join(text_out.splitlines()[4:]))


if __name__ == "__main__":
    main()
