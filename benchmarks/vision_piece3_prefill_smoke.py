"""Live end-to-end smoke test for HANDOFF section 16 piece 3 steps 2-3:
per-token bias_vl selection in route_topk_rows/moe_prefill_grouped, and
image_mask threading through TextDecodeRuntime's prefill path.

Two prefills on the real shipped 2-bit bank:

  1. A plain text prompt through the unmodified path (image_rows=None) --
     confirms the new kwarg threading through every one of prefill's five
     block variants (layer0, sliding window, and the three compressed
     variants) has not broken ordinary text prefill.
  2. The same token ids with three interior positions overwritten to
     IMAGE_TOKEN_ID and synthetic random image_rows supplied -- exercises
     merge_image_embeddings, the image_mask build, and per-token bias_vl
     selection in all 40 layers' routers, end to end. No vision encoder is
     wired yet (piece 4 is not started) so the "image" rows are random, not
     real aligner output -- this checks no crash and sane (non-NaN) output,
     not numerical correctness of a real image, the same standard
     qkv_fusion_live_smoke.py used before HANDOFF section 9.34 shipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import (  # noqa: E402
    IMAGE_TOKEN_ID,
    TextDecodeRuntime,
)

DIM = 5120
N_IMAGE_TOKENS = 3


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

        # ---- arm 1: plain text, unchanged path ---------------------------
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        assert bool(mx.isfinite(res.logits).all().item()), "text prefill produced non-finite logits"
        print(f"text prefill ok, {len(ids)} tokens, first decode token {tok} {rt.tokenizer.decode([tok])!r}")

        # ---- arm 2: synthetic image span ----------------------------------
        if len(ids) <= N_IMAGE_TOKENS + 2:
            raise RuntimeError("prompt too short to splice a synthetic image span into")

        image_ids = list(ids)
        splice_at = 2  # leave at least one real token before/after the span
        for i in range(N_IMAGE_TOKENS):
            image_ids[splice_at + i] = IMAGE_TOKEN_ID

        image_rows = mx.random.normal(shape=(N_IMAGE_TOKENS, DIM)).astype(mx.float32)
        mx.eval(image_rows)

        rt.reset()
        res_img = rt.prefill_tokens(image_ids, image_rows=image_rows, image_token_id=IMAGE_TOKEN_ID)
        tok_img = int(res_img.logits.argmax().item())
        assert bool(mx.isfinite(res_img.logits).all().item()), "image-bearing prefill produced non-finite logits"
        print(
            f"image-bearing prefill ok, {len(image_ids)} tokens "
            f"({N_IMAGE_TOKENS} image positions at index {splice_at}), "
            f"first decode token {tok_img} {rt.tokenizer.decode([tok_img])!r}"
        )

        out = []
        t = tok_img
        for _ in range(8):
            r = rt.decode_token(t)
            t = int(r.logits.argmax().item())
            assert bool(mx.isfinite(r.logits).all().item()), "decode after image prefill produced non-finite logits"
            out.append(t)
        print(f"8 more decode tokens after image prefill: {out}")
        print(f"decoded: {rt.tokenizer.decode(out)!r}")


if __name__ == "__main__":
    main()
