"""merge_image_embeddings, HANDOFF section 16 piece 3 step 1.

Not wired into TextDecodeRuntime yet (that needs the per-token bias_vl
selection and image_mask threading, piece 3's remaining two steps) -- this
proves the splice point itself: for a text position it must be bit-identical
to embed_token_decode(), and for an image_token_id position it must place
vision_embed()'s row, broadcast hc_mult times the same way a text embedding
is, in reading order, without disturbing any other position.
"""

from __future__ import annotations

import mlx.core as mx

from cachalot.model.model_boundary_mlx import embed_token_decode
from cachalot.model.vision_mlx import merge_image_embeddings

VOCAB = 256
DIM = 32
HC_MULT = 4
IMAGE_TOKEN_ID = 129


def _embed_weight():
    w = mx.random.normal(shape=(VOCAB, DIM)).astype(mx.bfloat16)
    mx.eval(w)
    return w


def test_all_text_prompt_is_bit_identical_to_embed_token_decode():
    embed_weight = _embed_weight()
    token_ids = [3, 17, 200, 5, 5]

    got = merge_image_embeddings(
        token_ids, embed_weight, IMAGE_TOKEN_ID, image_rows=None, hc_mult=HC_MULT
    )

    want = mx.stack(
        [embed_token_decode(t, embed_weight, hc_mult=HC_MULT) for t in token_ids],
        axis=0,
    )
    mx.eval(got, want)

    assert got.shape == want.shape
    assert bool(mx.all(got == want).item())


def test_image_positions_carry_vision_rows_in_reading_order():
    embed_weight = _embed_weight()
    image_rows = mx.random.normal(shape=(3, DIM)).astype(mx.float32)
    mx.eval(image_rows)

    token_ids = [10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 20, IMAGE_TOKEN_ID]

    got = merge_image_embeddings(
        token_ids, embed_weight, IMAGE_TOKEN_ID, image_rows=image_rows, hc_mult=HC_MULT
    )
    mx.eval(got)

    assert got.shape == (5, HC_MULT, DIM)

    # text positions unaffected
    want_10 = embed_token_decode(10, embed_weight, hc_mult=HC_MULT)
    want_20 = embed_token_decode(20, embed_weight, hc_mult=HC_MULT)
    mx.eval(want_10, want_20)
    assert bool(mx.all(got[0] == want_10).item())
    assert bool(mx.all(got[3] == want_20).item())

    # image positions carry image_rows in reading order, each broadcast
    # hc_mult times and cast to embed_weight's dtype
    for slot, row_idx in ((1, 0), (2, 1), (4, 2)):
        want_row = mx.broadcast_to(
            image_rows[row_idx][None, :], (HC_MULT, DIM)
        ).astype(embed_weight.dtype)
        mx.eval(want_row)
        assert bool(mx.all(got[slot] == want_row).item())


def test_too_few_image_rows_raises():
    embed_weight = _embed_weight()
    image_rows = mx.random.normal(shape=(1, DIM)).astype(mx.float32)
    mx.eval(image_rows)

    token_ids = [IMAGE_TOKEN_ID, IMAGE_TOKEN_ID]

    try:
        merge_image_embeddings(
            token_ids, embed_weight, IMAGE_TOKEN_ID, image_rows=image_rows, hc_mult=HC_MULT
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for exhausted image_rows")


def test_too_many_image_rows_raises():
    embed_weight = _embed_weight()
    image_rows = mx.random.normal(shape=(2, DIM)).astype(mx.float32)
    mx.eval(image_rows)

    token_ids = [IMAGE_TOKEN_ID]

    try:
        merge_image_embeddings(
            token_ids, embed_weight, IMAGE_TOKEN_ID, image_rows=image_rows, hc_mult=HC_MULT
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unconsumed image_rows")


def test_no_image_tokens_with_image_rows_supplied_raises():
    embed_weight = _embed_weight()
    image_rows = mx.random.normal(shape=(1, DIM)).astype(mx.float32)
    mx.eval(image_rows)

    token_ids = [1, 2, 3]

    try:
        merge_image_embeddings(
            token_ids, embed_weight, IMAGE_TOKEN_ID, image_rows=image_rows, hc_mult=HC_MULT
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError: image_rows supplied but never consumed")
