"""Vision piece 4 (HANDOFF section 16.4): prompt expansion, span layout, the
Engram image mask, and the prefix cache's image identity."""

import mlx.core as mx
import pytest

from cachalot.model.engram_mlx import engram_forward_batched
from cachalot.model.prefix_cache import PrefixCache, SequenceSnapshot
from cachalot.model.resident_layer import ResidentLayer
from cachalot.model.vision_prompt import (
    IMAGE_TOKEN_ID,
    ImageSpan,
    PromptImages,
    expand_prompt_images,
    image_span_rows,
)
from cachalot.storage.tensor_loader import ResidentTensor

DIM = 8


def test_span_rows_follow_the_reference_layout():
    # [IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h + [IMAGE_END]
    h, w = 2, 3
    aligner = mx.arange(h * w * DIM, dtype=mx.float32).reshape(h * w, DIM) + 100
    start, newline, end = (mx.full((DIM,), v, dtype=mx.float32) for v in (-1.0, -2.0, -3.0))
    rows = image_span_rows(aligner, h, w, start, newline, end)
    assert rows.shape == (h * (w + 1) + 2, DIM)
    assert mx.array_equal(rows[0], start)
    assert mx.array_equal(rows[1:4], aligner[0:3])
    assert mx.array_equal(rows[4], newline)
    assert mx.array_equal(rows[5:8], aligner[3:6])
    assert mx.array_equal(rows[8], newline)
    assert mx.array_equal(rows[9], end)


def test_span_rows_reject_a_grid_mismatch():
    with pytest.raises(ValueError):
        image_span_rows(mx.zeros((5, DIM)), 2, 3, mx.zeros((DIM,)), mx.zeros((DIM,)), mx.zeros((DIM,)))


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, record):
        self.calls.append(record)
        n = record["n"]
        return mx.full((n, DIM), float(n)), f"digest-{record['id']}"


def test_expand_replaces_each_placeholder_with_its_span():
    enc = FakeEncoder()
    tokens = [1, IMAGE_TOKEN_ID, 2, 3, IMAGE_TOKEN_ID, 4]
    out = expand_prompt_images(tokens, [{"n": 4, "id": "a"}, {"n": 2, "id": "b"}], enc)
    assert out.tokens == [1] + [IMAGE_TOKEN_ID] * 4 + [2, 3] + [IMAGE_TOKEN_ID] * 2 + [4]
    assert out.keys == ((1, 4, "digest-a"), (7, 2, "digest-b"))
    assert out.rows_from(0).shape == (6, DIM)
    # a prefill starting after the first span only needs the second
    assert out.rows_from(5).shape == (2, DIM)
    assert out.rows_from(9) is None
    with pytest.raises(ValueError):
        out.rows_from(3)  # inside the first span


def test_expand_text_only_is_untouched_and_needs_no_encoder():
    out = expand_prompt_images([5, 6, 7], [], None)
    assert out.tokens == [5, 6, 7] and out.spans == []


@pytest.mark.parametrize(
    "tokens,records",
    [
        ([1, IMAGE_TOKEN_ID], []),
        ([1, 2], [{"n": 1, "id": "a"}]),
        ([IMAGE_TOKEN_ID, IMAGE_TOKEN_ID], [{"n": 1, "id": "a"}]),
    ],
)
def test_expand_rejects_placeholder_image_count_mismatch(tokens, records):
    with pytest.raises(ValueError):
        expand_prompt_images(tokens, records, FakeEncoder())


def test_expand_without_an_encoder_refuses_images():
    with pytest.raises(ValueError):
        expand_prompt_images([IMAGE_TOKEN_ID], [{"n": 1, "id": "a"}], None)


def _snap(tokens, spans=()):
    return SequenceSnapshot(
        tokens=tuple(tokens), position=len(tokens), logits=None,
        windows={}, compressed_caches={}, compressor_kv={}, compressor_score={}, indexer_k={},
        engram_history=[], shared_compress_kv=None, shared_index_k_layer=None,
        shared_topk_idxs=None, shared_candidates=None, image_spans=tuple(spans),
    )


def test_prefix_cache_never_matches_a_different_image():
    img = [IMAGE_TOKEN_ID] * 3
    tokens = (1, *img, 2)
    cache = PrefixCache()
    cache.add(_snap(tokens, [(1, 3, "cat")]))
    longer = tokens + (9, 9)
    assert cache.find(longer, ((1, 3, "cat"),)) is not None
    # same token ids, different picture
    assert cache.find(longer, ((1, 3, "dog"),)) is None
    # a text-only snapshot is still a valid prefix before any image
    cache.add(_snap((1,)))
    assert len(cache.find(longer, ((1, 3, "dog"),)).tokens) == 1


def test_prefix_cache_text_only_unchanged():
    cache = PrefixCache()
    cache.add(_snap((1, 2, 3)))
    assert cache.find((1, 2, 3, 4)) is not None
    assert cache.find((1, 2, 4)) is None


def _engram_layer(dim, hc_mult, n_cols, head_dim):
    k = n_cols * head_dim
    n = hc_mult * dim + dim
    mx.random.seed(0)
    w = mx.to_fp8(mx.random.normal((n, k)) * 0.5)
    scales = mx.full((n // 32, k // 32), 127, dtype=mx.uint8)
    tensors = {
        "wkv.weight": w,
        "wkv.scale": scales,
        "q_weight": mx.random.normal((hc_mult, dim)).astype(mx.bfloat16),
        "k_weight": mx.random.normal((hc_mult, dim)).astype(mx.bfloat16),
    }
    return ResidentLayer(
        layer_id=1,
        tensors={
            f"layers.1.engram.{name}": ResidentTensor(f"layers.1.engram.{name}", "", tuple(t.shape), t)
            for name, t in tensors.items()
        },
    )


def test_engram_mask_passes_image_positions_through_and_leaves_text_bit_identical():
    dim, hc_mult, n_cols, head_dim, n_tok = 32, 4, 2, 32, 5
    layer = _engram_layer(dim, hc_mult, n_cols, head_dim)
    x = mx.random.normal((n_tok, hc_mult, dim)).astype(mx.bfloat16)
    rows = mx.random.normal((n_tok, n_cols, head_dim))
    mask = mx.array([True, False, False, True, True])

    plain = engram_forward_batched(x, rows, layer)
    masked = engram_forward_batched(x, rows, layer, token_mask=mask)

    assert not mx.array_equal(plain[1], x[1])  # the gate really does something
    assert mx.array_equal(masked[1], x[1]) and mx.array_equal(masked[2], x[2])
    for t in (0, 3, 4):
        assert mx.array_equal(masked[t], plain[t])


def test_prompt_images_keys_are_the_cache_identity():
    span = ImageSpan(start=3, length=2, digest="d", rows=mx.zeros((2, DIM)))
    assert PromptImages(tokens=[0] * 5, spans=[span]).keys == ((3, 2, "d"),)


def _stream(rt, images):
    from cachalot.model.generation import SamplingParams, stream_tokens

    params = SamplingParams(max_new_tokens=2, temperature=0.0)
    return [e for e in stream_tokens(rt, images.tokens, params, images=images)]


def _images(tokens_before, digest, after=()):
    enc = FakeEncoder()
    enc.encode = lambda rec: (mx.zeros((3, DIM)), rec["id"])
    return expand_prompt_images([*tokens_before, IMAGE_TOKEN_ID, *after], [{"id": digest}], enc)


def test_generation_sends_image_rows_and_reuses_only_the_same_image():
    from fake_model import ScriptedRuntime

    rt = ScriptedRuntime(reply="ok")
    first = _images([10, 11], "cat", after=[12])
    ev = _stream(rt, first)
    assert ev[0].reused_prefix_tokens == 0
    assert rt.image_prefills[-1] == 3

    # the same conversation, extended: the image span is inside the reused prefix
    longer = PromptImages(tokens=first.tokens + [13, 14], spans=first.spans)
    ev = _stream(rt, longer)
    assert ev[0].reused_prefix_tokens == len(first.tokens)
    assert rt.image_prefills[-1] is None  # the suffix is text only

    # the same token ids with a different picture must not reuse the cat
    dog = _images([10, 11], "dog", after=[12])
    ev = _stream(rt, dog)
    assert ev[0].reused_prefix_tokens == 0
    assert rt.image_prefills[-1] == 3


def test_encoder_reuses_span_rows_for_the_same_image_bytes(monkeypatch):
    # An agent resends every image in its history each turn; the tower runs
    # once per distinct image (HANDOFF section 15.5).
    from cachalot.model import vision_prompt as vp

    calls = []

    def fake_load_image(record, cfg):
        calls.append(record["data"])
        return None, 1, 1, 1, 1

    monkeypatch.setattr(vp, "load_image", fake_load_image)
    monkeypatch.setattr(vp, "vision_embed", lambda *a, **k: mx.zeros((1, 4)))
    monkeypatch.setattr(vp, "image_span_rows", lambda rows, *a: mx.ones((3, 4)) * len(calls))
    enc = vp.VisionEncoder({})
    enc._weights = object()
    enc._delims = {"image_start": None, "image_newline": None, "image_end": None}

    r1, d1 = enc.encode({"data": b"cat"})
    r2, d2 = enc.encode({"data": b"cat"})
    r3, d3 = enc.encode({"data": b"dog"})
    assert calls == [b"cat", b"dog"]
    assert d1 == d2 != d3
    assert r1 is r2 and enc.cache_hits == 1
    assert not mx.array_equal(r1, r3)


def test_vision_ablation_switches(monkeypatch):
    from cachalot.model import vision_ablation

    monkeypatch.delenv("CACHALOT_VISION_ABLATE", raising=False)
    assert not vision_ablation.ablated("delims")
    monkeypatch.setenv("CACHALOT_VISION_ABLATE", "delims, bias_vl")
    assert vision_ablation.ablated("delims") and vision_ablation.ablated("bias_vl")
    assert not vision_ablation.ablated("engram_mask")
    monkeypatch.setenv("CACHALOT_VISION_ABLATE", "delim")
    import pytest

    with pytest.raises(ValueError):
        vision_ablation.announce()
