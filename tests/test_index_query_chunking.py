"""HANDOFF section 15.2: the prefill indexer scores in query-row chunks.

Row chunking must not change a single score, top-k index or candidate mask:
the Hermes prompt that exposed the quadratic scoring memory is exactly the
case where nobody could otherwise check it."""

import mlx.core as mx
import numpy as np
import pytest

from cachalot.model import source_prefill_batched as spb


def _inputs(n_tokens, cmax, hidden=256, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((n_tokens, spb.INDEX_N_HEADS, spb.INDEX_HEAD_DIM)) * 0.5).astype(mx.bfloat16)
    x = mx.random.normal((n_tokens, hidden)).astype(mx.bfloat16)
    w = (mx.random.normal((spb.INDEX_N_HEADS, hidden)) * 0.05).astype(mx.bfloat16)
    k = (mx.random.normal((cmax, spb.INDEX_HEAD_DIM)) * 0.5).astype(mx.bfloat16)
    start = 2 * cmax - n_tokens  # the last token sees every compressed slot at ratio 2
    compress_len = (np.arange(start, start + n_tokens) + 1) // 2
    return q, x, w, k, compress_len


@pytest.mark.parametrize("n_tokens,cmax,chunk", [(600, 400, 300), (3000, 2000, 1024), (2100, 1500, 1024)])
def test_row_chunked_scores_topk_and_candidates_are_identical(monkeypatch, n_tokens, cmax, chunk):
    monkeypatch.setattr(spb, "INDEX_Q_CHUNK", chunk)
    q, x, w, k, compress_len = _inputs(n_tokens, cmax)

    whole = spb.index_scores_chunk(q, x, w, k, compress_len)
    topk_whole = spb.topk_rows(whole, np.minimum(64, compress_len), 64)
    cand_whole = spb.candidate_masks_chunk(whole, compress_len, 16, 8)

    parts, topk_parts, cand_parts = [], [], []
    slices = spb._row_slices(n_tokens)
    assert len(slices) > 1
    for rows in slices:
        s = spb.index_scores_chunk(q[rows], x[rows], w, k, compress_len[rows])
        parts.append(s)
        topk_parts += spb.topk_rows(s, np.minimum(64, compress_len[rows]), 64)
        cand_parts += spb.candidate_masks_chunk(s, compress_len[rows], 16, 8)

    assert mx.array_equal(mx.concatenate(parts, axis=0), whole, equal_nan=True)
    assert all(mx.array_equal(a, b) for a, b in zip(topk_parts, topk_whole, strict=True))
    assert all(mx.array_equal(a, b) for a, b in zip(cand_parts, cand_whole, strict=True))


def test_row_slices_are_balanced_and_cover_every_token_once(monkeypatch):
    monkeypatch.setattr(spb, "INDEX_Q_CHUNK", 1024)
    slices = spb._row_slices(13504)
    covered = [i for s in slices for i in range(s.start, s.stop)]
    assert covered == list(range(13504))
    assert all(512 <= s.stop - s.start <= 1024 for s in slices)
    assert spb._row_slices(1025) == [slice(0, 513), slice(513, 1025)]
    # a prompt of at most one chunk is the unchunked computation
    assert spb._row_slices(1024) == [slice(0, 1024)]
    monkeypatch.setattr(spb, "INDEX_Q_CHUNK", 0)
    assert spb._row_slices(2500) == [slice(0, 2500)]
