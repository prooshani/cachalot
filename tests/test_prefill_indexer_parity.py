"""Pins the prefill indexer's selection against the shipped reference.

The handoff names this as the one difference already known to live in prefill:
the reference masks with `where(idxs < compress_lens, idxs + offset, -1)`, and
`compress_lens` is a plain integer at decode -- where it cannot bite -- but a
per-query column vector during prefill. Our batched prefill takes a different
route to the same answer: it clamps k per row and returns only the reachable
positions, where the reference returns a fixed-width row padded with -1.

Those are equivalent, and this file is the record that they were compared. The
reference expressions are transcribed from `model.py` (lines 563-580, the
indexer tail) and `model.py` lines 583-610, `select_candidate_blocks`.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="the prefill indexer is MLX code")

import mlx.core as mx  # noqa: E402

from cachalot.model.source_prefill_batched import (  # noqa: E402
    candidate_masks_chunk,
    topk_rows,
)

RATIO = 4


def _reference_compress_lens(start_pos: int, n_tokens: int) -> np.ndarray:
    """`model.py` line 564, for a prefill beginning at start_pos:

        compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
    """
    return (np.arange(start_pos, start_pos + n_tokens) + 1) // RATIO


def _reference_topk_row(score_row: np.ndarray, compress_len: int, index_topk: int, cmax: int) -> np.ndarray:
    """`model.py` lines 565 and 578-580, for one query.

    Positions the query cannot reach are already -inf, `topk` is one number for
    the whole chunk, and the picks that land in the -inf region are mapped to
    -1 afterwards. Dropping the -1s leaves the reachable positions in order.
    """
    masked = np.where(np.arange(cmax) < compress_len, score_row, -np.inf)
    topk = min(index_topk, cmax)
    idxs = np.sort(np.argsort(-masked, kind="stable")[:topk])
    return idxs[idxs < compress_len]


def test_prefill_topk_returns_exactly_the_reachable_reference_picks():
    rng = np.random.default_rng(21)
    start_pos, n_tokens, index_topk = 0, 16, 3
    compress_len = _reference_compress_lens(start_pos, n_tokens)
    cmax = int(compress_len.max())

    score = rng.normal(size=(n_tokens, cmax)).astype(np.float32)
    visible = np.arange(cmax)[None, :] < compress_len[:, None]
    masked = np.where(visible, score, -np.inf).astype(np.float32)

    got = topk_rows(mx.array(masked), np.minimum(index_topk, compress_len), index_topk)

    for t in range(n_tokens):
        want = _reference_topk_row(score[t], int(compress_len[t]), index_topk, cmax)
        assert np.array_equal(np.array(got[t]), want), f"token {t}"


def test_a_query_that_can_reach_nothing_selects_nothing():
    """The first `ratio` tokens of a prefill have compress_len 0. The reference
    returns a row that is entirely -1; ours must return an empty selection
    rather than position 0, which no query can see yet."""
    n_tokens = RATIO
    compress_len = _reference_compress_lens(0, n_tokens)
    assert compress_len[0] == 0
    score = np.full((n_tokens, 1), -np.inf, dtype=np.float32)

    got = topk_rows(mx.array(score), np.minimum(8, compress_len), 8)

    for t in range(n_tokens):
        if compress_len[t] == 0:
            assert np.array(got[t]).size == 0, f"token {t}"


def _reference_candidate_mask(score_row: np.ndarray, compress_len: int, topk_blocks: int, block_size: int) -> np.ndarray:
    """`model.py` lines 583-610, select_candidate_blocks, for one query."""
    width = score_row.size
    pad = -width % block_size
    padded = np.concatenate([score_row, np.full(pad, -np.inf, dtype=np.float32)])
    scores = padded.reshape(-1, block_size).max(axis=-1)
    num_blocks = scores.size

    # the partly filled block holding this query's newest position is pinned in
    last = (compress_len - 1) // block_size
    scores = np.where(np.arange(num_blocks) == last, np.inf, scores)

    k = min(topk_blocks, num_blocks)
    top = np.argsort(-scores, kind="stable")[:k]
    keep = np.zeros(num_blocks, dtype=bool)
    keep[top] = scores[top] > -np.inf
    return np.repeat(keep, block_size)[:width]


def test_prefill_candidate_blocks_match_the_reference_including_the_pinned_block():
    rng = np.random.default_rng(22)
    block_size, topk_blocks = 8, 3
    start_pos, n_tokens = 0, 24
    compress_len = _reference_compress_lens(start_pos, n_tokens)
    cmax = int(compress_len.max())

    score = rng.normal(size=(n_tokens, cmax)).astype(np.float32)
    visible = np.arange(cmax)[None, :] < compress_len[:, None]
    masked = np.where(visible, score, -np.inf).astype(np.float32)

    got = candidate_masks_chunk(mx.array(masked), compress_len, topk_blocks, block_size)

    for t in range(n_tokens):
        want = _reference_candidate_mask(masked[t], int(compress_len[t]), topk_blocks, block_size)
        assert np.array_equal(np.array(got[t]), want[: int(compress_len[t])]), f"token {t}"
