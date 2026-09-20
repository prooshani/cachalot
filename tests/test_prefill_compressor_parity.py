"""Pins the prefill compressor's group alignment against the shipped reference.

`Compressor.forward` (`model.py` lines 458-484) pools `compress_ratio`
consecutive tokens into one KV latent with a learned softmax gate. Its prefill
branch assumes the whole prompt arrives at once: it groups from absolute
position 0 and parks the trailing partial group in `kv_state` / `score_state`.
We prefill in chunks, so the same grouping has to survive a chunk boundary that
falls in the middle of a group, and that is code the reference has no
counterpart for and nothing had checked.

Two things are pinned here. The single-chunk case must equal the reference
expression, and splitting the same prompt at a position that is deliberately
not a multiple of the ratio must produce exactly the same latents -- if the
carried partial group were dropped or double-counted, every compressed position
after the boundary would shift, and the failure would look like a vague quality
loss rather than like a bug.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="the prefill compressor is MLX code")

import mlx.core as mx  # noqa: E402

from cachalot.model.compressor_mlx import CompressorState  # noqa: E402
from cachalot.model.source_prefill_batched import compressor_chunk  # noqa: E402

RATIO, HIDDEN, HEAD_DIM, EPS = 4, 8, 6, 1e-20


def _weights(seed: int):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(HEAD_DIM, HIDDEN)).astype(np.float32) * 0.1,
        rng.normal(size=(HEAD_DIM, HIDDEN)).astype(np.float32) * 0.1,
        rng.random(HEAD_DIM).astype(np.float32) + 0.5,
    )


def _state():
    return CompressorState.create(max_batch_size=1, compress_ratio=RATIO, head_dim=HEAD_DIM)


def _reference_prefill(x, wkv, wgate, norm_weight):
    """`model.py` lines 464-473, the start_pos == 0 branch, as explicit numpy.

        kv, score = self.wkv(x), self.wgate(x)
        cutoff = seqlen - seqlen % ratio
        kv = kv.unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        return self.norm(kv)
    """
    kv, score = x @ wkv.T, x @ wgate.T
    groups = x.shape[0] // RATIO
    kv_g = kv[: groups * RATIO].reshape(groups, RATIO, HEAD_DIM)
    sc_g = score[: groups * RATIO].reshape(groups, RATIO, HEAD_DIM)
    shifted = sc_g - sc_g.max(axis=1, keepdims=True)
    soft = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    pooled = (kv_g * soft).sum(axis=1)
    rms = np.sqrt((pooled.astype(np.float32) ** 2).mean(axis=-1, keepdims=True) + EPS)
    return (pooled / rms) * norm_weight


def test_single_chunk_prefill_pools_the_reference_groups():
    wkv, wgate, norm_weight = _weights(31)
    n_tokens = 3 * RATIO + 2  # a deliberate trailing partial group
    x = np.random.default_rng(32).normal(size=(n_tokens, HIDDEN)).astype(np.float32)

    latents, ends = compressor_chunk(
        mx.array(x), start_pos=0, compress_ratio=RATIO,
        norm_weight=mx.array(norm_weight), wkv_weight=mx.array(wkv),
        wgate_weight=mx.array(wgate), state=_state(), eps=EPS,
    )

    assert np.allclose(np.array(latents), _reference_prefill(x, wkv, wgate, norm_weight), atol=1e-4)
    # group g closes at the position of its last token
    assert np.array_equal(ends, np.array([RATIO - 1, 2 * RATIO - 1, 3 * RATIO - 1]))


def test_splitting_a_prompt_mid_group_changes_nothing():
    """The reference has no chunked prefill, so this compares us against
    ourselves: the boundary is the only thing that moves."""
    wkv, wgate, norm_weight = _weights(33)
    n_tokens = 4 * RATIO
    x = np.random.default_rng(34).normal(size=(n_tokens, HIDDEN)).astype(np.float32)
    kwargs = dict(
        compress_ratio=RATIO, norm_weight=mx.array(norm_weight),
        wkv_weight=mx.array(wkv), wgate_weight=mx.array(wgate), eps=EPS,
    )

    whole, whole_ends = compressor_chunk(mx.array(x), start_pos=0, state=_state(), **kwargs)

    split = RATIO + 1  # not a multiple of the ratio: a group straddles the boundary
    state = _state()
    first, first_ends = compressor_chunk(mx.array(x[:split]), start_pos=0, state=state, **kwargs)
    second, second_ends = compressor_chunk(mx.array(x[split:]), start_pos=split, state=state, **kwargs)

    chunked = np.concatenate([np.array(first), np.array(second)], axis=0)
    assert np.allclose(chunked, np.array(whole), atol=1e-4)
    assert np.array_equal(np.concatenate([first_ends, second_ends]), whole_ends)


def test_a_chunk_that_completes_no_group_yields_nothing_and_still_carries():
    wkv, wgate, norm_weight = _weights(35)
    x = np.random.default_rng(36).normal(size=(RATIO, HIDDEN)).astype(np.float32)
    kwargs = dict(
        compress_ratio=RATIO, norm_weight=mx.array(norm_weight),
        wkv_weight=mx.array(wkv), wgate_weight=mx.array(wgate), eps=EPS,
    )
    state = _state()

    latents, ends = compressor_chunk(mx.array(x[: RATIO - 1]), start_pos=0, state=state, **kwargs)
    assert latents is None and ends.size == 0

    latents, ends = compressor_chunk(mx.array(x[RATIO - 1 :]), start_pos=RATIO - 1, state=state, **kwargs)
    assert np.array_equal(ends, np.array([RATIO - 1]))
    assert np.allclose(np.array(latents), _reference_prefill(x, wkv, wgate, norm_weight), atol=1e-4)
