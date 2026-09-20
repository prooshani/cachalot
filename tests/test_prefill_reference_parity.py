"""Pins the prefill path against the shipped reference implementation.

Section 7.4.8 found a transposed contraction in `hc_post` that survived months
of work because it preserved every aggregate statistic this project measured.
The handoff's standing instruction after that is to read our code against
`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` and to **write a test
whenever the comparison finds a match, not only when it finds a defect** --
`hc_post` had no test, and no A/B could have caught it because both arms shared
the error.

The decode path had been read against the reference at about forty points. The
prefill path had not been read at all, and it reaches the same arithmetic
through different code: batched hyper-connection coefficients instead of the
one-token Metal kernel, and a chunk-at-a-time indexer where the reference works
on a whole sequence with a per-query visible length. These tests record that
those two places agree, so that a later change has something to fail against.

Each reference expression below is transcribed from the shipped files and cited
by file and line.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="the prefill path is MLX code")

import mlx.core as mx  # noqa: E402

from cachalot.model.hyper_connection_mlx import (  # noqa: E402
    hc_mixes,
    hc_post,
    hc_split_sinkhorn,
)

HC, DIM = 4, 6
MIX_HC = (2 + HC) * HC


# ---------------------------------------------------------------------------
# Hyper connections, at the shapes prefill uses
# ---------------------------------------------------------------------------


def _reference_hc_post_one_token(x, residual, post, comb):
    """`model.py` line 962, Block.hc_post, written as explicit loops.

        y = post.unsqueeze(-1) * x.unsqueeze(-2)
            + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    """
    out = np.empty((HC, DIM), dtype=np.float32)
    for j in range(HC):
        acc = post[j] * x
        for i in range(HC):
            acc = acc + comb[i, j] * residual[i]
        out[j] = acc
    return out


def test_hc_post_is_the_reference_at_every_token_of_a_prefill_chunk():
    """Decode calls `hc_post` with one token; prefill calls it with a leading
    token axis, and a broadcast that is right for one shape can be wrong for
    the other. The reference is applied token by token."""
    rng = np.random.default_rng(11)
    tokens = 7
    x = rng.normal(size=(tokens, DIM)).astype(np.float32)
    residual = rng.normal(size=(tokens, HC, DIM)).astype(np.float32)
    post = rng.normal(size=(tokens, HC)).astype(np.float32)
    # deliberately asymmetric: a symmetric comb cannot tell the two
    # contractions apart, which is exactly how the defect survived
    comb = rng.random((tokens, HC, HC)).astype(np.float32)

    got = np.array(hc_post(mx.array(x), mx.array(residual), mx.array(post), mx.array(comb)))

    assert got.shape == (tokens, HC, DIM)
    for t in range(tokens):
        want = _reference_hc_post_one_token(x[t], residual[t], post[t], comb[t])
        assert np.allclose(got[t], want, atol=1e-5), f"token {t}"


def test_batched_hc_post_is_not_the_untransposed_mix():
    """The specific defect that shipped, at prefill shape: comb @ residual."""
    rng = np.random.default_rng(12)
    tokens = 3
    x = rng.normal(size=(tokens, DIM)).astype(np.float32)
    residual = rng.normal(size=(tokens, HC, DIM)).astype(np.float32)
    post = rng.normal(size=(tokens, HC)).astype(np.float32)
    comb = rng.random((tokens, HC, HC)).astype(np.float32)

    got = np.array(hc_post(mx.array(x), mx.array(residual), mx.array(post), mx.array(comb)))
    wrong = post[:, :, None] * x[:, None, :] + np.einsum("tij,tjd->tid", comb, residual)

    assert not np.allclose(got, wrong, atol=1e-5)


def _reference_hc_split_sinkhorn(mixes, hc_scale, hc_base, iters, eps):
    """`kernel.py` lines 407-462, hc_split_sinkhorn_kernel_, as explicit numpy.

    The comb block is read as `mixes[j * hc + k + hc * 2]`, so the flat tail of
    `mixes` is row-major over (j, k) -- the layout the sinkhorn then normalizes
    rows-then-columns in.
    """
    pre = 1.0 / (1.0 + np.exp(-(mixes[:HC] * hc_scale[0] + hc_base[:HC]))) + eps
    post = 2.0 / (1.0 + np.exp(-(mixes[HC:2 * HC] * hc_scale[1] + hc_base[HC:2 * HC])))

    comb = np.empty((HC, HC), dtype=np.float32)
    for j in range(HC):
        for k in range(HC):
            flat = j * HC + k + HC * 2
            comb[j, k] = mixes[flat] * hc_scale[2] + hc_base[flat]

    # comb = comb.softmax(-1) + eps
    comb = np.exp(comb - comb.max(axis=1, keepdims=True))
    comb = comb / comb.sum(axis=1, keepdims=True) + eps
    # comb = comb / (comb.sum(-2) + eps)
    comb = comb / (comb.sum(axis=0, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=0, keepdims=True) + eps)
    return pre, post, comb


def test_batched_sinkhorn_matches_the_reference_kernel():
    """Decode runs the one-token Metal sinkhorn; prefill runs the MLX one, and
    only the Metal one had ever been compared to anything."""
    rng = np.random.default_rng(13)
    tokens = 5
    mixes = rng.normal(size=(tokens, MIX_HC)).astype(np.float32)
    hc_scale = rng.random(3).astype(np.float32) + 0.5
    hc_base = rng.normal(size=(MIX_HC,)).astype(np.float32)
    iters, eps = 20, 1e-6

    pre, post, comb = hc_split_sinkhorn(
        mx.array(mixes), mx.array(hc_scale), mx.array(hc_base),
        hc_mult=HC, sinkhorn_iters=iters, eps=eps,
    )
    pre, post, comb = np.array(pre), np.array(post), np.array(comb)

    for t in range(tokens):
        want_pre, want_post, want_comb = _reference_hc_split_sinkhorn(
            mixes[t], hc_scale, hc_base, iters, eps
        )
        assert np.allclose(pre[t], want_pre, atol=1e-5), f"pre, token {t}"
        assert np.allclose(post[t], want_post, atol=1e-5), f"post, token {t}"
        assert np.allclose(comb[t], want_comb, atol=1e-5), f"comb, token {t}"


def test_the_prefill_and_decode_sinkhorns_agree():
    """`hc_mixes` dispatches to the Metal kernel at one token and to the MLX
    path at any other shape. They are the two arms of a screen that cleared
    `hc_post` while sharing its defect, so neither is the other's reference --
    but they must still agree, and nothing checked that they did."""
    rng = np.random.default_rng(14)
    tokens = 4
    x = rng.normal(size=(tokens, HC, DIM)).astype(np.float32)
    hc_fn = rng.normal(size=(MIX_HC, HC * DIM)).astype(np.float32)
    hc_scale = rng.random(3).astype(np.float32) + 0.5
    hc_base = rng.normal(size=(MIX_HC,)).astype(np.float32)
    kwargs = dict(norm_eps=1e-20, hc_mult=HC, sinkhorn_iters=20, hc_eps=1e-6)

    batched = [np.array(a) for a in hc_mixes(mx.array(x), mx.array(hc_fn), mx.array(hc_scale), mx.array(hc_base), **kwargs)]

    for t in range(tokens):
        one = hc_mixes(mx.array(x[t]), mx.array(hc_fn), mx.array(hc_scale), mx.array(hc_base), **kwargs)
        for name, got, want in zip(("pre", "post", "comb"), batched, one):
            assert np.allclose(got[t], np.array(want), atol=1e-4), f"{name}, token {t}"


def test_hc_mixes_normalizes_after_the_projection_not_before():
    """`model.py` lines 949-955:

        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt

    The projection is applied to the *unnormalized* stream and the scale is
    folded in afterwards. For a linear map those commute exactly, but the
    statistic is taken over the flattened `hc * dim` vector rather than over
    each stream, which does not commute with anything.
    """
    rng = np.random.default_rng(15)
    x = rng.normal(size=(HC, DIM)).astype(np.float32)
    hc_fn = rng.normal(size=(MIX_HC, HC * DIM)).astype(np.float32)
    hc_scale = np.ones(3, dtype=np.float32)
    hc_base = np.zeros(MIX_HC, dtype=np.float32)
    norm_eps = 1e-20

    flat = x.reshape(-1)
    want_mixes = (hc_fn @ flat) / np.sqrt((flat * flat).mean() + norm_eps)
    want = _reference_hc_split_sinkhorn(want_mixes, hc_scale, hc_base, 20, 1e-6)

    got = hc_mixes(
        mx.array(x), mx.array(hc_fn), mx.array(hc_scale), mx.array(hc_base),
        norm_eps=norm_eps, hc_mult=HC, sinkhorn_iters=20, hc_eps=1e-6,
    )
    for name, g, w in zip(("pre", "post", "comb"), got, want):
        assert np.allclose(np.array(g), w, atol=1e-4), name
