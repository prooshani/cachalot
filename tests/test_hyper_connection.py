"""Pins the hyper-connection residual mixing against the shipped reference.

`hc_post` writes a sub-layer's output back into the `hc_mult` residual streams
and mixes the incoming streams through `comb`. The official implementation
(`inference/model.py`, `Block.hc_post`) is

    torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)

which broadcasts to `elem[i, j, :] = comb[i, j] * residual[i, :]` and sums over
`i`, so output stream `j` is `sum_i comb[i, j] * residual[i]`. **comb is
contracted over its first index.**

Until 2026-09-20 both of this runtime's implementations contracted the second
index instead -- `comb @ residual` where the reference does `comb.T @ residual`.
It survived for months because `comb` comes out of a sinkhorn normalization and
is close to doubly stochastic: every stream still receives roughly the right
total weight, so the model stayed fluent and scored 88 % top-1 on ordinary
tokens. What a transpose destroys is *which* stream a given piece of information
lands in, and that shows up only in predictions that need one specific earlier
fact -- copying a word the text has already spelled out. HANDOFF section 7.4.7.

comb is asymmetric, so these tests fail loudly if the contraction flips back.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="hyper connections are MLX code")

import mlx.core as mx  # noqa: E402

from cachalot.model.hyper_connection_mlx import hc_post, hc_pre  # noqa: E402

HC, DIM = 4, 6


def _inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(DIM,)).astype(np.float32),
        rng.normal(size=(HC, DIM)).astype(np.float32),
        rng.normal(size=(HC,)).astype(np.float32),
        # deliberately asymmetric: a symmetric comb could not tell the two
        # contractions apart, which is exactly how the bug survived
        rng.random((HC, HC)).astype(np.float32),
    )


def _reference_hc_post(x, residual, post, comb):
    """model.py Block.hc_post, written as explicit loops."""
    out = np.empty((HC, DIM), dtype=np.float32)
    for j in range(HC):
        acc = post[j] * x
        for i in range(HC):
            acc = acc + comb[i, j] * residual[i]
        out[j] = acc
    return out


def test_hc_post_contracts_comb_over_its_first_index():
    x, residual, post, comb = _inputs()
    got = np.array(hc_post(mx.array(x), mx.array(residual), mx.array(post), mx.array(comb)))
    assert np.allclose(got, _reference_hc_post(x, residual, post, comb), atol=1e-5)


def test_hc_post_is_not_the_untransposed_mix():
    """The specific defect that was shipped: comb @ residual."""
    x, residual, post, comb = _inputs(1)
    wrong = np.empty((HC, DIM), dtype=np.float32)
    for i in range(HC):
        acc = post[i] * x
        for j in range(HC):
            acc = acc + comb[i, j] * residual[j]
        wrong[i] = acc
    got = np.array(hc_post(mx.array(x), mx.array(residual), mx.array(post), mx.array(comb)))
    assert not np.allclose(got, wrong, atol=1e-5)


def test_the_fused_kernel_agrees_with_the_mlx_path():
    """Both implementations carried the same transpose, so neither can be the
    other's reference; each is checked against model.py and then against the
    other."""
    from cachalot.model.decode_fused_metal import hc_post_1d

    x, residual, post, comb = _inputs(2)
    fused = np.array(hc_post_1d(mx.array(x), mx.array(residual), mx.array(post), mx.array(comb)))
    assert np.allclose(fused, _reference_hc_post(x, residual, post, comb), atol=1e-4)
    plain = np.array(hc_post(mx.array(x), mx.array(residual), mx.array(post), mx.array(comb)))
    assert np.allclose(fused, plain, atol=1e-4)


def test_hc_pre_is_a_plain_weighted_sum_over_streams():
    """model.py: sum(pre_mix.unsqueeze(-1) * x.float(), dim=2). No transpose to
    get wrong here, but it is the other half of the pair and cheap to pin."""
    _, streams, pre_mix, _ = _inputs(3)
    got = np.array(hc_pre(mx.array(streams), mx.array(pre_mix)))
    assert np.allclose(got, (pre_mix[:, None] * streams).sum(axis=0), atol=1e-5)
