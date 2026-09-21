"""The memoised scalar kernel parameters must be values, not just fast.

`cachalot.model.kernel_consts` hands every fused Metal kernel the same
one-element array object for a repeated parameter instead of building a new one
per call. That is only safe because an MLX array is immutable, so these tests
pin both halves of the claim: the cache really does return one object, and the
kernels that consume it produce bit-identical output either way.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from cachalot.model import kernel_consts
from cachalot.model.decode_fused_metal import (
    hc_mixes_1d,
    hc_post_1d,
    hc_pre_norm_1d,
    rms_norm_1d,
    rope_apply_1d,
)

HC = 4
HIDDEN = 256


def test_memoised_arrays_are_reused_and_correct():
    a, b = kernel_consts.u32(5120), kernel_consts.u32(5120)
    assert a is b
    assert a.dtype == mx.uint32 and a.shape == (1,)
    assert int(a.item()) == 5120

    c, d = kernel_consts.f32(1e-20), kernel_consts.f32(1e-20)
    assert c is d
    assert c.dtype == mx.float32 and c.shape == (1,)
    assert float(c.item()) == float(np.float32(1e-20))

    assert kernel_consts.u32(5120) is not kernel_consts.u32(4096)


def test_switch_off_rebuilds_a_fresh_array(monkeypatch):
    monkeypatch.setattr(kernel_consts, "MEMOISE", False)
    a, b = kernel_consts.u32(777), kernel_consts.u32(777)
    assert a is not b
    assert int(a.item()) == int(b.item()) == 777


def _fused_outputs():
    mx.random.seed(11)
    x = mx.random.normal((HC, HIDDEN)).astype(mx.bfloat16)
    hc_fn = (mx.random.normal(((2 + HC) * HC, HC * HIDDEN)) * 0.01).astype(mx.bfloat16)
    hc_scale = mx.array([0.1, 0.1, 0.1], dtype=mx.float32)
    hc_base = mx.random.normal(((2 + HC) * HC,)).astype(mx.float32)
    norm_w = mx.ones((HIDDEN,)).astype(mx.bfloat16)
    pre_mix = mx.array([0.4, 0.3, 0.2, 0.1], dtype=mx.float32)
    y = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    rope_in = mx.random.normal((2, 64)).astype(mx.bfloat16)
    cos_row = mx.random.normal((32,)).astype(mx.float32)
    sin_row = mx.random.normal((32,)).astype(mx.float32)

    pre, post, comb = hc_mixes_1d(
        x, hc_fn, hc_scale, hc_base, norm_eps=1e-20, hc_mult=HC, sinkhorn_iters=20, hc_eps=1e-6
    )
    outs = [
        pre,
        post,
        comb,
        hc_pre_norm_1d(x, pre_mix, norm_w, eps=1e-20),
        hc_post_1d(y, x, post, comb),
        rms_norm_1d(y, norm_w, eps=1e-20),
        rope_apply_1d(rope_in, cos_row, sin_row),
    ]
    mx.eval(outs)
    return [np.array(o.astype(mx.float32)) for o in outs]


def test_fused_decode_kernels_are_bit_identical_either_way(monkeypatch):
    memoised = _fused_outputs()
    monkeypatch.setattr(kernel_consts, "MEMOISE", False)
    rebuilt = _fused_outputs()
    for got, want in zip(rebuilt, memoised, strict=True):
        assert np.array_equal(got, want)
