"""The GQA decode attention kernel against MLX's SDPA (HANDOFF 18.3): same result up to float32 rounding."""

import mlx.core as mx
import pytest

from cachalot.minimax.gqa_decode import gqa_decode_attention

H, KVH, D = 64, 4, 128


@pytest.mark.parametrize(
    "length,cap",
    [(1, 256), (100, 256), (256, 256), (257, 512), (1000, 1000), (5000, 5120), (70001, 70144)],
)
def test_matches_sdpa(length, cap):
    mx.random.seed(length)
    k = mx.random.normal((1, KVH, cap, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, KVH, cap, D)).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, 1, D)) * 3).astype(mx.bfloat16)
    scale = D**-0.5
    ref = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k[..., :length, :].astype(mx.float32), v[..., :length, :].astype(mx.float32), scale=scale
    )
    got = gqa_decode_attention(q, k, v, length, scale)
    assert got.shape == (1, H, 1, D) and got.dtype == mx.bfloat16
    assert mx.abs(got.astype(mx.float32) - ref).max().item() < 2e-2


def test_keys_past_the_length_are_ignored():
    k = mx.random.normal((1, KVH, 512, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, KVH, 512, D)).astype(mx.bfloat16)
    q = mx.random.normal((1, H, 1, D)).astype(mx.bfloat16)
    a = gqa_decode_attention(q, k, v, 300, D**-0.5)
    k2 = mx.concatenate([k[..., :300, :], mx.full((1, KVH, 212, D), 50.0, dtype=mx.bfloat16)], axis=2)
    v2 = mx.concatenate([v[..., :300, :], mx.full((1, KVH, 212, D), 50.0, dtype=mx.bfloat16)], axis=2)
    b = gqa_decode_attention(q, k2, v2, 300, D**-0.5)
    assert mx.array_equal(a, b).item()
