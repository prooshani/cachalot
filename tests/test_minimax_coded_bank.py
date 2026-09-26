"""The bias-free MiniMax bank's encode/decode (cachalot.minimax.coded_bank, HANDOFF 18.4)."""

import numpy as np

from cachalot.minimax.coded_bank import BankLayout, bf16_bits_rne, decode_biases, encode_biases


def _bf16(x):
    return bf16_bits_rne(np.asarray(x, np.float32))


def _f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def test_roundtrip_is_exact_for_whole_multiples():
    rng = np.random.default_rng(0)
    scales = _bf16(rng.uniform(1e-4, 5e-2, 4096))
    k = rng.integers(-6, -2, 4096).astype(np.float32)
    biases = _bf16(k * _f32(scales))
    packed = encode_biases(scales, biases)
    assert packed is not None and packed.nbytes == 4096 // 4
    assert np.array_equal(decode_biases(scales, packed), biases)


def test_decode_writes_into_the_given_buffer():
    scales = _bf16(np.full(8, 0.01))
    biases = _bf16(np.float32(-4) * _f32(scales))
    out = np.zeros(8, np.uint16)
    decode_biases(scales, encode_biases(scales, biases), out)
    assert np.array_equal(out, biases)


def test_out_of_range_or_off_grid_biases_are_refused():
    scales = _bf16(np.full(8, 0.01))
    assert encode_biases(scales, _bf16(np.float32(-7) * _f32(scales))) is None  # k = -7: a raw record
    off = _bf16(np.float32(-4) * _f32(scales) + np.float32(1e-3))
    assert encode_biases(scales, off) is None


def test_rounding_matches_mlx_cast():
    import mlx.core as mx

    x = np.random.default_rng(1).standard_normal(10000).astype(np.float32) * 1e-2
    ours = bf16_bits_rne(x)
    theirs = np.array(mx.array(x).astype(mx.bfloat16).view(mx.uint16))
    assert np.array_equal(ours, theirs)


def test_layout_sizes_match_minimax():
    lay = BankLayout(weight=7077888, scales=589824)
    assert lay.codes == 73728
    assert lay.record("coded") == 23232512  # 22.16 MiB against 23.62 in the checkpoint
    assert lay.record("raw") > lay.record("coded")
