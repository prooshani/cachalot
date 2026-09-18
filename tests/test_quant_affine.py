"""The affine quantizer in benchmarks/quant_affine.py: packing and fit quality.

The packing is the part that must be exactly right: a bank built with it is
read back by mx.dequantize and mx.quantized_matmul, which assume MLX's own
layout. These tests pin the layout against mx.quantize itself, at 2 bits where
it aligns to a word and above 2 bits where it does not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from quant_affine import (  # noqa: E402
    fit_minmax, fit_search, fit_search_lsq, fit_search_wide_lsq, pack_2bit, pack_3bit,
    pack_bits, quantize_2bit, quantize_affine,
)


def _unpack_via_dequantize(q, scales, biases, group_size, bits=2):
    """Recover the level of every weight from what mx.dequantize produces."""
    dense = mx.dequantize(q, scales, biases, group_size=group_size, bits=bits).astype(mx.float32)
    s = scales.astype(mx.float32).reshape(-1, 1)
    b = biases.astype(mx.float32).reshape(-1, 1)
    return mx.round((dense.reshape(-1, group_size) - b) / s)


def _relative_error(dense, w):
    return float(mx.sqrt(mx.sum((dense - w) ** 2) / mx.sum(w**2)))


def test_pack_2bit_matches_mlx_word_layout():
    mx.random.seed(11)
    w = mx.random.normal((8, 256), dtype=mx.float32)
    q, scales, biases = mx.quantize(w, group_size=64, bits=2)
    levels = _unpack_via_dequantize(q, scales, biases, 64)

    assert mx.all(pack_2bit(levels).reshape(q.shape) == q).item()


def test_quantize_2bit_round_trips_through_dequantize():
    mx.random.seed(12)
    w = mx.random.normal((16, 512), dtype=mx.float32)
    q, scales, biases = quantize_2bit(w, group_size=64)
    reference = mx.quantize(w, group_size=64, bits=2)

    assert q.shape == reference[0].shape and q.dtype == mx.uint32
    assert scales.shape == reference[1].shape and scales.dtype == mx.bfloat16
    assert biases.shape == reference[2].shape and biases.dtype == mx.bfloat16

    levels = _unpack_via_dequantize(q, scales, biases, 64)
    assert float(mx.min(levels).item()) >= 0.0
    assert float(mx.max(levels).item()) <= 3.0


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_search_fit_beats_mlx_and_minmax(group_size):
    mx.random.seed(13)
    w = mx.random.normal((32, 512), dtype=mx.float32)

    errors = {}
    for name, fit in (("minmax", fit_minmax), ("search", fit_search)):
        q, scales, biases = quantize_2bit(w, group_size=group_size, fit=fit)
        errors[name] = _relative_error(
            mx.dequantize(q, scales, biases, group_size=group_size, bits=2).astype(mx.float32), w
        )
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=2)
    errors["mlx"] = _relative_error(
        mx.dequantize(q, scales, biases, group_size=group_size, bits=2).astype(mx.float32), w
    )

    assert errors["search"] < errors["mlx"]
    assert errors["search"] < errors["minmax"]


def test_quantized_matmul_accepts_the_packed_output():
    mx.random.seed(14)
    w = mx.random.normal((64, 256), dtype=mx.float32)
    x = mx.random.normal((256,), dtype=mx.float32)
    q, scales, biases = quantize_2bit(w, group_size=64)

    got = mx.quantized_matmul(x.astype(mx.bfloat16), q, scales, biases,
                              transpose=True, group_size=64, bits=2)
    dense = mx.dequantize(q, scales, biases, group_size=64, bits=2).astype(mx.float32)
    want = dense @ x

    mx.eval(got, want)
    # quantized_matmul returns bf16 and accumulates in it, so agreement is to
    # bf16 precision, not to the fp32 reference product.
    assert _relative_error(got.astype(mx.float32), want) < 0.02


def test_pack_2bit_rejects_a_group_that_is_not_a_multiple_of_sixteen():
    with pytest.raises(ValueError):
        pack_2bit(mx.zeros((2, 24)))


@pytest.mark.parametrize("group_size", [64, 128])
def test_lsq_refinement_never_loses_to_the_grid_it_refines(group_size):
    """Least-squares refinement keeps the grid's fit per group unless it beats it."""
    mx.random.seed(15)
    w = mx.random.normal((32, 512), dtype=mx.float32)

    errors = {}
    for name, fit in (("search", fit_search), ("search-lsq", fit_search_lsq),
                      ("wide-lsq", fit_search_wide_lsq)):
        q, scales, biases = quantize_2bit(w, group_size=group_size, fit=fit)
        errors[name] = _relative_error(
            mx.dequantize(q, scales, biases, group_size=group_size, bits=2).astype(mx.float32), w
        )

    assert errors["search-lsq"] <= errors["search"]
    assert errors["wide-lsq"] <= errors["search-lsq"]


def test_wide_lsq_output_stays_inside_the_two_bit_levels():
    """A refined fit still has to pack: every level in 0..3, and the layout unchanged."""
    mx.random.seed(16)
    w = mx.random.normal((16, 512), dtype=mx.float32)
    q, scales, biases = quantize_2bit(w, group_size=128, fit=fit_search_wide_lsq)
    reference = mx.quantize(w, group_size=128, bits=2)

    assert q.shape == reference[0].shape and q.dtype == mx.uint32
    assert scales.shape == reference[1].shape and biases.shape == reference[2].shape
    levels = _unpack_via_dequantize(q, scales, biases, 128)
    assert float(mx.min(levels).item()) >= 0.0
    assert float(mx.max(levels).item()) <= 3.0
    assert mx.all(pack_2bit(levels).reshape(q.shape) == q).item()


# --- Packing above 2 bits -----------------------------------------------------
#
# At 3, 5 and 6 bits MLX's layout is one contiguous little-endian bit stream per
# row, so a level straddles a word boundary and the 2-bit packer does not
# generalise. These tests are the contract: a bank written with the wrong layout
# reads back as noise through mx.dequantize with no error anywhere, which is the
# failure mode that cost a whole build on 2026-09-18.


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_pack_bits_reproduces_mlx_layout_at_every_width(bits, group_size):
    mx.random.seed(31)
    w = mx.random.normal((8, 512), dtype=mx.float32)
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    levels = _unpack_via_dequantize(q, scales, biases, group_size, bits)

    assert q.shape[-1] == 512 * bits // 32
    assert mx.all(pack_bits(levels, bits).reshape(q.shape) == q).item()


def test_pack_3bit_is_pack_bits_at_three():
    mx.random.seed(32)
    levels = mx.random.randint(0, 8, (4, 64))
    assert mx.all(pack_3bit(levels) == pack_bits(levels, 3)).item()


def test_pack_3bit_straddles_the_word_boundary():
    """Level 10 of a chunk occupies bits 30, 31 and 32, so it spans two words."""
    levels = mx.zeros((1, 32), dtype=mx.uint32)
    levels[0, 10] = 7
    packed = pack_3bit(levels)

    assert packed.shape == (1, 3)
    assert int(packed[0, 0].item()) == 0xC0000000   # bits 30 and 31
    assert int(packed[0, 1].item()) == 0x1          # bit 32, the third bit of the level
    assert int(packed[0, 2].item()) == 0


def test_pack_bits_rejects_a_group_that_is_not_a_whole_number_of_words():
    with pytest.raises(ValueError):
        pack_bits(mx.zeros((2, 20)), 3)     # 20 x 3 bits is not a multiple of 32


@pytest.mark.parametrize("group_size", [64, 128])
def test_quantize_affine_at_three_bits_matches_mlx_shapes_and_levels(group_size):
    mx.random.seed(33)
    w = mx.random.normal((16, 512), dtype=mx.float32)
    q, scales, biases = quantize_affine(w, group_size=group_size, bits=3)
    reference = mx.quantize(w, group_size=group_size, bits=3)

    assert q.shape == reference[0].shape and q.dtype == mx.uint32
    assert scales.shape == reference[1].shape and scales.dtype == mx.bfloat16
    assert biases.shape == reference[2].shape and biases.dtype == mx.bfloat16

    levels = _unpack_via_dequantize(q, scales, biases, group_size, 3)
    assert float(mx.min(levels).item()) >= 0.0
    assert float(mx.max(levels).item()) <= 7.0
    assert mx.all(pack_3bit(levels).reshape(q.shape) == q).item()


@pytest.mark.parametrize("group_size", [64, 128])
def test_searched_fit_beats_mlx_at_three_bits(group_size):
    """Section 8.1's claim at 3 bits: MLX wastes a level at every width, not only at 2."""
    mx.random.seed(34)
    w = mx.random.normal((32, 512), dtype=mx.float32)

    q, scales, biases = quantize_affine(w, group_size=group_size, bits=3, fit=fit_search)
    searched = _relative_error(
        mx.dequantize(q, scales, biases, group_size=group_size, bits=3).astype(mx.float32), w
    )
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=3)
    mlx = _relative_error(
        mx.dequantize(q, scales, biases, group_size=group_size, bits=3).astype(mx.float32), w
    )

    assert searched < mlx


def test_quantized_matmul_accepts_three_bit_packed_output():
    mx.random.seed(35)
    w = mx.random.normal((64, 256), dtype=mx.float32)
    x = mx.random.normal((256,), dtype=mx.float32)
    q, scales, biases = quantize_affine(w, group_size=64, bits=3)

    got = mx.quantized_matmul(x.astype(mx.bfloat16), q, scales, biases,
                              transpose=True, group_size=64, bits=3)
    dense = mx.dequantize(q, scales, biases, group_size=64, bits=3).astype(mx.float32)
    want = dense @ x

    mx.eval(got, want)
    assert _relative_error(got.astype(mx.float32), want) < 0.02


def test_quantize_2bit_is_quantize_affine_at_two_bits():
    """The 2-bit name is a wrapper now; its callers must see byte-identical output."""
    mx.random.seed(36)
    w = mx.random.normal((16, 512), dtype=mx.float32)
    a = quantize_2bit(w, group_size=128, fit=fit_search_wide_lsq)
    b = quantize_affine(w, group_size=128, bits=2, fit=fit_search_wide_lsq)

    for got, want in zip(a, b):
        assert got.dtype == want.dtype and got.shape == want.shape
        assert mx.all(got == want).item()
