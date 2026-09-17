"""The 2-bit quantizer in benchmarks/quant_affine.py: packing and fit quality.

The packing is the part that must be exactly right: a bank built with it is
read back by mx.dequantize and mx.quantized_matmul, which assume MLX's own
layout. These tests pin the layout against mx.quantize itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from quant_affine import fit_minmax, fit_search, pack_2bit, quantize_2bit  # noqa: E402


def _unpack_via_dequantize(q, scales, biases, group_size):
    """Recover the 0..3 level of every weight from what mx.dequantize produces."""
    dense = mx.dequantize(q, scales, biases, group_size=group_size, bits=2).astype(mx.float32)
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
