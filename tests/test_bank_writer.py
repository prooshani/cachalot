"""The bank writer's on-disk contract.

A 3-bit bank built on 2026-09-18 came out corrupt and the cause is worth a test.
mx.quantize returns scales in the dtype of its input; the dense weights are
fp32, so the `mlx` fit produced fp32 scales while the shard header declares
BF16 and reserves two bytes per element. The writer took its stride from the
array rather than from the header, so every expert was written at double
stride, over the tensors that followed. Every 2-bit bank was fine only because
quantize_affine casts to bf16 itself.

These check the contract rather than the symptom: what the quantizer emits must
match, byte for byte, what the shard plan reserved. Above 2 bits there is a
second way to get the bytes right and the contents wrong -- MLX packs 3-bit
levels across word boundaries -- so the searched widths are parametrized here
too, and tests/test_quant_affine.py pins the packing itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from build_affine_bank import (  # noqa: E402
    FIELD_DTYPE,
    ITEM_BYTES,
    PROJ_ROWS,
    quantize_expert,
    tensor_shape,
)
from quant_affine import SHRINKS, fit_search, quantize_affine  # noqa: E402


def _dense_expert() -> dict[str, mx.array]:
    mx.random.seed(31)
    return {
        proj: mx.random.normal(PROJ_ROWS[proj], dtype=mx.float32)
        for proj in ("w1", "w2", "w3")
    }


@pytest.mark.parametrize(
    "bits,group,fit",
    [(3, 64, "mlx"), (3, 128, "mlx"), (4, 64, "mlx"), (2, 128, "search"), (2, 64, "minmax"),
     (3, 64, "search"), (3, 128, "search")],
)
def test_quantizer_output_fills_exactly_what_the_shard_plan_reserved(bits, group, fit):
    packed = quantize_expert(_dense_expert(), bits, group, fit)
    for (proj, field), arr in packed.items():
        shape = tensor_shape(proj, field, bits, group)
        reserved = shape[1] * shape[2] * ITEM_BYTES[FIELD_DTYPE[field]]
        assert arr.nbytes == reserved, (
            f"{proj}.{field} at {bits}-bit g{group} ({fit}): quantizer emitted "
            f"{arr.nbytes} B but the header reserves {reserved} B"
        )


@pytest.mark.parametrize("bits,group,fit",
                         [(3, 64, "mlx"), (3, 64, "search"), (2, 128, "search")])
def test_scales_and_biases_are_stored_bf16_whatever_the_fit(bits, group, fit):
    """The failure was an fp32 scale in a BF16 slot; pin the dtype directly."""
    packed = quantize_expert(_dense_expert(), bits, group, fit)
    for (_proj, field), arr in packed.items():
        if field in ("scales", "biases"):
            assert arr.dtype == mx.bfloat16
        else:
            assert arr.dtype == mx.uint32


def test_the_mlx_fit_survives_an_fp32_input():
    """mx.quantize inherits its input dtype; the dense weights are always fp32."""
    dense = _dense_expert()
    assert all(a.dtype == mx.float32 for a in dense.values())
    packed = quantize_expert(dense, 3, 64, "mlx")
    assert packed[("w1", "scales")].dtype == mx.bfloat16


def test_a_three_bit_searched_expert_reads_back_as_the_fit_intended():
    """The packing is the other way to write the right number of wrong bytes.

    quantize_expert emits what goes on disk; mx.dequantize is what the runtime
    reads it with. If the 3-bit layout were wrong the bytes would still fill the
    header exactly and the shard would still load, so the only thing that
    catches it is reading the round trip back and recovering the levels. The
    comparison is on levels rather than on values because mx.dequantize
    accumulates in bf16 while the fit is computed in fp32, which differs in the
    last mantissa bit and says nothing about the layout.
    """
    dense = {proj: w[:8] for proj, w in _dense_expert().items()}
    for proj, w in dense.items():
        q, scales, biases = quantize_affine(w, group_size=64, bits=3, fit=fit_search)
        back = mx.dequantize(q, scales, biases, group_size=64, bits=3).astype(mx.float32)
        s = scales.astype(mx.float32).reshape(-1, 1)
        b = biases.astype(mx.float32).reshape(-1, 1)
        got = mx.round((back.reshape(-1, 64) - b) / s)

        scale, bias = fit_search(w.astype(mx.float32).reshape(-1, 64), SHRINKS, 3)
        scale = scale.astype(mx.bfloat16).astype(mx.float32)
        bias = bias.astype(mx.bfloat16).astype(mx.float32)
        want = mx.clip(mx.round((w.astype(mx.float32).reshape(-1, 64) - bias) / scale), 0, 7)

        assert mx.all(got == want).item(), proj


# --- Activation-weighted fits -------------------------------------------------
#
# Weighting changes only the scales and biases, so it cannot change the byte
# layout -- which is exactly why it needs pinning: if the importance vector were
# silently dropped or misaligned the bank would still be the right size and
# would still load.


def _fake_importance() -> dict[str, mx.array]:
    """Per-column importance for one expert, in the shapes expert_importance returns."""
    mx.random.seed(37)
    rows_w1, cols_w1 = PROJ_ROWS["w1"]
    rows_w2, cols_w2 = PROJ_ROWS["w2"]
    return {
        "w1": mx.abs(mx.random.normal((cols_w1,), dtype=mx.float32)) + 0.1,
        "w3": mx.abs(mx.random.normal((cols_w1,), dtype=mx.float32)) + 0.1,
        "w2": mx.abs(mx.random.normal((cols_w2,), dtype=mx.float32)) + 0.1,
    }


@pytest.mark.parametrize("bits,group", [(3, 64), (2, 128)])
def test_a_weighted_fit_still_fills_exactly_what_the_shard_plan_reserved(bits, group):
    packed = quantize_expert(_dense_expert(), bits, group, "search", _fake_importance())
    for (proj, field), arr in packed.items():
        shape = tensor_shape(proj, field, bits, group)
        reserved = shape[1] * shape[2] * ITEM_BYTES[FIELD_DTYPE[field]]
        assert arr.nbytes == reserved, f"{proj}.{field} at {bits}-bit g{group}"


def test_the_importance_vector_actually_reaches_the_fit():
    """A dropped weighting is invisible in the shapes, so compare the bytes."""
    dense = _dense_expert()
    plain = quantize_expert(dense, 3, 64, "search")
    weighted = quantize_expert(dense, 3, 64, "search", _fake_importance())

    differ = [key for key, arr in plain.items()
              if not mx.all(arr == weighted[key]).item()]
    assert {proj for proj, _field in differ} == {"w1", "w2", "w3"}
    assert {field for _proj, field in differ} == {"weight", "scales", "biases"}
