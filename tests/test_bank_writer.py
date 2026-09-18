"""The bank writer's on-disk contract.

A 3-bit bank built on 2026-09-18 came out corrupt and the cause is worth a test.
mx.quantize returns scales in the dtype of its input; the dense weights are
fp32, so the `mlx` fit produced fp32 scales while the shard header declares
BF16 and reserves two bytes per element. The writer took its stride from the
array rather than from the header, so every expert was written at double
stride, over the tensors that followed. Every 2-bit bank was fine only because
quantize_2bit casts to bf16 itself.

These check the contract rather than the symptom: what the quantizer emits must
match, byte for byte, what the shard plan reserved.
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


def _dense_expert() -> dict[str, mx.array]:
    mx.random.seed(31)
    return {
        proj: mx.random.normal(PROJ_ROWS[proj], dtype=mx.float32)
        for proj in ("w1", "w2", "w3")
    }


@pytest.mark.parametrize(
    "bits,group,fit",
    [(3, 64, "mlx"), (3, 128, "mlx"), (4, 64, "mlx"), (2, 128, "search"), (2, 64, "minmax")],
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


@pytest.mark.parametrize("bits,group", [(3, 64), (2, 128)])
def test_scales_and_biases_are_stored_bf16_whatever_the_fit(bits, group):
    """The failure was an fp32 scale in a BF16 slot; pin the dtype directly."""
    fit = "mlx" if bits != 2 else "search"
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
