"""Recorded activations to the per-column weighting a fit is applied with.

The weighting is per column, and the columns are the axis the groups run along,
so an importance vector that is the right length but the wrong orientation
would be applied to the wrong weights and would still produce a bank of the
right size. These pin the shapes and the one derived quantity, w2's importance,
which is the SwiGLU hidden rather than the recorded input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from activation_importance import (  # noqa: E402
    expert_importance,
    load_activations,
    mean_square,
    swiglu_hidden,
)
from quant_affine import fit_search, group_weights, quantize_affine  # noqa: E402

HIDDEN, INTER = 64, 32     # the real shapes' proportions, small enough to test


def _dense():
    mx.random.seed(41)
    return {"w1": mx.random.normal((INTER, HIDDEN), dtype=mx.float32),
            "w3": mx.random.normal((INTER, HIDDEN), dtype=mx.float32),
            "w2": mx.random.normal((HIDDEN, INTER), dtype=mx.float32)}


def test_mean_square_is_normalized_to_mean_one():
    mx.random.seed(42)
    v = mx.random.normal((16, 8), dtype=mx.float32) * 3.0
    m = mean_square(v)
    assert m.shape == (16,)
    assert abs(float(mx.mean(m).item()) - 1.0) < 1e-5


def test_expert_importance_has_one_weight_per_column_of_each_matrix():
    dense = _dense()
    mx.random.seed(43)
    acts = {"importance": mx.abs(mx.random.normal((HIDDEN,), dtype=mx.float32)),
            "x": mx.random.normal((HIDDEN, 8), dtype=mx.float32)}
    imp = expert_importance(dense, acts)

    for proj, w in dense.items():
        assert imp[proj].shape == (w.shape[1],), proj
    # w1 and w3 share the recorded input; w2's is derived and therefore different.
    assert mx.all(imp["w1"] == imp["w3"]).item()
    assert imp["w2"].shape == (INTER,)


def test_w2_importance_follows_the_swiglu_hidden_not_the_input():
    dense = _dense()
    mx.random.seed(44)
    x = mx.random.normal((HIDDEN, 8), dtype=mx.float32)
    acts = {"importance": mx.ones((HIDDEN,), dtype=mx.float32), "x": x}

    imp = expert_importance(dense, acts)
    want = mean_square(swiglu_hidden(x, dense["w1"], dense["w3"]))
    assert mx.allclose(imp["w2"], want).item()


def test_group_weights_lines_up_with_the_group_reshape():
    """Row r, group g of the reshape covers columns [g*G, (g+1)*G) for every r."""
    group = 8
    cols, rows = 32, 3
    importance = mx.arange(cols, dtype=mx.float32)
    w = group_weights(importance, rows, group)

    assert w.shape == (rows * cols // group, group)
    for r in range(rows):
        for g in range(cols // group):
            got = w[r * (cols // group) + g]
            want = importance[g * group:(g + 1) * group]
            assert mx.all(got == want).item(), (r, g)


def test_weighting_lowers_the_error_on_the_columns_it_says_matter():
    """The whole point: a weighted fit trades error on quiet columns for loud ones."""
    mx.random.seed(45)
    w = mx.random.normal((8, 128), dtype=mx.float32)
    importance = mx.concatenate([mx.full((64,), 100.0), mx.full((64,), 0.01)])

    errors = {}
    for label, imp in (("plain", None), ("weighted", importance)):
        q, sc, bi = quantize_affine(w, group_size=128, bits=3, fit=fit_search, importance=imp)
        back = mx.dequantize(q, sc, bi, group_size=128, bits=3).astype(mx.float32)
        d = (back - w) ** 2
        errors[label] = (float(mx.sum(d[:, :64])), float(mx.sum(d[:, 64:])))

    loud_plain, quiet_plain = errors["plain"]
    loud_weighted, quiet_weighted = errors["weighted"]
    assert loud_weighted < loud_plain
    assert quiet_weighted > quiet_plain


def test_load_activations_reads_what_capture_activations_writes(tmp_path):
    path = tmp_path / "acts.npz"
    np.savez(path,
             sumsq_0=np.arange(HIDDEN, dtype=np.float32) + 1.0,
             tokens_0=np.array(4),
             samples_0=np.ones((5, HIDDEN), dtype=np.float32),
             sample_expert_0=np.zeros(5, dtype=np.int32))

    acts = load_activations(path, layers=2)
    assert set(acts) == {0}
    assert acts[0]["x"].shape == (HIDDEN, 5)
    assert abs(float(mx.mean(acts[0]["importance"]).item()) - 1.0) < 1e-5


def test_load_activations_skips_a_layer_with_no_samples(tmp_path):
    path = tmp_path / "partial.npz"
    np.savez(path, sumsq_0=np.ones(HIDDEN, dtype=np.float32), tokens_0=np.array(1))
    assert load_activations(path, layers=2) == {}


@pytest.mark.parametrize("bits", [2, 3])
def test_a_weighting_of_all_ones_is_the_unweighted_fit(bits):
    """The weighted normal equations must reduce to the plain ones, exactly."""
    mx.random.seed(46)
    w = mx.random.normal((8, 128), dtype=mx.float32)
    plain = quantize_affine(w, group_size=64, bits=bits, fit=fit_search)
    ones = quantize_affine(w, group_size=64, bits=bits, fit=fit_search,
                           importance=mx.ones((128,), dtype=mx.float32))

    for got, want in zip(ones, plain, strict=True):
        assert mx.all(got == want).item()
