"""route_topk_rows's per-token bias_vl selection, HANDOFF section 16 piece 3
step 2.

route_topk_rows previously broadcast one `bias` [n_experts] array to every
row. bias_vl/image_mask let a row select a *different* [n_experts] bias --
the per-token correction bias the 40 vision-bearing MoE layers carry
(`layers.N.ffn.gate.bias_vl`, section 16.2) -- without changing the scores
themselves (selection only, matching the official semantics' "correction
bias affects selection only").

Correctness is checked against router_mlx.route_topk called once per row
with the bias that row should have used -- the reference the batched kernel
must reduce to.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from cachalot.model.moe_prefill_batched import route_topk_rows
from cachalot.model.router_mlx import route_topk

N_EXPERTS = 16
HIDDEN = 12
TOPK = 4


def setup_function(_fn):
    # Fixed seed: near-tied scores can flip which expert wins top-k under
    # the batched-vs-single-row fp32 accumulation-order difference (see the
    # weights/scores tolerance note below), so an unseeded run would flake.
    mx.random.seed(0)


def _weight_bias():
    weight = mx.random.normal(shape=(N_EXPERTS, HIDDEN))
    bias = mx.random.normal(shape=(N_EXPERTS,)) * 0.1
    bias_vl = mx.random.normal(shape=(N_EXPERTS,)) * 0.1
    mx.eval(weight, bias, bias_vl)
    return weight, bias, bias_vl


def test_no_bias_vl_is_bit_identical_to_before():
    weight, bias, _ = _weight_bias()
    x = mx.random.normal(shape=(5, HIDDEN))
    mx.eval(x)

    indices_before, weights_before, scores_before = route_topk_rows(
        x, weight, bias, topk=TOPK
    )
    indices_after, weights_after, scores_after = route_topk_rows(
        x, weight, bias, topk=TOPK, bias_vl=None, image_mask=None
    )
    mx.eval(indices_before, weights_before, scores_before, indices_after, weights_after, scores_after)

    assert bool(mx.all(indices_before == indices_after).item())
    assert bool(mx.all(weights_before == weights_after).item())
    assert bool(mx.all(scores_before == scores_after).item())


def test_bias_vl_alone_or_image_mask_alone_raises():
    weight, bias, bias_vl = _weight_bias()
    x = mx.random.normal(shape=(3, HIDDEN))
    mx.eval(x)

    with pytest.raises(ValueError):
        route_topk_rows(x, weight, bias, topk=TOPK, bias_vl=bias_vl, image_mask=None)

    with pytest.raises(ValueError):
        route_topk_rows(
            x, weight, bias, topk=TOPK,
            bias_vl=None, image_mask=mx.array([True, False, True]),
        )


def test_image_mask_wrong_shape_raises():
    weight, bias, bias_vl = _weight_bias()
    x = mx.random.normal(shape=(3, HIDDEN))
    mx.eval(x)

    with pytest.raises(ValueError):
        route_topk_rows(
            x, weight, bias, topk=TOPK,
            bias_vl=bias_vl, image_mask=mx.array([True, False]),
        )


def test_text_rows_use_bias_image_rows_use_bias_vl():
    weight, bias, bias_vl = _weight_bias()
    x = mx.random.normal(shape=(6, HIDDEN))
    mx.eval(x)

    image_mask = mx.array([False, True, False, True, True, False])

    indices, weights, scores = route_topk_rows(
        x, weight, bias, topk=TOPK, bias_vl=bias_vl, image_mask=image_mask
    )
    mx.eval(indices, weights, scores)

    mask_list = image_mask.tolist()
    for row in range(x.shape[0]):
        row_bias = bias_vl if mask_list[row] else bias
        want = route_topk(x[row], weight, row_bias, topk=TOPK)
        mx.eval(want.indices, want.weights, want.scores)

        assert bool(mx.all(indices[row] == want.indices).item()), f"row {row} indices mismatch"
        # weights and scores are compared with a tolerance, not exact
        # equality: the batched matmul (all rows at once) and route_topk's
        # single-row matmul sum in a different order, the same
        # fp32-accumulation-order caveat moe_prefill_batched.py's own module
        # docstring already documents for this file. Selection (indices) is
        # exact because bias_vl/bias are added before argsort, unaffected by
        # which matmul path computed the underlying scores.
        assert bool(
            mx.allclose(weights[row], want.weights, atol=1e-5, rtol=1e-5).item()
        ), f"row {row} weights mismatch"
        # scores never depend on image_mask -- selection only.
        assert bool(
            mx.allclose(scores[row], want.scores, atol=1e-5, rtol=1e-5).item()
        ), f"row {row} scores mismatch"


def test_all_image_rows_matches_bias_vl_broadcast_uniformly():
    weight, bias, bias_vl = _weight_bias()
    x = mx.random.normal(shape=(4, HIDDEN))
    mx.eval(x)

    image_mask = mx.array([True, True, True, True])

    got_indices, got_weights, _ = route_topk_rows(
        x, weight, bias, topk=TOPK, bias_vl=bias_vl, image_mask=image_mask
    )
    want_indices, want_weights, _ = route_topk_rows(x, weight, bias_vl, topk=TOPK)
    mx.eval(got_indices, got_weights, want_indices, want_weights)

    assert bool(mx.all(got_indices == want_indices).item())
    assert bool(mx.all(got_weights == want_weights).item())
