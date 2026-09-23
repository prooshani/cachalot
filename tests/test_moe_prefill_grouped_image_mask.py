"""moe_prefill_grouped's image_mask/gate_bias_vl guard, HANDOFF section 16
piece 3 step 2.

image_mask is only wired through the batched (route_topk_rows) path;
batched=False (route_topk_batch, the per-token Python-loop fallback no
production caller sets -- see moe_prefill_grouped's own `batched` default)
has no per-token bias selection. The guard must fire before touching
expert_index/expert_store, so this needs no real resident expert fixture --
matching this module's existing lack of dedicated unit tests, which rely on
benchmarks/*_live_smoke.py for the expert-dispatch path instead.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from cachalot.model.moe_prefill_grouped import moe_prefill_grouped

N_EXPERTS = 8
HIDDEN = 12


def test_image_mask_with_batched_false_raises_before_touching_experts():
    x = mx.random.normal(shape=(3, HIDDEN))
    gate_weight = mx.random.normal(shape=(N_EXPERTS, HIDDEN))
    gate_bias = mx.random.normal(shape=(N_EXPERTS,))
    mx.eval(x, gate_weight, gate_bias)

    with pytest.raises(NotImplementedError):
        moe_prefill_grouped(
            x,
            layer_id=0,
            gate_weight=gate_weight,
            gate_bias=gate_bias,
            expert_index={},  # never reached if the guard fires first
            expert_store=None,
            shared_w1=mx.zeros((1,)),
            shared_w1_scales=mx.zeros((1,)),
            shared_w2=mx.zeros((1,)),
            shared_w2_scales=mx.zeros((1,)),
            shared_w3=mx.zeros((1,)),
            shared_w3_scales=mx.zeros((1,)),
            batched=False,
            image_mask=mx.array([True, False, True]),
        )
