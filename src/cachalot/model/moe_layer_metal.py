from __future__ import annotations

import mlx.core as mx

from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.model.expert_metal import routed_expert_forward
from cachalot.model.moe_fused_metal import fused_routed_experts
from cachalot.model.router_mlx import RouterResult, route_topk
from cachalot.model.shared_expert_metal import shared_expert_forward
from cachalot.storage.index import ExpertEntry


def moe_layer_forward(
    x: mx.array,
    *,
    layer_id: int,
    gate_weight: mx.array,
    gate_bias: mx.array,
    expert_index: dict[tuple[int, int], ExpertEntry],
    expert_store: ResidentExpertStore,
    shared_w1: mx.array,
    shared_w1_scales: mx.array,
    shared_w2: mx.array,
    shared_w2_scales: mx.array,
    shared_w3: mx.array,
    shared_w3_scales: mx.array,
    topk: int = 6,
    gate_temp: float = 1.0,
    route_scale: float = 1.5,
    norm_topk_prob: bool = True,
    swiglu_limit: float = 10.0,
    fused: bool = True,
) -> tuple[mx.array, RouterResult]:
    """
    Complete DeepSeek V4.1 MoE path for one decode token:

      BF16 x
        -> FP32 router
        -> top-k routed FP4 experts
        -> resident FP8 shared expert
        -> sum
        -> BF16 output
    """

    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D for decode, got {x.shape}"
        )

    route = route_topk(
        x,
        gate_weight,
        gate_bias,
        topk=topk,
        gate_temp=gate_temp,
        route_scale=route_scale,
        norm_topk_prob=norm_topk_prob,
    )

    # Routing is needed on the CPU to address the resident store.
    mx.eval(
        route.indices,
        route.weights,
    )

    expert_ids = route.indices.tolist()
    router_weights = route.weights.tolist()

    routed = mx.zeros(
        x.shape,
        dtype=mx.float32,
    )

    entries = []

    for expert_id in expert_ids:
        key = (
            layer_id,
            int(expert_id),
        )

        entry = expert_index.get(key)

        if entry is None:
            raise KeyError(
                f"Missing routed expert "
                f"layer={layer_id}, expert={expert_id}"
            )

        entries.append(entry)

    # All misses of this layer are read concurrently instead of one
    # blocking SSD read per expert on the main thread.
    miss_budget = getattr(expert_store, "decode_miss_budget", None)
    experts = expert_store.get_many(
        entries,
        max_misses=miss_budget,
        priorities=router_weights,
    )

    weights = route.weights

    if miss_budget is not None and any(e is None for e in experts):
        # Approximate mode: skipped experts are dropped and the remaining
        # router weights are rescaled to keep the official total mass.
        kept = [i for i, e in enumerate(experts) if e is not None]
        experts = [experts[i] for i in kept]
        total = sum(router_weights)
        kept_sum = sum(router_weights[i] for i in kept) or 1.0
        weights = mx.array(
            [router_weights[i] * total / kept_sum for i in kept],
            dtype=mx.float32,
        )
        router_weights = weights.tolist()

    if not experts:
        routed = mx.zeros(x.shape, dtype=mx.float32)
    elif fused:
        # Two launches for all top-k experts (see moe_fused_metal).
        routed = fused_routed_experts(
            x,
            experts,
            weights,
            hidden_size=x.shape[0],
            intermediate=2304,
            swiglu_limit=swiglu_limit,
        )
    else:
        for expert, router_weight in zip(
            experts,
            router_weights,
            strict=True,
        ):
            model = expert.as_model_dict()

            y = routed_expert_forward(
                x,
                w1_packed=model["w1.weight"],
                w1_scales=model["w1.scale"],
                w2_packed=model["w2.weight"],
                w2_scales=model["w2.scale"],
                w3_packed=model["w3.weight"],
                w3_scales=model["w3.scale"],
                weight=float(router_weight),
                swiglu_limit=swiglu_limit,
            )

            routed = (
                routed
                + y.astype(mx.float32)
            )

    shared = shared_expert_forward(
        x,
        w1=shared_w1,
        w1_scales=shared_w1_scales,
        w2=shared_w2,
        w2_scales=shared_w2_scales,
        w3=shared_w3,
        w3_scales=shared_w3_scales,
        swiglu_limit=swiglu_limit,
    )

    output = (
        routed
        + shared.astype(mx.float32)
    ).astype(x.dtype)

    return output, route
