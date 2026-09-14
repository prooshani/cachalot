from __future__ import annotations

import mlx.core as mx

from cachalot.model.expert_metal import routed_expert_forward


def routed_topk_forward(
    x: mx.array,
    experts: list[dict[str, mx.array]],
    weights: mx.array,
    *,
    swiglu_limit: float = 10.0,
) -> mx.array:
    if len(experts) != weights.size:
        raise ValueError(
            f"{len(experts)=} does not match {weights.size=}"
        )

    outputs: list[mx.array] = []

    for i, expert in enumerate(experts):
        outputs.append(
            routed_expert_forward(
                x,
                w1_packed=expert["w1.weight"],
                w1_scales=expert["w1.scale"],
                w2_packed=expert["w2.weight"],
                w2_scales=expert["w2.scale"],
                w3_packed=expert["w3.weight"],
                w3_scales=expert["w3.scale"],
                weight=weights[i],
                swiglu_limit=swiglu_limit,
            )
        )

    return mx.sum(mx.stack(outputs, axis=0), axis=0)
