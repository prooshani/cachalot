from __future__ import annotations

from collections import defaultdict

import mlx.core as mx

from v41runtime.cache.resident_store import (
    ResidentExpertStore,
)
from v41runtime.io.resident_prefetch import (
    ResidentExpertPrefetcher,
)
from v41runtime.model.expert_metal import (
    routed_expert_forward,
)
from v41runtime.model.router_mlx import (
    RouterResult,
    route_topk_batch,
)
from v41runtime.model.shared_expert_metal import (
    shared_expert_forward,
)
from v41runtime.storage.index import ExpertEntry


def moe_prefill_grouped(
    x: mx.array,
    *,
    layer_id: int,
    gate_weight: mx.array,
    gate_bias: mx.array,
    expert_index: dict[
        tuple[int, int],
        ExpertEntry,
    ],
    expert_store: ResidentExpertStore,
    expert_prefetcher: ResidentExpertPrefetcher | None = None,
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
) -> tuple[
    mx.array,
    RouterResult,
]:
    """
    Expert-major MoE execution for prompt prefill.

    This is intentionally separate from moe_layer_forward().

    Current implementation keeps the already-validated single-vector
    FP4 GEMV kernel, but changes scheduling from:

        token -> top-k experts

    to:

        route all tokens
        -> expert -> all routed tokens

    This means a routed expert is acquired from ResidentExpertStore
    once for all tokens using that expert in the current layer.

    x:
        [tokens, hidden]

    Returns:
        output:
            [tokens, hidden]

        route:
            batched RouterResult
    """
    if x.ndim != 2:
        raise ValueError(
            f"x must be 2D [tokens, hidden], got {x.shape}"
        )

    n_tokens = x.shape[0]

    if n_tokens == 0:
        raise ValueError(
            "x must contain at least one token"
        )

    route = route_topk_batch(
        x,
        gate_weight,
        gate_bias,
        topk=topk,
        gate_temp=gate_temp,
        route_scale=route_scale,
        norm_topk_prob=norm_topk_prob,
    )

    # Routing decisions are needed on the CPU for expert-store
    # addressing and expert-major scheduling.
    mx.eval(
        route.indices,
        route.weights,
    )

    expert_ids = route.indices.tolist()
    router_weights = route.weights.tolist()

    # expert_id -> [(token_index, route_slot, router_weight), ...]
    #
    # Expert execution remains expert-major so each resident expert
    # is fetched once for the whole prompt chunk.
    #
    # route_slot preserves the exact per-token accumulation order
    # used by moe_layer_forward().
    assignments: dict[
        int,
        list[tuple[int, int, float]],
    ] = defaultdict(list)

    for token_index in range(n_tokens):
        for route_slot, (
            expert_id,
            router_weight,
        ) in enumerate(
            zip(
                expert_ids[token_index],
                router_weights[token_index],
            )
        ):
            assignments[
                int(expert_id)
            ].append(
                (
                    token_index,
                    route_slot,
                    float(router_weight),
                )
            )

    # Store each routed-expert result in its original top-k slot.
    #
    # Do NOT accumulate here: expert-major traversal order differs
    # from decode order and can introduce floating-point rounding
    # differences even when routing is identical.
    routed_by_slot: list[
        list[mx.array | None]
    ] = [
        [
            None
            for _ in range(topk)
        ]
        for _ in range(n_tokens)
    ]

    # ------------------------------------------------------------
    # Expert-major execution.
    #
    # Preserve the existing deterministic expert traversal order,
    # but overlap upcoming SSD reads/promotions with computation on
    # the current resident expert.
    #
    # The prefetcher only changes WHEN an expert becomes resident.
    # It does not change:
    #   - routing
    #   - expert execution order
    #   - token execution order within each expert
    #   - per-token top-k accumulation order
    # ------------------------------------------------------------
    expert_work = []

    for expert_id, token_assignments in (
        assignments.items()
    ):
        key = (
            layer_id,
            expert_id,
        )

        entry = expert_index.get(
            key
        )

        if entry is None:
            raise KeyError(
                "Missing routed expert "
                f"layer={layer_id}, "
                f"expert={expert_id}"
            )

        expert_work.append(
            (
                entry,
                token_assignments,
            )
        )

    # ------------------------------------------------------------
    # Decide the layer's cache working set BEFORE asynchronous
    # prefetch begins.
    #
    # This keeps cache membership deterministic even though SSD
    # reads/promotions may complete in a different worker order.
    # ------------------------------------------------------------
    if (
        expert_prefetcher is not None
        and expert_work
    ):
        expert_store.prepare_prefill_layer(
            layer_id,
            [
                entry
                for entry, _ in expert_work
            ],
            num_layers=40,
        )

    prefetch_depth = (
        expert_prefetcher.workers
        if expert_prefetcher is not None
        else 0
    )

    if (
        expert_prefetcher is not None
        and expert_work
    ):
        # Seed the full configured prefetch pipeline.
        for prefetch_index in range(
            min(
                prefetch_depth,
                len(expert_work),
            )
        ):
            expert_prefetcher.prefetch(
                expert_work[
                    prefetch_index
                ][0]
            )

    for work_index, (
        entry,
        token_assignments,
    ) in enumerate(
        expert_work
    ):
        if expert_prefetcher is None:
            expert = expert_store.get(
                entry
            )
        else:
            expert = expert_prefetcher.get(
                entry
            )

            # Maintain the configured prefetch depth.
            next_prefetch_index = (
                work_index
                + prefetch_depth
            )

            if (
                next_prefetch_index
                < len(expert_work)
            ):
                expert_prefetcher.prefetch(
                    expert_work[
                        next_prefetch_index
                    ][0]
                )

        model = expert.as_model_dict()

        for (
            token_index,
            route_slot,
            router_weight,
        ) in token_assignments:
            y = routed_expert_forward(
                x[token_index],
                w1_packed=model[
                    "w1.weight"
                ],
                w1_scales=model[
                    "w1.scale"
                ],
                w2_packed=model[
                    "w2.weight"
                ],
                w2_scales=model[
                    "w2.scale"
                ],
                w3_packed=model[
                    "w3.weight"
                ],
                w3_scales=model[
                    "w3.scale"
                ],
                weight=router_weight,
                swiglu_limit=(
                    swiglu_limit
                ),
            )

            routed_by_slot[
                token_index
            ][
                route_slot
            ] = y.astype(
                mx.float32
            )

    # The shared FP8 expert is still single-token today.
    #
    # Keep it on the validated path for this first locality
    # experiment. We can batch it separately after measuring
    # the routed-expert scheduling improvement.
    outputs = []

    for token_index in range(
        n_tokens
    ):
        # Reproduce moe_layer_forward() accumulation exactly:
        #
        #   zero
        #   + route slot 0
        #   + route slot 1
        #   ...
        #
        # Expert computation happened expert-major above, but
        # floating-point addition happens in decode order here.
        routed = mx.zeros(
            (x.shape[1],),
            dtype=mx.float32,
        )

        for route_slot in range(
            topk
        ):
            contribution = (
                routed_by_slot[
                    token_index
                ][
                    route_slot
                ]
            )

            if contribution is None:
                raise RuntimeError(
                    "Missing routed contribution "
                    f"token={token_index}, "
                    f"route_slot={route_slot}"
                )

            routed = (
                routed
                + contribution
            )

        shared = shared_expert_forward(
            x[token_index],
            w1=shared_w1,
            w1_scales=shared_w1_scales,
            w2=shared_w2,
            w2_scales=shared_w2_scales,
            w3=shared_w3,
            w3_scales=shared_w3_scales,
            swiglu_limit=(
                swiglu_limit
            ),
        )

        output = (
            routed
            + shared.astype(
                mx.float32
            )
        ).astype(
            x.dtype
        )

        outputs.append(
            output
        )

    output = mx.stack(
        outputs,
        axis=0,
    )

    return (
        output,
        route,
    )
