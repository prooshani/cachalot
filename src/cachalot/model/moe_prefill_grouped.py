from __future__ import annotations

import os
from collections import defaultdict
from time import perf_counter

import mlx.core as mx

from cachalot.cache.resident_store import (
    ResidentExpertStore,
)
from cachalot.io.resident_prefetch import (
    ResidentExpertPrefetcher,
)
from cachalot.model.expert_metal import (
    routed_expert_forward,
)
from cachalot.model.fp4_sgmm_metal import (
    routed_expert_forward_sgmm,
)
from cachalot.model.moe_prefill_batched import (
    route_topk_rows,
    routed_expert_forward_batched,
    shared_expert_forward_batched,
)
from cachalot.model.router_mlx import (
    RouterResult,
    route_topk_batch,
)
from cachalot.model.shared_expert_metal import (
    shared_expert_forward,
)
from cachalot.storage.index import ExpertEntry

PREFILL_SGMM = os.environ.get("CACHALOT_PREFILL_SGMM", "1") != "0"
# Speculative loading of the next layer's experts into transient slots while
# the router of that layer is still being computed (the SSD is idle then).
SPECULATIVE_PREFILL = os.environ.get("CACHALOT_SPECULATIVE_PREFILL", "1") != "0"
SPEC_MIN_UTILIZATION = float(os.environ.get("CACHALOT_SPEC_MIN_UTIL", "0.5"))
SPEC_SECONDS_PER_EXPERT = 0.0034      # one 18.8 MB expert at ~5.5 GB/s
SPEC_TRANSIENT_RESERVE = 8            # transient slots left free for the layer's own misses
N_LAYERS = 40
N_ROUTED_EXPERTS = 384


def _speculate_next_layer(layer_id, expert_index, expert_store, expert_prefetcher, utilization):
    """Submit loads for the next layer's most-used non-resident experts, sized to the measured gap."""
    if not SPECULATIVE_PREFILL or expert_prefetcher is None or layer_id + 1 >= N_LAYERS:
        return 0
    if utilization < SPEC_MIN_UTILIZATION:
        return 0
    gap = getattr(expert_prefetcher, "layer_gap_s", 0.3)
    budget = min(
        expert_store.transient_free() - SPEC_TRANSIENT_RESERVE,
        int(gap / SPEC_SECONDS_PER_EXPERT) + expert_prefetcher.workers,
    )
    if budget <= 0:
        return 0
    entries = [
        expert_index[(layer_id + 1, e)]
        for e in range(N_ROUTED_EXPERTS)
        if (layer_id + 1, e) in expert_index
    ]
    submitted = 0
    for entry in expert_store.speculative_candidates(entries, budget):
        if expert_prefetcher.prefetch(entry):
            submitted += 1
    expert_prefetcher.speculated = getattr(expert_prefetcher, "speculated", 0) + submitted
    return submitted
SGMM_MAX_ROWS = int(os.environ.get("CACHALOT_SGMM_MAX_ROWS", "64"))



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
    batched: bool = True,
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

    if batched:
        indices, weights, scores = route_topk_rows(
            x,
            gate_weight,
            gate_bias,
            topk=topk,
            gate_temp=gate_temp,
            route_scale=route_scale,
            norm_topk_prob=norm_topk_prob,
        )
        route = RouterResult(
            indices=indices,
            weights=weights,
            scores=scores,
        )
    else:
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
                strict=True,
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

    # Batched path accumulates per-(token, slot) contributions here and
    # sums the slots in decode order afterwards.
    slot_buffer = mx.zeros(
        (n_tokens, topk, x.shape[1]),
        dtype=mx.float32,
    )

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
        expert_prefetcher.depth
        if expert_prefetcher is not None
        else 0
    )

    # Miss-aware lookahead: keep `prefetch_depth` real loads outstanding.
    # Resident hits do not count, so a warm cache no longer starves the
    # SSD queue (positional lookahead did: 8 positions of hits = 0 loads).
    next_prefetch_index = 0
    outstanding_keys: set[tuple[int, int]] = set()
    max_scan_ahead = max(prefetch_depth * 8, 64)

    def top_up_prefetch(consumed_index: int) -> None:
        nonlocal next_prefetch_index
        while (
            next_prefetch_index < len(expert_work)
            and len(outstanding_keys) < prefetch_depth
            and next_prefetch_index - consumed_index <= max_scan_ahead
        ):
            entry_ahead = expert_work[next_prefetch_index][0]
            next_prefetch_index += 1
            if expert_prefetcher.prefetch(entry_ahead):
                outstanding_keys.add(
                    (entry_ahead.layer, entry_ahead.expert)
                )

    if (
        expert_prefetcher is not None
        and expert_work
    ):
        # cancel speculative loads this layer does not need, and measure
        # the SSD-idle gap since the previous layer's MoE ended
        expert_prefetcher.discard_pending(layer_id, {(e.layer, e.expert) for e, _ in expert_work})
        # Consume in the order the data arrives: resident experts first, then
        # the speculative loads in submission order, then the rest. Each
        # (token, slot) pair receives exactly one contribution, so the expert
        # traversal order does not change any sum.
        pending_rank = {k: i for i, k in enumerate(expert_prefetcher.pending_keys())}

        def arrival_rank(item):
            key = (item[0].layer, item[0].expert)
            if expert_store.is_resident(key):
                return (0, 0)
            if key in pending_rank:
                return (1, pending_rank[key])
            return (2, 0)

        expert_work.sort(key=arrival_rank)
        last_end = getattr(expert_prefetcher, "layer_end_s", None)
        if last_end is not None:
            gap = perf_counter() - last_end
            prev = getattr(expert_prefetcher, "layer_gap_s", gap)
            expert_prefetcher.layer_gap_s = 0.5 * prev + 0.5 * gap
        top_up_prefetch(0)

    # Bypass ("transient") loads hold pool slots until the MLX ops that
    # read them are evaluated. Track consumed transients and their outputs;
    # when the pool runs low, evaluate and hand the slots back.
    consumed_transients: list[tuple[int, int]] = []
    outputs_since_eval: list[mx.array] = []

    # Evaluate the routed outputs in small batches. Without this the whole
    # layer's MoE graph would run at the next layer's route eval while the
    # SSD sits idle; with it the GPU works on expert i while the loader
    # threads stream experts i+1.. (and transient slots return promptly).
    release_batch = 8
    low_water = (
        expert_prefetcher.workers + 2
        if expert_prefetcher is not None
        else 0
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
            if outputs_since_eval and (
                len(outputs_since_eval) >= release_batch
                or expert_store.transient_free() < low_water
            ):
                mx.eval(*outputs_since_eval)
                if consumed_transients:
                    expert_store.release_transients(
                        consumed_transients
                    )
                    consumed_transients = []
                outputs_since_eval = []

            expert = expert_prefetcher.get(
                entry
            )

            outstanding_keys.discard(
                (entry.layer, entry.expert)
            )

            if next_prefetch_index <= work_index:
                next_prefetch_index = work_index + 1

            top_up_prefetch(work_index)

        model = expert.as_model_dict()

        if batched:
            # One dequantize + GEMM pass for every token routed here.
            token_idx = mx.array(
                [t for t, _, _ in token_assignments],
                dtype=mx.int32,
            )
            slot_idx = mx.array(
                [s for _, s, _ in token_assignments],
                dtype=mx.int32,
            )
            w_rows = mx.array(
                [w for _, _, w in token_assignments],
                dtype=mx.float32,
            )

            # simdgroup-matrix FP4 kernels read the expert once; above
            # SGMM_MAX_ROWS the tiled quantized_matmul is as fast or faster.
            fmt = getattr(expert_store, "format", None)
            if fmt is not None and fmt.kind == "affine":
                from cachalot.model.expert_affine import affine_expert_forward_batched

                y = affine_expert_forward_batched(
                    x[token_idx], model, fmt, w_rows, swiglu_limit=swiglu_limit
                )
            else:
                expert_fn = (
                    routed_expert_forward_sgmm
                    if PREFILL_SGMM and len(token_assignments) <= SGMM_MAX_ROWS
                    else routed_expert_forward_batched
                )
                y = expert_fn(
                    x[token_idx],
                    w1_packed=model["w1.weight"],
                    w1_scales=model["w1.scale"],
                    w2_packed=model["w2.weight"],
                    w2_scales=model["w2.scale"],
                    w3_packed=model["w3.weight"],
                    w3_scales=model["w3.scale"],
                    weights=w_rows,
                    swiglu_limit=swiglu_limit,
                )

            slot_buffer = slot_buffer.at[
                token_idx,
                slot_idx,
            ].add(y)

            outputs_since_eval.append(y)
        else:
            if getattr(expert_store, "format", None) is not None and expert_store.format.kind == "affine":
                raise NotImplementedError("affine expert banks require batched prefill")
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

                y = y.astype(
                    mx.float32
                )

                routed_by_slot[
                    token_index
                ][
                    route_slot
                ] = y

                outputs_since_eval.append(y)

        if getattr(expert, "transient", False):
            consumed_transients.append(
                (entry.layer, entry.expert)
            )

    # The shared FP8 expert is still single-token today.
    #
    # Keep it on the validated path for this first locality
    # experiment. We can batch it separately after measuring
    # the routed-expert scheduling improvement.
    if batched:
        # Same accumulation order as decode: slot 0 + slot 1 + ... + shared.
        routed_all = slot_buffer[:, 0]
        for route_slot in range(1, topk):
            routed_all = routed_all + slot_buffer[:, route_slot]

        shared_all = shared_expert_forward_batched(
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
            routed_all + shared_all.astype(mx.float32)
        ).astype(x.dtype)

        if expert_prefetcher is not None:
            _speculate_next_layer(
                layer_id, expert_index, expert_store, expert_prefetcher,
                utilization=len(expert_work) / N_ROUTED_EXPERTS,
            )
            expert_prefetcher.layer_end_s = perf_counter()

        return (
            output,
            route,
        )

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
