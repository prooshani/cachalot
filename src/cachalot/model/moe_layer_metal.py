from __future__ import annotations

import os

import mlx.core as mx

from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.model.expert_metal import routed_expert_forward
from cachalot.model.moe_fused_metal import fused_routed_experts
from cachalot.model.router_fused_metal import route_topk_fused
from cachalot.model.router_mlx import RouterResult, route_topk
from cachalot.model.shared_expert_metal import shared_expert_forward
from cachalot.storage.index import ExpertEntry

ASYNC_MOE = os.environ.get("CACHALOT_ASYNC_MOE", "1") != "0"
# One-layer-early routing prediction: layer L+1's router applied to layer L's
# router input recalls ~73 % of L+1's experts (63 % of its cache misses) at
# top-6; those are loaded while the GPU runs layer L. 0 disables.
# The width that pays depends on how many bytes an expert costs, because every
# predicted expert that is not resident is a read: recall and bytes rise
# together, and the two settle at a saddle.
#
# On the 15.48 MiB 3-bit bank the saddle was at top-3. Decode A/B at a 28 GiB
# budget (512-token prompt, 64 tokens, 2026-09-16): off 2.26-2.34 tok/s; top-2
# 2.43 (86 % precision, 3.2 wasted loads/token); top-3 2.55 (80 %, 7.5 wasted);
# top-4 2.44-2.47 (72 %, 14.6 wasted).
#
# On the 9.49 MiB 2-bit bank it is at top-6, because the floor the extra reads
# raise stays under compute. Decode A/B at a 36 GiB budget, six runs each
# interleaved in both directions (2026-09-17): top-3 190-211 ms/token, median
# 192.5; top-6 180-184, median 182.5, 5.2 % faster and every run of it faster
# than every run of top-3. Wider overshoots: top-8 187, top-12 210.
PREDICT_TOPK = int(os.environ.get("CACHALOT_PREDICT_TOPK", "6"))
PREDICT_AHEAD = int(os.environ.get("CACHALOT_PREDICT_AHEAD", "1"))
N_LAYERS = 40


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
    fused: bool | None = None,
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

    if fused is None:
        fused = os.environ.get("CACHALOT_FUSED_MOE", "1") != "0"

    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D for decode, got {x.shape}"
        )

    from cachalot.model import decode_fused_metal as _dfm

    router = route_topk_fused if (_dfm.FUSED_DECODE and norm_topk_prob) else route_topk
    route = router(
        x,
        gate_weight,
        gate_bias,
        topk=topk,
        gate_temp=gate_temp,
        route_scale=route_scale,
        norm_topk_prob=norm_topk_prob,
    )

    # Predicted routing of the next layer(s), evaluated with this layer's
    # routing in one sync.
    predicted = []
    gates = expert_store.decode_gates if PREDICT_TOPK > 0 else None
    if gates:
        for ahead in range(1, PREDICT_AHEAD + 1):
            nxt = layer_id + ahead
            if nxt in gates:
                w_next, b_next = gates[nxt]
                predicted.append((nxt, route_topk_fused(x, w_next, b_next, topk=PREDICT_TOPK).indices))

    # Routing is needed on the CPU to address the resident store.
    mx.eval(
        route.indices,
        route.weights,
        *[p_idx for _, p_idx in predicted],
    )

    prefetch_entries = []
    for nxt, p_idx in predicted:
        # strongest first (route_topk orders ascending)
        for e in reversed(p_idx.tolist()):
            entry = expert_index.get((nxt, int(e)))
            if entry is not None:
                prefetch_entries.append(entry)

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
        prefetch=prefetch_entries or None,
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

    fmt = getattr(expert_store, "format", None)
    if not experts:
        routed = mx.zeros(x.shape, dtype=mx.float32)
    elif fmt is not None and fmt.kind == "affine":
        # Affine-quantized expert bank (e.g. oQ3e 3-bit): mx.quantized_matmul
        # on the slot views, see expert_affine.
        from cachalot.model.expert_affine import affine_expert_forward

        for expert, router_weight in zip(experts, router_weights, strict=True):
            routed = routed + affine_expert_forward(
                x, expert.as_model_dict(), fmt, float(router_weight), swiglu_limit
            )
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

    if ASYNC_MOE:
        # Dispatch the expert work now so the GPU runs it while the CPU
        # builds the next layer's attention graph (otherwise the GPU idles
        # until the next router eval forces the whole graph).
        mx.async_eval(output)

    return output, route
