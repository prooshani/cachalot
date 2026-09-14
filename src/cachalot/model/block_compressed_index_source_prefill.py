from __future__ import annotations

import mlx.core as mx

from cachalot.cache.resident_store import (
    ResidentExpertStore,
)
from cachalot.io.resident_prefetch import (
    ResidentExpertPrefetcher,
)
from cachalot.model.attention_compressed import (
    compressed_attention_decode_index_source,
)
from cachalot.model.hc_prefill_exact import (
    hc_mixes_prefill_exact,
)
from cachalot.model.hyper_connection_mlx import (
    hc_post,
    hc_pre,
)
from cachalot.model.indexer_mlx import (
    IndexerDecodeResult,
)
from cachalot.model.moe_prefill_grouped import (
    moe_prefill_grouped,
)
from cachalot.model.norm_rope_mlx import rms_norm
from cachalot.model.router_mlx import RouterResult
from cachalot.model.shared_attention import (
    SharedAttentionRuntime,
)
from cachalot.storage.index import ExpertEntry


def compressed_index_source_block_prefill(
    layer_id: int,
    x: mx.array,
    *,
    start_pos: int,
    pre_mix: mx.array,

    # Attention runtime state
    compress_ratio: int,
    window_cache: mx.array,
    shared_attn: SharedAttentionRuntime,

    # Position-dependent publication from layer 20.
    shared_candidates_by_token: tuple[
        mx.array,
        ...,
    ],

    # Hyper-connection
    hc_attn_fn: mx.array,
    hc_attn_scale: mx.array,
    hc_attn_base: mx.array,
    hc_ffn_fn: mx.array,
    hc_ffn_scale: mx.array,
    hc_ffn_base: mx.array,

    # Sublayer norms
    attn_norm_weight: mx.array,
    ffn_norm_weight: mx.array,

    # Attention
    rope_cos: mx.array,
    rope_sin: mx.array,
    attn_sink: mx.array,
    q_norm_weight: mx.array,
    kv_norm_weight: mx.array,
    wq_a: mx.array,
    wq_a_scales: mx.array,
    wq_b: mx.array,
    wq_b_scales: mx.array,
    wkv: mx.array,
    wkv_scales: mx.array,
    wo_a_bf16: mx.array,
    wo_b: mx.array,
    wo_b_scales: mx.array,

    # Index-only source
    indexer_weights_proj_weight: mx.array,
    indexer_wq_b_weight: mx.array,
    indexer_wq_b_scales: mx.array,

    # MoE
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

    # Architecture
    hc_mult: int = 4,
    hc_sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-20,
    window_size: int = 128,
    n_heads: int = 64,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    n_groups: int = 8,
    o_lora_rank: int = 1024,
    index_topk: int = 512,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    RouterResult,
    tuple[IndexerDecodeResult, ...],
]:
    """
    Layer-major prompt prefill for an INDEX-ONLY source layer.

    Released V4.1 Flash layers using this topology:

        24, 28, 32, 36

    The layer reuses the state originally published by the
    layer-20 candidate/KV source:

        compress_kv
        index_k
        candidates

    Important layer-major replay rule
    ---------------------------------
    The final layer-20 compress_kv and index_k caches may be reused
    for the whole prompt because this layer only reads visible
    prefixes:

        compress_kv[:compress_len]
        index_k[:compress_len]

    The candidate mask is different: it is a per-token publication
    from layer 20. Therefore it must be replayed token-by-token via
    shared_candidates_by_token.

    This layer itself publishes a new topk_idxs for every token.
    Those per-token results are returned so later reuse layers can
    replay the correct publication.

    Only the FFN/MoE phase is reorganized expert-major.
    """
    if x.ndim != 3:
        raise ValueError(
            f"x must be [tokens, hc, dim], got {x.shape}"
        )

    n_tokens = x.shape[0]

    if n_tokens == 0:
        raise ValueError(
            "x must contain at least one token"
        )

    if x.shape[1] != hc_mult:
        raise ValueError(
            f"x hc dimension {x.shape[1]} != {hc_mult}"
        )

    if pre_mix.shape != (
        n_tokens,
        hc_mult,
    ):
        raise ValueError(
            f"pre_mix shape {pre_mix.shape} != "
            f"{(n_tokens, hc_mult)}"
        )

    if len(
        shared_candidates_by_token
    ) != n_tokens:
        raise ValueError(
            "shared_candidates_by_token length "
            f"{len(shared_candidates_by_token)} "
            f"!= token count {n_tokens}"
        )

    if compress_ratio < 1:
        raise ValueError(
            f"compress_ratio must be >= 1, got {compress_ratio}"
        )

    if shared_attn.compress_kv is None:
        raise RuntimeError(
            "shared_attn.compress_kv is None"
        )

    if shared_attn.index_k is None:
        raise RuntimeError(
            "shared_attn.index_k is None"
        )

    if shared_attn.topk_idxs is None:
        raise RuntimeError(
            "shared_attn.topk_idxs is None"
        )

    # ============================================================
    # Attention HC coefficients
    # ============================================================

    residual = x

    (
        attn_pre,
        attn_post,
        attn_comb,
    ) = hc_mixes_prefill_exact(
        x,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        norm_eps=norm_eps,
        hc_mult=hc_mult,
        sinkhorn_iters=hc_sinkhorn_iters,
        hc_eps=hc_eps,
    )

    attn_input = hc_pre(
        x,
        pre_mix,
    )

    attn_input = rms_norm(
        attn_input,
        attn_norm_weight,
        eps=norm_eps,
    )

    # ============================================================
    # Index-only compressed attention
    #
    # Remains token-sequential so window-cache evolution and
    # per-token index publication are decode-exact.
    #
    # Final layer-20 compress_kv/index_k are safe because the
    # underlying attention/indexer functions slice them to the
    # current compress_len.
    #
    # Candidates must be restored separately for every token.
    # ============================================================

    cache = window_cache

    attn_outputs = []
    index_results = []

    final_compress_kv = (
        shared_attn.compress_kv
    )

    final_index_k = (
        shared_attn.index_k
    )

    # The incoming top-k is only required to be present by the
    # decode attention contract. It is replaced with this layer's
    # newly computed top-k before sparse attention is constructed.
    incoming_topk = (
        shared_attn.topk_idxs
    )

    for token_offset in range(
        n_tokens
    ):
        token_shared = (
            SharedAttentionRuntime()
        )

        token_shared.compress_kv = (
            final_compress_kv
        )

        token_shared.index_k = (
            final_index_k
        )

        token_shared.candidates = (
            shared_candidates_by_token[
                token_offset
            ]
        )

        token_shared.topk_idxs = (
            incoming_topk
        )

        (
            output,
            cache,
            index_result,
        ) = (
            compressed_attention_decode_index_source(
                attn_input[
                    token_offset
                ],
                start_pos=(
                    start_pos
                    + token_offset
                ),
                compress_ratio=(
                    compress_ratio
                ),
                window_cache=cache,
                shared_attn=token_shared,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                attn_sink=attn_sink,
                q_norm_weight=(
                    q_norm_weight
                ),
                kv_norm_weight=(
                    kv_norm_weight
                ),
                wq_a=wq_a,
                wq_a_scales=(
                    wq_a_scales
                ),
                wq_b=wq_b,
                wq_b_scales=(
                    wq_b_scales
                ),
                wkv=wkv,
                wkv_scales=(
                    wkv_scales
                ),
                wo_a_bf16=(
                    wo_a_bf16
                ),
                wo_b=wo_b,
                wo_b_scales=(
                    wo_b_scales
                ),
                indexer_weights_proj_weight=(
                    indexer_weights_proj_weight
                ),
                indexer_wq_b_weight=(
                    indexer_wq_b_weight
                ),
                indexer_wq_b_scales=(
                    indexer_wq_b_scales
                ),
                window_size=(
                    window_size
                ),
                n_heads=n_heads,
                head_dim=head_dim,
                rope_head_dim=(
                    rope_head_dim
                ),
                n_groups=n_groups,
                o_lora_rank=(
                    o_lora_rank
                ),
                norm_eps=norm_eps,
                index_topk=index_topk,
            )
        )

        attn_outputs.append(
            output
        )

        index_results.append(
            index_result
        )

    # Publish the final token's new top-k exactly as token-major
    # execution would leave SharedAttentionRuntime after the layer.
    shared_attn.topk_idxs = (
        index_results[-1].topk_idxs
    )

    # The index-only source does NOT replace these publications.
    shared_attn.compress_kv = (
        final_compress_kv
    )

    shared_attn.index_k = (
        final_index_k
    )

    shared_attn.candidates = (
        shared_candidates_by_token[-1]
    )

    attn_output = mx.stack(
        attn_outputs,
        axis=0,
    )

    # ============================================================
    # Attention HC post
    # ============================================================

    x = hc_post(
        attn_output,
        residual,
        attn_post,
        attn_comb,
    )

    # ============================================================
    # FFN HC coefficients
    # ============================================================

    residual = x

    (
        ffn_pre,
        ffn_post,
        ffn_comb,
    ) = hc_mixes_prefill_exact(
        x,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        norm_eps=norm_eps,
        hc_mult=hc_mult,
        sinkhorn_iters=hc_sinkhorn_iters,
        hc_eps=hc_eps,
    )

    ffn_input = hc_pre(
        x,
        attn_pre,
    )

    ffn_input = rms_norm(
        ffn_input,
        ffn_norm_weight,
        eps=norm_eps,
    )

    # ============================================================
    # Grouped routed + shared MoE
    # ============================================================

    ffn_output, route = (
        moe_prefill_grouped(
            ffn_input,
            layer_id=layer_id,
            gate_weight=gate_weight,
            gate_bias=gate_bias,
            expert_index=expert_index,
            expert_store=expert_store,
            expert_prefetcher=expert_prefetcher,
            shared_w1=shared_w1,
            shared_w1_scales=(
                shared_w1_scales
            ),
            shared_w2=shared_w2,
            shared_w2_scales=(
                shared_w2_scales
            ),
            shared_w3=shared_w3,
            shared_w3_scales=(
                shared_w3_scales
            ),
            topk=6,
            gate_temp=1.0,
            route_scale=1.5,
            norm_topk_prob=True,
            swiglu_limit=10.0,
        )
    )

    x = hc_post(
        ffn_output,
        residual,
        ffn_post,
        ffn_comb,
    )

    return (
        x,
        ffn_pre,
        cache,
        route,
        tuple(index_results),
    )
