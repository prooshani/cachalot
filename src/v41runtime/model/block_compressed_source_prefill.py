from __future__ import annotations

import mlx.core as mx

from v41runtime.cache.resident_store import ResidentExpertStore
from v41runtime.io.resident_prefetch import (
    ResidentExpertPrefetcher,
)
from v41runtime.model.attention_compressed import (
    compressed_attention_decode_source,
)
from v41runtime.model.compressor_mlx import CompressorState
from v41runtime.model.hc_prefill_exact import (
    hc_mixes_prefill_exact,
)
from v41runtime.model.hyper_connection_mlx import (
    hc_post,
    hc_pre,
)
from v41runtime.model.indexer_mlx import (
    IndexerDecodeResult,
    IndexerState,
)
from v41runtime.model.moe_prefill_grouped import (
    moe_prefill_grouped,
)
from v41runtime.model.norm_rope_mlx import rms_norm
from v41runtime.model.router_mlx import RouterResult
from v41runtime.model.shared_attention import (
    SharedAttentionRuntime,
)
from v41runtime.storage.index import ExpertEntry


def compressed_source_block_prefill(
    layer_id: int,
    x: mx.array,
    *,
    start_pos: int,
    pre_mix: mx.array,

    # Attention runtime state
    compress_ratio: int,
    window_cache: mx.array,
    compressed_cache: mx.array,
    compressor_state: CompressorState | None,
    indexer_state: IndexerState,
    shared_attn: SharedAttentionRuntime | None,

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

    # Compressor
    compressor_norm_weight: mx.array,
    compressor_wkv_weight: mx.array,
    compressor_wgate_weight: mx.array | None,

    # Indexer
    indexer_weights_proj_weight: mx.array,
    indexer_wq_b_weight: mx.array,
    indexer_wq_b_scales: mx.array,
    indexer_wk_weight: mx.array,
    indexer_k_norm_weight: mx.array,

    # MoE
    gate_weight: mx.array,
    gate_bias: mx.array,
    expert_index: dict[tuple[int, int], ExpertEntry],
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
    mx.array,
    RouterResult,
    tuple[IndexerDecodeResult, ...],
]:
    """
    Layer-major prompt prefill for a compressed-KV +
    Indexer source layer.

    Initially targets layer 2, but preserves the source-layer
    topology required by layers 2, 8, 14 and 20.

    Input shapes
    ------------
    x:
        [tokens, hc_mult, hidden]

    pre_mix:
        [tokens, hc_mult]

    Correctness strategy
    --------------------
    Hyper-connection coefficient generation uses the exact
    tokenwise 1D HC path already validated against decode.

    Compressed attention is replayed sequentially across prompt
    positions. This preserves exactly:

        raw window-cache evolution
        CompressorState evolution
        IndexerState evolution
        compressed-KV writes
        shared-attention publication
        sparse-attention visibility

    Only the FFN/MoE phase is reorganized expert-major across
    the prompt.
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

    if compress_ratio < 1:
        raise ValueError(
            f"compress_ratio must be >= 1, got {compress_ratio}"
        )

    # ============================================================
    # Attention HC coefficients
    #
    # Use the decode-exact tokenwise HC implementation.
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
    # Compressed attention
    #
    # This remains token-sequential intentionally.
    #
    # The source attention contains state whose precise ordering
    # matters:
    #
    #   raw window KV
    #   -> compressor
    #   -> indexer
    #   -> index publication
    #   -> compressed-KV write
    #   -> compressed-KV publication
    #   -> sparse attention
    #
    # Reusing the existing one-token implementation keeps all of
    # those semantics identical to decode.
    # ============================================================

    cache = window_cache
    compressed = compressed_cache

    attn_outputs = []
    index_results = []

    for token_offset in range(
        n_tokens
    ):
        (
            output,
            cache,
            compressed,
            index_result,
        ) = compressed_attention_decode_source(
            attn_input[token_offset],
            start_pos=(
                start_pos
                + token_offset
            ),
            compress_ratio=compress_ratio,
            window_cache=cache,
            compressed_cache=compressed,
            compressor_state=compressor_state,
            indexer_state=indexer_state,
            shared_attn=shared_attn,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attn_sink=attn_sink,
            q_norm_weight=q_norm_weight,
            kv_norm_weight=kv_norm_weight,
            wq_a=wq_a,
            wq_a_scales=wq_a_scales,
            wq_b=wq_b,
            wq_b_scales=wq_b_scales,
            wkv=wkv,
            wkv_scales=wkv_scales,
            wo_a_bf16=wo_a_bf16,
            wo_b=wo_b,
            wo_b_scales=wo_b_scales,
            compressor_norm_weight=(
                compressor_norm_weight
            ),
            compressor_wkv_weight=(
                compressor_wkv_weight
            ),
            compressor_wgate_weight=(
                compressor_wgate_weight
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
            indexer_wk_weight=(
                indexer_wk_weight
            ),
            indexer_k_norm_weight=(
                indexer_k_norm_weight
            ),
            window_size=window_size,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            n_groups=n_groups,
            o_lora_rank=o_lora_rank,
            norm_eps=norm_eps,
            index_topk=index_topk,
            candidate_source=(
                layer_id == 20
            ),
        )

        attn_outputs.append(
            output
        )

        index_results.append(
            index_result
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
    #
    # Again use the exact tokenwise coefficient path.
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
    # Routed + shared MoE
    #
    # This is the actual layer-major optimization:
    #
    # route all prompt tokens
    # -> group assignments by expert
    # -> load each required expert once for the layer/chunk
    # -> run all tokens assigned to it
    # ============================================================

    ffn_output, route = moe_prefill_grouped(
        ffn_input,
        layer_id=layer_id,
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        expert_index=expert_index,
        expert_store=expert_store,
        expert_prefetcher=expert_prefetcher,
        shared_w1=shared_w1,
        shared_w1_scales=shared_w1_scales,
        shared_w2=shared_w2,
        shared_w2_scales=shared_w2_scales,
        shared_w3=shared_w3,
        shared_w3_scales=shared_w3_scales,
        topk=6,
        gate_temp=1.0,
        route_scale=1.5,
        norm_topk_prob=True,
        swiglu_limit=10.0,
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
        compressed,
        route,
        tuple(index_results),
    )
