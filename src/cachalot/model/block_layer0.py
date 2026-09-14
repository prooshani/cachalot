from __future__ import annotations

import mlx.core as mx

from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.model.attention_layer0 import layer0_attention_decode
from cachalot.model.hyper_connection_mlx import (
    hc_mixes,
    hc_post,
    hc_pre,
)
from cachalot.model.moe_layer_metal import moe_layer_forward
from cachalot.model.norm_rope_mlx import rms_norm
from cachalot.model.router_mlx import RouterResult
from cachalot.storage.index import ExpertEntry


def layer0_block_decode(
    x: mx.array,
    *,
    start_pos: int,
    pre_mix: mx.array,
    window_cache: mx.array,

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

    # MoE
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
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    RouterResult,
]:
    """
    Complete DeepSeek V4.1 Flash backbone block 0 for one decode token.

    x:
        [hc_mult, hidden_size] BF16 residual stream.

    pre_mix:
        [hc_mult] FP32.
        For the very first block this is [1, 0, 0, 0].

    Returns:
        x:
            [hc_mult, hidden_size] BF16 block output.

        ffn_pre:
            [hc_mult] FP32, consumed as the next block's
            attention pre_mix.

        updated_window_cache:
            [window_size, head_dim] BF16.

        route:
            RouterResult for the block's MoE.
    """
    if x.ndim != 2:
        raise ValueError(
            f"x must have shape [hc, dim], got {x.shape}"
        )

    if x.shape[0] != hc_mult:
        raise ValueError(
            f"x hc dimension {x.shape[0]} != {hc_mult}"
        )

    if pre_mix.shape != (hc_mult,):
        raise ValueError(
            f"pre_mix shape {pre_mix.shape} != ({hc_mult},)"
        )

    # ==================================================
    # Attention HC coefficients
    #
    # Important official semantic:
    # attn_pre is produced now, but attention itself uses
    # the pre_mix received from the previous block.
    # ==================================================

    residual = x

    attn_pre, attn_post, attn_comb = hc_mixes(
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

    attn_output, updated_window_cache = (
        layer0_attention_decode(
            attn_input,
            start_pos=start_pos,
            window_cache=window_cache,
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
            window_size=window_size,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            n_groups=n_groups,
            o_lora_rank=o_lora_rank,
            norm_eps=norm_eps,
        )
    )

    x = hc_post(
        attn_output,
        residual,
        attn_post,
        attn_comb,
    )

    # ==================================================
    # FFN / MoE HC coefficients
    #
    # The attention-produced attn_pre becomes the FFN's
    # collapse coefficients.
    # ==================================================

    residual = x

    ffn_pre, ffn_post, ffn_comb = hc_mixes(
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

    ffn_output, route = moe_layer_forward(
        ffn_input,
        layer_id=0,
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        expert_index=expert_index,
        expert_store=expert_store,
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
        updated_window_cache,
        route,
    )
