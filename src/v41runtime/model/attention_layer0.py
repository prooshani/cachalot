from __future__ import annotations

import mlx.core as mx

from v41runtime.model.fp8_act_mlx import fp8_roundtrip_activation_mlx
from v41runtime.model.fp8_linear_metal import fp8_linear
from v41runtime.model.norm_rope_mlx import apply_rotary_emb, rms_norm
from v41runtime.model.sparse_attn_mlx import (
    get_window_topk_idxs,
    sparse_attention,
)


def layer0_attention_decode(
    x: mx.array,
    *,
    start_pos: int,
    window_cache: mx.array,
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
    window_size: int = 128,
    n_heads: int = 64,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    n_groups: int = 8,
    o_lora_rank: int = 1024,
    norm_eps: float = 1e-20,
) -> tuple[mx.array, mx.array]:
    """
    One-token layer-0 attention matching official DeepSeek V4.1 Flash.

    x:
        [5120] BF16

    window_cache:
        [128, 512] BF16

    Returns:
        output       [5120] BF16
        updated_cache [128,512] BF16
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {x.shape}")

    if window_cache.shape != (window_size, head_dim):
        raise ValueError(
            f"window_cache shape {window_cache.shape} "
            f"!= {(window_size, head_dim)}"
        )

    # --------------------------------------------------
    # Q low-rank path
    # --------------------------------------------------

    qr = fp8_linear(
        x,
        wq_a,
        wq_a_scales,
    )

    qr = rms_norm(
        qr,
        q_norm_weight,
        eps=norm_eps,
    )

    q = fp8_linear(
        qr,
        wq_b,
        wq_b_scales,
    ).reshape(
        (n_heads, head_dim)
    )

    # Current-token RoPE frequencies.
    cos = rope_cos[
        start_pos:start_pos + 1
    ]

    sin = rope_sin[
        start_pos:start_pos + 1
    ]

    # apply_rotary_emb expects sequence dimension.
    q_nope = q[:, :-rope_head_dim]

    q_rope = apply_rotary_emb(
        q[
            None,
            :,
            -rope_head_dim:,
        ],
        cos,
        sin,
    )[0]

    q = mx.concatenate(
        [q_nope, q_rope],
        axis=-1,
    )


    # --------------------------------------------------
    # Sliding-window KV
    # --------------------------------------------------

    kv = fp8_linear(
        x,
        wkv,
        wkv_scales,
    )

    kv = rms_norm(
        kv,
        kv_norm_weight,
        eps=norm_eps,
    )

    kv_nope = kv[
        :-rope_head_dim
    ]

    kv_rope = apply_rotary_emb(
        kv[
            None,
            -rope_head_dim:,
        ],
        cos,
        sin,
    )[0]

    kv = mx.concatenate(
        [kv_nope, kv_rope],
        axis=-1,
    )

    # Official:
    #
    # act_quant(
    #     kv,
    #     block_size=32,
    #     scale_fmt="ue8m0",
    #     scale_dtype=E8M0,
    #     inplace=True,
    # )
    #
    # Therefore the ring stores BF16 FP8-rounded values.
    kv = fp8_roundtrip_activation_mlx(kv)


    # --------------------------------------------------
    # Ring-cache write + attention view
    # --------------------------------------------------

    slot = start_pos % window_size

    updated_cache = mx.concatenate(
        [
            window_cache[:slot],
            kv[None, :],
            window_cache[slot + 1:],
        ],
        axis=0,
    )

    if start_pos == 0:
        # Official prefill behavior for a one-token first call:
        # attention sees only this token, not the full ring.
        attention_kv = kv[None, :]

        topk_idxs = get_window_topk_idxs(
            window_size,
            seqlen=1,
            start_pos=0,
        )

    else:
        attention_kv = updated_cache

        topk_idxs = get_window_topk_idxs(
            window_size,
            seqlen=1,
            start_pos=start_pos,
        )


    # --------------------------------------------------
    # Sparse attention
    # --------------------------------------------------

    o = sparse_attention(
        q[None, :, :],
        attention_kv,
        attn_sink,
        topk_idxs,
        head_dim ** -0.5,
    )[0]

    # Official removes the query rotation from the output.
    o_nope = o[:, :-rope_head_dim]

    o_rope = apply_rotary_emb(
        o[
            None,
            :,
            -rope_head_dim:,
        ],
        cos,
        sin,
        inverse=True,
    )[0]

    o = mx.concatenate(
        [o_nope, o_rope],
        axis=-1,
    )


    # --------------------------------------------------
    # Grouped BF16 wo_a
    #
    # Official converted layout:
    #
    #   wo_a:
    #       [8192,4096]
    #
    # viewed as:
    #       [8,1024,4096]
    #
    # each group receives 8 heads:
    #       8 * 512 = 4096
    # --------------------------------------------------

    group_input_dim = (
        n_heads * head_dim
    ) // n_groups

    grouped_o = o.reshape(
        (
            n_groups,
            group_input_dim,
        )
    )

    grouped_wo_a = wo_a_bf16.reshape(
        (
            n_groups,
            o_lora_rank,
            group_input_dim,
        )
    )

    # One batched grouped matmul is exact and substantially
    # faster than launching one matmul per group:
    #
    #   [G,R,D] @ [G,D,1] -> [G,R,1]
    #
    low_rank_o = mx.matmul(
        grouped_wo_a,
        grouped_o[..., None],
    )[..., 0].reshape(-1)

    # --------------------------------------------------
    # Final FP8 output projection
    # --------------------------------------------------

    output = fp8_linear(
        low_rank_o,
        wo_b,
        wo_b_scales,
    )

    return output, updated_cache
