from __future__ import annotations

import mlx.core as mx

from v41runtime.model.compressor_mlx import (
    CompressorState,
    compressor_forward,
)
from v41runtime.model.fp4_act_mlx import (
    fp4_act_roundtrip_mlx,
)
from v41runtime.model.fp8_act_mlx import (
    fp8_roundtrip_activation_mlx,
)
from v41runtime.model.fp8_linear_metal import (
    fp8_linear,
)
from v41runtime.model.indexer_mlx import (
    IndexerDecodeResult,
    IndexerState,
    indexer_decode_base,
    indexer_decode_candidate_consumer,
    indexer_decode_candidate_source,
)
from v41runtime.model.norm_rope_mlx import (
    apply_rotary_emb,
    rms_norm,
)
from v41runtime.model.sparse_attn_mlx import (
    get_window_topk_idxs,
    sparse_attention,
)
from v41runtime.model.shared_attention import (
    SharedAttentionRuntime,
)


def compressed_attention_decode_source(
    x: mx.array,
    *,
    start_pos: int,
    compress_ratio: int,
    window_cache: mx.array,
    compressed_cache: mx.array,
    compressor_state: CompressorState,
    indexer_state: IndexerState,
    shared_attn: SharedAttentionRuntime | None = None,
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
    compressor_norm_weight: mx.array,
    compressor_wkv_weight: mx.array,
    compressor_wgate_weight: mx.array,
    indexer_weights_proj_weight: mx.array,
    indexer_wq_b_weight: mx.array,
    indexer_wq_b_scales: mx.array,
    indexer_wk_weight: mx.array,
    indexer_k_norm_weight: mx.array,
    window_size: int = 128,
    n_heads: int = 64,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    n_groups: int = 8,
    o_lora_rank: int = 1024,
    norm_eps: float = 1e-20,
    index_topk: int = 512,
    candidate_source: bool = False,
    candidate_topk_blocks: int = 2048,
    candidate_block_size: int = 8,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    IndexerDecodeResult,
]:
    """
    One-token compressed-attention decode for a layer that is
    BOTH a KV source and an Indexer source.

    This is the layer-2 / layer-8 / layer-14 / layer-20 source
    topology in released DeepSeek V4.1 Flash.

    Currently intended first for layer 2, where candidate
    pre-filtering does not apply.

    Inputs
    ------
    x:
        [5120] BF16

    window_cache:
        [window_size, head_dim] BF16

    compressed_cache:
        [max_compressed_positions, head_dim] BF16

    compressor_state:
        Persistent partial-group state.

    indexer_state:
        Persistent index-K state.

    Returns
    -------
    output:
        [5120] BF16

    updated_window_cache:
        [window_size, head_dim] BF16

    updated_compressed_cache:
        Same shape as compressed_cache.

    index_result:
        IndexerDecodeResult for validation/debugging.

    Important ordering
    ------------------
    The Indexer sees the compressor latent BEFORE compressed-KV
    RoPE and FP4 quantization, exactly like the official model.

    The compressed attention cache then stores:

        latent
        -> RoPE(last 64)
        -> FP4 activation round-trip
           block_size=16
           E4M3 scale
        -> BF16 cache
    """
    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D, got {x.shape}"
        )

    if compress_ratio <= 0:
        raise ValueError(
            f"compress_ratio must be > 0, got {compress_ratio}"
        )

    if window_cache.shape != (
        window_size,
        head_dim,
    ):
        raise ValueError(
            f"window_cache shape {window_cache.shape} "
            f"!= {(window_size, head_dim)}"
        )

    if (
        compressed_cache.ndim != 2
        or compressed_cache.shape[1] != head_dim
    ):
        raise ValueError(
            "compressed_cache must have shape "
            f"[N,{head_dim}], got {compressed_cache.shape}"
        )

    # ========================================================
    # Q low-rank path
    #
    # qr is shared conceptually between:
    #
    #   attention wq_b
    #   indexer wq_b
    # ========================================================

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
        n_heads,
        head_dim,
    )

    # Ratio > 0 attention uses the compressed-attention RoPE
    # frequencies supplied by the caller.
    cos = rope_cos[
        start_pos:start_pos + 1
    ]

    sin = rope_sin[
        start_pos:start_pos + 1
    ]

    q_nope = q[
        :,
        :-rope_head_dim,
    ]

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
        [
            q_nope,
            q_rope,
        ],
        axis=-1,
    )

    # ========================================================
    # Raw sliding-window KV
    # ========================================================

    window_kv = fp8_linear(
        x,
        wkv,
        wkv_scales,
    )

    window_kv = rms_norm(
        window_kv,
        kv_norm_weight,
        eps=norm_eps,
    )

    window_kv_nope = window_kv[
        :-rope_head_dim
    ]

    window_kv_rope = apply_rotary_emb(
        window_kv[
            None,
            -rope_head_dim:,
        ],
        cos,
        sin,
    )[0]

    window_kv = mx.concatenate(
        [
            window_kv_nope,
            window_kv_rope,
        ],
        axis=-1,
    )

    # Official raw-window cache precision:
    #
    # FP8 activation quant/dequant inplace, E8M0 scales.
    window_kv = fp8_roundtrip_activation_mlx(
        window_kv
    )

    window_slot = (
        start_pos % window_size
    )

    updated_window_cache = mx.concatenate(
        [
            window_cache[
                :window_slot
            ],
            window_kv[
                None,
                :,
            ],
            window_cache[
                window_slot + 1:
            ],
        ],
        axis=0,
    )

    # ========================================================
    # Compressor
    #
    # Returned latent is PRE-RoPE.
    # ========================================================

    latent = compressor_forward(
        x.reshape(
            1,
            1,
            x.shape[0],
        ),
        start_pos=start_pos,
        compress_ratio=compress_ratio,
        norm_weight=compressor_norm_weight,
        wkv_weight=compressor_wkv_weight,
        wgate_weight=compressor_wgate_weight,
        state=compressor_state,
        eps=norm_eps,
    )

    if latent is None:
        latent_1d = None
    else:
        if latent.shape != (
            1,
            1,
            head_dim,
        ):
            raise ValueError(
                f"Unexpected compressor latent shape "
                f"{latent.shape}"
            )

        latent_1d = latent.reshape(
            head_dim
        )

    # ========================================================
    # Indexer
    #
    # IMPORTANT:
    #
    # The Indexer receives the PRE-RoPE compressor latent.
    # It maintains its own projected/rotated/FP4 index-K cache.
    # ========================================================

    compress_len = (
        start_pos + 1
    ) // compress_ratio

    if compress_len == 0:
        # Official _compress_topk_idxs() returns an empty
        # int32 tensor without running the Indexer.
        index_result = IndexerDecodeResult(
            topk_idxs=mx.empty(
                (0,),
                dtype=mx.int32,
            ),
            compress_len=0,
        )
    else:
        if candidate_source:
            index_result = (
                indexer_decode_candidate_source(
                    x=x,
                    qr=qr,
                    latent=latent_1d,
                    start_pos=start_pos,
                    compress_ratio=compress_ratio,
                    state=indexer_state,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    weights_proj_weight=(
                        indexer_weights_proj_weight
                    ),
                    wq_b_weight=indexer_wq_b_weight,
                    wq_b_scales=indexer_wq_b_scales,
                    wk_weight=indexer_wk_weight,
                    k_norm_weight=indexer_k_norm_weight,
                    norm_eps=norm_eps,
                    index_topk=index_topk,
                    candidate_topk_blocks=(
                        candidate_topk_blocks
                    ),
                    candidate_block_size=(
                        candidate_block_size
                    ),
                )
            )
        else:
            index_result = indexer_decode_base(
                x=x,
                qr=qr,
                latent=latent_1d,
                start_pos=start_pos,
                compress_ratio=compress_ratio,
                state=indexer_state,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                weights_proj_weight=(
                    indexer_weights_proj_weight
                ),
                wq_b_weight=indexer_wq_b_weight,
                wq_b_scales=indexer_wq_b_scales,
                wk_weight=indexer_wk_weight,
                k_norm_weight=indexer_k_norm_weight,
                norm_eps=norm_eps,
                index_topk=index_topk,
            )

    # Publish latest index source results.
    #
    # Keep top-k relative here; consumers add their own
    # current window-KV offset when constructing
    # [window | compressed].
    if shared_attn is not None:
        shared_attn.index_k = indexer_state.k_cache
        shared_attn.topk_idxs = index_result.topk_idxs

        if candidate_source:
            shared_attn.candidates = (
                index_result.candidates
            )

    # ========================================================
    # Compressed KV cache
    #
    # Official ordering:
    #
    #   latent
    #   -> RoPE at FIRST token of compression group
    #   -> FP4 act quant inplace
    #      block_size=16
    #      E4M3 scale
    #   -> cache
    # ========================================================

    updated_compressed_cache = (
        compressed_cache
    )

    if latent_1d is not None:
        end_pos = (
            start_pos + 1
        )

        if (
            end_pos
            % compress_ratio
            != 0
        ):
            raise ValueError(
                "Compressor emitted latent at an invalid "
                f"position: start_pos={start_pos}, "
                f"ratio={compress_ratio}"
            )

        compressed_slot = (
            end_pos
            // compress_ratio
            - 1
        )

        if (
            compressed_slot
            >= compressed_cache.shape[0]
        ):
            raise IndexError(
                f"compressed slot {compressed_slot} exceeds "
                f"cache length {compressed_cache.shape[0]}"
            )

        # One compressed latent represents a group whose RoPE
        # position is the FIRST raw-token position in the group.
        compressed_rope_pos = (
            end_pos
            - compress_ratio
        )

        latent_nope = latent_1d[
            :-rope_head_dim
        ]

        latent_rope = apply_rotary_emb(
            latent_1d[
                None,
                -rope_head_dim:,
            ],
            rope_cos[
                compressed_rope_pos:
                compressed_rope_pos + 1
            ],
            rope_sin[
                compressed_rope_pos:
                compressed_rope_pos + 1
            ],
        )[0]

        compressed_kv = mx.concatenate(
            [
                latent_nope,
                latent_rope,
            ],
            axis=-1,
        )

        compressed_kv = (
            fp4_act_roundtrip_mlx(
                compressed_kv,
                block_size=16,
                scale_dtype="e4m3",
            )
        )

        updated_compressed_cache = (
            mx.concatenate(
                [
                    compressed_cache[
                        :compressed_slot
                    ],
                    compressed_kv[
                        None,
                        :,
                    ],
                    compressed_cache[
                        compressed_slot + 1:
                    ],
                ],
                axis=0,
            )
        )

    # Publish the KV source cache only after any current-token
    # compressed write has completed, matching the official
    # source-before-consumer ordering.
    if shared_attn is not None:
        shared_attn.compress_kv = updated_compressed_cache

    # ========================================================
    # Construct sparse-attention KV view + indices
    # ========================================================

    compress_len = (
        index_result.compress_len
    )

    if start_pos == 0:
        # One-token first call:
        #
        # exactly like the validated pure-window path, only the
        # current raw KV is visible. At ratio >= 1 there cannot
        # be an earlier compressed position here.
        attention_kv = window_kv[
            None,
            :,
        ]

        window_idxs = get_window_topk_idxs(
            window_size,
            seqlen=1,
            start_pos=0,
        )

        combined_idxs = window_idxs

    else:
        # Decode ring occupies [0, window_size).
        #
        # Compressed cache is appended after the FULL ring, so
        # compressed relative index i becomes:
        #
        #     window_size + i
        #
        if compress_len > 0:
            attention_kv = mx.concatenate(
                [
                    updated_window_cache,
                    updated_compressed_cache[
                        :compress_len
                    ],
                ],
                axis=0,
            )
        else:
            attention_kv = (
                updated_window_cache
            )

        window_idxs = get_window_topk_idxs(
            window_size,
            seqlen=1,
            start_pos=start_pos,
        )

        if (
            index_result.topk_idxs.shape[0]
            > 0
        ):
            compressed_idxs = (
                index_result.topk_idxs
                + window_size
            ).astype(
                mx.int32
            )

            combined_idxs = mx.concatenate(
                [
                    window_idxs,
                    compressed_idxs[
                        None,
                        :,
                    ],
                ],
                axis=1,
            )
        else:
            combined_idxs = window_idxs

    # ========================================================
    # Sparse attention
    # ========================================================

    o = sparse_attention(
        q[
            None,
            :,
            :,
        ],
        attention_kv,
        attn_sink,
        combined_idxs,
        head_dim ** -0.5,
    )[0]

    # Official removes the query rotation after attention.
    o_nope = o[
        :,
        :-rope_head_dim,
    ]

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
        [
            o_nope,
            o_rope,
        ],
        axis=-1,
    )

    # ========================================================
    # Grouped BF16 wo_a
    # ========================================================

    group_input_dim = (
        n_heads
        * head_dim
    ) // n_groups

    grouped_o = o.reshape(
        n_groups,
        group_input_dim,
    )

    grouped_wo_a = (
        wo_a_bf16.reshape(
            n_groups,
            o_lora_rank,
            group_input_dim,
        )
    )

    low_rank_o = mx.matmul(
        grouped_wo_a,
        grouped_o[
            ...,
            None,
        ],
    )[
        ...,
        0
    ].reshape(
        -1
    )

    # ========================================================
    # Final FP8 output projection
    # ========================================================

    output = fp8_linear(
        low_rank_o,
        wo_b,
        wo_b_scales,
    )

    return (
        output,
        updated_window_cache,
        updated_compressed_cache,
        index_result,
    )


def compressed_attention_decode_reuse(
    x: mx.array,
    *,
    start_pos: int,
    compress_ratio: int,
    window_cache: mx.array,
    shared_attn: SharedAttentionRuntime,
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
) -> tuple[
    mx.array,
    mx.array,
]:
    """
    One-token compressed-attention decode for a layer that
    REUSES compressed KV and compressed top-k published by an
    earlier source layer.

    This is initially the layer-3..7 topology after layer 2.

    It does NOT run:

        Compressor
        Indexer
        compressed-KV writes
        shared-attention publication

    It computes only this layer's own:

        Q
        raw/window KV
        sparse attention
        inverse RoPE
        wo_a
        wo_b

    shared_attn.topk_idxs are stored as RELATIVE compressed
    positions. During decode they are offset by window_size
    because attention KV is:

        [full window ring | compressed KV]
    """
    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D, got {x.shape}"
        )

    if compress_ratio <= 0:
        raise ValueError(
            f"compress_ratio must be > 0, got {compress_ratio}"
        )

    if window_cache.shape != (
        window_size,
        head_dim,
    ):
        raise ValueError(
            f"window_cache shape {window_cache.shape} "
            f"!= {(window_size, head_dim)}"
        )

    if shared_attn.compress_kv is None:
        raise RuntimeError(
            "shared_attn.compress_kv is None; "
            "a KV-source layer must run before this consumer"
        )

    if shared_attn.topk_idxs is None:
        raise RuntimeError(
            "shared_attn.topk_idxs is None; "
            "an index-source layer must run before this consumer"
        )

    compressed_cache = shared_attn.compress_kv

    if (
        compressed_cache.ndim != 2
        or compressed_cache.shape[1] != head_dim
    ):
        raise ValueError(
            "shared compressed KV must have shape "
            f"[N,{head_dim}], got {compressed_cache.shape}"
        )

    # ========================================================
    # Q low-rank path
    # ========================================================

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
        n_heads,
        head_dim,
    )

    cos = rope_cos[
        start_pos:start_pos + 1
    ]

    sin = rope_sin[
        start_pos:start_pos + 1
    ]

    q_nope = q[
        :,
        :-rope_head_dim,
    ]

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
        [
            q_nope,
            q_rope,
        ],
        axis=-1,
    )

    # ========================================================
    # Raw sliding-window KV
    # ========================================================

    window_kv = fp8_linear(
        x,
        wkv,
        wkv_scales,
    )

    window_kv = rms_norm(
        window_kv,
        kv_norm_weight,
        eps=norm_eps,
    )

    window_kv_nope = window_kv[
        :-rope_head_dim
    ]

    window_kv_rope = apply_rotary_emb(
        window_kv[
            None,
            -rope_head_dim:,
        ],
        cos,
        sin,
    )[0]

    window_kv = mx.concatenate(
        [
            window_kv_nope,
            window_kv_rope,
        ],
        axis=-1,
    )

    window_kv = fp8_roundtrip_activation_mlx(
        window_kv
    )

    window_slot = (
        start_pos % window_size
    )

    updated_window_cache = mx.concatenate(
        [
            window_cache[
                :window_slot
            ],
            window_kv[
                None,
                :,
            ],
            window_cache[
                window_slot + 1:
            ],
        ],
        axis=0,
    )

    # ========================================================
    # Reuse latest published compressed state
    # ========================================================

    compress_len = (
        start_pos + 1
    ) // compress_ratio

    if compress_len > compressed_cache.shape[0]:
        raise IndexError(
            f"compress_len {compress_len} exceeds shared "
            f"compressed cache length {compressed_cache.shape[0]}"
        )

    relative_compressed_idxs = (
        shared_attn.topk_idxs
    )

    # ========================================================
    # Construct sparse-attention KV view + indices
    # ========================================================

    if start_pos == 0:
        # First decode token sees only its raw KV.
        attention_kv = window_kv[
            None,
            :,
        ]

        combined_idxs = get_window_topk_idxs(
            window_size,
            seqlen=1,
            start_pos=0,
        )

    else:
        if compress_len > 0:
            attention_kv = mx.concatenate(
                [
                    updated_window_cache,
                    compressed_cache[
                        :compress_len
                    ],
                ],
                axis=0,
            )
        else:
            attention_kv = (
                updated_window_cache
            )

        window_idxs = get_window_topk_idxs(
            window_size,
            seqlen=1,
            start_pos=start_pos,
        )

        if (
            relative_compressed_idxs.shape[0]
            > 0
        ):
            compressed_idxs = (
                relative_compressed_idxs
                + window_size
            ).astype(
                mx.int32
            )

            combined_idxs = mx.concatenate(
                [
                    window_idxs,
                    compressed_idxs[
                        None,
                        :,
                    ],
                ],
                axis=1,
            )
        else:
            combined_idxs = window_idxs

    # ========================================================
    # Sparse attention
    # ========================================================

    o = sparse_attention(
        q[
            None,
            :,
            :,
        ],
        attention_kv,
        attn_sink,
        combined_idxs,
        head_dim ** -0.5,
    )[0]

    # Official removes the query rotation after attention.
    o_nope = o[
        :,
        :-rope_head_dim,
    ]

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
        [
            o_nope,
            o_rope,
        ],
        axis=-1,
    )

    # ========================================================
    # Grouped BF16 wo_a
    # ========================================================

    group_input_dim = (
        n_heads
        * head_dim
    ) // n_groups

    grouped_o = o.reshape(
        n_groups,
        group_input_dim,
    )

    grouped_wo_a = (
        wo_a_bf16.reshape(
            n_groups,
            o_lora_rank,
            group_input_dim,
        )
    )

    low_rank_o = mx.matmul(
        grouped_wo_a,
        grouped_o[
            ...,
            None,
        ],
    )[
        ...,
        0
    ].reshape(
        -1
    )

    # ========================================================
    # Final FP8 output projection
    # ========================================================

    output = fp8_linear(
        low_rank_o,
        wo_b,
        wo_b_scales,
    )

    return (
        output,
        updated_window_cache,
    )


def compressed_attention_decode_index_source(
    x: mx.array,
    *,
    start_pos: int,
    compress_ratio: int,
    window_cache: mx.array,
    shared_attn: SharedAttentionRuntime,
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

    # Index-only source projection tensors
    indexer_weights_proj_weight: mx.array,
    indexer_wq_b_weight: mx.array,
    indexer_wq_b_scales: mx.array,

    window_size: int = 128,
    n_heads: int = 64,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    n_groups: int = 8,
    o_lora_rank: int = 1024,
    norm_eps: float = 1e-20,
    index_topk: int = 512,
) -> tuple[
    mx.array,
    mx.array,
    IndexerDecodeResult,
]:
    """
    One-token compressed-attention decode for an INDEX-ONLY
    source layer.

    Released V4.1 Flash topology:

        layer 20:
            publishes compress_kv
            publishes index_k
            publishes candidates
            publishes topk_idxs

        layer 24:
            reuses layer-20 compress_kv
            reuses layer-20 index_k
            reuses layer-20 candidates
            computes a NEW candidate-filtered topk_idxs

    This layer does not own a Compressor, compressed-KV cache,
    or index-K cache.

    This is initially the layer-3..7 topology after layer 2.

    It does NOT run:

        Compressor
        Indexer
        compressed-KV writes
        shared-attention publication

    It computes only this layer's own:

        Q
        raw/window KV
        sparse attention
        inverse RoPE
        wo_a
        wo_b

    shared_attn.topk_idxs are stored as RELATIVE compressed
    positions. During decode they are offset by window_size
    because attention KV is:

        [full window ring | compressed KV]
    """
    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D, got {x.shape}"
        )

    if compress_ratio <= 0:
        raise ValueError(
            f"compress_ratio must be > 0, got {compress_ratio}"
        )

    if window_cache.shape != (
        window_size,
        head_dim,
    ):
        raise ValueError(
            f"window_cache shape {window_cache.shape} "
            f"!= {(window_size, head_dim)}"
        )

    if shared_attn.compress_kv is None:
        raise RuntimeError(
            "shared_attn.compress_kv is None; "
            "a KV-source layer must run before this consumer"
        )

    if shared_attn.topk_idxs is None:
        raise RuntimeError(
            "shared_attn.topk_idxs is None; "
            "an index-source layer must run before this consumer"
        )

    if shared_attn.index_k is None:
        raise RuntimeError(
            "shared_attn.index_k is None; "
            "layer 20 must publish index K before "
            "an index-only source layer"
        )

    if shared_attn.candidates is None:
        raise RuntimeError(
            "shared_attn.candidates is None; "
            "layer 20 must publish candidate blocks before "
            "an index-only source layer"
        )

    compressed_cache = shared_attn.compress_kv

    if (
        compressed_cache.ndim != 2
        or compressed_cache.shape[1] != head_dim
    ):
        raise ValueError(
            "shared compressed KV must have shape "
            f"[N,{head_dim}], got {compressed_cache.shape}"
        )

    # ========================================================
    # Q low-rank path
    # ========================================================

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
        n_heads,
        head_dim,
    )

    cos = rope_cos[
        start_pos:start_pos + 1
    ]

    sin = rope_sin[
        start_pos:start_pos + 1
    ]

    q_nope = q[
        :,
        :-rope_head_dim,
    ]

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
        [
            q_nope,
            q_rope,
        ],
        axis=-1,
    )

    # ========================================================
    # Raw sliding-window KV
    # ========================================================

    window_kv = fp8_linear(
        x,
        wkv,
        wkv_scales,
    )

    window_kv = rms_norm(
        window_kv,
        kv_norm_weight,
        eps=norm_eps,
    )

    window_kv_nope = window_kv[
        :-rope_head_dim
    ]

    window_kv_rope = apply_rotary_emb(
        window_kv[
            None,
            -rope_head_dim:,
        ],
        cos,
        sin,
    )[0]

    window_kv = mx.concatenate(
        [
            window_kv_nope,
            window_kv_rope,
        ],
        axis=-1,
    )

    window_kv = fp8_roundtrip_activation_mlx(
        window_kv
    )

    window_slot = (
        start_pos % window_size
    )

    updated_window_cache = mx.concatenate(
        [
            window_cache[
                :window_slot
            ],
            window_kv[
                None,
                :,
            ],
            window_cache[
                window_slot + 1:
            ],
        ],
        axis=0,
    )

    # ========================================================
    # Index-only source
    #
    # Reuse:
    #   shared_attn.index_k
    #   shared_attn.candidates
    #
    # Publish only:
    #   shared_attn.topk_idxs
    # ========================================================

    index_result = indexer_decode_candidate_consumer(
        x=x,
        qr=qr,
        start_pos=start_pos,
        compress_ratio=compress_ratio,
        index_k=shared_attn.index_k,
        candidates=shared_attn.candidates,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        weights_proj_weight=indexer_weights_proj_weight,
        wq_b_weight=indexer_wq_b_weight,
        wq_b_scales=indexer_wq_b_scales,
        index_topk=index_topk,
    )

    shared_attn.topk_idxs = (
        index_result.topk_idxs
    )

    # ========================================================
    # Reuse latest published compressed state
    # ========================================================

    compress_len = (
        start_pos + 1
    ) // compress_ratio

    if compress_len > compressed_cache.shape[0]:
        raise IndexError(
            f"compress_len {compress_len} exceeds shared "
            f"compressed cache length {compressed_cache.shape[0]}"
        )

    relative_compressed_idxs = (
        index_result.topk_idxs
    )

    # ========================================================
    # Construct sparse-attention KV view + indices
    # ========================================================

    if start_pos == 0:
        # First decode token sees only its raw KV.
        attention_kv = window_kv[
            None,
            :,
        ]

        combined_idxs = get_window_topk_idxs(
            window_size,
            seqlen=1,
            start_pos=0,
        )

    else:
        if compress_len > 0:
            attention_kv = mx.concatenate(
                [
                    updated_window_cache,
                    compressed_cache[
                        :compress_len
                    ],
                ],
                axis=0,
            )
        else:
            attention_kv = (
                updated_window_cache
            )

        window_idxs = get_window_topk_idxs(
            window_size,
            seqlen=1,
            start_pos=start_pos,
        )

        if (
            relative_compressed_idxs.shape[0]
            > 0
        ):
            compressed_idxs = (
                relative_compressed_idxs
                + window_size
            ).astype(
                mx.int32
            )

            combined_idxs = mx.concatenate(
                [
                    window_idxs,
                    compressed_idxs[
                        None,
                        :,
                    ],
                ],
                axis=1,
            )
        else:
            combined_idxs = window_idxs

    # ========================================================
    # Sparse attention
    # ========================================================

    o = sparse_attention(
        q[
            None,
            :,
            :,
        ],
        attention_kv,
        attn_sink,
        combined_idxs,
        head_dim ** -0.5,
    )[0]

    # Official removes the query rotation after attention.
    o_nope = o[
        :,
        :-rope_head_dim,
    ]

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
        [
            o_nope,
            o_rope,
        ],
        axis=-1,
    )

    # ========================================================
    # Grouped BF16 wo_a
    # ========================================================

    group_input_dim = (
        n_heads
        * head_dim
    ) // n_groups

    grouped_o = o.reshape(
        n_groups,
        group_input_dim,
    )

    grouped_wo_a = (
        wo_a_bf16.reshape(
            n_groups,
            o_lora_rank,
            group_input_dim,
        )
    )

    low_rank_o = mx.matmul(
        grouped_wo_a,
        grouped_o[
            ...,
            None,
        ],
    )[
        ...,
        0
    ].reshape(
        -1
    )

    # ========================================================
    # Final FP8 output projection
    # ========================================================

    output = fp8_linear(
        low_rank_o,
        wo_b,
        wo_b_scales,
    )

    return (
        output,
        updated_window_cache,
        index_result,
    )
