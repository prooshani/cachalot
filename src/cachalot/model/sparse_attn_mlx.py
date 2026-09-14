from __future__ import annotations

import mlx.core as mx


def get_window_topk_idxs(
    window_size: int,
    seqlen: int,
    start_pos: int,
) -> mx.array:
    """
    Single-batch equivalent of DeepSeek get_window_topk_idxs().

    Returns:
        int32 [seqlen, topk] for prefill
        int32 [1, window_size] for decode

    -1 marks an empty cache slot.
    """
    if start_pos == 0:
        width = min(
            seqlen,
            window_size,
        )

        rows = []

        for pos in range(seqlen):
            begin = max(
                pos - window_size + 1,
                0,
            )

            row = []

            for offset in range(width):
                idx = begin + offset

                if idx > pos:
                    idx = -1

                row.append(idx)

            rows.append(row)

        return mx.array(
            rows,
            dtype=mx.int32,
        )

    oldest = (
        start_pos % window_size
    ) + 1

    order = (
        list(range(oldest, window_size))
        + list(range(oldest))
    )

    order = [
        idx if idx <= start_pos else -1
        for idx in order
    ]

    return mx.array(
        [order],
        dtype=mx.int32,
    )


def sparse_attention(
    q: mx.array,
    kv: mx.array,
    attn_sink: mx.array,
    topk_idxs: mx.array,
    softmax_scale: float,
) -> mx.array:
    """
    Straightforward MLX reference for DeepSeek sparse_attn.

    q:
        [S, H, D] BF16

    kv:
        [N, D] BF16

    attn_sink:
        [H] FP32

    topk_idxs:
        [S, K] int32; -1 marks invalid positions.

    Returns:
        [S, H, D] BF16

    The sink contributes to the softmax denominator only,
    with no corresponding value vector.
    """
    if q.ndim != 3:
        raise ValueError(
            f"q must have shape [S,H,D], got {q.shape}"
        )

    if kv.ndim != 2:
        raise ValueError(
            f"kv must have shape [N,D], got {kv.shape}"
        )

    seqlen, n_heads, head_dim = q.shape

    if kv.shape[1] != head_dim:
        raise ValueError(
            f"KV dim {kv.shape[1]} != query dim {head_dim}"
        )

    if attn_sink.shape != (n_heads,):
        raise ValueError(
            f"attn_sink shape {attn_sink.shape} "
            f"!= ({n_heads},)"
        )

    if topk_idxs.shape[0] != seqlen:
        raise ValueError(
            f"topk rows {topk_idxs.shape[0]} "
            f"!= query seqlen {seqlen}"
        )

    outputs = []

    for pos in range(seqlen):
        idxs = topk_idxs[pos]

        valid = idxs >= 0

        # Replace -1 with zero for safe gathering.
        safe_idxs = mx.where(
            valid,
            idxs,
            mx.zeros_like(idxs),
        )

        selected_kv = mx.take(
            kv,
            safe_idxs,
            axis=0,
        ).astype(mx.float32)

        q_pos = q[pos].astype(
            mx.float32
        )

        # [H,K]
        scores = (
            mx.matmul(
                q_pos,
                mx.transpose(selected_kv),
            )
            * softmax_scale
        )

        scores = mx.where(
            valid[None, :],
            scores,
            mx.full_like(
                scores,
                -mx.inf,
            ),
        )

        # Equivalent to the TileLang online-softmax result:
        # add sink as an extra logit with value vector = 0.
        sink = attn_sink.astype(
            mx.float32
        )[:, None]

        max_scores = mx.maximum(
            mx.max(
                scores,
                axis=-1,
                keepdims=True,
            ),
            sink,
        )

        weights = mx.exp(
            scores - max_scores
        )

        sink_weight = mx.exp(
            sink - max_scores
        )

        denom = (
            mx.sum(
                weights,
                axis=-1,
                keepdims=True,
            )
            + sink_weight
        )

        # The official TileLang kernel keeps the softmax
        # exponentials/sum in FP32, but casts the exponentials
        # to BF16 before the value GEMM:
        #
        #     T.copy(acc_s, acc_s_cast)   # FP32 -> BF16
        #     T.gemm(acc_s_cast, kv_shared, acc_o)
        #
        # Preserve that precision boundary here.
        value_weights = weights.astype(
            mx.bfloat16
        )

        value_sum = mx.matmul(
            value_weights,
            selected_kv.astype(mx.bfloat16),
        ).astype(mx.float32)

        # [H,D]
        out = value_sum / denom

        outputs.append(
            out.astype(q.dtype)
        )

    return mx.stack(
        outputs,
        axis=0,
    )
