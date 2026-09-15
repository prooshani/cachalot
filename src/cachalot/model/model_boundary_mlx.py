from __future__ import annotations

import mlx.core as mx

from cachalot.model.bf16_gemv_metal import bf16_gemv_f32
from cachalot.model.hyper_connection_mlx import hc_pre
from cachalot.model.norm_rope_mlx import rms_norm


def embed_token_decode(
    token_id: int,
    embed_weight: mx.array,
    *,
    hc_mult: int = 4,
) -> mx.array:
    """
    Official Transformer input boundary for one decode token.

    Checkpoint:
        embed.weight: [vocab_size, dim] BF16

    Official model:
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(..., hc_mult, ...)

    Decode representation used by this runtime:
        [hc_mult, dim]
    """
    if embed_weight.ndim != 2:
        raise ValueError(
            "embed_weight must have shape [vocab, dim], "
            f"got {embed_weight.shape}"
        )

    vocab_size = embed_weight.shape[0]

    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(
            f"token_id {token_id} outside vocab [0, {vocab_size})"
        )

    row = embed_weight[token_id]

    # Do not create four independent copies in Python.
    # MLX will represent the broadcast lazily until evaluation.
    h = mx.broadcast_to(
        row[None, :],
        (
            hc_mult,
            row.shape[0],
        ),
    )

    return h.astype(
        embed_weight.dtype
    )


def make_identity_pre_mix_decode(
    *,
    hc_mult: int = 4,
) -> mx.array:
    """
    Decode equivalent of official make_identity_pre_mix():

        [1, 0, 0, 0]
    """
    values = [
        1.0,
        *([0.0] * (hc_mult - 1)),
    ]

    return mx.array(
        values,
        dtype=mx.float32,
    )


def collapse_final_hidden_decode(
    x: mx.array,
    pre_mix: mx.array,
    norm_weight: mx.array,
    *,
    norm_eps: float = 1e-20,
) -> mx.array:
    """
    Exact top-level Transformer tail before ParallelHead:

        h = layer.hc_pre(h, pre_mix)
        h = self.norm(h)

    Returns one [dim] hidden vector.
    """
    if x.ndim != 2:
        raise ValueError(
            f"x must have shape [hc, dim], got {x.shape}"
        )

    if pre_mix.shape != (x.shape[0],):
        raise ValueError(
            "pre_mix must have shape "
            f"({x.shape[0]},), got {pre_mix.shape}"
        )

    h = hc_pre(
        x,
        pre_mix,
    )

    h = rms_norm(
        h,
        norm_weight,
        eps=norm_eps,
    )

    return h


def parallel_head_logits_decode(
    hidden: mx.array,
    head_weight: mx.array,
    *,
    chunk_size: int = 4096,
) -> mx.array:
    """
    Memory-bounded equivalent of official ParallelHead:

        logits = F.linear(
            x.float(),
            self.weight,
        )

    The checkpoint stores head.weight in BF16, while the official
    PyTorch module loads it into FP32.

    Instead of permanently materializing the full ~129k x 5120
    matrix as FP32, convert one vocabulary chunk at a time.

    hidden:
        [dim]

    head_weight:
        [vocab_size, dim] BF16

    returns:
        [vocab_size] FP32
    """
    if hidden.ndim != 1:
        raise ValueError(
            f"hidden must have shape [dim], got {hidden.shape}"
        )

    if head_weight.ndim != 2:
        raise ValueError(
            "head_weight must have shape [vocab, dim], "
            f"got {head_weight.shape}"
        )

    if head_weight.shape[1] != hidden.shape[0]:
        raise ValueError(
            "head/input dimension mismatch: "
            f"{head_weight.shape[1]} != {hidden.shape[0]}"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be > 0"
        )

    hidden_f32 = hidden.astype(
        mx.float32
    )

    # Fast path: read the bf16 head directly with fp32 accumulation
    # (exact widening, no per-token conversion of the 1.3 GB matrix).
    if head_weight.dtype == mx.bfloat16 and head_weight.shape[1] % 64 == 0:
        return bf16_gemv_f32(hidden_f32, head_weight)

    vocab_size = head_weight.shape[0]
    pieces: list[mx.array] = []

    for start in range(
        0,
        vocab_size,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            vocab_size,
        )

        weight_f32 = head_weight[
            start:end
        ].astype(
            mx.float32
        )

        logits_chunk = (
            weight_f32
            @ hidden_f32
        )

        pieces.append(
            logits_chunk
        )

    logits = mx.concatenate(
        pieces,
        axis=0,
    )

    return logits.astype(
        mx.float32
    )


def final_logits_decode(
    x: mx.array,
    pre_mix: mx.array,
    norm_weight: mx.array,
    head_weight: mx.array,
    *,
    norm_eps: float = 1e-20,
    head_chunk_size: int = 4096,
) -> tuple[
    mx.array,
    mx.array,
]:
    """
    Full official Transformer tail for one decode position.

    Returns:
        hidden:
            final normalized [dim] vector

        logits:
            FP32 [vocab_size]
    """
    hidden = collapse_final_hidden_decode(
        x,
        pre_mix,
        norm_weight,
        norm_eps=norm_eps,
    )

    logits = parallel_head_logits_decode(
        hidden,
        head_weight,
        chunk_size=head_chunk_size,
    )

    return hidden, logits
