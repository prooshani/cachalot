from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from cachalot.model.norm_rope_mlx import rms_norm


@dataclass
class CompressorState:
    """
    Persistent partial-group state for compress_ratio > 1.

    Shapes:
        kv_state:    [max_batch_size, compress_ratio, head_dim]
        score_state: [max_batch_size, compress_ratio, head_dim]

    Matches the official DeepSeek Compressor buffers.
    """

    kv_state: mx.array
    score_state: mx.array

    @classmethod
    def create(
        cls,
        *,
        max_batch_size: int,
        compress_ratio: int,
        head_dim: int,
    ) -> CompressorState:
        if compress_ratio <= 1:
            raise ValueError(
                "CompressorState is only needed for compress_ratio > 1"
            )

        shape = (
            max_batch_size,
            compress_ratio,
            head_dim,
        )

        return cls(
            kv_state=mx.zeros(
                shape,
                dtype=mx.float32,
            ),
            score_state=mx.full(
                shape,
                -mx.inf,
                dtype=mx.float32,
            ),
        )

    def reset(self) -> None:
        """
        Reset state before beginning a new independent sequence.
        """
        self.kv_state = mx.zeros_like(
            self.kv_state
        )

        self.score_state = mx.full_like(
            self.score_state,
            -mx.inf,
        )


def compressor_forward(
    x: mx.array,
    *,
    start_pos: int,
    compress_ratio: int,
    norm_weight: mx.array,
    wkv_weight: mx.array,
    wgate_weight: mx.array | None = None,
    state: CompressorState | None = None,
    eps: float = 1e-20,
) -> mx.array | None:
    """
    DeepSeek V4.1 Flash compressed-KV pooling.

    Mirrors inference/model.py::Compressor.forward.

    Input:
        x:
            [batch, sequence, hidden_dim]

    Returns:
        [batch, compressed_sequence, head_dim]

        or None when compress_ratio > 1 and the current
        partial group has not completed yet.

    Important semantics:

    * ratio == 1:
        BF16/plain projection, no gate, no FP32 pooling.

    * ratio > 1:
        x, wkv and wgate are promoted to FP32.
        Each latent channel gets its own softmax over
        the token/group dimension.

    * The pooled latent is converted back to x.dtype
      BEFORE RMSNorm.

    * Returned latent is pre-RoPE.
    """

    if x.ndim != 3:
        raise ValueError(
            f"x must have shape [batch, sequence, dim], got {x.shape}"
        )

    if compress_ratio < 1:
        raise ValueError(
            f"compress_ratio must be >= 1, got {compress_ratio}"
        )

    bsz, seqlen, _ = x.shape

    if seqlen < 1:
        raise ValueError(
            "compressor_forward requires at least one token"
        )

    input_dtype = x.dtype

    #
    # Ratio 1:
    #
    # Official implementation:
    #
    #     return self.norm(self.wkv(x))
    #
    # No gate and no FP32 promotion.
    #
    if compress_ratio == 1:
        kv = mx.matmul(
            x,
            mx.swapaxes(
                wkv_weight,
                -1,
                -2,
            ),
        )

        # Preserve the checkpoint/input path's dtype explicitly.
        kv = kv.astype(input_dtype)

        return rms_norm(
            kv,
            norm_weight,
            eps,
        )

    #
    # Ratio > 1.
    #
    if wgate_weight is None:
        raise ValueError(
            "wgate_weight is required when compress_ratio > 1"
        )

    if state is None:
        raise ValueError(
            "CompressorState is required when compress_ratio > 1"
        )

    if state.kv_state.shape[1] != compress_ratio:
        raise ValueError(
            "state compress_ratio does not match compressor"
        )

    #
    # Official model promotes these projections to FP32.
    #
    x_f32 = x.astype(mx.float32)

    kv = mx.matmul(
        x_f32,
        mx.swapaxes(
            wkv_weight.astype(mx.float32),
            -1,
            -2,
        ),
    )

    score = mx.matmul(
        x_f32,
        mx.swapaxes(
            wgate_weight.astype(mx.float32),
            -1,
            -2,
        ),
    )

    #
    # Prefill.
    #
    if start_pos == 0:
        should_compress = seqlen >= compress_ratio

        remainder = seqlen % compress_ratio
        cutoff = seqlen - remainder

        #
        # Preserve the incomplete trailing group for later decode.
        #
        if remainder:
            state.kv_state[
                :bsz,
                :remainder,
            ] = kv[
                :,
                cutoff:,
            ]

            state.score_state[
                :bsz,
                :remainder,
            ] = score[
                :,
                cutoff:,
            ]

            kv = kv[
                :,
                :cutoff,
            ]

            score = score[
                :,
                :cutoff,
            ]

        #
        # Equivalent to:
        #
        #   kv = kv.unflatten(1, (-1, ratio))
        #   score = score.unflatten(1, (-1, ratio))
        #   kv = (kv * score.softmax(dim=2)).sum(dim=2)
        #
        if cutoff:
            groups = cutoff // compress_ratio
            head_dim = kv.shape[-1]

            kv = mx.reshape(
                kv,
                (
                    bsz,
                    groups,
                    compress_ratio,
                    head_dim,
                ),
            )

            score = mx.reshape(
                score,
                (
                    bsz,
                    groups,
                    compress_ratio,
                    head_dim,
                ),
            )

            weights = mx.softmax(
                score,
                axis=2,
            )

            kv = mx.sum(
                kv * weights,
                axis=2,
            )

    #
    # Decode.
    #
    else:
        #
        # Official decode path assumes one token per invocation.
        #
        if seqlen != 1:
            raise ValueError(
                "start_pos > 0 compressor decode expects seqlen == 1"
            )

        should_compress = (
            (start_pos + 1) % compress_ratio
            == 0
        )

        slot = start_pos % compress_ratio

        state.kv_state[
            :bsz,
            slot,
        ] = kv[:, 0]

        state.score_state[
            :bsz,
            slot,
        ] = score[:, 0]

        if should_compress:
            weights = mx.softmax(
                state.score_state[:bsz],
                axis=1,
            )

            kv = mx.sum(
                state.kv_state[:bsz]
                * weights,
                axis=1,
                keepdims=True,
            )

    #
    # Incomplete group during either prefill or decode.
    #
    if not should_compress:
        return None

    #
    # Critical official ordering:
    #
    #     return self.norm(kv.to(dtype))
    #
    # i.e. FP32 pooling -> input dtype -> RMSNorm.
    #
    return rms_norm(
        kv.astype(input_dtype),
        norm_weight,
        eps,
    )
