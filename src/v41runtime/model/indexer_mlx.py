from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from v41runtime.model.fp4_act_mlx import (
    fp4_act_roundtrip_mlx,
)
from v41runtime.model.fp8_linear_metal import (
    fp8_linear,
)
from v41runtime.model.norm_rope_mlx import (
    apply_rotary_emb,
    rms_norm,
)


INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_ROPE_DIM = 64
INDEX_TOPK = 512
FP4_BLOCK_SIZE = 32


@dataclass
class IndexerState:
    """
    Decode-time compressed-key cache.

    Cache indices are compressed-position indices:

        raw positions 0,1   -> compressed position 0
        raw positions 2,3   -> compressed position 1
        ...

    for compress_ratio=2.
    """

    k_cache: mx.array

    @classmethod
    def create(
        cls,
        *,
        max_seq_len: int,
        compress_ratio: int,
        index_head_dim: int = INDEX_HEAD_DIM,
    ) -> "IndexerState":
        if compress_ratio <= 0:
            raise ValueError(
                f"compress_ratio must be > 0, got {compress_ratio}"
            )

        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be > 0, got {max_seq_len}"
            )

        cache_len = (
            max_seq_len + compress_ratio - 1
        ) // compress_ratio

        return cls(
            k_cache=mx.zeros(
                (cache_len, index_head_dim),
                dtype=mx.bfloat16,
            )
        )

    def reset(self) -> None:
        self.k_cache[...] = mx.array(
            0.0,
            dtype=self.k_cache.dtype,
        )


@dataclass(frozen=True)
class IndexerDecodeResult:
    """
    topk_idxs are relative compressed-cache positions.

    They are returned in ascending sequence-position order,
    matching the official Indexer after its top-k selection.

    candidates is populated only by the candidate-source
    Indexer (official layer 20). It is a boolean mask over
    visible compressed positions.
    """

    topk_idxs: mx.array
    compress_len: int
    candidates: mx.array | None = None


def _bf16_linear(
    x: mx.array,
    weight: mx.array,
) -> mx.array:
    """
    F.linear(x, weight) for the decode-only 1D case.
    """
    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D, got {x.shape}"
        )

    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2D, got {weight.shape}"
        )

    if weight.shape[1] != x.shape[0]:
        raise ValueError(
            f"weight expects {weight.shape[1]} inputs, "
            f"x has {x.shape[0]}"
        )

    return mx.matmul(
        weight,
        x,
    ).astype(mx.bfloat16)


def _apply_index_rope(
    x: mx.array,
    cos_row: mx.array,
    sin_row: mx.array,
) -> mx.array:
    """
    Rotate only the final 64 dimensions.

    x:
        [..., 128]

    cos_row/sin_row:
        [32] corresponding to 64 rotary dimensions.
    """
    if x.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError(
            f"Expected index head dim {INDEX_HEAD_DIM}, "
            f"got {x.shape[-1]}"
        )

    rope = x[..., -INDEX_ROPE_DIM:]
    passthrough = x[..., :-INDEX_ROPE_DIM]

    # apply_rotary_emb expects a sequence dimension.
    #
    # q comes in as [32, 128], i.e. 32 heads for one token.
    # Convert that to [B=1, S=1, H=32, D=64].
    if x.ndim == 2:
        rope4 = rope.reshape(
            1,
            1,
            x.shape[0],
            INDEX_ROPE_DIM,
        )

        rotated = apply_rotary_emb(
            rope4,
            cos_row.reshape(1, -1),
            sin_row.reshape(1, -1),
        ).reshape(
            x.shape[0],
            INDEX_ROPE_DIM,
        )

    elif x.ndim == 1:
        # One K vector:
        # [128] -> [B=1, S=1, D=64]
        rope3 = rope.reshape(
            1,
            1,
            INDEX_ROPE_DIM,
        )

        rotated = apply_rotary_emb(
            rope3,
            cos_row.reshape(1, -1),
            sin_row.reshape(1, -1),
        ).reshape(
            INDEX_ROPE_DIM,
        )

    else:
        raise ValueError(
            f"Unsupported index RoPE shape {x.shape}"
        )

    return mx.concatenate(
        [
            passthrough,
            rotated,
        ],
        axis=-1,
    )


def _select_topk_relative(
    scores: mx.array,
    topk: int,
) -> mx.array:
    """
    Select the highest-scoring compressed positions, then return
    their positions sorted chronologically.

    The official implementation performs top-k selection first
    and then sorts the selected indices by position.
    """
    if scores.ndim != 1:
        raise ValueError(
            f"scores must be 1D, got {scores.shape}"
        )

    n = scores.shape[0]

    if topk < 0:
        raise ValueError(
            f"topk must be >= 0, got {topk}"
        )

    k = min(
        topk,
        n,
    )

    if k == 0:
        return mx.zeros(
            (0,),
            dtype=mx.int32,
        )

    # Sorting the complete score vector is acceptable here:
    #
    # - decode-only correctness path
    # - index_topk <= 512
    # - compressed sequence is much smaller than model compute
    #
    # We can replace this with a dedicated partial-select kernel
    # after correctness is locked down.
    ranked = mx.argsort(
        scores,
        axis=-1,
    )

    selected = ranked[-k:]

    # Official Indexer restores chronological order after top-k.
    return mx.sort(
        selected.astype(mx.int32),
        axis=-1,
    )



def select_candidate_blocks_decode(
    logits: mx.array,
    *,
    compress_len: int,
    topk_blocks: int,
    block_size: int,
) -> mx.array:
    """
    Decode-only MLX equivalent of the official
    select_candidate_blocks().

    This is level one of DeepSeek V4.1 Flash's two-level
    compressed-position selection.

    Parameters
    ----------
    logits:
        1D index scores for all currently visible compressed
        positions.

    compress_len:
        Number of compressed positions currently visible.

        During one-token decode this is a plain integer and
        should match logits.shape[0].

    topk_blocks:
        Maximum number of candidate blocks to retain.

    block_size:
        Number of compressed positions per candidate block.

    Returns
    -------
    bool mask:
        Shape [compress_len].

        True positions belong to blocks selected by the
        candidate source layer.

    Official semantics preserved here:

      1. Block score = maximum position score in the block.
      2. The partially filled newest block is forcibly retained
         by assigning its block score +inf.
      3. Select top-k blocks by block score.
      4. Blocks whose score is still -inf are discarded.
      5. Expand the selected block mask back to position space.
    """
    if logits.ndim != 1:
        raise ValueError(
            f"logits must be 1D, got {logits.shape}"
        )

    if compress_len < 0:
        raise ValueError(
            f"compress_len must be >= 0, got {compress_len}"
        )

    if topk_blocks < 0:
        raise ValueError(
            f"topk_blocks must be >= 0, got {topk_blocks}"
        )

    if block_size <= 0:
        raise ValueError(
            f"block_size must be > 0, got {block_size}"
        )

    width = logits.shape[0]

    if width != compress_len:
        raise ValueError(
            "decode candidate selector expects "
            "logits.shape[0] == compress_len, got "
            f"{width} != {compress_len}"
        )

    if width == 0:
        return mx.zeros(
            (0,),
            dtype=mx.bool_,
        )

    # Official:
    #
    #   F.pad(
    #       logits,
    #       (0, -width % block_size),
    #       value=-inf,
    #   )
    #
    # then reshape into blocks and take max.
    pad = (
        -width
    ) % block_size

    if pad:
        padded = mx.concatenate(
            [
                logits,
                mx.full(
                    (pad,),
                    float("-inf"),
                    dtype=logits.dtype,
                ),
            ],
            axis=0,
        )
    else:
        padded = logits

    num_blocks = (
        padded.shape[0]
        // block_size
    )

    block_scores = mx.max(
        padded.reshape(
            num_blocks,
            block_size,
        ),
        axis=-1,
    )

    block_ids = mx.arange(
        num_blocks,
        dtype=mx.int32,
    )

    # Official:
    #
    #   last = (compress_lens - 1) // block_size
    #   scores[last] = +inf
    #
    # This pins the newest partially filled block into the
    # candidate set.
    last = (
        compress_len - 1
    ) // block_size

    block_scores = mx.where(
        block_ids == last,
        mx.array(
            float("inf"),
            dtype=block_scores.dtype,
        ),
        block_scores,
    )

    k = min(
        topk_blocks,
        num_blocks,
    )

    if k == 0:
        return mx.zeros(
            (width,),
            dtype=mx.bool_,
        )

    # Same top-k semantics as the rest of our correctness path:
    # full argsort is acceptable here and avoids depending on
    # a separate partial-selection primitive.
    ranked = mx.argsort(
        block_scores,
        axis=-1,
    )

    selected = ranked[
        -k:
    ].astype(
        mx.int32
    )

    selected_values = block_scores[
        selected
    ]

    # Official drops leftover top-k selections whose block
    # score is -inf:
    #
    #   top.values > -inf
    #
    valid = (
        selected_values
        > mx.array(
            float("-inf"),
            dtype=selected_values.dtype,
        )
    )

    # MLX equivalent of:
    #
    #   zeros.scatter_(..., selected, valid)
    #
    # without requiring mutable scatter.
    matches = (
        block_ids[:, None]
        == selected[None, :]
    )

    keep_blocks = mx.any(
        matches
        & valid[None, :],
        axis=1,
    )

    # repeat_interleave(block_size)[..., :width]
    keep_positions = mx.repeat(
        keep_blocks,
        block_size,
        axis=0,
    )

    return keep_positions[
        :width
    ]




def indexer_decode_candidate_source(
    *,
    x: mx.array,
    qr: mx.array,
    latent: mx.array | None,
    start_pos: int,
    compress_ratio: int,
    state: IndexerState,
    rope_cos: mx.array,
    rope_sin: mx.array,
    weights_proj_weight: mx.array,
    wq_b_weight: mx.array,
    wq_b_scales: mx.array,
    wk_weight: mx.array,
    k_norm_weight: mx.array,
    norm_eps: float = 1e-20,
    index_topk: int = INDEX_TOPK,
    candidate_topk_blocks: int = 2048,
    candidate_block_size: int = 8,
) -> IndexerDecodeResult:
    """
    Decode-only candidate-source Indexer.

    This is the released DeepSeek V4.1 Flash layer-20 behavior.

    Unlike indexer_decode_base(), this path additionally runs
    level-one candidate BLOCK selection and returns the mask
    that must be published as shared_attn.candidates.

    Important:
        The candidate mask does NOT restrict layer 20's own
        final top-k.

        Official ordering is:

            compute index_score
              -> publish candidate blocks
              -> normal position-level top-k

        Only later index-source layers use the candidate mask
        to filter their own scores.
    """
    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D decode state, got {x.shape}"
        )

    if qr.ndim != 1:
        raise ValueError(
            f"qr must be 1D decode state, got {qr.shape}"
        )

    if compress_ratio <= 0:
        raise ValueError(
            f"compress_ratio must be > 0, got {compress_ratio}"
        )

    if start_pos < 0:
        raise ValueError(
            f"start_pos must be >= 0, got {start_pos}"
        )

    end_pos = start_pos + 1

    # ========================================================
    # 1. Own/update compressed index K.
    #
    # Identical semantics to the already validated base path.
    # ========================================================

    if latent is not None:
        if latent.ndim != 1:
            raise ValueError(
                f"latent must be 1D, got {latent.shape}"
            )

        if end_pos % compress_ratio != 0:
            raise ValueError(
                "latent was supplied before the compression "
                f"group completed: start_pos={start_pos}, "
                f"compress_ratio={compress_ratio}"
            )

        compressed_pos = (
            end_pos // compress_ratio
        ) - 1

        if compressed_pos >= state.k_cache.shape[0]:
            raise IndexError(
                f"compressed position {compressed_pos} exceeds "
                f"k_cache length {state.k_cache.shape[0]}"
            )

        k = _bf16_linear(
            latent,
            wk_weight,
        )

        k = rms_norm(
            k,
            k_norm_weight,
            eps=norm_eps,
        )

        k_rope_pos = (
            end_pos
            - compress_ratio
        )

        k = _apply_index_rope(
            k,
            rope_cos[k_rope_pos],
            rope_sin[k_rope_pos],
        )

        k = fp4_act_roundtrip_mlx(
            k,
            block_size=FP4_BLOCK_SIZE,
            scale_dtype="e8m0",
        )

        state.k_cache[
            compressed_pos
        ] = k.astype(mx.bfloat16)

    # ========================================================
    # 2. Current index query.
    # ========================================================

    q = fp8_linear(
        qr,
        wq_b_weight,
        wq_b_scales,
    ).reshape(
        INDEX_N_HEADS,
        INDEX_HEAD_DIM,
    )

    q = _apply_index_rope(
        q,
        rope_cos[start_pos],
        rope_sin[start_pos],
    )

    q = fp4_act_roundtrip_mlx(
        q,
        block_size=FP4_BLOCK_SIZE,
        scale_dtype="e8m0",
    )

    compress_len = (
        end_pos // compress_ratio
    )

    if compress_len == 0:
        return IndexerDecodeResult(
            topk_idxs=mx.zeros(
                (0,),
                dtype=mx.int32,
            ),
            compress_len=0,
            candidates=mx.zeros(
                (0,),
                dtype=mx.bool_,
            ),
        )

    index_k = state.k_cache[
        :compress_len
    ]

    # ========================================================
    # 3. Per-head weights.
    # ========================================================

    weights = _bf16_linear(
        x,
        weights_proj_weight,
    )

    scale = (
        INDEX_HEAD_DIM ** -0.5
        * INDEX_N_HEADS ** -0.5
    )

    weights = (
        weights
        * mx.array(
            scale,
            dtype=mx.float32,
        )
    ).astype(mx.bfloat16)

    # ========================================================
    # 4. Index score.
    # ========================================================

    per_head_score = mx.matmul(
        q,
        mx.transpose(index_k),
    )

    per_head_score = mx.maximum(
        per_head_score,
        mx.array(
            0.0,
            dtype=per_head_score.dtype,
        ),
    )

    index_score = mx.sum(
        per_head_score
        * weights[:, None],
        axis=0,
    )

    # ========================================================
    # 5. Level-one candidate blocks.
    #
    # Official layer 20 publishes this mask but does NOT use
    # it to filter its own top-k.
    # ========================================================

    candidates = select_candidate_blocks_decode(
        index_score,
        compress_len=compress_len,
        topk_blocks=candidate_topk_blocks,
        block_size=candidate_block_size,
    )

    # ========================================================
    # 6. Normal position-level top-k.
    #
    # Same semantics as indexer_decode_base().
    # ========================================================

    topk = min(
        index_topk,
        compress_len,
    )

    topk_idxs = _select_topk_relative(
        index_score,
        topk,
    )

    return IndexerDecodeResult(
        topk_idxs=topk_idxs,
        compress_len=compress_len,
        candidates=candidates,
    )




def indexer_decode_candidate_consumer(
    *,
    x: mx.array,
    qr: mx.array,
    start_pos: int,
    compress_ratio: int,
    index_k: mx.array,
    candidates: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    weights_proj_weight: mx.array,
    wq_b_weight: mx.array,
    wq_b_scales: mx.array,
    index_topk: int = INDEX_TOPK,
) -> IndexerDecodeResult:
    """
    Decode-only candidate-filtered Indexer for an INDEX-ONLY
    source layer.

    This is the released DeepSeek V4.1 Flash layer-24 behavior
    (and later the same topology is reused by layers 28/32/36).

    Unlike indexer_decode_base() and
    indexer_decode_candidate_source():

      * does NOT own compressed index K
      * does NOT receive a compressor latent
      * does NOT update any IndexerState
      * reuses shared_attn.index_k from layer 20
      * applies shared_attn.candidates BEFORE final top-k
      * returns a new topk_idxs publication

    `index_k` and `candidates` are the shared state produced by
    the candidate/KV source layer.
    """
    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D decode state, got {x.shape}"
        )

    if qr.ndim != 1:
        raise ValueError(
            f"qr must be 1D decode state, got {qr.shape}"
        )

    if start_pos < 0:
        raise ValueError(
            f"start_pos must be >= 0, got {start_pos}"
        )

    if compress_ratio <= 0:
        raise ValueError(
            f"compress_ratio must be > 0, got {compress_ratio}"
        )

    if index_k.ndim != 2:
        raise ValueError(
            f"index_k must be 2D, got {index_k.shape}"
        )

    if index_k.shape[1] != INDEX_HEAD_DIM:
        raise ValueError(
            f"index_k last dim must be {INDEX_HEAD_DIM}, "
            f"got {index_k.shape[1]}"
        )

    if candidates.ndim != 1:
        raise ValueError(
            f"candidates must be 1D, got {candidates.shape}"
        )

    end_pos = start_pos + 1

    compress_len = (
        end_pos // compress_ratio
    )

    if compress_len == 0:
        return IndexerDecodeResult(
            topk_idxs=mx.zeros(
                (0,),
                dtype=mx.int32,
            ),
            compress_len=0,
        )

    if compress_len > index_k.shape[0]:
        raise ValueError(
            f"compress_len {compress_len} exceeds "
            f"shared index_k length {index_k.shape[0]}"
        )

    if candidates.shape[0] < compress_len:
        raise ValueError(
            f"candidate mask length {candidates.shape[0]} "
            f"is smaller than compress_len {compress_len}"
        )

    # ========================================================
    # 1. Current index query.
    #
    # Exact same query path as the owning Indexers.
    # ========================================================

    q = fp8_linear(
        qr,
        wq_b_weight,
        wq_b_scales,
    ).reshape(
        INDEX_N_HEADS,
        INDEX_HEAD_DIM,
    )

    q = _apply_index_rope(
        q,
        rope_cos[start_pos],
        rope_sin[start_pos],
    )

    q = fp4_act_roundtrip_mlx(
        q,
        block_size=FP4_BLOCK_SIZE,
        scale_dtype="e8m0",
    )

    visible_index_k = index_k[
        :compress_len
    ]

    # ========================================================
    # 2. Per-head weights.
    # ========================================================

    weights = _bf16_linear(
        x,
        weights_proj_weight,
    )

    scale = (
        INDEX_HEAD_DIM ** -0.5
        * INDEX_N_HEADS ** -0.5
    )

    weights = (
        weights
        * mx.array(
            scale,
            dtype=mx.float32,
        )
    ).astype(
        mx.bfloat16
    )

    # ========================================================
    # 3. Position-level index score.
    # ========================================================

    per_head_score = mx.matmul(
        q,
        mx.transpose(
            visible_index_k
        ),
    )

    per_head_score = mx.maximum(
        per_head_score,
        mx.array(
            0.0,
            dtype=per_head_score.dtype,
        ),
    )

    index_score = mx.sum(
        per_head_score
        * weights[:, None],
        axis=0,
    )

    # ========================================================
    # 4. Candidate filtering.
    #
    # Official:
    #
    #   index_score = index_score.masked_fill(
    #       ~shared_attn.candidates,
    #       -inf,
    #   )
    #
    # This happens BEFORE final top-k selection.
    # ========================================================

    visible_candidates = candidates[
        :compress_len
    ].astype(
        mx.bool_
    )

    filtered_score = mx.where(
        visible_candidates,
        index_score,
        mx.array(
            float("-inf"),
            dtype=index_score.dtype,
        ),
    )

    # The official top-k can technically include -inf entries
    # when fewer than index_topk candidates exist. The later
    # implementation converts unreachable positions to -1.
    #
    # Our relative-index runtime must not expose such invalid
    # positions to sparse attention, so select only the number
    # of actually eligible candidate positions.
    candidate_count = int(
        mx.sum(
            visible_candidates.astype(
                mx.int32
            )
        ).item()
    )

    topk = min(
        index_topk,
        compress_len,
        candidate_count,
    )

    topk_idxs = _select_topk_relative(
        filtered_score,
        topk,
    )

    return IndexerDecodeResult(
        topk_idxs=topk_idxs,
        compress_len=compress_len,
    )

def indexer_decode_base(
    *,
    x: mx.array,
    qr: mx.array,
    latent: mx.array | None,
    start_pos: int,
    compress_ratio: int,
    state: IndexerState,
    rope_cos: mx.array,
    rope_sin: mx.array,
    weights_proj_weight: mx.array,
    wq_b_weight: mx.array,
    wq_b_scales: mx.array,
    wk_weight: mx.array,
    k_norm_weight: mx.array,
    norm_eps: float = 1e-20,
    index_topk: int = INDEX_TOPK,
) -> IndexerDecodeResult:
    """
    Decode-only base Indexer used by layer 2.

    This implements the released V4.1 Flash layer-2 behavior:

      * owns compressed index K
      * produces its own index top-k
      * no candidate-block prefilter
      * 32 index heads
      * index head dim 128
      * RoPE on final 64 dimensions
      * FP4 activation round-trip with:
            block_size=32
            E8M0 scales

    Inputs:
        x:
            Current hidden state [5120], BF16.

        qr:
            Current q-LoRA activation [1280], BF16.

        latent:
            Newly emitted compressor latent [512], BF16,
            or None when this raw token does not finish a
            compression group.

        start_pos:
            Zero-based raw token position.

    Returned topk indices are COMPRESSED-CACHE RELATIVE.
    """
    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D decode state, got {x.shape}"
        )

    if qr.ndim != 1:
        raise ValueError(
            f"qr must be 1D decode state, got {qr.shape}"
        )

    if compress_ratio <= 0:
        raise ValueError(
            f"compress_ratio must be > 0, got {compress_ratio}"
        )

    if start_pos < 0:
        raise ValueError(
            f"start_pos must be >= 0, got {start_pos}"
        )

    end_pos = start_pos + 1

    # --------------------------------------------------------
    # 1. Publish a new compressed index K whenever the
    #    compressor emits a latent.
    #
    # A compressed group is positioned at the FIRST raw token
    # belonging to that group.
    #
    # ratio=2:
    #   emitted at raw position 1 -> RoPE position 0
    #   emitted at raw position 3 -> RoPE position 2
    # --------------------------------------------------------
    if latent is not None:
        if latent.ndim != 1:
            raise ValueError(
                f"latent must be 1D, got {latent.shape}"
            )

        if end_pos % compress_ratio != 0:
            raise ValueError(
                "latent was supplied before the compression "
                f"group completed: start_pos={start_pos}, "
                f"compress_ratio={compress_ratio}"
            )

        compressed_pos = (
            end_pos // compress_ratio
        ) - 1

        if compressed_pos >= state.k_cache.shape[0]:
            raise IndexError(
                f"compressed position {compressed_pos} exceeds "
                f"k_cache length {state.k_cache.shape[0]}"
            )

        # Official:
        #
        #   k = k_norm(wk(latent))
        #
        k = _bf16_linear(
            latent,
            wk_weight,
        )

        k = rms_norm(
            k,
            k_norm_weight,
            eps=norm_eps,
        )

        k_rope_pos = (
            end_pos - compress_ratio
        )

        k = _apply_index_rope(
            k,
            rope_cos[k_rope_pos],
            rope_sin[k_rope_pos],
        )

        # Official index Q/K cache precision:
        #
        #   fp4_act_quant(
        #       k,
        #       block_size=32,
        #       inplace=True,
        #       scale_dtype=E8M0,
        #   )
        #
        k = fp4_act_roundtrip_mlx(
            k,
            block_size=FP4_BLOCK_SIZE,
            scale_dtype="e8m0",
        )

        state.k_cache[
            compressed_pos
        ] = k.astype(mx.bfloat16)

    # --------------------------------------------------------
    # 2. Build index Q for the current raw-token position.
    #
    # Official:
    #
    #   q = wq_b(qr)
    #   q = q.unflatten(-1, (32, 128))
    #   RoPE(last 64)
    #   fp4_act_quant(... block32, E8M0, inplace=True)
    # --------------------------------------------------------
    q = fp8_linear(
        qr,
        wq_b_weight,
        wq_b_scales,
    ).reshape(
        INDEX_N_HEADS,
        INDEX_HEAD_DIM,
    )

    q = _apply_index_rope(
        q,
        rope_cos[start_pos],
        rope_sin[start_pos],
    )

    q = fp4_act_roundtrip_mlx(
        q,
        block_size=FP4_BLOCK_SIZE,
        scale_dtype="e8m0",
    )

    # Only COMPLETE compressed groups are visible.
    compress_len = (
        end_pos // compress_ratio
    )

    if compress_len == 0:
        return IndexerDecodeResult(
            topk_idxs=mx.zeros(
                (0,),
                dtype=mx.int32,
            ),
            compress_len=0,
        )

    index_k = state.k_cache[
        :compress_len
    ]

    # --------------------------------------------------------
    # 3. Learned per-head weighting.
    #
    # Official:
    #
    #   weights = weights_proj(x)
    #   weights *= (
    #       index_head_dim^-0.5
    #       * index_n_heads^-0.5
    #   )
    # --------------------------------------------------------
    weights = _bf16_linear(
        x,
        weights_proj_weight,
    )

    scale = (
        INDEX_HEAD_DIM ** -0.5
        * INDEX_N_HEADS ** -0.5
    )

    weights = (
        weights
        * mx.array(
            scale,
            dtype=mx.float32,
        )
    ).astype(mx.bfloat16)

    # --------------------------------------------------------
    # 4. Index score.
    #
    # Equivalent to:
    #
    #   score = einsum(
    #       "hd,td->ht",
    #       q,
    #       index_k,
    #   )
    #
    #   score = (
    #       relu(score)
    #       * weights[:, None]
    #   ).sum(dim=0)
    # --------------------------------------------------------
    per_head_score = mx.matmul(
        q,
        mx.transpose(index_k),
    )

    per_head_score = mx.maximum(
        per_head_score,
        mx.array(
            0.0,
            dtype=per_head_score.dtype,
        ),
    )

    index_score = mx.sum(
        per_head_score
        * weights[:, None],
        axis=0,
    )

    # --------------------------------------------------------
    # 5. Direct top-k.
    #
    # Layer 2 is before candidate_source_layer_id=20,
    # therefore no candidate pre-filter applies.
    # --------------------------------------------------------
    topk = min(
        index_topk,
        compress_len,
    )

    topk_idxs = _select_topk_relative(
        index_score,
        topk,
    )

    return IndexerDecodeResult(
        topk_idxs=topk_idxs,
        compress_len=compress_len,
    )
