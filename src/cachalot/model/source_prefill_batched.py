"""
Chunk-batched compressor, indexer and attention for the compressed-KV
source layers (2, 8, 14, 20) and the index-only source layers (24..36).

Mirrors attention_compressed.compressed_attention_decode_source /
compressed_attention_decode_index_source and indexer_mlx for T tokens at
once, keeping the sequential path's dtype boundaries:

  compressor : fp32 projections and softmax pooling, bf16 before RMSNorm
  indexer    : bf16 linears, bf16 scores (relu, head weights, head sum),
               top-k on bf16 scores, chronological order after selection
  caches     : FP4 round-trips (index K: block 32 E8M0; compressed KV:
               block 16 E4M3) stored as bf16
  attention  : window keys p-127..p plus the token's own top-k, fp32
               scores, sink logit, bf16 probabilities (see
               attention_prefill_batched)

Only fp32/bf16 accumulation order differs from the per-token path.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from cachalot.model.attention_prefill_batched import (
    _rope_rows,
    attention_prefill_batched,
    fp8_linear_rows,
)
from cachalot.model.compressor_mlx import CompressorState
from cachalot.model.fp4_act_mlx import fp4_act_roundtrip_mlx
from cachalot.model.indexer_mlx import (
    INDEX_HEAD_DIM,
    INDEX_N_HEADS,
    INDEX_ROPE_DIM,
    IndexerDecodeResult,
    IndexerState,
)
from cachalot.model.norm_rope_mlx import rms_norm

# Query rows per indexer scoring pass. The scores are [T, 32, cmax] bf16 before
# the head sum, so a whole-prompt pass is quadratic in the prompt: 5.8 GB of
# per-head scores alone for a 13.5k-token Hermes prompt, plus two temporaries of
# the same size, which is what ran the Metal heap out (HANDOFF section 15.2).
# Rows are independent and every pass keeps the full cmax width, so the top-k
# and candidate selection see exactly the arrays they saw unchunked.
# 0 disables chunking.
INDEX_Q_CHUNK = int(os.environ.get("CACHALOT_INDEX_Q_CHUNK", "1024"))


def _row_slices(n_tokens: int) -> list[slice]:
    """
    Near-equal row slices of at most INDEX_Q_CHUNK rows. A prompt shorter
    than one chunk is a single slice -- exactly the unchunked computation --
    and no slice of a longer prompt drops below half a chunk, because MLX
    picks a different matmul kernel for a handful of rows and the scores
    would stop being bit-identical to a whole-prompt pass.
    """
    if INDEX_Q_CHUNK <= 0 or n_tokens <= INDEX_Q_CHUNK:
        return [slice(0, n_tokens)]
    n_chunks = -(-n_tokens // INDEX_Q_CHUNK)
    base, extra = divmod(n_tokens, n_chunks)
    out, r0 = [], 0
    for i in range(n_chunks):
        r1 = r0 + base + (1 if i < extra else 0)
        out.append(slice(r0, r1))
        r0 = r1
    return out


# ---------------------------------------------------------------------------
# compressor
# ---------------------------------------------------------------------------
def compressor_chunk(
    x: mx.array,
    *,
    start_pos: int,
    compress_ratio: int,
    norm_weight: mx.array,
    wkv_weight: mx.array,
    wgate_weight: mx.array | None,
    state: CompressorState | None,
    eps: float,
) -> tuple[mx.array | None, np.ndarray]:
    """
    x: [T, hidden] bf16 attention input.
    Returns (latents [G, head_dim] bf16 pre-RoPE, group_end_positions [G])
    where group g completed at absolute position group_end_positions[g].
    Updates `state` with the trailing partial group (ratio > 1).
    """
    n_tokens = x.shape[0]
    positions = np.arange(start_pos, start_pos + n_tokens)

    if compress_ratio == 1:
        kv = (x @ mx.swapaxes(wkv_weight, -1, -2)).astype(x.dtype)
        return rms_norm(kv, norm_weight, eps), positions

    assert wgate_weight is not None and state is not None
    xf = x.astype(mx.float32)
    kv = xf @ mx.swapaxes(wkv_weight.astype(mx.float32), -1, -2)       # [T, D]
    score = xf @ mx.swapaxes(wgate_weight.astype(mx.float32), -1, -2)  # [T, D]

    # prepend pending partial-group rows kept in the state
    pending = start_pos % compress_ratio
    if pending:
        kv = mx.concatenate([state.kv_state[0, :pending], kv], axis=0)
        score = mx.concatenate([state.score_state[0, :pending], score], axis=0)
    total = pending + n_tokens
    groups = total // compress_ratio
    remainder = total - groups * compress_ratio

    latents = None
    ends = np.zeros((0,), dtype=np.int64)
    if groups:
        kv_g = kv[: groups * compress_ratio].reshape(groups, compress_ratio, -1)
        sc_g = score[: groups * compress_ratio].reshape(groups, compress_ratio, -1)
        pooled = mx.sum(kv_g * mx.softmax(sc_g, axis=1), axis=1)
        latents = rms_norm(pooled.astype(x.dtype), norm_weight, eps)
        first_group_end = start_pos - pending + compress_ratio - 1
        ends = first_group_end + compress_ratio * np.arange(groups)

    # trailing partial group -> state
    new_kv = mx.zeros_like(state.kv_state[0])
    new_sc = mx.full(state.score_state[0].shape, -mx.inf, dtype=mx.float32)
    if remainder:
        tail_kv = kv[groups * compress_ratio :]
        tail_sc = score[groups * compress_ratio :]
        new_kv = mx.concatenate([tail_kv, new_kv[remainder:]], axis=0)
        new_sc = mx.concatenate([tail_sc, new_sc[remainder:]], axis=0)
    state.kv_state = new_kv[None]
    state.score_state = new_sc[None]
    return latents, ends


# ---------------------------------------------------------------------------
# indexer pieces
# ---------------------------------------------------------------------------
def _index_rope_rows(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x [T, ..., 128]: rotate the last 64 dims with per-token cos/sin [T, 32]."""
    return _rope_rows(x, cos, sin, INDEX_ROPE_DIM)


def index_queries_chunk(qr: mx.array, wq_b_weight, wq_b_scales, cos, sin) -> mx.array:
    """qr [T, 1280] bf16 -> fp4-rounded index queries [T, 32, 128] bf16."""
    n = qr.shape[0]
    q = fp8_linear_rows(qr, wq_b_weight, wq_b_scales).reshape(n, INDEX_N_HEADS, INDEX_HEAD_DIM)
    q = _index_rope_rows(q, cos, sin)
    return fp4_act_roundtrip_mlx(q, block_size=32, scale_dtype="e8m0")


def index_keys_chunk(latents: mx.array, wk_weight, k_norm_weight, cos, sin, eps: float) -> mx.array:
    """latents [G, 512] -> index K rows [G, 128] bf16 (norm, rope at group start, fp4 round-trip)."""
    k = (latents @ mx.swapaxes(wk_weight, -1, -2)).astype(mx.bfloat16)
    k = rms_norm(k, k_norm_weight, eps=eps)
    k = _index_rope_rows(k, cos, sin)
    return fp4_act_roundtrip_mlx(k, block_size=32, scale_dtype="e8m0").astype(mx.bfloat16)


def index_scores_chunk(q: mx.array, x: mx.array, weights_proj_weight, index_k: mx.array, compress_len: np.ndarray) -> mx.array:
    """
    q [T, 32, 128], x [T, hidden], index_k [Cmax, 128] -> bf16 scores [T, Cmax]
    with positions >= compress_len[t] masked to -inf.
    """
    weights = (x @ mx.swapaxes(weights_proj_weight, -1, -2)).astype(mx.bfloat16)
    scale = INDEX_HEAD_DIM**-0.5 * INDEX_N_HEADS**-0.5
    weights = (weights * mx.array(scale, dtype=mx.float32)).astype(mx.bfloat16)  # [T, 32]
    per_head = mx.matmul(q, mx.transpose(index_k))                                  # [T, 32, C] bf16
    per_head = mx.maximum(per_head, mx.array(0.0, dtype=per_head.dtype))
    score = mx.sum(per_head * weights[:, :, None], axis=1)                          # [T, C] bf16
    cmax = index_k.shape[0]
    visible = mx.array(np.arange(cmax)[None, :] < compress_len[:, None])
    return mx.where(visible, score, mx.array(-mx.inf, dtype=score.dtype))


def topk_rows(score: mx.array, k_per_row: np.ndarray, topk: int) -> list[mx.array]:
    """
    Per row: take the k_per_row[t] highest-scoring visible positions, then
    sort them chronologically (indexer semantics). Returns int32 arrays.
    """
    n, cmax = score.shape
    kmax = int(min(topk, cmax))
    if kmax == 0:
        return [mx.zeros((0,), dtype=mx.int32) for _ in range(n)]
    order = mx.argsort(score, axis=-1)                # ascending, -inf first
    top = order[:, cmax - kmax :]                      # highest kmax
    top_np = np.array(top.astype(mx.int32))
    out = []
    for t in range(n):
        k = int(min(k_per_row[t], kmax))
        sel = top_np[t, kmax - k :] if k else np.zeros((0,), dtype=np.int32)
        out.append(mx.array(np.sort(sel).astype(np.int32)))
    return out


def candidate_masks_chunk(score: mx.array, compress_len: np.ndarray, topk_blocks: int, block_size: int) -> list[mx.array]:
    """Per-token candidate block masks (official select_candidate_blocks), bool [compress_len[t]]."""
    n, cmax = score.shape
    nb = (cmax + block_size - 1) // block_size
    pad = nb * block_size - cmax
    padded = mx.concatenate([score, mx.full((n, pad), -mx.inf, dtype=score.dtype)], axis=1) if pad else score
    block_scores = mx.max(padded.reshape(n, nb, block_size), axis=-1).astype(mx.float32)
    last = (compress_len - 1) // block_size
    pin = mx.array(np.arange(nb)[None, :] == last[:, None])
    block_scores = mx.where(pin, mx.array(mx.inf, dtype=mx.float32), block_scores)
    k = min(topk_blocks, nb)
    order = mx.argsort(block_scores, axis=-1)
    selected = order[:, nb - k :]
    valid = mx.take_along_axis(block_scores, selected, axis=-1) > -mx.inf
    keep = mx.zeros((n, nb), dtype=mx.bool_)
    keep = keep.at[mx.arange(n)[:, None], selected].add(valid)
    keep_positions = mx.repeat(keep, block_size, axis=1)[:, :cmax]
    kp = np.array(keep_positions)
    return [mx.array(kp[t, : int(compress_len[t])]) for t in range(n)]


# ---------------------------------------------------------------------------
# source layer (KV + index source; layer 20 also publishes candidates)
# ---------------------------------------------------------------------------
def compressed_source_chunk(
    x: mx.array,
    *,
    start_pos: int,
    compress_ratio: int,
    window_cache: mx.array,
    compressed_cache: mx.array,
    compressor_state: CompressorState | None,
    indexer_state: IndexerState,
    rope_cos: mx.array,
    rope_sin: mx.array,
    attn_sink, q_norm_weight, kv_norm_weight,
    wq_a, wq_a_scales, wq_b, wq_b_scales, wkv, wkv_scales, wo_a_bf16, wo_b, wo_b_scales,
    compressor_norm_weight, compressor_wkv_weight, compressor_wgate_weight,
    indexer_weights_proj_weight, indexer_wq_b_weight, indexer_wq_b_scales, indexer_wk_weight, indexer_k_norm_weight,
    window_size: int = 128, n_heads: int = 64, head_dim: int = 512, rope_head_dim: int = 64,
    n_groups: int = 8, o_lora_rank: int = 1024, norm_eps: float = 1e-20, index_topk: int = 512,
    candidate_source: bool = False, candidate_topk_blocks: int = 2048, candidate_block_size: int = 8,
) -> tuple[mx.array, mx.array, mx.array, tuple[IndexerDecodeResult, ...]]:
    n_tokens = x.shape[0]
    positions = np.arange(start_pos, start_pos + n_tokens)
    cos = rope_cos[start_pos : start_pos + n_tokens]
    sin = rope_sin[start_pos : start_pos + n_tokens]

    # qr is shared by attention Q and the indexer
    qr = rms_norm(fp8_linear_rows(x, wq_a, wq_a_scales), q_norm_weight, eps=norm_eps)

    # ---- compressor -> latents at group ends ----
    latents, ends = compressor_chunk(
        x, start_pos=start_pos, compress_ratio=compress_ratio, norm_weight=compressor_norm_weight,
        wkv_weight=compressor_wkv_weight, wgate_weight=compressor_wgate_weight, state=compressor_state, eps=norm_eps,
    )
    updated_compressed = compressed_cache
    if latents is not None and len(ends):
        slots = (ends + 1) // compress_ratio - 1                # compressed positions
        group_pos = ends + 1 - compress_ratio                   # RoPE position: first token of group
        gp = mx.array(group_pos.astype(np.int32))
        # index K
        k_rows = index_keys_chunk(latents, indexer_wk_weight, indexer_k_norm_weight, rope_cos[gp], rope_sin[gp], norm_eps)
        # Functional updates (copy, then indexed assignment) so prefix-cache
        # snapshots holding the previous arrays are never mutated.
        slot_idx = mx.array(slots.astype(np.int32))
        k_cache = mx.array(indexer_state.k_cache)
        k_cache[slot_idx] = k_rows
        indexer_state.k_cache = k_cache
        # compressed KV: rope at group start, fp4 block 16 e4m3
        ckv = _rope_rows(latents, rope_cos[gp], rope_sin[gp], rope_head_dim)
        ckv = fp4_act_roundtrip_mlx(ckv, block_size=16, scale_dtype="e4m3")
        updated_compressed = mx.array(compressed_cache)
        updated_compressed[slot_idx] = ckv.astype(compressed_cache.dtype)

    # ---- indexer scores / top-k / candidates ----
    compress_len = (positions + 1) // compress_ratio
    cmax = int(compress_len.max())
    results: list[IndexerDecodeResult] = []
    if cmax == 0:
        topk_by_token = [mx.zeros((0,), dtype=mx.int32) for _ in range(n_tokens)]
        cands = [mx.zeros((0,), dtype=mx.bool_) for _ in range(n_tokens)] if candidate_source else None
    else:
        q_idx = index_queries_chunk(qr, indexer_wq_b_weight, indexer_wq_b_scales, cos, sin)
        index_k = indexer_state.k_cache[:cmax]
        topk_by_token = []
        cands = [] if candidate_source else None
        for rows in _row_slices(n_tokens):
            score = index_scores_chunk(q_idx[rows], x[rows], indexer_weights_proj_weight, index_k, compress_len[rows])
            topk_by_token += topk_rows(score, np.minimum(index_topk, compress_len[rows]), index_topk)
            if candidate_source:
                cands += candidate_masks_chunk(
                    score, compress_len[rows], candidate_topk_blocks, candidate_block_size
                )
    for t in range(n_tokens):
        results.append(IndexerDecodeResult(
            topk_idxs=topk_by_token[t], compress_len=int(compress_len[t]),
            candidates=(cands[t] if cands is not None else None),
        ))

    # ---- attention over window + own compressed top-k ----
    output, updated_window = attention_prefill_batched(
        x, start_pos=start_pos, window_cache=window_cache, rope_cos=rope_cos, rope_sin=rope_sin, attn_sink=attn_sink,
        q_norm_weight=q_norm_weight, kv_norm_weight=kv_norm_weight, wq_a=wq_a, wq_a_scales=wq_a_scales, wq_b=wq_b,
        wq_b_scales=wq_b_scales, wkv=wkv, wkv_scales=wkv_scales, wo_a_bf16=wo_a_bf16, wo_b=wo_b, wo_b_scales=wo_b_scales,
        compressed_cache=updated_compressed, compressed_idxs_by_token=tuple(topk_by_token),
        window_size=window_size, n_heads=n_heads, head_dim=head_dim, rope_head_dim=rope_head_dim, n_groups=n_groups,
        o_lora_rank=o_lora_rank, norm_eps=norm_eps,
    )
    return output, updated_window, updated_compressed, tuple(results)


# ---------------------------------------------------------------------------
# index-only source layer (24, 28, 32, 36)
# ---------------------------------------------------------------------------
def index_source_chunk(
    x: mx.array,
    *,
    start_pos: int,
    compress_ratio: int,
    window_cache: mx.array,
    compressed_cache: mx.array,
    index_k: mx.array,
    candidates_by_token: tuple[mx.array, ...],
    rope_cos, rope_sin, attn_sink, q_norm_weight, kv_norm_weight,
    wq_a, wq_a_scales, wq_b, wq_b_scales, wkv, wkv_scales, wo_a_bf16, wo_b, wo_b_scales,
    indexer_weights_proj_weight, indexer_wq_b_weight, indexer_wq_b_scales,
    window_size: int = 128, n_heads: int = 64, head_dim: int = 512, rope_head_dim: int = 64,
    n_groups: int = 8, o_lora_rank: int = 1024, norm_eps: float = 1e-20, index_topk: int = 512,
) -> tuple[mx.array, mx.array, tuple[IndexerDecodeResult, ...]]:
    n_tokens = x.shape[0]
    positions = np.arange(start_pos, start_pos + n_tokens)
    cos = rope_cos[start_pos : start_pos + n_tokens]
    sin = rope_sin[start_pos : start_pos + n_tokens]
    qr = rms_norm(fp8_linear_rows(x, wq_a, wq_a_scales), q_norm_weight, eps=norm_eps)
    compress_len = (positions + 1) // compress_ratio
    cmax = int(compress_len.max())

    if cmax == 0:
        topk_by_token = [mx.zeros((0,), dtype=mx.int32) for _ in range(n_tokens)]
    else:
        q_idx = index_queries_chunk(qr, indexer_wq_b_weight, indexer_wq_b_scales, cos, sin)
        index_k_visible = index_k[:cmax]
        topk_by_token = []
        for rows in _row_slices(n_tokens):
            score = index_scores_chunk(
                q_idx[rows], x[rows], indexer_weights_proj_weight, index_k_visible, compress_len[rows]
            )
            # candidate filter: -inf outside the token's candidate mask
            n_rows = score.shape[0]
            cand = np.zeros((n_rows, cmax), dtype=bool)
            counts = np.zeros(n_rows, dtype=np.int64)
            for i, t in enumerate(range(rows.start, rows.stop)):
                c = np.array(candidates_by_token[t]).astype(bool)[: int(compress_len[t])]
                cand[i, : c.shape[0]] = c
                counts[i] = int(c.sum())
            score = mx.where(mx.array(cand), score, mx.array(-mx.inf, dtype=score.dtype))
            k_rows = np.minimum(np.minimum(index_topk, compress_len[rows]), counts)
            topk_by_token += topk_rows(score, k_rows, index_topk)

    results = tuple(
        IndexerDecodeResult(topk_idxs=topk_by_token[t], compress_len=int(compress_len[t])) for t in range(n_tokens)
    )
    output, updated_window = attention_prefill_batched(
        x, start_pos=start_pos, window_cache=window_cache, rope_cos=rope_cos, rope_sin=rope_sin, attn_sink=attn_sink,
        q_norm_weight=q_norm_weight, kv_norm_weight=kv_norm_weight, wq_a=wq_a, wq_a_scales=wq_a_scales, wq_b=wq_b,
        wq_b_scales=wq_b_scales, wkv=wkv, wkv_scales=wkv_scales, wo_a_bf16=wo_a_bf16, wo_b=wo_b, wo_b_scales=wo_b_scales,
        compressed_cache=compressed_cache, compressed_idxs_by_token=tuple(topk_by_token),
        window_size=window_size, n_heads=n_heads, head_dim=head_dim, rope_head_dim=rope_head_dim, n_groups=n_groups,
        o_lora_rank=o_lora_rank, norm_eps=norm_eps,
    )
    return output, updated_window, results
