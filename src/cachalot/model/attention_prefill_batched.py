"""
Chunk-batched attention for layer-major prefill (sliding-window and
compressed-reuse layers).

Mirrors attention_sliding_window.sliding_window_attention_decode and
attention_compressed.compressed_attention_decode_reuse for T tokens at once:

  * FP8 projections: official per-32-block activation quantization, fp32
    accumulation, bf16 outputs (same cast points as fp8_linear).
  * Window keys: for token at position p, positions p-127..p in ascending
    order (what the ring cache + get_window_topk_idxs produce), invalid
    positions masked to -inf.
  * Compressed keys (reuse layers): the source layer's per-token top-k,
    padded with -1 and masked.
  * sparse_attention semantics: fp32 scores, sink as an extra logit with a
    zero value vector, bf16-cast probabilities before the value GEMM.

Only the fp32 summation order differs from the token-sequential path.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from cachalot.model.moe_prefill_batched import (
    dequantize_fp8_weight,
    quantize_activation_fp8_rows,
)
from cachalot.model.norm_rope_mlx import apply_rotary_emb, rms_norm


def fp8_linear_rows(x: mx.array, weight: mx.array, weight_scales: mx.array) -> mx.array:
    """x [T, K] (bf16) -> bf16 [T, N], official FP8 GEMM semantics."""
    qx = quantize_activation_fp8_rows(x)
    return (qx @ dequantize_fp8_weight(weight, weight_scales).T).astype(mx.bfloat16)


def fp8_roundtrip_rows(x: mx.array) -> mx.array:
    return quantize_activation_fp8_rows(x).astype(x.dtype)


def _rope_rows(x: mx.array, cos: mx.array, sin: mx.array, rope_dim: int, inverse: bool = False) -> mx.array:
    """Rotate the last rope_dim features of x [T, ..., D] with per-token cos/sin [T, rope_dim/2]."""
    nope = x[..., :-rope_dim]
    rot = x[..., -rope_dim:]
    if x.ndim == 2:      # [T, D]
        rotated = apply_rotary_emb(rot[None], cos, sin, inverse=inverse)[0]
    elif x.ndim == 3:    # [T, H, D]
        rotated = apply_rotary_emb(rot[None], cos, sin, inverse=inverse)[0]
    else:
        raise ValueError(f"unsupported shape {x.shape}")
    return mx.concatenate([nope, rotated], axis=-1)


def attention_prefill_batched(
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
    compressed_cache: mx.array | None = None,
    compressed_idxs_by_token: tuple[mx.array, ...] | None = None,
    window_size: int = 128,
    n_heads: int = 64,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    n_groups: int = 8,
    o_lora_rank: int = 1024,
    norm_eps: float = 1e-20,
) -> tuple[mx.array, mx.array]:
    """
    x: [T, hidden] bf16 (already hc_pre + rms_norm'ed attention input)
    Returns (output [T, hidden] bf16, updated_window_cache [window_size, head_dim]).
    """
    n_tokens = x.shape[0]
    positions = np.arange(start_pos, start_pos + n_tokens)
    cos = rope_cos[start_pos : start_pos + n_tokens]
    sin = rope_sin[start_pos : start_pos + n_tokens]

    # ---- Q ----
    qr = rms_norm(fp8_linear_rows(x, wq_a, wq_a_scales), q_norm_weight, eps=norm_eps)
    q = fp8_linear_rows(qr, wq_b, wq_b_scales).reshape(n_tokens, n_heads, head_dim)
    q = _rope_rows(q, cos, sin, rope_head_dim)

    # ---- KV ----
    kv = rms_norm(fp8_linear_rows(x, wkv, wkv_scales), kv_norm_weight, eps=norm_eps)
    kv = _rope_rows(kv, cos, sin, rope_head_dim)
    kv = fp8_roundtrip_rows(kv)  # [T, head_dim] bf16

    # ---- key pool: previous ring by position, then the new tokens ----
    # pool index j <-> position start_pos - window_size + j
    old_positions = positions[0] - window_size + np.arange(window_size)
    ring_index = mx.array((old_positions % window_size).astype(np.int32))
    pool = mx.concatenate([window_cache[ring_index], kv], axis=0)  # [W + T, D]
    pool_positions = np.concatenate([old_positions, positions])
    pool_valid = pool_positions >= 0

    # token t attends to pool[t + 1 : t + W + 1]  (positions p-W+1 .. p, ascending)
    win_idx = np.arange(n_tokens)[:, None] + 1 + np.arange(window_size)[None, :]  # [T, W]
    win_valid = pool_valid[win_idx]
    key_idx = [mx.array(win_idx.astype(np.int32))]
    key_valid = [mx.array(win_valid)]
    key_pool = [pool]

    if compressed_idxs_by_token is not None:
        assert compressed_cache is not None
        widths = [int(i.shape[0]) for i in compressed_idxs_by_token]
        kc = max(widths + [1])
        padded = np.full((n_tokens, kc), -1, dtype=np.int32)
        for t, idxs in enumerate(compressed_idxs_by_token):
            if widths[t]:
                padded[t, : widths[t]] = np.asarray(idxs, dtype=np.int32)
        c_valid = padded >= 0
        c_idx = np.where(c_valid, padded, 0)
        key_idx.append(mx.array(c_idx + pool.shape[0]))
        key_valid.append(mx.array(c_valid))
        key_pool.append(compressed_cache[: int(padded.max()) + 1] if c_valid.any() else compressed_cache[:1])

    all_pool = mx.concatenate(key_pool, axis=0) if len(key_pool) > 1 else pool
    idx = mx.concatenate(key_idx, axis=1)         # [T, K]
    valid = mx.concatenate(key_valid, axis=1)     # [T, K] bool
    selected = all_pool[idx]                      # [T, K, D] bf16

    # ---- sparse attention (fp32 scores, sink, bf16 probabilities) ----
    scale = head_dim ** -0.5
    qf = q.astype(mx.float32)                                   # [T, H, D]
    kf = selected.astype(mx.float32)                            # [T, K, D]
    scores = mx.matmul(qf, mx.swapaxes(kf, 1, 2)) * scale       # [T, H, K]
    scores = mx.where(valid[:, None, :], scores, mx.array(-mx.inf, dtype=mx.float32))
    sink = attn_sink.astype(mx.float32)[None, :, None]          # [1, H, 1]
    max_scores = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
    weights = mx.exp(scores - max_scores)
    sink_weight = mx.exp(sink - max_scores)
    denom = mx.sum(weights, axis=-1, keepdims=True) + sink_weight
    value_sum = mx.matmul(weights.astype(mx.bfloat16), selected).astype(mx.float32)  # [T, H, D]
    o = (value_sum / denom).astype(q.dtype)

    # ---- inverse rope, grouped wo_a, wo_b ----
    o = _rope_rows(o, cos, sin, rope_head_dim, inverse=True)
    group_input_dim = n_heads * head_dim // n_groups
    grouped_o = o.reshape(n_tokens, n_groups, group_input_dim)
    grouped_wo_a = wo_a_bf16.reshape(n_groups, o_lora_rank, group_input_dim)
    low_rank = mx.matmul(grouped_wo_a[None], grouped_o[..., None])[..., 0].reshape(n_tokens, -1)
    output = fp8_linear_rows(low_rank, wo_b, wo_b_scales)

    # ---- ring cache after the chunk ----
    end = positions[-1]
    slot_src = np.empty(window_size, dtype=np.int64)
    for s in range(window_size):
        # latest position <= end with position % W == s; from the pool if it is
        # a new token, else keep the old ring entry
        last = end - ((end - s) % window_size)
        slot_src[s] = last - old_positions[0] if last >= positions[0] else s
    new_from_pool = np.array([last >= positions[0] for last in (end - ((end - np.arange(window_size)) % window_size))])
    pool_take = pool[mx.array(np.where(new_from_pool, slot_src, 0).astype(np.int32))]
    updated_cache = mx.where(mx.array(new_from_pool)[:, None], pool_take, window_cache)

    return output, updated_cache
