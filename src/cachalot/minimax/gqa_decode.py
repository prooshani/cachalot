"""
One decode token's attention for MiniMax-M3, reading each KV head once for its whole query group (HANDOFF 18.3).

M3 has 64 query heads over 4 KV heads (a group of 16). MLX's vector SDPA gives every query head its own
threadgroups, so each K and V row is fetched 16 times, and at 64k tokens the 60 layers' decode attention costs
~40 ms per token (~200 GB/s over ~8 GB of KV, HANDOFF 18.2). Here one threadgroup takes 256-key blocks of one
KV head and scores them against all 16 of its query heads with simdgroup 8x8 matrix multiplies (Q K^T, then P V),
walking several blocks with an online softmax, and a second, small kernel combines the threadgroups (split-K
"flash decoding"). 60 layers at 64k: 44 -> 23 ms; at 32k 24 -> 13; at 8k 9 -> 6; slower below ~3k, so the
model uses it from 4,096 cached tokens (`language.GQA_DECODE_MIN`). The arithmetic is float32 throughout; the
result differs from MLX's kernel by rounding only (a different reduction order), and was gated like the fast
norm: KL against the model's own chunking noise on the same text (HANDOFF 18.3).

`gqa_decode_attention(q, keys, values, length, scale)`: q [1, H, 1, D]; keys/values the cache's full buffers
[1, KVH, capacity, D] (not the sliced view: a strided input would be copied whole on every call), `length` the
valid prefix. D must be 128 and H / KVH 16.
"""

from __future__ import annotations

import os

import mlx.core as mx

NSG = int(os.environ.get("CACHALOT_MINIMAX_GQA_NSG", "8"))  # simdgroups per threadgroup, 32 keys each
TARGET_TGS = int(os.environ.get("CACHALOT_MINIMAX_GQA_TGS", "32"))  # threadgroups per KV head, at most
GROUP = 16
DIM = 128

_SCORE_SRC = """
    // One threadgroup: R consecutive blocks of NSG*32 keys of one KV head against its 16 query heads, with an
    // online softmax across the blocks; NSG simdgroups, 32 keys each per block.
    constexpr int BK = NSG * 32;
    constexpr int DV = 128 / NSG;        // output dims per simdgroup in P @ V
    constexpr int DT = DV / 8;
    const uint blk = threadgroup_position_in_grid.x;
    const uint kvh = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int L = meta[0];
    const int cap = meta[1];
    const int nb = meta[2];
    const int R = meta[3];

    threadgroup float S[16 * BK];                  // scores, then probabilities
    threadgroup float st[NSG * 256];               // per-simdgroup staging: K^T [8 d][32 keys], V [8 keys][DV d]
    threadgroup float mh[16], lh[16], diag[2 * 64];

    const device float* qg = qf + kvh * 16 * 128;  // this group's queries, float, pre-scaled
    threadgroup float* sst = st + sg * 256;
    const int dv = int(sg) * DV;
    simdgroup_float8x8 o[2][DT];
    for (int a = 0; a < 2; ++a) for (int c = 0; c < DT; ++c) o[a][c] = simdgroup_float8x8(0);
    if (sg == 0 && lane < 16) { mh[lane] = -INFINITY; lh[lane] = 0.0f; }
    for (uint i = thread_index_in_threadgroup; i < 128; i += NSG * 32) diag[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int rb = 0; rb < R; ++rb) {
        const int base = (int(blk) * R + rb) * BK;
        if (base >= L) break;
        // Q [16 x 128] @ K^T [128 x 32] for this simdgroup's 32 keys
        simdgroup_float8x8 acc[2][4];
        for (int a = 0; a < 2; ++a) for (int c = 0; c < 4; ++c) acc[a][c] = simdgroup_float8x8(0);
        const int j = base + int(sg) * 32 + int(lane);
        const bool live = j < L && j < cap;
        const device T* kp = k + ((size_t)kvh * cap + (live ? j : 0)) * 128;
        for (int d0 = 0; d0 < 128; d0 += 8) {
            uint4 w = live ? *((const device uint4*)(kp + d0)) : uint4(0);
            for (int e = 0; e < 4; ++e) {
                sst[(2 * e) * 32 + lane] = as_type<float>(w[e] << 16);
                sst[(2 * e + 1) * 32 + lane] = as_type<float>(w[e] & 0xffff0000u);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 qa, qb, kt;
            simdgroup_load(qa, qg + 0 * 128 + d0, 128);
            simdgroup_load(qb, qg + 8 * 128 + d0, 128);
            for (int c = 0; c < 4; ++c) {
                simdgroup_load(kt, sst + c * 8, 32);
                simdgroup_multiply_accumulate(acc[0][c], qa, kt, acc[0][c]);
                simdgroup_multiply_accumulate(acc[1][c], qb, kt, acc[1][c]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (int a = 0; a < 2; ++a)
            for (int c = 0; c < 4; ++c)
                simdgroup_store(acc[a][c], S + (a * 8) * BK + sg * 32 + c * 8, BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax per head; each simdgroup handles 16 / NSG heads
        for (uint hh = 0; hh < 16 / NSG; ++hh) {
            uint h = sg * (16 / NSG) + hh;
            float m = -INFINITY;
            for (uint i = lane; i < BK; i += 32) if (base + int(i) < L) m = max(m, S[h * BK + i]);
            m = simd_max(m);
            const float m_old = mh[h];
            const float m_new = max(m_old, m);
            float l = 0;
            for (uint i = lane; i < BK; i += 32) {
                float p = (base + int(i) < L) ? exp(S[h * BK + i] - m_new) : 0.0f;
                S[h * BK + i] = p;
                l += p;
            }
            l = simd_sum(l);
            if (lane == 0) {
                const float alpha = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
                lh[h] = lh[h] * alpha + l;
                mh[h] = m_new;
                diag[(h / 8) * 64 + (h % 8) * 9] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // rescale the running output, then add P [16 x BK] @ V [BK x DV]
        if (rb > 0) {
            simdgroup_float8x8 da, db, t;
            simdgroup_load(da, diag + 0, 8);
            simdgroup_load(db, diag + 64, 8);
            for (int c = 0; c < DT; ++c) {
                simdgroup_multiply(t, da, o[0][c]); o[0][c] = t;
                simdgroup_multiply(t, db, o[1][c]); o[1][c] = t;
            }
        }
        for (int r0 = 0; r0 < BK; r0 += 8) {
            for (int idx = int(lane) * 4; idx < 8 * DV; idx += 128) {
                int r = idx / DV, d = idx % DV;
                int jj = base + r0 + r;
                uint2 w = (jj < L && jj < cap) ? *((const device uint2*)(v + ((size_t)kvh * cap + jj) * 128 + dv + d)) : uint2(0);
                sst[r * DV + d + 0] = as_type<float>(w[0] << 16);
                sst[r * DV + d + 1] = as_type<float>(w[0] & 0xffff0000u);
                sst[r * DV + d + 2] = as_type<float>(w[1] << 16);
                sst[r * DV + d + 3] = as_type<float>(w[1] & 0xffff0000u);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 pa, pb, vt;
            simdgroup_load(pa, S + 0 * BK + r0, BK);
            simdgroup_load(pb, S + 8 * BK + r0, BK);
            for (int c = 0; c < DT; ++c) {
                simdgroup_load(vt, sst + c * 8, DV);
                simdgroup_multiply_accumulate(o[0][c], pa, vt, o[0][c]);
                simdgroup_multiply_accumulate(o[1][c], pb, vt, o[1][c]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // S and diag are rewritten by the next block
    }
    for (int a = 0; a < 2; ++a)
        for (int c = 0; c < DT; ++c)
            simdgroup_store(o[a][c], O + ((size_t)(kvh * 16 + a * 8) * nb + blk) * 128 + dv + c * 8, (ulong)nb * 128);
    if (sg == 0 && lane < 16) {
        Ms[(kvh * 16 + lane) * nb + blk] = mh[lane];
        Ls[(kvh * 16 + lane) * nb + blk] = lh[lane];
    }
"""

_COMBINE_SRC = """
    const uint d = thread_position_in_threadgroup.x;
    const uint hq = threadgroup_position_in_grid.x;
    const int nb = meta[2];
    float m = -INFINITY;
    for (int b = 0; b < nb; ++b) m = max(m, Ms[hq * nb + b]);
    float acc = 0, l = 0;
    for (int b = 0; b < nb; ++b) {
        float mb = Ms[hq * nb + b];
        if (mb == -INFINITY) continue;
        float w = exp(mb - m);
        l += w * Ls[hq * nb + b];
        acc += w * O[((size_t)hq * nb + b) * 128 + d];
    }
    out[hq * 128 + d] = T(acc / l);
"""

_score = None
_combine = None


def _kernels():
    global _score, _combine
    if _score is None:
        _score = mx.fast.metal_kernel(
            name="minimax_gqa_decode_score",
            input_names=["qf", "k", "v", "meta"],
            output_names=["O", "Ms", "Ls"],
            source=_SCORE_SRC,
            header="#include <metal_simdgroup_matrix>\nusing namespace metal;\n",
        )
        _combine = mx.fast.metal_kernel(
            name="minimax_gqa_decode_combine",
            input_names=["O", "Ms", "Ls", "meta"],
            output_names=["out"],
            source=_COMBINE_SRC,
        )
    return _score, _combine


def gqa_decode_attention(q: mx.array, keys: mx.array, values: mx.array, length: int, scale: float) -> mx.array:
    _, H, _, D = q.shape
    _, KVH, cap, _ = keys.shape
    if keys.dtype != mx.bfloat16:
        raise ValueError(f"gqa_decode_attention reads bfloat16 keys and values, got {keys.dtype}")
    if D != DIM or H != KVH * GROUP:
        raise ValueError(f"gqa_decode_attention expects D={DIM}, H=16*KVH; got D={D}, H={H}, KVH={KVH}")
    block = NSG * 32
    blocks = (length + block - 1) // block
    per_tg = max(1, (blocks + TARGET_TGS - 1) // TARGET_TGS)  # blocks walked by one threadgroup
    nb = (blocks + per_tg - 1) // per_tg
    meta = mx.array([length, cap, nb, per_tg], dtype=mx.int32)
    score, combine = _kernels()
    qf = (q.astype(mx.float32) * scale).reshape(H * D)
    O, Ms, Ls = score(
        inputs=[qf, keys, values, meta],
        template=[("T", keys.dtype), ("NSG", NSG)],
        grid=(nb * block, KVH, 1),
        threadgroup=(block, 1, 1),
        output_shapes=[(H, nb, D), (H, nb), (H, nb)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = combine(
        inputs=[O, Ms, Ls, meta],
        template=[("T", q.dtype)],
        grid=(D * H, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(H * D,)],
        output_dtypes=[q.dtype],
    )
    return out.reshape(1, H, 1, D)
