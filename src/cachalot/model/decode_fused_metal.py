"""
Single-launch Metal kernels for the small per-layer glue of one decode token.

Each replaces a chain of 4-10 MLX ops with one dispatch, computing exactly
the same fp32 arithmetic as the MLX reference functions:

  rope_apply_1d    : apply_rotary_emb on [..., 64] with one position (q, kv, o^-1)
  rms_norm_1d      : rms_norm for a 1-D vector
  hc_pre_norm      : hc_pre(x, pre_mix) followed by rms_norm (attention / FFN input)
  hc_post_1d       : hc_post(y, residual, post, comb)
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
@cache
def _rope_kernel(inverse: bool):
    sign = "-" if inverse else ""
    source = f"""
        uint i = thread_position_in_grid.x;          // pair index over the flattened input
        if (i >= n_pairs[0]) return;
        uint rope_pairs = dims[0];                    // rope_dim / 2 (32)
        uint p = i % rope_pairs;                      // pair within the rotated tail
        float re = float(x[2 * i]);
        float im = float(x[2 * i + 1]);
        float c = cos_row[p];
        float s_ = {sign}sin_row[p];
        out[2 * i] = T(re * c - im * s_);
        out[2 * i + 1] = T(re * s_ + im * c);
    """
    return mx.fast.metal_kernel(
        name=f"rope_1d_{'inv' if inverse else 'fwd'}",
        input_names=["x", "cos_row", "sin_row", "n_pairs", "dims"],
        output_names=["out"],
        source=source,
        header=HEADER,
    )


def rope_apply_1d(x: mx.array, cos_row: mx.array, sin_row: mx.array, *, inverse: bool = False) -> mx.array:
    """x [..., D] with D even, cos_row/sin_row [D/2] for one position. Same math as apply_rotary_emb."""
    d = x.shape[-1]
    n_pairs = x.size // 2
    out = _rope_kernel(inverse)(
        inputs=[x.reshape(-1), cos_row.astype(mx.float32), sin_row.astype(mx.float32),
                mx.array([n_pairs], dtype=mx.uint32), mx.array([d // 2], dtype=mx.uint32)],
        template=[("T", x.dtype)],
        grid=(n_pairs, 1, 1),
        threadgroup=(min(256, n_pairs), 1, 1),
        output_shapes=[(x.size,)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# RMSNorm / hyper-connection glue (one threadgroup, cooperative reduction)
# ---------------------------------------------------------------------------
@cache
def _hc_pre_norm_kernel(hc_mult: int, has_pre: bool):
    # y[d] = sum_h pre[h] * x[h, d]   (fp32)     ; then rms_norm over d
    combine = (
        "\n".join(f"            v += pre[{h}] * float(x[{h} * n + d]);" for h in range(hc_mult))
        if has_pre else "            v = float(x[d]);"
    )
    source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint n = dims[0];
        float eps = eps_in[0];
        threadgroup float partial[256];
        float acc = 0.0f;
        for (uint d = tid; d < n; d += 256) {{
            float v = 0.0f;
{combine}
            v = float(T(v));   // hc_pre returns x.dtype before the norm
            acc += v * v;
        }}
        partial[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {{
            if (tid < stride) partial[tid] += partial[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float rs = metal::rsqrt(partial[0] / float(n) + eps);
        for (uint d = tid; d < n; d += 256) {{
            float v = 0.0f;
{combine}
            v = float(T(v));
            out[d] = T(float(weight[d]) * (v * rs));
        }}
    """
    return mx.fast.metal_kernel(
        name=f"hc_pre_norm_h{hc_mult}_{int(has_pre)}",
        input_names=["x", "pre", "weight", "dims", "eps_in"] if has_pre else ["x", "weight", "dims", "eps_in"],
        output_names=["out"],
        source=source,
        header=HEADER,
    )


def hc_pre_norm_1d(x: mx.array, pre_mix: mx.array, weight: mx.array, *, eps: float) -> mx.array:
    """rms_norm(hc_pre(x [hc, n], pre_mix [hc]), weight) -> [n] in x.dtype."""
    hc, n = x.shape
    return _hc_pre_norm_kernel(hc, True)(
        inputs=[x, pre_mix.astype(mx.float32), weight, mx.array([n], dtype=mx.uint32), mx.array([eps], dtype=mx.float32)],
        template=[("T", x.dtype)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
    )[0]


def rms_norm_1d(x: mx.array, weight: mx.array, *, eps: float) -> mx.array:
    n = x.shape[-1]
    return _hc_pre_norm_kernel(1, False)(
        inputs=[x.reshape(-1), weight, mx.array([n], dtype=mx.uint32), mx.array([eps], dtype=mx.float32)],
        template=[("T", x.dtype)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
    )[0].reshape(x.shape)


@cache
def _hc_post_kernel(hc_mult: int):
    rows = []
    for h in range(hc_mult):
        terms = " + ".join(f"comb[{h} * {hc_mult} + {j}] * float(residual[{j} * n + d])" for j in range(hc_mult))
        rows.append(f"        out[{h} * n + d] = T(post[{h}] * yv + ({terms}));")
    source = f"""
        uint d = thread_position_in_grid.x;
        uint n = dims[0];
        if (d >= n) return;
        float yv = float(y[d]);
{chr(10).join(rows)}
    """
    return mx.fast.metal_kernel(
        name=f"hc_post_h{hc_mult}",
        input_names=["y", "residual", "post", "comb", "dims"],
        output_names=["out"],
        source=source,
        header=HEADER,
    )


def hc_post_1d(y: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    """hc_post for one token: y [n], residual [hc, n], post [hc], comb [hc, hc] -> [hc, n]."""
    hc, n = residual.shape
    return _hc_post_kernel(hc)(
        inputs=[y, residual, post.astype(mx.float32), comb.astype(mx.float32).reshape(-1), mx.array([n], dtype=mx.uint32)],
        template=[("T", residual.dtype)],
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(hc * n,)],
        output_dtypes=[residual.dtype],
    )[0].reshape(hc, n)


# ---------------------------------------------------------------------------
# switchable entry points used by the decode blocks (CACHALOT_FUSED_DECODE=0
# restores the MLX op chains; both produce identical bits)
# ---------------------------------------------------------------------------
import os  # noqa: E402

from cachalot.model.hyper_connection_mlx import hc_post, hc_pre  # noqa: E402
from cachalot.model.norm_rope_mlx import apply_rotary_emb, rms_norm  # noqa: E402

FUSED_DECODE = os.environ.get("CACHALOT_FUSED_DECODE", "1") != "0"


def rope_decode(x: mx.array, cos_row: mx.array, sin_row: mx.array, *, inverse: bool = False) -> mx.array:
    """x [D] or [H, D] for one position."""
    if FUSED_DECODE:
        return rope_apply_1d(x, cos_row, sin_row, inverse=inverse)
    return apply_rotary_emb(x[None], cos_row[None], sin_row[None], inverse=inverse)[0]


def rms_norm_decode(x: mx.array, weight: mx.array, *, eps: float) -> mx.array:
    if FUSED_DECODE and x.ndim == 1:
        return rms_norm_1d(x, weight, eps=eps)
    return rms_norm(x, weight, eps=eps)


def hc_pre_norm_decode(x: mx.array, pre_mix: mx.array, weight: mx.array, *, eps: float) -> mx.array:
    if FUSED_DECODE and x.ndim == 2:
        return hc_pre_norm_1d(x, pre_mix, weight, eps=eps)
    return rms_norm(hc_pre(x, pre_mix), weight, eps=eps)


def hc_post_decode(y: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    if FUSED_DECODE and residual.ndim == 2 and y.ndim == 1:
        return hc_post_1d(y, residual, post, comb)
    return hc_post(y, residual, post, comb)


# ---------------------------------------------------------------------------
# Sparse attention for one query position (all heads) in a single launch
# ---------------------------------------------------------------------------
@cache
def _sparse_attention_kernel(head_dim: int, max_keys: int, n_threads: int = 256):
    # head_dim must be a multiple of 256 (each of 32 lanes owns whole uint4 blocks); n_threads:
    # phase 1 one simdgroup per key (vectorized dot), phase 4 = n_threads/d_blocks key-slices x d_blocks
    # d-blocks of 8 elements, reduced through threadgroup memory.
    d_blocks = head_dim // 8
    n_sg = n_threads // 32
    n_slices = n_threads // d_blocks
    source = f"""
        uint head = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint n_keys = dims[0];
        float scale = scale_in[0];

        threadgroup float scores[{max_keys}];
        threadgroup float red[{n_threads}];
        threadgroup float part_acc[{n_slices} * {head_dim}];

        // phase 1: one simdgroup per key, lanes split head_dim (coalesced row reads)
        uint sg = tid >> 5;
        uint lane = tid & 31;
        constexpr uint LANE_BLOCKS = {d_blocks // 32};
        const device uint4* qv = reinterpret_cast<const device uint4*>(q + head * {head_dim});
        float qf[LANE_BLOCKS * 8];
        for (uint b = 0; b < LANE_BLOCKS; ++b) {{
            uint4 w4 = qv[lane + 32 * b];
            uint ws[4] = {{w4.x, w4.y, w4.z, w4.w}};
            for (uint j = 0; j < 4; ++j) {{
                qf[b * 8 + 2 * j] = as_type<float>(ws[j] << 16);
                qf[b * 8 + 2 * j + 1] = as_type<float>(ws[j] & 0xFFFF0000u);
            }}
        }}
        for (uint k = sg; k < n_keys; k += {n_sg}) {{
            int idx = idxs[k];
            float acc = 0.0f;
            if (idx >= 0) {{
                const device uint4* kr = reinterpret_cast<const device uint4*>(kv + uint(idx) * {head_dim});
                for (uint b = 0; b < LANE_BLOCKS; ++b) {{
                    uint4 w4 = kr[lane + 32 * b];
                    uint ws[4] = {{w4.x, w4.y, w4.z, w4.w}};
                    for (uint j = 0; j < 4; ++j) {{
                        acc += qf[b * 8 + 2 * j] * as_type<float>(ws[j] << 16);
                        acc += qf[b * 8 + 2 * j + 1] * as_type<float>(ws[j] & 0xFFFF0000u);
                    }}
                }}
            }}
            acc = simd_sum(acc);
            if (lane == 0) scores[k] = (idx < 0) ? -INFINITY : acc * scale;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // phase 2: max
        float m = -INFINITY;
        for (uint k = tid; k < n_keys; k += {n_threads}) m = metal::max(m, scores[k]);
        red[tid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = {n_threads // 2}; s > 0; s >>= 1) {{
            if (tid < s) red[tid] = metal::max(red[tid], red[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float sink = sink_in[head];
        float max_score = metal::max(red[0], sink);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // phase 3: bf16-rounded weights, fp32 denominator
        float part = 0.0f;
        for (uint k = tid; k < n_keys; k += {n_threads}) {{
            float w = metal::exp(scores[k] - max_score);
            part += w;
            scores[k] = float(bfloat16_t(w));
        }}
        red[tid] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = {n_threads // 2}; s > 0; s >>= 1) {{
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float denom = red[0] + metal::exp(sink - max_score);

        // phase 4: value sum, {n_slices} key slices x {d_blocks} blocks of 8 dims
        uint slice = tid / {d_blocks};
        uint blk = tid % {d_blocks};
        if (slice < {n_slices}) {{
            float acc[8];
            for (uint j = 0; j < 8; ++j) acc[j] = 0.0f;
            for (uint k = slice; k < n_keys; k += {n_slices}) {{
                int idx = idxs[k];
                if (idx < 0) continue;
                float w = scores[k];
                uint4 w4 = reinterpret_cast<const device uint4*>(kv + uint(idx) * {head_dim})[blk];
                uint ws[4] = {{w4.x, w4.y, w4.z, w4.w}};
                for (uint j = 0; j < 4; ++j) {{
                    acc[2 * j] += w * as_type<float>(ws[j] << 16);
                    acc[2 * j + 1] += w * as_type<float>(ws[j] & 0xFFFF0000u);
                }}
            }}
            for (uint j = 0; j < 8; ++j) part_acc[slice * {head_dim} + blk * 8 + j] = acc[j];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < {head_dim}; d += {n_threads}) {{
            float total = 0.0f;
            for (uint sl = 0; sl < {n_slices}; ++sl) total += part_acc[sl * {head_dim} + d];
            float vs = float(bfloat16_t(total));
            out[head * {head_dim} + d] = T(vs / denom);
        }}
    """
    return mx.fast.metal_kernel(
        name=f"sparse_attn_decode_d{head_dim}_k{max_keys}_t{n_threads}",
        input_names=["q", "kv", "idxs", "sink_in", "dims", "scale_in"],
        output_names=["out"],
        source=source,
        header=HEADER,
    )


ATTN_THREADS = int(os.environ.get("CACHALOT_ATTN_THREADS", "256"))


def sparse_attention_decode_1d(q: mx.array, kv: mx.array, attn_sink: mx.array, idxs: mx.array, softmax_scale: float,
                               n_threads: int | None = None) -> mx.array:
    """
    q [H, D] bf16, kv [N, D] bf16, idxs int32 [K] (-1 = masked) -> [H, D] bf16.
    Same semantics as sparse_attn_mlx.sparse_attention for one position.
    """
    n_heads, head_dim = q.shape
    n_keys = int(idxs.shape[0])
    max_keys = 1024 if n_keys <= 1024 else ((n_keys + 255) // 256) * 256
    n_threads = ATTN_THREADS if n_threads is None else n_threads
    return _sparse_attention_kernel(head_dim, max_keys, n_threads)(
        inputs=[q, kv, idxs.astype(mx.int32), attn_sink.astype(mx.float32),
                mx.array([n_keys], dtype=mx.uint32), mx.array([float(softmax_scale)], dtype=mx.float32)],
        template=[("T", q.dtype)],
        grid=(n_heads * n_threads, 1, 1),
        threadgroup=(n_threads, 1, 1),
        output_shapes=[(n_heads * head_dim,)],
        output_dtypes=[q.dtype],
    )[0].reshape(n_heads, head_dim)


# ---------------------------------------------------------------------------
# Hyper-connection mixes for one token: norm + 24 dot products + Sinkhorn
# ---------------------------------------------------------------------------
@cache
def _hc_mixes_kernel(hc_mult: int, sinkhorn_iters: int):
    mix_hc = (2 + hc_mult) * hc_mult
    source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint n = dims[0];                       // hc_mult * hidden
        threadgroup float red[256 * {mix_hc + 1}];
        float acc[{mix_hc}];
        for (uint j = 0; j < {mix_hc}; ++j) acc[j] = 0.0f;
        float sq = 0.0f;
        for (uint i = tid; i < n; i += 256) {{
            float xv = float(x[i]);
            sq += xv * xv;
            for (uint j = 0; j < {mix_hc}; ++j) {{
                acc[j] += xv * float(hc_fn[j * n + i]);
            }}
        }}
        for (uint j = 0; j < {mix_hc}; ++j) red[j * 256 + tid] = acc[j];
        red[{mix_hc} * 256 + tid] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128; s > 0; s >>= 1) {{
            if (tid < s) {{
                for (uint j = 0; j <= {mix_hc}; ++j) red[j * 256 + tid] += red[j * 256 + tid + s];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0) {{
            float rs = metal::rsqrt(red[{mix_hc} * 256] / float(n) + norm_eps[0]);
            float mixes[{mix_hc}];
            for (uint j = 0; j < {mix_hc}; ++j) mixes[j] = red[j * 256] * rs;
            float eps = hc_eps[0];
            for (uint j = 0; j < {hc_mult}; ++j) {{
                float z = mixes[j] * hc_scale[0] + hc_base[j];
                pre[j] = 1.0f / (1.0f + metal::exp(-z)) + eps;
                uint idx = {hc_mult} + j;
                float z2 = mixes[idx] * hc_scale[1] + hc_base[idx];
                post[j] = 2.0f / (1.0f + metal::exp(-z2));
            }}
            float c[{hc_mult * hc_mult}];
            for (uint j = 0; j < {hc_mult * hc_mult}; ++j) {{
                uint idx = 2 * {hc_mult} + j;
                c[j] = mixes[idx] * hc_scale[2] + hc_base[idx];
            }}
            for (uint r = 0; r < {hc_mult}; ++r) {{
                float rm = -INFINITY;
                for (uint k = 0; k < {hc_mult}; ++k) rm = metal::max(rm, c[r * {hc_mult} + k]);
                float sum = 0.0f;
                for (uint k = 0; k < {hc_mult}; ++k) {{ float v = metal::exp(c[r * {hc_mult} + k] - rm); c[r * {hc_mult} + k] = v; sum += v; }}
                for (uint k = 0; k < {hc_mult}; ++k) c[r * {hc_mult} + k] = c[r * {hc_mult} + k] / sum + eps;
            }}
            for (uint k = 0; k < {hc_mult}; ++k) {{
                float cs = 0.0f;
                for (uint r = 0; r < {hc_mult}; ++r) cs += c[r * {hc_mult} + k];
                for (uint r = 0; r < {hc_mult}; ++r) c[r * {hc_mult} + k] = c[r * {hc_mult} + k] / (cs + eps);
            }}
            for (uint it = 1; it < {sinkhorn_iters}; ++it) {{
                for (uint r = 0; r < {hc_mult}; ++r) {{
                    float rsum = 0.0f;
                    for (uint k = 0; k < {hc_mult}; ++k) rsum += c[r * {hc_mult} + k];
                    for (uint k = 0; k < {hc_mult}; ++k) c[r * {hc_mult} + k] = c[r * {hc_mult} + k] / (rsum + eps);
                }}
                for (uint k = 0; k < {hc_mult}; ++k) {{
                    float cs = 0.0f;
                    for (uint r = 0; r < {hc_mult}; ++r) cs += c[r * {hc_mult} + k];
                    for (uint r = 0; r < {hc_mult}; ++r) c[r * {hc_mult} + k] = c[r * {hc_mult} + k] / (cs + eps);
                }}
            }}
            for (uint j = 0; j < {hc_mult * hc_mult}; ++j) comb[j] = c[j];
        }}
    """
    return mx.fast.metal_kernel(
        name=f"hc_mixes_1d_h{hc_mult}_i{sinkhorn_iters}",
        input_names=["x", "hc_fn", "hc_scale", "hc_base", "dims", "norm_eps", "hc_eps"],
        output_names=["pre", "post", "comb"],
        source=source,
        header=HEADER,
    )


def hc_mixes_1d(x: mx.array, hc_fn: mx.array, hc_scale: mx.array, hc_base: mx.array, *, norm_eps: float,
                hc_mult: int, sinkhorn_iters: int, hc_eps: float):
    """hc_mixes for one token (x [hc, hidden]) in a single launch."""
    n = x.size
    pre, post, comb = _hc_mixes_kernel(hc_mult, sinkhorn_iters)(
        inputs=[x.reshape(-1), hc_fn, hc_scale.astype(mx.float32), hc_base.astype(mx.float32),
                mx.array([n], dtype=mx.uint32), mx.array([norm_eps], dtype=mx.float32), mx.array([hc_eps], dtype=mx.float32)],
        template=[],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(hc_mult,), (hc_mult,), (hc_mult, hc_mult)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return pre, post, comb


def sparse_attention_decode(q: mx.array, kv: mx.array, attn_sink: mx.array, idxs: mx.array, softmax_scale: float) -> mx.array:
    """q [1, H, D] / idxs [1, K] as sparse_attention takes them; returns [1, H, D]."""
    if FUSED_DECODE and q.shape[0] == 1 and q.shape[-1] % 256 == 0:
        return sparse_attention_decode_1d(q[0], kv, attn_sink, idxs[0], softmax_scale)[None]
    from cachalot.model.sparse_attn_mlx import sparse_attention

    return sparse_attention(q, kv, attn_sink, idxs, softmax_scale)


def hc_mixes_decode(x, hc_fn, hc_scale, hc_base, *, norm_eps, hc_mult, sinkhorn_iters, hc_eps):
    if FUSED_DECODE and x.ndim == 2:
        return hc_mixes_1d(x, hc_fn, hc_scale, hc_base, norm_eps=norm_eps, hc_mult=hc_mult,
                           sinkhorn_iters=sinkhorn_iters, hc_eps=hc_eps)
    from cachalot.model.hyper_connection_mlx import hc_mixes

    return hc_mixes(x, hc_fn, hc_scale, hc_base, norm_eps=norm_eps, hc_mult=hc_mult,
                    sinkhorn_iters=sinkhorn_iters, hc_eps=hc_eps)
