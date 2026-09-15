"""
Fused DeepSeek V4.1 router for one decode token (two Metal launches).

The MLX chain (matmul, softplus, sqrt, add, argsort, take, sum, div, mul)
costs ~0.35 ms of GPU time per layer, dominated by the argsort of 384
scores and the ~9 launches. Here:

    router_scores : one simdgroup per expert row, fp32 dot(x, w[row]) / temp,
                    scores = sqrt(softplus(.))
    router_topk   : one threadgroup; k rounds of argmax over scores + bias
                    with the stable-argsort tie rule (equal value: larger
                    index wins), emitting indices in ascending selection
                    order exactly like `argsort(...)[-k:]`, then the
                    normalized, scaled weights.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from cachalot.model.router_mlx import RouterResult

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


@cache
def _scores_kernel(hidden: int):
    source = f"""
        uint global_tid = thread_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        uint row = global_tid >> 5;
        if (row >= n_experts[0]) return;
        const device T* wrow = weight + row * {hidden};
        float acc = 0.0f;
        for (uint k = lane; k < {hidden}; k += 32) {{
            acc += float(x[k]) * float(wrow[k]);
        }}
        acc = simd_sum(acc);
        if (lane == 0) {{
            float s = acc / gate_temp[0];
            // softplus(s) = log1p(exp(s)), numerically stable form
            float sp = metal::max(s, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(s)));
            scores[row] = metal::sqrt(sp);
        }}
    """
    return mx.fast.metal_kernel(
        name=f"router_scores_k{hidden}",
        input_names=["x", "weight", "n_experts", "gate_temp"],
        output_names=["scores"],
        source=source,
        header=_HEADER,
    )


@cache
def _topk_kernel(n_threads: int, topk: int):
    source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint n = n_experts[0];
        threadgroup float best_v[{n_threads}];
        threadgroup uint best_i[{n_threads}];
        float sel = (tid < n) ? scores[tid] + bias[tid] : -INFINITY;
        bool taken = false;
        for (uint r = 0; r < {topk}; ++r) {{
            best_v[tid] = taken ? -INFINITY : sel;
            best_i[tid] = tid;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = {n_threads // 2}; s > 0; s >>= 1) {{
                if (tid < s) {{
                    float ov = best_v[tid + s];
                    uint oi = best_i[tid + s];
                    // stable argsort tie rule: equal values keep index order,
                    // so the larger index is the later (bigger) element
                    if (ov > best_v[tid] || (ov == best_v[tid] && oi > best_i[tid])) {{
                        best_v[tid] = ov;
                        best_i[tid] = oi;
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            uint winner = best_i[0];
            if (tid == winner) taken = true;
            if (tid == 0) {{
                indices[{topk} - 1 - r] = int(winner);
                weights[{topk} - 1 - r] = scores[winner];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0) {{
            float total = 0.0f;
            for (uint j = 0; j < {topk}; ++j) total += weights[j];
            float denom = total + 1e-20f;
            for (uint j = 0; j < {topk}; ++j) weights[j] = weights[j] / denom * route_scale[0];
        }}
    """
    return mx.fast.metal_kernel(
        name=f"router_topk_t{n_threads}_k{topk}",
        input_names=["scores", "bias", "n_experts", "route_scale"],
        output_names=["indices", "weights"],
        source=source,
        header=_HEADER,
    )


def route_topk_fused(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    *,
    topk: int = 6,
    gate_temp: float = 1.0,
    route_scale: float = 1.5,
    norm_topk_prob: bool = True,
) -> RouterResult:
    """Same contract as router_mlx.route_topk (1-D x); norm_topk_prob must be True."""
    if not norm_topk_prob:
        raise ValueError("route_topk_fused implements norm_topk_prob=True only")
    n_experts, hidden = weight.shape
    if x.ndim != 1 or x.size != hidden:
        raise ValueError(f"x must be 1-D with {hidden} elements, got {x.shape}")
    scores = _scores_kernel(hidden)(
        inputs=[x, weight, mx.array([n_experts], dtype=mx.uint32), mx.array([float(gate_temp)], dtype=mx.float32)],
        template=[("T", weight.dtype)],
        grid=(n_experts * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_experts,)],
        output_dtypes=[mx.float32],
    )[0]
    n_threads = 1
    while n_threads < n_experts:
        n_threads *= 2
    n_threads = max(n_threads, 32)
    indices, weights = _topk_kernel(n_threads, topk)(
        inputs=[scores, bias.astype(mx.float32), mx.array([n_experts], dtype=mx.uint32),
                mx.array([float(route_scale)], dtype=mx.float32)],
        template=[],
        grid=(n_threads, 1, 1),
        threadgroup=(n_threads, 1, 1),
        output_shapes=[(topk,), (topk,)],
        output_dtypes=[mx.int32, mx.float32],
    )
    return RouterResult(indices=indices, weights=weights, scores=scores)
