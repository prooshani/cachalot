"""
Fused multi-expert kernels for layer-major prefill (EXPERIMENT, not wired in).

Status (2026-09-15): numerically matches the decode GEMV path to fp32 ulp, but
runs ~1.5x slower per expert than the affine-8-bit mx.quantized_matmul path
across ROWS/TILE settings (best 2.75 ms vs 1.83 ms for a 4-expert chunk);
see benchmarks/micro_prefill_fused.py. Kept as an exact reference and a
starting point for a tiled/simdgroup-matrix rewrite.

A chunk of C experts (default 4) is processed in two launches:

  gate_up : for every (expert, token-row) pair, gate = W1 x, up = W3 x with
            FP4 weights dequantized inline, then
            hidden = silu(min(gate, L)) * clip(up, -L, L) * router_weight    (fp32)
  down    : out = W2 hidden                                                   (fp32)

Work decomposition: one simdgroup owns ROWS output rows of one expert and
loops over that expert's tokens in tiles of TILE; the 32 lanes split the
K dimension by 32-element blocks (lane owns blocks lane, lane+32, ...),
exactly like fp4_gemv_metal, and elements are accumulated in the same
order (x * table * scale), so per-element results are bit-identical to the
decode GEMV path. x is read once per (simdgroup, token, block) and reused
across the simdgroup's rows.

Chunk experts are passed as separate buffers (slots live in separate MLX
arrays); Metal's ~31 buffer bindings bound the chunk size.
"""

from __future__ import annotations

import os
from functools import cache

import mlx.core as mx

FP4_HEADER = """
#include <metal_stdlib>
using namespace metal;
constant float fp4_table[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,
     2.0f,  3.0f,  4.0f,  6.0f,
     0.0f, -0.5f, -1.0f, -1.5f,
    -2.0f, -3.0f, -4.0f, -6.0f
};
"""

CHUNK = 4      # experts per launch
ROWS = int(os.environ.get("CACHALOT_FUSED_ROWS", "4"))       # output rows per simdgroup
TILE = int(os.environ.get("CACHALOT_FUSED_TILE", "8"))       # tokens per pass


def _switch(prefix: str, n: int) -> str:
    cases = "\n".join(
        f"            case {e}: packed = {prefix}_{e}; scales = s{prefix[1]}_{e}; break;" for e in range(n)
    )
    return f"""
        const device uchar* packed;
        const device uchar* scales;
        switch (expert) {{
{cases}
            default: packed = {prefix}_0; scales = s{prefix[1]}_0; break;
        }}
"""


def _dot_tile(k: int, x_loader: str, acc: str, packed: str, scales: str) -> str:
    """
    Accumulate acc[row][tok] += dot(x_tok_block, dequant(W[n0+row] block))
    for the lane's blocks, ROWS rows and up to TILE tokens. `x_loader` must
    fill float xv[32] for token index (t0 + t) and block start k_base.
    """
    return f"""
        for (uint block = lane; block < {k // 32}; block += 32) {{
            uint k_base = block * 32;
            float wv[{ROWS}][32];
            for (uint i = 0; i < {ROWS}; ++i) {{
                uint n = n0 + i;
                uchar scale_raw = {scales}[n * {k // 32} + block];
                float scale = metal::exp2(float(int(scale_raw) - 127));
                const device uint4* prow = reinterpret_cast<const device uint4*>({packed} + n * {k // 2});
                uint4 v = prow[block];
                uint words[4] = {{v.x, v.y, v.z, v.w}};
                for (uint wi = 0; wi < 4; ++wi) {{
                    uint word = words[wi];
                    for (uint bb = 0; bb < 4; ++bb) {{
                        uint byte = (word >> (8 * bb)) & 0xFF;
                        uint e0 = (wi * 4 + bb) * 2;
                        wv[i][e0] = fp4_table[byte & 0x0F] * scale;
                        wv[i][e0 + 1] = fp4_table[(byte >> 4) & 0x0F] * scale;
                    }}
                }}
            }}
            for (uint t = 0; t < {TILE}; ++t) {{
                if (t0 + t >= r_end) break;
                float xv[32];
                {x_loader}
                for (uint e0 = 0; e0 < 32; ++e0) {{
                    for (uint i = 0; i < {ROWS}; ++i) {{
                        {acc}[i][t] += xv[e0] * wv[i][e0];
                    }}
                }}
            }}
        }}
"""


X_LOADER_BF16 = """
                {
                    const device uint4* xr = reinterpret_cast<const device uint4*>(
                        x + row_tok[t0 + t] * K_IN + k_base);
                    for (uint q = 0; q < 4; ++q) {
                        uint4 w4 = xr[q];
                        uint ws[4] = {w4.x, w4.y, w4.z, w4.w};
                        for (uint j = 0; j < 4; ++j) {
                            xv[q * 8 + j * 2] = as_type<float>(ws[j] << 16);
                            xv[q * 8 + j * 2 + 1] = as_type<float>(ws[j] & 0xFFFF0000u);
                        }
                    }
                }
"""

X_LOADER_F32 = """
                {
                    const device float4* xr = reinterpret_cast<const device float4*>(
                        hidden + (t0 + t) * K_IN + k_base);
                    for (uint q = 0; q < 8; ++q) {
                        float4 f = xr[q];
                        xv[q * 4] = f.x; xv[q * 4 + 1] = f.y; xv[q * 4 + 2] = f.z; xv[q * 4 + 3] = f.w;
                    }
                }
"""


@cache
def _gate_up_kernel(n_experts: int, in_features: int, out_features: int):
    names = ["x", "row_tok", "row_weight", "expert_row_start", "limit"]
    names += [f"w1_{e}" for e in range(n_experts)] + [f"s1_{e}" for e in range(n_experts)]
    names += [f"w3_{e}" for e in range(n_experts)] + [f"s3_{e}" for e in range(n_experts)]
    row_groups = out_features // ROWS
    x_loader = X_LOADER_BF16.replace("K_IN", str(in_features))
    source = f"""
        uint global_tid = thread_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = global_tid >> 5;
        uint expert = sg / {row_groups};
        uint n0 = (sg % {row_groups}) * {ROWS};
        if (expert >= {n_experts}) return;
        uint r_begin = expert_row_start[expert];
        uint r_end = expert_row_start[expert + 1];
        float L = limit[0];

        for (uint t0 = r_begin; t0 < r_end; t0 += {TILE}) {{
            float acc_g[{ROWS}][{TILE}];
            float acc_u[{ROWS}][{TILE}];
            for (uint i = 0; i < {ROWS}; ++i)
                for (uint t = 0; t < {TILE}; ++t) {{ acc_g[i][t] = 0.0f; acc_u[i][t] = 0.0f; }}
            {{
                {_switch("w1", n_experts)}
                {_dot_tile(in_features, x_loader, "acc_g", "packed", "scales")}
            }}
            {{
                {_switch("w3", n_experts)}
                {_dot_tile(in_features, x_loader, "acc_u", "packed", "scales")}
            }}
            for (uint i = 0; i < {ROWS}; ++i) {{
                for (uint t = 0; t < {TILE}; ++t) {{
                    float g = simd_sum(acc_g[i][t]);
                    float u = simd_sum(acc_u[i][t]);
                    if (lane == 0 && t0 + t < r_end) {{
                        if (L > 0.0f) {{
                            u = metal::min(metal::max(u, -L), L);
                            g = metal::min(g, L);
                        }}
                        float h = g / (1.0f + metal::exp(-g)) * u;
                        hidden[(t0 + t) * {out_features} + n0 + i] = h * row_weight[t0 + t];
                    }}
                }}
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"prefill_gate_up_c{n_experts}_k{in_features}_n{out_features}",
        input_names=names,
        output_names=["hidden"],
        source=source,
        header=FP4_HEADER,
    )


@cache
def _down_kernel(n_experts: int, in_features: int, out_features: int):
    names = ["hidden", "expert_row_start"]
    names += [f"w2_{e}" for e in range(n_experts)] + [f"s2_{e}" for e in range(n_experts)]
    row_groups = out_features // ROWS
    x_loader = X_LOADER_F32.replace("K_IN", str(in_features))
    source = f"""
        uint global_tid = thread_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = global_tid >> 5;
        uint expert = sg / {row_groups};
        uint n0 = (sg % {row_groups}) * {ROWS};
        if (expert >= {n_experts}) return;
        uint r_begin = expert_row_start[expert];
        uint r_end = expert_row_start[expert + 1];

        for (uint t0 = r_begin; t0 < r_end; t0 += {TILE}) {{
            float acc[{ROWS}][{TILE}];
            for (uint i = 0; i < {ROWS}; ++i)
                for (uint t = 0; t < {TILE}; ++t) acc[i][t] = 0.0f;
            {{
                {_switch("w2", n_experts)}
                {_dot_tile(in_features, x_loader, "acc", "packed", "scales")}
            }}
            for (uint i = 0; i < {ROWS}; ++i) {{
                for (uint t = 0; t < {TILE}; ++t) {{
                    float v = simd_sum(acc[i][t]);
                    if (lane == 0 && t0 + t < r_end) {{
                        out[(t0 + t) * {out_features} + n0 + i] = v;
                    }}
                }}
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"prefill_down_c{n_experts}_k{in_features}_n{out_features}",
        input_names=names,
        output_names=["out"],
        source=source,
        header=FP4_HEADER,
    )


def fused_experts_chunk(
    x: mx.array,
    experts,
    row_tok: mx.array,
    row_weight: mx.array,
    expert_row_start: mx.array,
    *,
    hidden_size: int = 5120,
    intermediate: int = 2304,
    swiglu_limit: float = 10.0,
) -> mx.array:
    """
    x: [T, hidden_size] (bf16) chunk activations
    experts: C resident experts (slot arrays)
    row_tok: int32 [R] token index of each row (rows grouped by expert, in expert order)
    row_weight: fp32 [R] router weight per row
    expert_row_start: int32 [C+1] row offsets per expert
    Returns fp32 [R, hidden_size] routed outputs (already weighted).
    """
    c = len(experts)
    n_rows = int(row_tok.shape[0])
    inputs = [x, row_tok, row_weight, expert_row_start, mx.array([float(swiglu_limit)], dtype=mx.float32)]
    inputs += [e.w1_weight for e in experts] + [e.w1_scale for e in experts]
    inputs += [e.w3_weight for e in experts] + [e.w3_scale for e in experts]
    sg = c * (intermediate // ROWS)
    hidden = _gate_up_kernel(c, hidden_size, intermediate)(
        inputs=inputs,
        template=[],
        grid=(sg * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_rows * intermediate,)],
        output_dtypes=[mx.float32],
    )[0]
    inputs = [hidden, expert_row_start] + [e.w2_weight for e in experts] + [e.w2_scale for e in experts]
    sg = c * (hidden_size // ROWS)
    out = _down_kernel(c, intermediate, hidden_size)(
        inputs=inputs,
        template=[],
        grid=(sg * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_rows * hidden_size,)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(n_rows, hidden_size)
