"""
Row-tiled FP4 expert kernels for prefill (few tokens per expert).

In a 512-token prefill each routed expert sees ~8 tokens on average, so the
expert "GEMM" is really a multi-row GEMV: the cost is streaming the 18.8 MB of
FP4 weights, not arithmetic. The affine-8 path (fp4_affine8_metal.py) repacks
the weights (read 1x, write 2x) and then `mx.quantized_matmul` reads the 8-bit
copy (2x) - about five times the bytes of the FP4 tensor. These kernels read
the FP4 bytes exactly once per row tile and dequantize in registers:

    fp4_gate_up_rows : hidden[m, n] = bf16( silu(min(gate, L)) * clip(up, -L, L) * w[m] )
                       gate/up = bf16( sum_k x[m, k] * W1/W3[n, k] )   (one simdgroup per n)
    fp4_gemm_rows    : out[m, n] = bf16( sum_k h[m, k] * W2[n, k] )

Rows are processed in tiles of MT (8/16/32) so each lane keeps MT fp32
accumulators; the weight row is loaded once per tile as uint4 vectors.
Numerics: fp32 accumulation like the affine-8 GEMM, bf16 rounding at the
same points as the official model; only the fp32 summation order differs.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from cachalot.model.moe_fused_metal import FP4_HEADER

HIDDEN = 5120
INTERMEDIATE = 2304


ROWS_PER_TG = 8          # simdgroups (output rows) per threadgroup
CHUNK = 1024             # activation elements staged per chunk (32 blocks, one per lane)


def _kernel_body(mt: int, k: int, n: int, mats: list[tuple[str, str, str]], epilogue: str) -> str:
    """
    Shared body: a threadgroup owns ROWS_PER_TG consecutive output rows and a
    tile of MT activation rows. The activation tile is staged through
    threadgroup memory in CHUNK-element pieces so its bytes are read from L2
    once per threadgroup instead of once per output row. Each lane owns one
    32-element block of the chunk, dequantizes it into registers once per
    weight matrix and accumulates MT dot products.

    mats: (packed_name, scales_name, acc_name) per weight matrix sharing x.
    epilogue: code run per (mi, m) after the simd reductions; has `mi`, `m`,
    `row`, `lane` and the reduced totals `tot_<acc>` in scope.
    """
    n_chunks = (k + CHUNK - 1) // CHUNK
    decl = "\n".join(f"        float {acc}[{mt}];\n        for (uint mi = 0; mi < {mt}; ++mi) {acc}[mi] = 0.0f;" for _, _, acc in mats)
    ptrs = "\n".join(
        f"        const device uint4* prow_{acc} = reinterpret_cast<const device uint4*>({pk} + row * (K / 2));\n"
        f"        const device uchar* srow_{acc} = {sc} + row * SCALE_K;"
        for pk, sc, acc in mats
    )
    per_mat = "\n".join(f"""
                {{
                    float scale = metal::exp2(float(int(srow_{acc}[block]) - 127));
                    uint4 v = prow_{acc}[block];
                    uint words[4] = {{v.x, v.y, v.z, v.w}};
                    float wv[32];
                    for (uint wi = 0; wi < 4; ++wi) {{
                        uint word = words[wi];
                        for (uint bb = 0; bb < 4; ++bb) {{
                            uint byte = (word >> (8 * bb)) & 0xFF;
                            wv[wi * 8 + 2 * bb] = fp4_table[byte & 0x0F] * scale;
                            wv[wi * 8 + 2 * bb + 1] = fp4_table[byte >> 4] * scale;
                        }}
                    }}
                    for (uint mi = 0; mi < {mt}; ++mi) {{
                        const threadgroup uint4* xr = xs + mi * {CHUNK // 8} + lane * 4;
                        float a = 0.0f;
                        for (uint wi = 0; wi < 4; ++wi) {{
                            uint4 xv = xr[wi];
                            uint xw[4] = {{xv.x, xv.y, xv.z, xv.w}};
                            for (uint bb = 0; bb < 4; ++bb) {{
                                a += as_type<float>(xw[bb] << 16) * wv[wi * 8 + 2 * bb];
                                a += as_type<float>(xw[bb] & 0xFFFF0000u) * wv[wi * 8 + 2 * bb + 1];
                            }}
                        }}
                        {acc}[mi] += a;
                    }}
                }}""" for _, _, acc in mats)
    totals = "\n".join(f"            float tot_{acc} = simd_sum({acc}[mi]);" for _, _, acc in mats)
    return f"""
        constexpr uint K = {k};
        constexpr uint SCALE_K = {k // 32};
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = tid >> 5;
        uint row = threadgroup_position_in_grid.x * {ROWS_PER_TG} + sg;
        bool row_ok = row < {n};
        uint n_rows = dims[0];
        uint m0 = threadgroup_position_in_grid.y * {mt};

        threadgroup uint4 xs[{mt} * {CHUNK // 8}];
{decl}
        uint safe_row = row_ok ? row : 0;
        {{
            uint row = safe_row;
{ptrs}
            for (uint c = 0; c < {n_chunks}; ++c) {{
                uint k_start = c * {CHUNK};
                uint chunk = metal::min(uint({CHUNK}), K - k_start);
                uint chunk_vec = chunk / 8;                       // uint4 vectors per row
                // stage x[m0 .. m0+MT, k_start .. k_start+chunk] cooperatively
                for (uint i = tid; i < {mt} * chunk_vec; i += {ROWS_PER_TG * 32}) {{
                    uint mi = i / chunk_vec;
                    uint vi = i - mi * chunk_vec;
                    uint m = metal::min(m0 + mi, n_rows - 1);
                    xs[mi * {CHUNK // 8} + vi] =
                        reinterpret_cast<const device uint4*>(x + m * K + k_start)[vi];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane < chunk / 32) {{
                    uint block = k_start / 32 + lane;
{per_mat}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }}
        for (uint mi = 0; mi < {mt}; ++mi) {{
{totals}
            uint m = m0 + mi;
            if (row_ok && lane == mi % 32 && m < n_rows) {{
{epilogue}
            }}
        }}
    """


@cache
def _gemm_rows_kernel(mt: int, k: int, n: int):
    source = _kernel_body(mt, k, n, [("packed", "scales", "acc")],
                          f"                out[m * {n} + row] = bfloat16_t(tot_acc);")
    return mx.fast.metal_kernel(
        name=f"fp4_gemm_rows_mt{mt}_k{k}_n{n}",
        input_names=["x", "packed", "scales", "dims"],
        output_names=["out"],
        source=source,
        header=FP4_HEADER,
    )


@cache
def _gate_up_rows_kernel(mt: int, k: int, n: int):
    epilogue = f"""
                // official: linears return bf16, gating math in fp32
                float g = float(bfloat16_t(tot_acc_g));
                float u = float(bfloat16_t(tot_acc_u));
                float L = limit[0];
                if (L > 0.0f) {{
                    u = metal::min(metal::max(u, -L), L);
                    g = metal::min(g, L);
                }}
                float h = g / (1.0f + metal::exp(-g)) * u * router_w[m];
                hidden[m * {n} + row] = bfloat16_t(h);"""
    source = _kernel_body(mt, k, n, [("w1", "s1", "acc_g"), ("w3", "s3", "acc_u")], epilogue)
    return mx.fast.metal_kernel(
        name=f"fp4_gate_up_rows_mt{mt}_k{k}_n{n}",
        input_names=["x", "w1", "s1", "w3", "s3", "router_w", "limit", "dims"],
        output_names=["hidden"],
        source=source,
        header=FP4_HEADER,
    )


def _row_tile(m: int) -> int:
    if m <= 4:
        return 4
    if m <= 8:
        return 8
    return 16


def fp4_gemm_rows(x: mx.array, packed: mx.array, scales: mx.array, *, out_features: int, in_features: int,
                  row_tile: int | None = None) -> mx.array:
    """x bf16 [M, K] @ dequant(W)^T -> bf16 [M, N]."""
    m = x.shape[0]
    mt = _row_tile(m) if row_tile is None else row_tile
    return _gemm_rows_kernel(mt, in_features, out_features)(
        inputs=[x, packed.reshape(-1), scales.reshape(-1), mx.array([m], dtype=mx.uint32)],
        template=[],
        grid=(((out_features + ROWS_PER_TG - 1) // ROWS_PER_TG) * ROWS_PER_TG * 32, (m + mt - 1) // mt, 1),
        threadgroup=(ROWS_PER_TG * 32, 1, 1),
        output_shapes=[(m, out_features)],
        output_dtypes=[mx.bfloat16],
    )[0]


def fp4_gate_up_rows(x: mx.array, w1: mx.array, s1: mx.array, w3: mx.array, s3: mx.array, weights: mx.array, *,
                     swiglu_limit: float, in_features: int = HIDDEN, out_features: int = INTERMEDIATE,
                     row_tile: int | None = None) -> mx.array:
    """x bf16 [M, HIDDEN], weights fp32 [M] -> hidden bf16 [M, INTERMEDIATE] (gated, router-weighted)."""
    m = x.shape[0]
    mt = _row_tile(m) if row_tile is None else row_tile
    return _gate_up_rows_kernel(mt, in_features, out_features)(
        inputs=[x, w1.reshape(-1), s1.reshape(-1), w3.reshape(-1), s3.reshape(-1), weights.astype(mx.float32),
                mx.array([float(swiglu_limit)], dtype=mx.float32), mx.array([m], dtype=mx.uint32)],
        template=[],
        grid=(((out_features + ROWS_PER_TG - 1) // ROWS_PER_TG) * ROWS_PER_TG * 32, (m + mt - 1) // mt, 1),
        threadgroup=(ROWS_PER_TG * 32, 1, 1),
        output_shapes=[(m, out_features)],
        output_dtypes=[mx.bfloat16],
    )[0]


def routed_expert_forward_fp4_rows(
    x: mx.array,
    *,
    w1_packed: mx.array,
    w1_scales: mx.array,
    w2_packed: mx.array,
    w2_scales: mx.array,
    w3_packed: mx.array,
    w3_scales: mx.array,
    weights: mx.array,
    swiglu_limit: float = 10.0,
    row_tile: int | None = None,
) -> mx.array:
    """Same contract as moe_prefill_batched.routed_expert_forward_batched: fp32 [M, HIDDEN]."""
    xb = x.astype(mx.bfloat16)
    hidden = fp4_gate_up_rows(xb, w1_packed, w1_scales, w3_packed, w3_scales, weights,
                              swiglu_limit=swiglu_limit, row_tile=row_tile)
    return fp4_gemm_rows(hidden, w2_packed, w2_scales, out_features=HIDDEN, in_features=INTERMEDIATE,
                         row_tile=row_tile).astype(mx.float32)
