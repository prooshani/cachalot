"""
FP4 expert GEMM on the simdgroup matrix units (prefill, few rows per expert).

Each simdgroup owns 8 output columns (expert rows n) and *all* activation rows
of the launch (up to 16 tiles of 8 = 128 rows), so every FP4 byte is read and
dequantized exactly once per launch. Per 32-element block a lane turns one
uint32 (8 nibbles) of its column into 8 bf16 values in a threadgroup scratch
tile; the tile is then consumed by `simdgroup_multiply_accumulate` against
bf16 8x8 activation tiles loaded straight from device memory. FP4 x E8M0
values are exact in bf16, and the accumulation is fp32, so the numerics are
those of a bf16 GEMM with fp32 accumulation - the same class as
`mx.quantized_matmul`; only the summation order differs.

    fp4_sgmm         : out[m, n] = bf16( sum_k x[m, k] * W[n, k] )
    fp4_sgmm_gate_up : hidden[m, n] = bf16( silu(min(g, L)) * clip(u, -L, L) * w[m] ),
                       g = bf16(x W1^T), u = bf16(x W3^T)

With N/8 simdgroups an expert GEMM (N = 2304) would fill a tenth of the GPU,
so K is split across simdgroups (fp32 partials [S, M, N]) and a small reduce
kernel sums the partials in a fixed order and applies the epilogue.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from cachalot.model.moe_fused_metal import FP4_HEADER

HIDDEN = 5120
INTERMEDIATE = 2304
MAX_TILES = 8            # activation rows per launch = 8 * MAX_TILES
SG_PER_TG = 4
TARGET_SIMDGROUPS = 2048  # split K until the launch has about this many simdgroups

_HEADER = FP4_HEADER + """
#include <metal_simdgroup_matrix>
"""


GROUP_BLOCKS = 4         # 32-element blocks dequantized per iteration (one uint4 per lane)


def _body(k: int, n: int, m_tiles: int, splits: int, mats: list[tuple[str, str, str]]) -> str:
    """
    mats: (packed, scales, tag) per weight matrix sharing the activation.
    Writes fp32 partial sums part_<tag>[split, m, col] for this simdgroup's K range.

    Lane (c, s) = (lane >> 2, lane & 3) owns column c of the simdgroup's 8 and
    block 4g+s of each group g: one uint4 load (32 nibbles) and one scale byte
    per iteration, dequantized into a [8][128] bf16 tile. The next group's
    bytes are loaded before the current group's MMAs so DRAM latency overlaps
    the matrix work. Activation tiles come straight from device memory.

    Variants measured and rejected on the M3 Ultra (per expert, 8 rows):
    staging the activation tile per threadgroup (no gain), 16-block groups
    with 64 B contiguous per lane (2.5x slower: register pressure, half the
    lanes idle in dequantization).
    """
    scale_k = k // 32
    groups = scale_k // GROUP_BLOCKS
    if groups * GROUP_BLOCKS != scale_k:
        raise ValueError(f"K={k} must be a multiple of {32 * GROUP_BLOCKS}")
    groups_per_split = (groups + splits - 1) // splits
    gk = 32 * GROUP_BLOCKS
    decl = "\n".join(
        f"        simdgroup_float8x8 C_{tag}[{m_tiles}];\n"
        f"        for (uint t = 0; t < {m_tiles}; ++t) C_{tag}[t] = simdgroup_float8x8(0.0f);"
        for _, _, tag in mats
    )

    def load(g: str) -> str:
        return "\n".join(f"""
            {{
                uint block = ({g}) * {GROUP_BLOCKS} + my_sub;
                nxt_scale_{tag} = {sc}[my_col * {scale_k} + block];
                nxt_w_{tag} = reinterpret_cast<const device uint4*>({pk} + my_col * (K / 2))[block];
            }}""" for pk, sc, tag in mats)

    dequant = "\n".join(f"""
            {{
                float scale = metal::exp2(float(int(cur_scale_{tag}) - 127));
                uint words[4] = {{cur_w_{tag}.x, cur_w_{tag}.y, cur_w_{tag}.z, cur_w_{tag}.w}};
                threadgroup bfloat16_t* dst = Wt_{tag} + my_row * {gk} + my_sub * 32;
                for (uint wi = 0; wi < 4; ++wi) {{
                    uint word = words[wi];
                    for (uint bb = 0; bb < 4; ++bb) {{
                        uint byte = (word >> (8 * bb)) & 0xFF;
                        dst[wi * 8 + 2 * bb] = bfloat16_t(fp4_table[byte & 0x0F] * scale);
                        dst[wi * 8 + 2 * bb + 1] = bfloat16_t(fp4_table[byte >> 4] * scale);
                    }}
                }}
            }}""" for _, _, tag in mats)
    regs = "\n".join(f"        uint4 nxt_w_{tag}; uchar nxt_scale_{tag}; uint4 cur_w_{tag}; uchar cur_scale_{tag};"
                      for _, _, tag in mats)
    rotate = "\n".join(f"            cur_w_{tag} = nxt_w_{tag}; cur_scale_{tag} = nxt_scale_{tag};" for _, _, tag in mats)
    mma = "\n".join(f"""
                    simdgroup_bfloat8x8 B_{tag};
                    simdgroup_load(B_{tag}, Wt_{tag} + step * 8, {gk}, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(C_{tag}[t], A, B_{tag}, C_{tag}[t]);""" for _, _, tag in mats)
    stores = "\n".join(f"            simdgroup_store(C_{tag}[t], Ct_{tag}, 8);" for _, _, tag in mats)
    scratch = "\n".join(
        f"        threadgroup bfloat16_t Wt_{tag}_all[{SG_PER_TG} * 8 * {gk}];\n"
        f"        threadgroup float Ct_{tag}_all[{SG_PER_TG} * 64];\n"
        f"        threadgroup bfloat16_t* Wt_{tag} = Wt_{tag}_all + sg * 8 * {gk};\n"
        f"        threadgroup float* Ct_{tag} = Ct_{tag}_all + sg * 64;"
        for _, _, tag in mats
    )
    values = "\n".join(
        f"                    part_{tag}[(split * n_rows + m) * {n} + col] = Ct_{tag}[e];" for _, _, tag in mats
    )
    return f"""
        constexpr uint K = {k};
        uint lane = thread_index_in_simdgroup;
        uint sg = thread_position_in_threadgroup.x >> 5;
        uint gsg = threadgroup_position_in_grid.x * {SG_PER_TG} + sg;
        uint split = gsg % {splits};
        uint col0 = (gsg / {splits}) * 8;   // first expert row (output column)
        if (col0 >= {n}) return;
        uint n_rows = dims[0];
        uint my_row = lane >> 2;          // 0..7: which of the 8 columns this lane dequantizes
        uint my_sub = lane & 3;           // 0..3: which block of the group
        uint my_col = metal::min(col0 + my_row, uint({n} - 1));
        uint g_begin = split * {groups_per_split};
        uint g_end = metal::min(g_begin + {groups_per_split}, uint({groups}));
{scratch}
{decl}
{regs}
        if (g_begin < g_end) {{
{load("g_begin")}
        }}
        for (uint g = g_begin; g < g_end; ++g) {{
{rotate}
            if (g + 1 < g_end) {{
{load("g + 1")}
            }}
{dequant}
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = 0; t < {m_tiles}; ++t) {{
                for (uint step = 0; step < {GROUP_BLOCKS * 4}; ++step) {{
                    simdgroup_bfloat8x8 A;
                    simdgroup_load(A, x + (t * 8) * K + g * {gk} + step * 8, K, ulong2(0, 0), false);
{mma}
                }}
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}
        for (uint t = 0; t < {m_tiles}; ++t) {{
{stores}
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {{
                uint m = t * 8 + (e >> 3);
                uint col = col0 + (e & 7);
                if (m < n_rows && col < {n}) {{
{values}
                }}
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}
    """


@cache
def _sgmm_kernel(k: int, n: int, m_tiles: int, splits: int):
    return mx.fast.metal_kernel(
        name=f"fp4_sgmm_k{k}_n{n}_t{m_tiles}_s{splits}",
        input_names=["x", "packed", "scales", "dims"],
        output_names=["part_w"],
        source=_body(k, n, m_tiles, splits, [("packed", "scales", "w")]),
        header=_HEADER,
    )


@cache
def _sgmm_gate_up_kernel(k: int, n: int, m_tiles: int, splits: int):
    return mx.fast.metal_kernel(
        name=f"fp4_sgmm_gate_up_k{k}_n{n}_t{m_tiles}_s{splits}",
        input_names=["x", "w1", "s1", "w3", "s3", "dims"],
        output_names=["part_g", "part_u"],
        source=_body(k, n, m_tiles, splits, [("w1", "s1", "g"), ("w3", "s3", "u")]),
        header=_HEADER,
    )


@cache
def _reduce_kernel(splits: int):
    """out[i] = bf16(sum_s part[s, i]) in fixed order."""
    source = f"""
        uint i = thread_position_in_grid.x;
        if (i >= count[0]) return;
        uint stride = count[0];
        float total = 0.0f;
        for (uint s = 0; s < {splits}; ++s) total += part[s * stride + i];
        out[i] = bfloat16_t(total);
    """
    return mx.fast.metal_kernel(
        name=f"fp4_sgmm_reduce_s{splits}", input_names=["part", "count"], output_names=["out"],
        source=source, header=_HEADER,
    )


@cache
def _reduce_gate_up_kernel(splits: int, n: int):
    source = f"""
        uint i = thread_position_in_grid.x;
        if (i >= count[0]) return;
        uint stride = count[0];
        float g = 0.0f;
        float u = 0.0f;
        for (uint s = 0; s < {splits}; ++s) {{
            g += part_g[s * stride + i];
            u += part_u[s * stride + i];
        }}
        // official: linears return bf16, gating math in fp32
        g = float(bfloat16_t(g));
        u = float(bfloat16_t(u));
        float L = limit[0];
        if (L > 0.0f) {{
            u = metal::min(metal::max(u, -L), L);
            g = metal::min(g, L);
        }}
        uint m = i / {n};
        float h = g / (1.0f + metal::exp(-g)) * u * router_w[m];
        hidden[i] = bfloat16_t(h);
    """
    return mx.fast.metal_kernel(
        name=f"fp4_sgmm_reduce_gate_up_s{splits}_n{n}",
        input_names=["part_g", "part_u", "router_w", "limit", "count"],
        output_names=["hidden"],
        source=source,
        header=_HEADER,
    )


def _splits(n: int, k: int) -> int:
    n_sg = (n + 7) // 8
    s = max(1, min(8, TARGET_SIMDGROUPS // n_sg))
    while (k // (32 * GROUP_BLOCKS)) % s:
        s -= 1
    return s


def _pad_rows(x: mx.array, tiles: int) -> mx.array:
    rows = tiles * 8
    if x.shape[0] == rows:
        return x
    return mx.concatenate([x, mx.zeros((rows - x.shape[0], x.shape[1]), dtype=x.dtype)], axis=0)


def _grid(n: int, splits: int) -> tuple[int, int, int]:
    n_sg = ((n + 7) // 8) * splits
    n_tg = (n_sg + SG_PER_TG - 1) // SG_PER_TG
    return (n_tg * SG_PER_TG * 32, 1, 1)


def fp4_sgmm(x: mx.array, packed: mx.array, scales: mx.array, *, out_features: int, in_features: int) -> mx.array:
    """x bf16 [M, K] @ dequant(W)^T -> bf16 [M, N]; M <= 8 * MAX_TILES per call."""
    m = x.shape[0]
    if m > 8 * MAX_TILES:
        parts = [fp4_sgmm(x[i:i + 8 * MAX_TILES], packed, scales, out_features=out_features, in_features=in_features)
                 for i in range(0, m, 8 * MAX_TILES)]
        return mx.concatenate(parts, axis=0)
    tiles = (m + 7) // 8
    splits = _splits(out_features, in_features)
    xp = _pad_rows(x.astype(mx.bfloat16), tiles)
    part = _sgmm_kernel(in_features, out_features, tiles, splits)(
        inputs=[xp, packed.reshape(-1), scales.reshape(-1), mx.array([m], dtype=mx.uint32)],
        template=[],
        grid=_grid(out_features, splits),
        threadgroup=(SG_PER_TG * 32, 1, 1),
        output_shapes=[(splits, m, out_features)],
        output_dtypes=[mx.float32],
    )[0]
    count = m * out_features
    return _reduce_kernel(splits)(
        inputs=[part, mx.array([count], dtype=mx.uint32)],
        template=[],
        grid=(count, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, out_features)],
        output_dtypes=[mx.bfloat16],
    )[0]


def fp4_sgmm_gate_up(x: mx.array, w1: mx.array, s1: mx.array, w3: mx.array, s3: mx.array, weights: mx.array, *,
                     swiglu_limit: float, in_features: int = HIDDEN, out_features: int = INTERMEDIATE) -> mx.array:
    """x bf16 [M, HIDDEN], weights fp32 [M] -> gated, router-weighted hidden bf16 [M, INTERMEDIATE]."""
    m = x.shape[0]
    if m > 8 * MAX_TILES:
        parts = [fp4_sgmm_gate_up(x[i:i + 8 * MAX_TILES], w1, s1, w3, s3, weights[i:i + 8 * MAX_TILES],
                                  swiglu_limit=swiglu_limit, in_features=in_features, out_features=out_features)
                 for i in range(0, m, 8 * MAX_TILES)]
        return mx.concatenate(parts, axis=0)
    tiles = (m + 7) // 8
    splits = _splits(out_features, in_features)
    xp = _pad_rows(x.astype(mx.bfloat16), tiles)
    part_g, part_u = _sgmm_gate_up_kernel(in_features, out_features, tiles, splits)(
        inputs=[xp, w1.reshape(-1), s1.reshape(-1), w3.reshape(-1), s3.reshape(-1), mx.array([m], dtype=mx.uint32)],
        template=[],
        grid=_grid(out_features, splits),
        threadgroup=(SG_PER_TG * 32, 1, 1),
        output_shapes=[(splits, m, out_features), (splits, m, out_features)],
        output_dtypes=[mx.float32, mx.float32],
    )
    count = m * out_features
    return _reduce_gate_up_kernel(splits, out_features)(
        inputs=[part_g, part_u, weights.astype(mx.float32), mx.array([float(swiglu_limit)], dtype=mx.float32),
                mx.array([count], dtype=mx.uint32)],
        template=[],
        grid=(count, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, out_features)],
        output_dtypes=[mx.bfloat16],
    )[0]


def routed_expert_forward_sgmm(
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
) -> mx.array:
    """Same contract as moe_prefill_batched.routed_expert_forward_batched: fp32 [M, HIDDEN]."""
    xb = x.astype(mx.bfloat16)
    hidden = fp4_sgmm_gate_up(xb, w1_packed, w1_scales, w3_packed, w3_scales, weights, swiglu_limit=swiglu_limit)
    return fp4_sgmm(hidden, w2_packed, w2_scales, out_features=HIDDEN, in_features=INTERMEDIATE).astype(mx.float32)
