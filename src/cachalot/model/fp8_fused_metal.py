"""
FP8 (E4M3) linear for decode with the activation quantization fused into
the GEMV kernel.

The unfused path runs ~8 kernels to quantize x (abs/max/log2/ceil/pow/div/
clip/to_fp8) before every FP8 GEMV; there are seven such linears per layer.
Here each lane owns whole 32-element blocks of the activation, so it can
compute the official act_quant for its blocks locally:

    amax  = max(max|x_block|, 1e-4)
    scale = 2 ** ceil(log2(amax / 448))
    q     = E4M3_rne(clip(x / scale, -448, 448))

and dot the re-decoded q with the E4M3 weights, exactly as fp8_gemv_metal
does with a pre-quantized activation. The E4M3 rounding matches mx.to_fp8
(round-to-nearest-even, subnormals, saturating clamp is never reached).
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from cachalot.model.fp8_gemv_metal import BLOCK_SIZE

E4M3_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

// float -> E4M3FN bits, round to nearest even, |x| <= 448 assumed (clipped by caller).
inline uint f32_to_e4m3(float f) {
    uint bits = as_type<uint>(f);
    uint sign = (bits >> 31) & 1u;
    float a = metal::abs(f);
    if (a == 0.0f) return sign << 7;
    // scale so that the E4M3 quantum becomes an integer step: values are
    // rounded on a grid of 2^(e-3) for normals (e = floor(log2 a), e >= -6)
    // and 2^-9 for subnormals (a < 2^-6).
    int e = int(metal::floor(metal::log2(a)));
    if (e < -6) e = -6;                 // subnormal range shares the 2^-9 quantum
    float quantum = metal::exp2(float(e - 3));
    float r = metal::rint(a / quantum);   // RNE on the grid
    float v = r * quantum;
    if (v >= 480.0f) v = 448.0f;          // cannot happen after the 448 clip
    // re-derive fields from v
    uint vb = as_type<uint>(v);
    int ve = int((vb >> 23) & 0xFF) - 127;
    uint mant;
    uint ebits;
    if (ve < -6) {                        // subnormal: v = mant * 2^-9
        ebits = 0u;
        mant = uint(metal::rint(v / metal::exp2(-9.0f)));
    } else {
        ebits = uint(ve + 7);
        mant = (vb >> 20) & 0x7u;
    }
    return (sign << 7) | (ebits << 3) | mant;
}

// E4M3FN bits -> float
inline float e4m3_to_f32(uint b) {
    uint sign = (b >> 7) & 1u;
    uint ebits = (b >> 3) & 0xFu;
    uint mant = b & 0x7u;
    float v;
    if (ebits == 0u) {
        v = float(mant) * metal::exp2(-9.0f);
    } else {
        v = (1.0f + float(mant) / 8.0f) * metal::exp2(float(int(ebits) - 7));
    }
    return sign ? -v : v;
}
"""


@cache
def _fused_kernel(in_features: int):
    if in_features % BLOCK_SIZE != 0:
        raise ValueError("in_features must be a multiple of 32")
    scale_k = in_features // BLOCK_SIZE
    source = f"""
        uint global_tid = thread_position_in_grid.x;
        uint row = global_tid >> 5;
        uint lane = thread_index_in_simdgroup;
        if (row >= n_rows[0]) return;

        const device uchar* wrow = weight + row * {in_features};
        uint weight_scale_base = (row >> 5) * {scale_k};
        float acc = 0.0f;

        for (uint block = lane; block < {scale_k}; block += 32) {{
            uint k_base = block * 32;
            // official act_quant for this block
            float amax = 0.0f;
            for (uint j = 0; j < 32; ++j) {{
                amax = metal::max(amax, metal::abs(float(x[k_base + j])));
            }}
            amax = metal::max(amax, 1e-4f);
            float act_scale = metal::exp2(metal::ceil(metal::log2(amax / 448.0f)));

            uchar weight_scale_raw = weight_scales[weight_scale_base + block];
            float weight_scale = metal::exp2(float(int(weight_scale_raw) - 127));

            float block_acc = 0.0f;
            for (uint j = 0; j < 32; ++j) {{
                float xs = float(x[k_base + j]) / act_scale;
                xs = metal::min(metal::max(xs, -448.0f), 448.0f);
                float a = e4m3_to_f32(f32_to_e4m3(xs));
                float w = e4m3_to_f32(uint(wrow[k_base + j]));
                block_acc += a * w;
            }}
            acc += block_acc * act_scale * weight_scale;
        }}
        acc = simd_sum(acc);
        if (lane == 0) {{
            out[row] = acc;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"fp8_gemv_fused_k{in_features}",
        input_names=["x", "weight", "weight_scales", "n_rows"],
        output_names=["out"],
        source=source,
        header=E4M3_HEADER,
    )


@cache
def _e4m3_test_kernel():
    return mx.fast.metal_kernel(
        name="e4m3_roundtrip_test",
        input_names=["x"],
        output_names=["q", "back"],
        source="""
            uint i = thread_position_in_grid.x;
            uint b = f32_to_e4m3(x[i]);
            q[i] = uchar(b);
            back[i] = e4m3_to_f32(b);
        """,
        header=E4M3_HEADER,
    )


def e4m3_roundtrip_test(x: mx.array) -> tuple[mx.array, mx.array]:
    """For validation: (e4m3 bits, decoded float) from the in-kernel conversion."""
    n = x.size
    q, back = _e4m3_test_kernel()(
        inputs=[x.astype(mx.float32).reshape(-1)],
        template=[],
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,), (n,)],
        output_dtypes=[mx.uint8, mx.float32],
    )
    return q, back


def fp8_linear_fused(x: mx.array, weight: mx.array, weight_scales: mx.array) -> mx.array:
    """x [K] (bf16/fp32), weight uint8 [N, K], weight_scales uint8 [ceil(N/32), K/32] -> bf16 [N]."""
    n, k = weight.shape
    out = _fused_kernel(k)(
        inputs=[x, weight, weight_scales, mx.array([n], dtype=mx.uint32)],
        template=[],
        grid=(n * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]
    return out.astype(mx.bfloat16)


@cache
def _quantize_kernel(roundtrip: bool):
    """One thread per 32-element block: official act_quant, emitting either
    (q bytes, fp32 scale) or the dequantized round-trip values."""
    body_out = (
        "out[k_base + j] = OUT_T(a);" if roundtrip
        else "uint qb = f32_to_e4m3(xs); q[k_base + j] = uchar(qb); deq[k_base + j] = e4m3_to_f32(qb);"
    )
    source = f"""
        uint block = thread_position_in_grid.x;
        if (block >= n_blocks[0]) return;
        uint k_base = block * 32;
        float amax = 0.0f;
        for (uint j = 0; j < 32; ++j) {{
            amax = metal::max(amax, metal::abs(float(x[k_base + j])));
        }}
        amax = metal::max(amax, 1e-4f);
        float scale = metal::exp2(metal::ceil(metal::log2(amax / 448.0f)));
        for (uint j = 0; j < 32; ++j) {{
            float xs = float(x[k_base + j]) / scale;
            xs = metal::min(metal::max(xs, -448.0f), 448.0f);
            {"float a = e4m3_to_f32(f32_to_e4m3(xs)) * scale;" if roundtrip else ""}
            {body_out}
        }}
        {"" if roundtrip else "scales[block] = scale;"}
    """
    if roundtrip:
        return mx.fast.metal_kernel(
            name="fp8_act_roundtrip_fused",
            input_names=["x", "n_blocks"],
            output_names=["out"],
            source=source.replace("OUT_T", "T"),
            header=E4M3_HEADER,
        )
    return mx.fast.metal_kernel(
        name="fp8_act_quantize_fused",
        input_names=["x", "n_blocks"],
        output_names=["q", "scales", "deq"],
        source=source,
        header=E4M3_HEADER,
    )


def quantize_activation_fp8_fused(x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """
    1-D x -> (E4M3 bytes [K], fp32 scales [K/32], decoded fp32 values [K]).
    Bytes and scales are bit-identical to quantize_activation_fp8_mlx; the
    decoded values are the unscaled E4M3 numbers the GEMV kernels consume.
    """
    k = x.size
    n_blocks = k // BLOCK_SIZE
    q, scales, deq = _quantize_kernel(False)(
        inputs=[x.reshape(-1), mx.array([n_blocks], dtype=mx.uint32)],
        template=[],
        grid=(n_blocks, 1, 1),
        threadgroup=(min(256, n_blocks), 1, 1),
        output_shapes=[(k,), (n_blocks,), (k,)],
        output_dtypes=[mx.uint8, mx.float32, mx.float32],
    )
    return q, scales, deq


def _e4m3_table_literal() -> str:
    vals = []
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = m * 2.0 ** -9
        elif e == 15 and m == 7:
            v = 0.0  # NaN encoding; checkpoints never contain it
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        vals.append(f"{sign * v!r}f")
    return ", ".join(vals)


E4M3_TABLE_HEADER = E4M3_HEADER + f"""
constant float e4m3_table[256] = {{ {_e4m3_table_literal()} }};
"""


def _lanes_per_row(n_blocks: int) -> int:
    for lanes in (32, 16, 8):
        if n_blocks % lanes == 0:
            return lanes
    return 32


@cache
def _gemv_v2_kernel(in_features: int, lanes_per_row: int):
    """
    E4M3 GEMV with a pre-decoded fp32 activation.

    Each lane owns whole 32-element blocks (two uint4 weight loads and eight
    float4 activation loads per block) and `lanes_per_row` lanes share one
    output row, so short rows (K = 1280 -> 40 blocks) stay balanced across
    lanes instead of leaving most of the simdgroup idle. Per-block arithmetic
    is the same as fp8_gemv_metal (block sum, then x act_scale x weight_scale).
    """
    scale_k = in_features // BLOCK_SIZE
    rows_per_sg = 32 // lanes_per_row
    source = f"""
        uint global_tid = thread_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = global_tid >> 5;
        uint sub = lane / {lanes_per_row};
        uint l = lane % {lanes_per_row};
        uint row = sg * {rows_per_sg} + sub;
        bool valid = row < n_rows[0];
        uint safe_row = valid ? row : 0;
        const device uint4* wrow = reinterpret_cast<const device uint4*>(weight + safe_row * {in_features});
        uint weight_scale_base = (safe_row >> 5) * {scale_k};
        float acc = 0.0f;
        for (uint block = l; block < {scale_k}; block += {lanes_per_row}) {{
            float act_scale = act_scales[block];
            float weight_scale = metal::exp2(float(int(weight_scales[weight_scale_base + block]) - 127));
            uint4 w0 = wrow[2 * block];
            uint4 w1 = wrow[2 * block + 1];
            uint words[8] = {{w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w}};
            const device float4* av = reinterpret_cast<const device float4*>(act + block * 32);
            float block_acc = 0.0f;
            for (uint wi = 0; wi < 8; ++wi) {{
                float4 a4 = av[wi];
                uint word = words[wi];
                block_acc += a4.x * e4m3_table[word & 0xFFu];
                block_acc += a4.y * e4m3_table[(word >> 8) & 0xFFu];
                block_acc += a4.z * e4m3_table[(word >> 16) & 0xFFu];
                block_acc += a4.w * e4m3_table[(word >> 24) & 0xFFu];
            }}
            acc += block_acc * act_scale * weight_scale;
        }}
        for (uint off = {lanes_per_row // 2}; off > 0; off >>= 1) {{
            acc += simd_shuffle_xor(acc, off);
        }}
        if (valid && l == 0) {{
            out[row] = acc;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"fp8_gemv_v2_k{in_features}_l{lanes_per_row}",
        input_names=["act", "act_scales", "weight", "weight_scales", "n_rows"],
        output_names=["out"],
        source=source,
        header=E4M3_TABLE_HEADER,
    )


def fp8_gemv_decoded(act: mx.array, act_scales: mx.array, weight: mx.array, weight_scales: mx.array) -> mx.array:
    """
    act: fp32 [K] decoded E4M3 values (from quantize_activation_fp8_fused),
    act_scales fp32 [K/32], weight uint8 [N, K], weight_scales uint8 [ceil(N/32), K/32]
    -> fp32 [N]. Same math as fp8_gemv_metal.fp8_gemv_quantized, different lane mapping.
    """
    n, k = weight.shape
    if k % BLOCK_SIZE != 0 or act.size != k:
        raise ValueError(f"bad shapes: weight {weight.shape}, act {act.shape}")
    lanes = _lanes_per_row(k // BLOCK_SIZE)
    rows_per_sg = 32 // lanes
    n_sg = (n + rows_per_sg - 1) // rows_per_sg
    return _gemv_v2_kernel(k, lanes)(
        inputs=[act, act_scales, weight, weight_scales, mx.array([n], dtype=mx.uint32)],
        template=[],
        grid=(n_sg * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]


def fp8_roundtrip_fused(x: mx.array) -> mx.array:
    """1-D quantize + dequantize in x.dtype; bit-identical to fp8_roundtrip_activation_mlx."""
    k = x.size
    n_blocks = k // BLOCK_SIZE
    out = _quantize_kernel(True)(
        inputs=[x.reshape(-1), mx.array([n_blocks], dtype=mx.uint32)],
        template=[("T", x.dtype)],
        grid=(n_blocks, 1, 1),
        threadgroup=(min(256, n_blocks), 1, 1),
        output_shapes=[(k,)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(x.shape)
