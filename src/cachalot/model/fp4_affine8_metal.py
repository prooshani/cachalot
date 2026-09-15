"""
Exact repacking of FP4 (E2M1 values, E8M0 block scales) into MLX's affine
8-bit quantization layout so routed experts can use mx.quantized_matmul.

Within a 32-element block with exponent e, every E2M1 value v is a multiple
of 0.5 * 2^e in [-6, 6] * 2^e, so

    v = scale * q + bias,   scale = 0.5 * 2^e,  bias = -6 * 2^e,  q = 2 * v / 2^e + 12 in [0, 24]

is exact, and scale/bias are exact in bf16. The dequantized values are
identical to the dense bf16 dequantization; only the matmul's accumulation
order differs.

Layout (bits=8, group_size=32): w_q uint32 [N, K/4] with 4 consecutive
elements per word (element i in bits 8i..8i+7), scales/biases [N, K/32].
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx


@cache
def _kernel():
    source = """
        uint idx = thread_position_in_grid.x;         // output uint32 word index
        uint n_words = dims[0];                       // N * K / 4
        if (idx >= n_words) return;
        uint k = dims[1];
        uint words_per_row = k / 4;
        uint row = idx / words_per_row;
        uint word_in_row = idx - row * words_per_row;
        uint col0 = word_in_row * 4;                  // first element of this word
        uint group = col0 >> 5;
        uchar e_raw = scales_in[row * (k / 32) + group];
        int e = int(e_raw) - 127;

        // two packed FP4 bytes -> four elements
        uint pbase = row * (k / 2) + word_in_row * 2;
        uchar b0 = packed[pbase];
        uchar b1 = packed[pbase + 1];
        uint n0 = b0 & 0x0F, n1 = (b0 >> 4) & 0x0F, n2 = b1 & 0x0F, n3 = (b1 >> 4) & 0x0F;
        uint q0 = uint(twice_table[n0] + 12);
        uint q1 = uint(twice_table[n1] + 12);
        uint q2 = uint(twice_table[n2] + 12);
        uint q3 = uint(twice_table[n3] + 12);
        w_q[idx] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);

        if ((col0 & 31) == 0) {
            float p = metal::exp2(float(e));
            uint gi = row * (k / 32) + group;
            scales_out[gi] = bfloat16_t(0.5f * p);
            biases_out[gi] = bfloat16_t(-6.0f * p);
        }
    """
    return mx.fast.metal_kernel(
        name="fp4_to_affine8",
        input_names=["packed", "scales_in", "dims"],
        output_names=["w_q", "scales_out", "biases_out"],
        source=source,
        header="""
            #include <metal_stdlib>
            using namespace metal;
            // 2 * E2M1 value for each nibble (sign in bit 3)
            constant int twice_table[16] = {
                 0,  1,  2,  3,  4,  6,  8,  12,
                 0, -1, -2, -3, -4, -6, -8, -12
            };
        """,
    )


def fp4_to_affine8(packed: mx.array, scales: mx.array, out_features: int, in_features: int):
    """Returns (w_q uint32 [N, K/4], scales bf16 [N, K/32], biases bf16 [N, K/32])."""
    n_words = out_features * in_features // 4
    n_groups = out_features * in_features // 32
    w_q, s, b = _kernel()(
        inputs=[
            packed.reshape(-1),
            scales.reshape(-1),
            mx.array([n_words, in_features], dtype=mx.uint32),
        ],
        template=[],
        grid=(n_words, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_words,), (n_groups,), (n_groups,)],
        output_dtypes=[mx.uint32, mx.bfloat16, mx.bfloat16],
    )
    return (
        w_q.reshape(out_features, in_features // 4),
        s.reshape(out_features, in_features // 32),
        b.reshape(out_features, in_features // 32),
    )


def fp4_affine8_matmul(x: mx.array, packed: mx.array, scales: mx.array, *, out_features: int, in_features: int) -> mx.array:
    """x [M, K] (bf16) @ dequant(W)^T -> [M, N] in x's dtype, via mx.quantized_matmul."""
    w_q, s, b = fp4_to_affine8(packed, scales, out_features, in_features)
    return mx.quantized_matmul(x, w_q, s, b, transpose=True, group_size=32, bits=8)
