"""FP4 E2M1 + E8M0 block-scale dequantization to a dense fp32 matrix (one launch)."""

from __future__ import annotations

from functools import cache

import mlx.core as mx


@cache
def _kernel():
    source = """
        uint idx = thread_position_in_grid.x;      // output element index
        if (idx >= n_elems[0]) return;
        uint k = k_dim[0];
        uint row = idx / k;
        uint col = idx - row * k;
        uchar byte = packed[row * (k / 2) + (col >> 1)];
        uint nibble = (col & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        uchar scale_raw = scales[row * (k / 32) + (col >> 5)];
        float scale = metal::exp2(float(int(scale_raw) - 127));
        out[idx] = fp4_table[nibble] * scale;
    """
    return mx.fast.metal_kernel(
        name="fp4_dequant_dense",
        input_names=["packed", "scales", "n_elems", "k_dim"],
        output_names=["out"],
        source=source,
        header="""
            #include <metal_stdlib>
            using namespace metal;
            constant float fp4_table[16] = {
                 0.0f,  0.5f,  1.0f,  1.5f,
                 2.0f,  3.0f,  4.0f,  6.0f,
                 0.0f, -0.5f, -1.0f, -1.5f,
                -2.0f, -3.0f, -4.0f, -6.0f
            };
        """,
    )


def dequantize_fp4_dense(packed: mx.array, scales: mx.array, out_features: int, in_features: int) -> mx.array:
    """packed: uint8 [N*K/2] (or [N, K/2]); scales: uint8 [N*K/32]. Returns fp32 [N, K]."""
    n_elems = out_features * in_features
    out = _kernel()(
        inputs=[
            packed.reshape(-1),
            scales.reshape(-1),
            mx.array([n_elems], dtype=mx.uint32),
            mx.array([in_features], dtype=mx.uint32),
        ],
        template=[],
        grid=(n_elems, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_elems,)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(out_features, in_features)
