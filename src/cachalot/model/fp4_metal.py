from __future__ import annotations

import mlx.core as mx

_FP4_DEQUANT_KERNEL = mx.fast.metal_kernel(
    name="fp4_dequant_block",
    input_names=["packed", "scales"],
    output_names=["out"],
    source=r"""
        uint idx = thread_position_in_grid.x;

        constexpr float fp4_table[16] = {
             0.0f,  0.5f,  1.0f,  1.5f,
             2.0f,  3.0f,  4.0f,  6.0f,
             0.0f, -0.5f, -1.0f, -1.5f,
            -2.0f, -3.0f, -4.0f, -6.0f
        };

        uint packed_idx = idx >> 1;
        uchar byte = packed[packed_idx];

        uint nibble = (idx & 1)
            ? ((byte >> 4) & 0x0F)
            : (byte & 0x0F);

        uint scale_idx = idx / 32;
        uchar scale_raw = scales[scale_idx];

        // E8M0: value = 2^(raw - 127)
        float scale = metal::exp2(float(int(scale_raw) - 127));

        out[idx] = fp4_table[nibble] * scale;
    """,
    header=r"""
        #include <metal_stdlib>
        using namespace metal;
    """,
)


def dequantize_fp4_block(
    packed: mx.array,
    scales: mx.array,
    logical_size: int,
) -> mx.array:
    result = _FP4_DEQUANT_KERNEL(
        inputs=[packed, scales],
        template=[],
        grid=(logical_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(logical_size,)],
        output_dtypes=[mx.float32],
    )

    return result[0]
