from __future__ import annotations

from functools import cache

import mlx.core as mx


@cache
def _make_fp4_gemv_kernel(in_features: int):
    if in_features % 32 != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by 32"
        )

    packed_k = in_features // 2
    scale_k = in_features // 32

    source = f"""
        constexpr uint K = {in_features};
        constexpr uint PACKED_K = {packed_k};
        constexpr uint SCALE_K = {scale_k};

        constexpr float fp4_table[16] = {{
             0.0f,  0.5f,  1.0f,  1.5f,
             2.0f,  3.0f,  4.0f,  6.0f,
             0.0f, -0.5f, -1.0f, -1.5f,
            -2.0f, -3.0f, -4.0f, -6.0f
        }};

        uint global_tid = thread_position_in_grid.x;
        uint row = global_tid >> 5;
        uint lane = thread_index_in_simdgroup;

        uint packed_base = row * PACKED_K;
        uint scale_base = row * SCALE_K;

        float acc = 0.0f;

        for (uint block = lane; block < SCALE_K; block += 32) {{
            uchar scale_raw = scales[scale_base + block];
            float scale = metal::exp2(float(int(scale_raw) - 127));

            uint k_base = block * 32;
            uint packed_block = packed_base + block * 16;

            for (uint j = 0; j < 16; ++j) {{
                uchar byte = packed[packed_block + j];

                uint low = byte & 0x0F;
                uint high = (byte >> 4) & 0x0F;

                uint k0 = k_base + j * 2;
                uint k1 = k0 + 1;

                acc += x[k0] * fp4_table[low] * scale;
                acc += x[k1] * fp4_table[high] * scale;
            }}
        }}

        acc = simd_sum(acc);

        if (lane == 0) {{
            out[row] = acc;
        }}
    """

    return mx.fast.metal_kernel(
        name=f"fp4_gemv_k{in_features}",
        input_names=["x", "packed", "scales"],
        output_names=["out"],
        source=source,
        header=r"""
            #include <metal_stdlib>
            using namespace metal;
        """,
    )


def fp4_gemv(
    x: mx.array,
    packed: mx.array,
    scales: mx.array,
    out_features: int,
    in_features: int | None = None,
) -> mx.array:
    if in_features is None:
        in_features = x.size

    if x.size != in_features:
        raise ValueError(
            f"x has {x.size} elements but in_features={in_features}"
        )

    kernel = _make_fp4_gemv_kernel(in_features)

    outputs = kernel(
        inputs=[x, packed, scales],
        template=[],
        grid=(out_features * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(out_features,)],
        output_dtypes=[mx.float32],
    )

    return outputs[0]
