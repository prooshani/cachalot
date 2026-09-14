from __future__ import annotations

from functools import cache

import mlx.core as mx

BLOCK_SIZE = 32


@cache
def _make_fp8_gemv_kernel(
    in_features: int,
):
    if in_features % BLOCK_SIZE != 0:
        raise ValueError(
            f"in_features={in_features} must be "
            f"divisible by {BLOCK_SIZE}"
        )

    scale_k = in_features // BLOCK_SIZE

    source = f"""
        constexpr uint K = {in_features};
        constexpr uint SCALE_K = {scale_k};
        uint global_tid =
            thread_position_in_grid.x;

        uint row =
            global_tid >> 5;

        uint lane =
            thread_index_in_simdgroup;

        uint weight_base =
            row * K;

        // Weight scales are shared across each
        // group of 32 output rows.
        uint weight_scale_base =
            (row >> 5) * SCALE_K;

        float acc = 0.0f;

        for (
            uint block = lane;
            block < SCALE_K;
            block += 32
        ) {{
            float act_scale =
                activation_scales[block];

            uchar weight_scale_raw =
                weight_scales[
                    weight_scale_base + block
                ];

            float weight_scale =
                metal::exp2(
                    float(
                        int(weight_scale_raw) - 127
                    )
                );

            uint k_base =
                block * 32;

            float block_acc = 0.0f;

            for (
                uint j = 0;
                j < 32;
                ++j
            ) {{
                float a =
                    decode_e4m3(
                        activation[
                            k_base + j
                        ]
                    );

                float b =
                    decode_e4m3(
                        weight[
                            weight_base
                            + k_base
                            + j
                        ]
                    );

                block_acc += a * b;
            }}

            acc += (
                block_acc
                * act_scale
                * weight_scale
            );
        }}

        acc = simd_sum(acc);

        if (lane == 0) {{
            out[row] = acc;
        }}
    """

    return mx.fast.metal_kernel(
        name=f"fp8_gemv_k{in_features}",
        input_names=[
            "activation",
            "activation_scales",
            "weight",
            "weight_scales",
        ],
        output_names=["out"],
        source=source,
        header=r"""
            #include <metal_stdlib>
            using namespace metal;

            constant float e4m3_table[256] = {
                0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
                0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
                0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
                0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
                0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
                0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
                0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
                1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
                2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
                4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
                8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
                32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
                64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
                128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
                256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f,
                0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
                -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
                -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
                -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
                -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
                -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
                -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
                -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
                -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
                -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
                -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
                -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
                -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
                -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
                -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
                -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, 0.0f
            };

            inline float decode_e4m3(uchar raw) {
                return e4m3_table[uint(raw)];
            }
        """,
    )


def fp8_gemv_quantized(
    activation: mx.array,
    activation_scales: mx.array,
    weight: mx.array,
    weight_scales: mx.array,
    *,
    out_features: int,
    in_features: int,
) -> mx.array:
    """
    GEMV using already-quantized E4M3 activation bytes.

    activation:
        uint8 [K]

    activation_scales:
        float32 [K / 32]

    weight:
        uint8 [N, K], raw E4M3 checkpoint bytes

    weight_scales:
        uint8 [ceil(N / 32), K / 32],
        raw E8M0 checkpoint bytes

    Returns:
        float32 [N]
    """
    if in_features % BLOCK_SIZE != 0:
        raise ValueError(
            f"in_features={in_features} must be "
            f"divisible by {BLOCK_SIZE}"
        )

    if activation.size != in_features:
        raise ValueError(
            f"activation has {activation.size} "
            f"elements, expected {in_features}"
        )

    expected_activation_scales = (
        in_features // BLOCK_SIZE
    )

    if (
        activation_scales.size
        != expected_activation_scales
    ):
        raise ValueError(
            "activation_scales has "
            f"{activation_scales.size} elements, "
            f"expected "
            f"{expected_activation_scales}"
        )

    if weight.shape != (
        out_features,
        in_features,
    ):
        raise ValueError(
            f"weight shape {weight.shape} != "
            f"({out_features}, {in_features})"
        )

    expected_weight_scale_shape = (
        (
            out_features
            + BLOCK_SIZE
            - 1
        )
        // BLOCK_SIZE,
        in_features // BLOCK_SIZE,
    )

    if (
        weight_scales.shape
        != expected_weight_scale_shape
    ):
        raise ValueError(
            f"weight_scales shape "
            f"{weight_scales.shape} != "
            f"{expected_weight_scale_shape}"
        )

    kernel = _make_fp8_gemv_kernel(
        in_features
    )

    outputs = kernel(
        inputs=[
            activation,
            activation_scales,
            weight,
            weight_scales,
        ],
        template=[],
        grid=(
            out_features * 32,
            1,
            1,
        ),
        threadgroup=(
            256,
            1,
            1,
        ),
        output_shapes=[
            (out_features,)
        ],
        output_dtypes=[
            mx.float32
        ],
    )

    return outputs[0]
