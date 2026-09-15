"""
Fused Metal kernels for the top-k routed experts of one decode token.

The unfused path launches, per layer, 6 experts x (2 FP4 GEMVs + clip + silu
+ mul + weight + GEMV + add) ~= 48 kernels. Two fused launches replace them:

    fused_gate_up : hidden[e, :] = silu(min(gate, L)) * clip(up, -L, L) * w_e
                    for all k experts (one simdgroup per (expert, row))
    fused_down    : out[:] = sum_e W2_e @ hidden[e, :]   (fp32, e = 0..k-1 in order)

Per-element arithmetic mirrors expert_metal.routed_expert_forward:
FP4 E2M1 table, E8M0 scales, fp32 accumulation with a simd_sum reduction.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

FP4_TABLE = """
        constexpr float fp4_table[16] = {
             0.0f,  0.5f,  1.0f,  1.5f,
             2.0f,  3.0f,  4.0f,  6.0f,
             0.0f, -0.5f, -1.0f, -1.5f,
            -2.0f, -3.0f, -4.0f, -6.0f
        };
"""


def _gemv_body(x_expr: str, packed_name: str, scales_name: str, k: int) -> str:
    """One simdgroup computes one output row: dot(x, dequant(row))."""
    return f"""
        {{
            constexpr uint SCALE_K = {k // 32};
            uint packed_base = row * {k // 2};
            uint scale_base = row * SCALE_K;
            float acc = 0.0f;
            for (uint block = lane; block < SCALE_K; block += 32) {{
                uchar scale_raw = {scales_name}[scale_base + block];
                float scale = metal::exp2(float(int(scale_raw) - 127));
                uint k_base = block * 32;
                uint packed_block = packed_base + block * 16;
                for (uint j = 0; j < 16; ++j) {{
                    uchar byte = {packed_name}[packed_block + j];
                    uint low = byte & 0x0F;
                    uint high = (byte >> 4) & 0x0F;
                    uint k0 = k_base + j * 2;
                    acc += {x_expr}(k0) * fp4_table[low] * scale;
                    acc += {x_expr}(k0 + 1) * fp4_table[high] * scale;
                }}
            }}
            acc = simd_sum(acc);
            result = acc;
        }}
"""


@cache
def _make_gate_up_kernel(n_experts: int, in_features: int, out_features: int):
    # inputs: x, w1_0..w1_{k-1}, s1_0.., w3_0.., s3_0.., weights(k fp32), limit(1 fp32)
    names = ["x"]
    names += [f"w1_{e}" for e in range(n_experts)]
    names += [f"s1_{e}" for e in range(n_experts)]
    names += [f"w3_{e}" for e in range(n_experts)]
    names += [f"s3_{e}" for e in range(n_experts)]
    names += ["router_w", "limit"]

    def dispatch(prefix: str) -> str:
        # select expert buffers by index with a switch (buffers cannot be indexed dynamically)
        cases = "\n".join(
            f"                case {e}: packed = {prefix}_{e}; scales = s{prefix[1]}_{e}; break;"
            for e in range(n_experts)
        )
        return f"""
            const device uchar* packed;
            const device uchar* scales;
            switch (expert) {{
{cases}
                default: packed = {prefix}_0; scales = s{prefix[1]}_0; break;
            }}
"""

    source = f"""
        {FP4_TABLE}
        uint global_tid = thread_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = global_tid >> 5;                 // simdgroup id
        uint expert = sg / {out_features};
        uint row = sg % {out_features};
        if (expert >= {n_experts}) return;

        float gate, up;
        {{
            {dispatch("w1")}
            float result;
            #define XV(i) x[(i)]
            {_gemv_body("XV", "packed", "scales", in_features)}
            #undef XV
            gate = result;
        }}
        {{
            {dispatch("w3")}
            float result;
            #define XV(i) x[(i)]
            {_gemv_body("XV", "packed", "scales", in_features)}
            #undef XV
            up = result;
        }}
        if (lane == 0) {{
            float L = limit[0];
            if (L > 0.0f) {{
                up = metal::min(metal::max(up, -L), L);
                gate = metal::min(gate, L);
            }}
            float h = gate / (1.0f + metal::exp(-gate)) * up;
            hidden[expert * {out_features} + row] = h * router_w[expert];
        }}
    """
    return mx.fast.metal_kernel(
        name=f"fused_gate_up_e{n_experts}_k{in_features}_n{out_features}",
        input_names=names,
        output_names=["hidden"],
        source=source,
        header="#include <metal_stdlib>\nusing namespace metal;\n",
    )


@cache
def _make_down_kernel(n_experts: int, in_features: int, out_features: int):
    names = ["hidden"]
    names += [f"w2_{e}" for e in range(n_experts)]
    names += [f"s2_{e}" for e in range(n_experts)]
    cases = "\n".join(
        f"                case {e}: packed = w2_{e}; scales = s2_{e}; break;" for e in range(n_experts)
    )
    source = f"""
        {FP4_TABLE}
        uint global_tid = thread_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        uint row = global_tid >> 5;
        if (row >= {out_features}) return;

        float total = 0.0f;
        for (uint expert = 0; expert < {n_experts}; ++expert) {{
            const device uchar* packed;
            const device uchar* scales;
            switch (expert) {{
{cases}
                default: packed = w2_0; scales = s2_0; break;
            }}
            const device float* hx = hidden + expert * {in_features};
            float result;
            #define XV(i) hx[(i)]
            {_gemv_body("XV", "packed", "scales", in_features)}
            #undef XV
            total += result;
        }}
        if (lane == 0) {{
            out[row] = total;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"fused_down_e{n_experts}_k{in_features}_n{out_features}",
        input_names=names,
        output_names=["out"],
        source=source,
        header="#include <metal_stdlib>\nusing namespace metal;\n",
    )


def fused_routed_experts(
    x: mx.array,
    experts,
    router_weights: mx.array,
    *,
    hidden_size: int = 5120,
    intermediate: int = 2304,
    swiglu_limit: float = 10.0,
) -> mx.array:
    """
    x: [hidden_size] (any float dtype; read as float32 by the kernel)
    experts: sequence of objects with w1_weight/w1_scale/w2_weight/w2_scale/w3_weight/w3_scale
    router_weights: [k] float32
    Returns float32 [hidden_size] = sum_e w_e * W2_e(silu(W1_e x) * W3_e x)  (official clamp semantics)
    """
    k = len(experts)
    xf = x.astype(mx.float32)
    gate_up = _make_gate_up_kernel(k, hidden_size, intermediate)
    inputs = [xf]
    inputs += [e.w1_weight for e in experts]
    inputs += [e.w1_scale for e in experts]
    inputs += [e.w3_weight for e in experts]
    inputs += [e.w3_scale for e in experts]
    inputs += [router_weights.astype(mx.float32), mx.array([float(swiglu_limit)], dtype=mx.float32)]
    hidden = gate_up(
        inputs=inputs,
        template=[],
        grid=(k * intermediate * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(k * intermediate,)],
        output_dtypes=[mx.float32],
    )[0]

    down = _make_down_kernel(k, intermediate, hidden_size)
    inputs = [hidden]
    inputs += [e.w2_weight for e in experts]
    inputs += [e.w2_scale for e in experts]
    return down(
        inputs=inputs,
        template=[],
        grid=(hidden_size * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(hidden_size,)],
        output_dtypes=[mx.float32],
    )[0]
