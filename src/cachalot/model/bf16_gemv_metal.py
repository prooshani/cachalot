"""
bf16-weight, fp32-activation GEMV with fp32 accumulation.

Used for the ParallelHead: the official module upcasts the bf16 head weight
to fp32 and runs F.linear(x.float(), weight). Converting the 129,280 x 5,120
matrix per token costs ~4 GB of memory traffic; this kernel reads the bf16
bits once and widens in registers (exact: bf16 -> fp32 is a shift).
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from cachalot.model.kernel_consts import u32


@cache
def _make_kernel(in_features: int):
    if in_features % 64 != 0:
        raise ValueError("in_features must be a multiple of 64")
    source = f"""
        uint global_tid = thread_position_in_grid.x;
        uint row = global_tid >> 5;
        uint lane = thread_index_in_simdgroup;
        if (row >= n_rows[0]) return;

        const device ushort* w = weight + row * {in_features};
        float acc = 0.0f;
        for (uint k = lane * 2; k < {in_features}; k += 64) {{
            float w0 = as_type<float>(uint(w[k]) << 16);
            float w1 = as_type<float>(uint(w[k + 1]) << 16);
            acc += w0 * x[k] + w1 * x[k + 1];
        }}
        acc = simd_sum(acc);
        if (lane == 0) {{
            out[row] = acc;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"bf16_gemv_k{in_features}",
        input_names=["x", "weight", "n_rows"],
        output_names=["out"],
        source=source,
        header="#include <metal_stdlib>\nusing namespace metal;\n",
    )


def bf16_gemv_f32(x: mx.array, weight_bf16: mx.array) -> mx.array:
    """x: [K] any float dtype; weight_bf16: [N, K] bfloat16. Returns fp32 [N]."""
    if weight_bf16.dtype != mx.bfloat16 or weight_bf16.ndim != 2:
        raise ValueError("weight must be 2D bfloat16")
    n, k = weight_bf16.shape
    kernel = _make_kernel(k)
    return kernel(
        inputs=[x.astype(mx.float32), weight_bf16.view(mx.uint16), u32(n)],
        template=[],
        grid=(n * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]
