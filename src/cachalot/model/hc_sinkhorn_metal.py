from __future__ import annotations

from functools import cache

import mlx.core as mx


@cache
def _make_hc_sinkhorn_kernel(
    hc_mult: int,
    sinkhorn_iters: int,
):
    if hc_mult <= 0:
        raise ValueError(
            f"hc_mult must be positive, got {hc_mult}"
        )

    if sinkhorn_iters <= 0:
        raise ValueError(
            f"sinkhorn_iters must be positive, got {sinkhorn_iters}"
        )

    mix_hc = (2 + hc_mult) * hc_mult
    comb_size = hc_mult * hc_mult

    source = f"""
        constexpr uint HC = {hc_mult};
        constexpr uint MIX_HC = {mix_hc};
        constexpr uint COMB_SIZE = {comb_size};
        constexpr uint SINKHORN_ITERS = {sinkhorn_iters};

        uint tid = thread_position_in_grid.x;

        // The whole HC state is tiny:
        //
        //   pre  = HC
        //   post = HC
        //   comb = HC x HC
        //
        // For HC=4 that is only 24 values. Run the complete
        // operation in one thread to avoid dozens of separate
        // GPU dispatches/reductions.
        if (tid == 0) {{
            float comb_local[COMB_SIZE];
            float row_sum[HC];
            float col_sum[HC];

            // ------------------------------------------
            // pre = sigmoid(mix * scale + base) + eps[0]
            // ------------------------------------------

            for (uint j = 0; j < HC; ++j) {{
                float z =
                    mixes[j] * hc_scale[0]
                    + hc_base[j];

                pre[j] =
                    1.0f / (1.0f + metal::exp(-z))
                    + eps[0];
            }}

            // ------------------------------------------
            // post = 2 * sigmoid(...)
            // ------------------------------------------

            for (uint j = 0; j < HC; ++j) {{
                uint idx = HC + j;

                float z =
                    mixes[idx] * hc_scale[1]
                    + hc_base[idx];

                post[j] =
                    2.0f
                    / (
                        1.0f
                        + metal::exp(-z)
                    );
            }}

            // ------------------------------------------
            // comb logits
            // ------------------------------------------

            for (uint j = 0; j < HC; ++j) {{
                for (uint k = 0; k < HC; ++k) {{
                    uint flat_idx =
                        j * HC + k;

                    uint mix_idx =
                        2 * HC + flat_idx;

                    comb_local[flat_idx] =
                        mixes[mix_idx]
                        * hc_scale[2]
                        + hc_base[mix_idx];
                }}
            }}

            // ------------------------------------------
            // First Sinkhorn pass:
            //
            // comb = softmax(-1) + eps[0]
            // ------------------------------------------

            for (uint j = 0; j < HC; ++j) {{
                float row_max = -INFINITY;

                for (uint k = 0; k < HC; ++k) {{
                    float v =
                        comb_local[j * HC + k];

                    row_max = metal::max(
                        row_max,
                        v
                    );
                }}

                float sum = 0.0f;

                for (uint k = 0; k < HC; ++k) {{
                    uint idx =
                        j * HC + k;

                    float v = metal::exp(
                        comb_local[idx]
                        - row_max
                    );

                    comb_local[idx] = v;
                    sum += v;
                }}

                for (uint k = 0; k < HC; ++k) {{
                    uint idx =
                        j * HC + k;

                    comb_local[idx] =
                        comb_local[idx] / sum
                        + eps[0];
                }}
            }}

            // ------------------------------------------
            // First column normalization
            // ------------------------------------------

            for (uint k = 0; k < HC; ++k) {{
                float sum = 0.0f;

                for (uint j = 0; j < HC; ++j) {{
                    sum +=
                        comb_local[
                            j * HC + k
                        ];
                }}

                col_sum[k] = sum;
            }}

            for (uint j = 0; j < HC; ++j) {{
                for (uint k = 0; k < HC; ++k) {{
                    uint idx =
                        j * HC + k;

                    comb_local[idx] =
                        comb_local[idx]
                        / (
                            col_sum[k]
                            + eps[0]
                        );
                }}
            }}

            // ------------------------------------------
            // Remaining iterations
            // ------------------------------------------

            for (
                uint iter = 1;
                iter < SINKHORN_ITERS;
                ++iter
            ) {{
                // Row normalization.
                for (uint j = 0; j < HC; ++j) {{
                    float sum = 0.0f;

                    for (uint k = 0; k < HC; ++k) {{
                        sum +=
                            comb_local[
                                j * HC + k
                            ];
                    }}

                    row_sum[j] = sum;
                }}

                for (uint j = 0; j < HC; ++j) {{
                    for (uint k = 0; k < HC; ++k) {{
                        uint idx =
                            j * HC + k;

                        comb_local[idx] =
                            comb_local[idx]
                            / (
                                row_sum[j]
                                + eps[0]
                            );
                    }}
                }}

                // Column normalization.
                for (uint k = 0; k < HC; ++k) {{
                    float sum = 0.0f;

                    for (uint j = 0; j < HC; ++j) {{
                        sum +=
                            comb_local[
                                j * HC + k
                            ];
                    }}

                    col_sum[k] = sum;
                }}

                for (uint j = 0; j < HC; ++j) {{
                    for (uint k = 0; k < HC; ++k) {{
                        uint idx =
                            j * HC + k;

                        comb_local[idx] =
                            comb_local[idx]
                            / (
                                col_sum[k]
                                + eps[0]
                            );
                    }}
                }}
            }}

            // ------------------------------------------
            // Output
            // ------------------------------------------

            for (uint i = 0; i < COMB_SIZE; ++i) {{
                comb[i] = comb_local[i];
            }}
        }}
    """

    return mx.fast.metal_kernel(
        name=(
            f"hc_sinkhorn_h{hc_mult}"
            f"_i{sinkhorn_iters}"
        ),
        input_names=[
            "mixes",
            "hc_scale",
            "hc_base",
            "eps",
        ],
        output_names=[
            "pre",
            "post",
            "comb",
        ],
        source=source,
        header=r"""
            #include <metal_stdlib>
            using namespace metal;
        """,
    )


def hc_split_sinkhorn_metal(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
]:
    mix_hc = (
        2 + hc_mult
    ) * hc_mult

    if mixes.shape != (mix_hc,):
        raise ValueError(
            f"mixes shape {mixes.shape} "
            f"!= ({mix_hc},)"
        )

    if hc_scale.shape != (3,):
        raise ValueError(
            f"hc_scale shape {hc_scale.shape} != (3,)"
        )

    if hc_base.shape != (mix_hc,):
        raise ValueError(
            f"hc_base shape {hc_base.shape} "
            f"!= ({mix_hc},)"
        )

    kernel = _make_hc_sinkhorn_kernel(
        hc_mult,
        sinkhorn_iters,
    )

    eps_array = mx.array(
        [eps],
        dtype=mx.float32,
    )

    outputs = kernel(
        inputs=[
            mixes.astype(mx.float32),
            hc_scale.astype(mx.float32),
            hc_base.astype(mx.float32),
            eps_array,
        ],
        template=[],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[
            (hc_mult,),
            (hc_mult,),
            (hc_mult, hc_mult),
        ],
        output_dtypes=[
            mx.float32,
            mx.float32,
            mx.float32,
        ],
    )

    return (
        outputs[0],
        outputs[1],
        outputs[2],
    )
