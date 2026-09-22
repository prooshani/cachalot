from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from cachalot.model.fp8_linear_metal import (
    fp8_linear_quantized,
    quantize_fp8_activation,
)

# w1 and w3 read the same quantized activation and neither depends on the
# other, so they concatenate into one [2I, H] GEMV instead of two [I, H]
# ones -- 80 launches per token instead of 120, bit-identical by
# construction (HANDOFF section 9.27; reproduced on real shapes by
# benchmarks/micro_shared_expert_roofline.py, 0.4 ms/token; confirmed live
# 2026-09-22, no crash, no quality regression, same greedy token ids on and
# off before this was the default).
#
# id(w1) -> (w1, w3, w13, w13_scales). The resident weight arrays never move
# or get reassigned once loaded, so the same w1 object recurs on every call;
# keeping w1/w3 alongside the fused pair guards against an id being reused by
# an unrelated array later in the process.
_fused_w13_cache: dict[int, tuple[mx.array, mx.array, mx.array, mx.array]] = {}


def _fused_w13(
    w1: mx.array,
    w1_scales: mx.array,
    w3: mx.array,
    w3_scales: mx.array,
) -> tuple[mx.array, mx.array]:
    key = id(w1)
    cached = _fused_w13_cache.get(key)
    if cached is not None and cached[0] is w1 and cached[1] is w3:
        return cached[2], cached[3]

    w13 = mx.concatenate([w1, w3], axis=0)
    w13_scales = mx.concatenate([w1_scales, w3_scales], axis=0)
    mx.eval(w13, w13_scales)

    _fused_w13_cache[key] = (w1, w3, w13, w13_scales)
    return w13, w13_scales


def shared_expert_forward(
    x: mx.array,
    *,
    w1: mx.array,
    w1_scales: mx.array,
    w2: mx.array,
    w2_scales: mx.array,
    w3: mx.array,
    w3_scales: mx.array,
    swiglu_limit: float = 10.0,
) -> mx.array:
    """
    DeepSeek shared FP8 expert, one token at a time.

    Decode calls `shared_expert_forward_fused` instead (one GEMV over `w1`
    and `w3` concatenated, HANDOFF section 9.27); this two-GEMV form is kept
    for `moe_prefill_grouped.py`'s per-token fallback, which was never
    benchmarked on the fused path.

    Official semantics:

        gate = w1(x).float()
        up   = w3(x).float()

        up   = clamp(up, -limit, +limit)
        gate = clamp(gate, max=limit)

        h = silu(gate) * up

        return w2(h.to(original_dtype))

    w1 and w3 reuse one activation quantization.
    """

    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D for decode, got {x.shape}"
        )

    original_dtype = x.dtype

    # w1 and w3 consume the same activation.
    qx = quantize_fp8_activation(x)

    gate = fp8_linear_quantized(
        qx,
        w1,
        w1_scales,
    ).astype(mx.float32)

    up = fp8_linear_quantized(
        qx,
        w3,
        w3_scales,
    ).astype(mx.float32)

    if swiglu_limit > 0:
        up = mx.clip(
            up,
            -swiglu_limit,
            swiglu_limit,
        )

        gate = mx.minimum(
            gate,
            swiglu_limit,
        )

    hidden = (
        nn.silu(gate)
        * up
    )

    # Match:
    #
    #     self.w2(x.to(dtype))
    #
    hidden = hidden.astype(
        original_dtype
    )

    qhidden = quantize_fp8_activation(
        hidden
    )

    return fp8_linear_quantized(
        qhidden,
        w2,
        w2_scales,
    )


def shared_expert_forward_fused(
    x: mx.array,
    *,
    w13: mx.array,
    w13_scales: mx.array,
    w2: mx.array,
    w2_scales: mx.array,
    swiglu_limit: float = 10.0,
) -> mx.array:
    """
    Same arithmetic as `shared_expert_forward`, issued as one GEMV over
    `w1` and `w3` concatenated along axis 0 (`_fused_w13`) instead of two.
    Bit-identical by construction: both read the same quantized activation
    and neither depends on the other (HANDOFF section 9.27).

    Callers must build `w13`/`w13_scales` with `_fused_w13` in eager Python
    *before* entering an `mx.compile`d trace -- `_fused_w13` calls `mx.eval`,
    which `mx.compile` forbids mid-trace.
    """

    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D for decode, got {x.shape}"
        )

    original_dtype = x.dtype
    intermediate = w13.shape[0] // 2

    qx = quantize_fp8_activation(x)

    gu = fp8_linear_quantized(
        qx,
        w13,
        w13_scales,
    ).astype(mx.float32)

    gate = gu[:intermediate]
    up = gu[intermediate:]

    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, swiglu_limit)

    hidden = (nn.silu(gate) * up).astype(original_dtype)

    qhidden = quantize_fp8_activation(hidden)

    return fp8_linear_quantized(
        qhidden,
        w2,
        w2_scales,
    )
