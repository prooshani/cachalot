"""
Routed-expert math for an affine-quantized expert bank (storage.index.ExpertFormat
kind "affine"), e.g. oMLX's oQ3e conversion: 3-bit weights, group 64, fp16
scales and biases.

Slot bytes are viewed in place (no copy) as MLX quantized matrices and
multiplied with mx.quantized_matmul. Numerics follow the official routed
expert: bf16 activations in, bf16 linear outputs, fp32 gating, router weight
applied before the down projection. With bf16 activations and fp16 scales MLX
returns fp32 products (measured the most accurate combination).
"""
from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from cachalot.storage.index import ExpertFormat

_DTYPES = {"uint32": mx.uint32, "float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32, "uint8": mx.uint8}


def affine_views(
    arrays: dict[str, mx.array],
    fmt: ExpertFormat,
    proj: str,
    cache: dict[str, tuple[mx.array, mx.array, mx.array]] | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    (weight, scales, biases) views of one projection's slot bytes.

    `cache` is a dict owned by the slot these bytes live in. The views are
    zero-copy and keep aliasing that memory when a different expert is read
    into the slot, so a cached entry stays correct for the slot's whole life;
    see ExpertSlot.typed.
    """
    if cache is not None:
        hit = cache.get(proj)
        if hit is not None:
            return hit
    out = []
    for field in ("weight", "scales", "biases"):
        short = f"{proj}.{field}"
        out.append(arrays[short].view(_DTYPES[fmt.dtypes[short]]).reshape(fmt.shapes[short]))
    views = (out[0], out[1], out[2])
    if cache is not None:
        cache[proj] = views
    return views


def _qmm(
    x: mx.array,
    arrays: dict[str, mx.array],
    fmt: ExpertFormat,
    proj: str,
    cache: dict[str, tuple[mx.array, mx.array, mx.array]] | None = None,
) -> mx.array:
    w, s, b = affine_views(arrays, fmt, proj, cache)
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=fmt.group_size, bits=fmt.bits)


def affine_expert_forward(
    x: mx.array,
    arrays: dict[str, mx.array],
    fmt: ExpertFormat,
    weight: float,
    swiglu_limit: float = 10.0,
    cache: dict[str, tuple[mx.array, mx.array, mx.array]] | None = None,
) -> mx.array:
    """x: [HIDDEN] -> fp32 [HIDDEN] = W2 (silu(min(W1 x, L)) * clip(W3 x, -L, L) * weight)."""
    xb = x.astype(mx.bfloat16)
    gate = _qmm(xb, arrays, fmt, "w1", cache).astype(mx.bfloat16).astype(mx.float32)
    up = _qmm(xb, arrays, fmt, "w3", cache).astype(mx.bfloat16).astype(mx.float32)
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, mx.array(swiglu_limit, dtype=mx.float32))
    hidden = (nn.silu(gate) * up * weight).astype(mx.bfloat16)
    return _qmm(hidden, arrays, fmt, "w2", cache).astype(mx.bfloat16).astype(mx.float32)


def affine_expert_forward_batched(
    x: mx.array,
    arrays: dict[str, mx.array],
    fmt: ExpertFormat,
    weights: mx.array,
    swiglu_limit: float = 10.0,
) -> mx.array:
    """x: [M, HIDDEN], weights: [M] -> fp32 [M, HIDDEN]; same contract as routed_expert_forward_batched."""
    xb = x.astype(mx.bfloat16)
    gate = _qmm(xb, arrays, fmt, "w1").astype(mx.bfloat16).astype(mx.float32)
    up = _qmm(xb, arrays, fmt, "w3").astype(mx.bfloat16).astype(mx.float32)
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, mx.array(swiglu_limit, dtype=mx.float32))
    hidden = (nn.silu(gate) * up * weights.astype(mx.float32)[:, None]).astype(mx.bfloat16)
    return _qmm(hidden, arrays, fmt, "w2").astype(mx.bfloat16).astype(mx.float32)


# ---------------------------------------------------------------------------
# Compiled top-k block
# ---------------------------------------------------------------------------
# One decode layer issues the six routed experts one at a time, and every call
# builds the same eighteen quantized_matmul / astype / clip / silu operations
# again on the CPU while the GPU has nothing queued (HANDOFF section 9.15).
# The shapes are identical on every layer and every token, so the whole block
# traces once under mx.compile and is replayed afterwards.
#
# Two conditions have to hold for the trace to be reused rather than rebuilt:
# the router weights enter as an array (a Python float would be a constant and
# retrace on every token), and the number of experts is part of the cache key.
COMPILE_MOE = os.environ.get("CACHALOT_COMPILE_MOE", "1") != "0"


def topk_core(
    x: mx.array,
    weights: mx.array,
    views: list[mx.array],
    *,
    n_experts: int,
    bits: int,
    group_size: int,
    hidden: int,
    swiglu_limit: float,
) -> mx.array:
    """
    The sum of `n_experts` affine experts, written as one function so it can be
    traced. `views` is the flat (w1, w2, w3) x (weight, scales, biases) list in
    expert order; `weights` is the router's fp32 array. Same operations and the
    same order as affine_expert_forward called once per expert.
    """
    kw = {"transpose": True, "group_size": group_size, "bits": bits}
    xb = x.astype(mx.bfloat16)
    routed = mx.zeros((hidden,), dtype=mx.float32)
    for i in range(n_experts):
        base = 9 * i
        gate = mx.quantized_matmul(
            xb, views[base], views[base + 1], views[base + 2], **kw
        ).astype(mx.bfloat16).astype(mx.float32)
        up = mx.quantized_matmul(
            xb, views[base + 6], views[base + 7], views[base + 8], **kw
        ).astype(mx.bfloat16).astype(mx.float32)
        if swiglu_limit > 0:
            up = mx.clip(up, -swiglu_limit, swiglu_limit)
            gate = mx.minimum(gate, swiglu_limit)
        hidden_act = (nn.silu(gate) * up * weights[i]).astype(mx.bfloat16)
        routed = routed + mx.quantized_matmul(
            hidden_act, views[base + 3], views[base + 4], views[base + 5], **kw
        ).astype(mx.bfloat16).astype(mx.float32)
    return routed


@lru_cache(maxsize=None)
def _compiled_topk(
    n_experts: int,
    bits: int,
    group_size: int,
    hidden: int,
    swiglu_limit: float,
) -> Callable[[mx.array, mx.array, list[mx.array]], mx.array]:
    """mx.compile of `n_experts` affine experts summed, weights as an array."""

    def core(x: mx.array, weights: mx.array, views: list[mx.array]) -> mx.array:
        return topk_core(
            x, weights, views,
            n_experts=n_experts, bits=bits, group_size=group_size,
            hidden=hidden, swiglu_limit=swiglu_limit,
        )

    return mx.compile(core)


def slot_views(experts: list, fmt: ExpertFormat) -> list[mx.array]:
    """The flat view list topk_core expects, taken from each slot's view cache."""
    views: list[mx.array] = []
    for expert in experts:
        arrays = expert.as_model_dict()
        cache = expert.slot.typed
        for proj in ("w1", "w2", "w3"):
            views.extend(affine_views(arrays, fmt, proj, cache))
    return views


def affine_routed_experts(
    x: mx.array,
    experts: list,
    fmt: ExpertFormat,
    weights: mx.array,
    swiglu_limit: float = 10.0,
) -> mx.array:
    """
    Sum of the routed experts for one decode token, traced once.

    `experts` are ResidentExpert objects whose slots carry the view cache;
    `weights` is the router's own fp32 array, one entry per expert, in the
    same order. Numerically identical to summing affine_expert_forward() over
    the same experts -- the operations and their order are unchanged, only
    their construction is skipped.
    """
    views = slot_views(experts, fmt)
    fn = _compiled_topk(
        len(experts), fmt.bits, fmt.group_size, int(x.shape[0]), float(swiglu_limit)
    )
    return fn(x, weights, views)
