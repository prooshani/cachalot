"""
Memoised scalar parameter arrays for the fused Metal kernels.

Every `mx.fast.metal_kernel` wrapper in the decode path hands its kernel a few
one-element arrays: the row count, an epsilon, a softmax scale. Building them
is cheap in isolation and expensive in aggregate -- a decoded token issues
roughly 1,600 of these constructions across the rope, norm, hyper-connection,
attention, FP8 and router kernels, and each one is a host allocation plus an
MLX array construction on the main thread, in the gap where the GPU has nothing
queued (section 9.15).

The values are immutable and there are only a few dozen distinct ones in a
session, so they can be built once and reused. The arrays are evaluated at
construction so that reusing one never re-enters the scheduler.

This is the same trick, and the same safety argument, as the memoised affine
slot views: an MLX array is a value, so feeding the identical array object to
many graph nodes is indistinguishable from feeding equal copies of it.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

__all__ = ["u32", "f32"]

# Bisecting switch only. 0 rebuilds every parameter array on every call, which
# is what the runtime did before 2026-09-21; the arrays are identical either
# way, so this changes timing and nothing else.
MEMOISE = os.environ.get("CACHALOT_KERNEL_CONSTS", "1") != "0"


@lru_cache(maxsize=4096)
def _u32_cached(value: int) -> mx.array:
    a = mx.array([value], dtype=mx.uint32)
    mx.eval(a)
    return a


@lru_cache(maxsize=4096)
def _f32_cached(value: float) -> mx.array:
    a = mx.array([value], dtype=mx.float32)
    mx.eval(a)
    return a


def u32(value: int) -> mx.array:
    """A one-element uint32 array holding ``value``, built at most once."""
    if not MEMOISE:
        return mx.array([int(value)], dtype=mx.uint32)
    return _u32_cached(int(value))


def f32(value: float) -> mx.array:
    """A one-element float32 array holding ``value``, built at most once."""
    if not MEMOISE:
        return mx.array([float(value)], dtype=mx.float32)
    return _f32_cached(float(value))
