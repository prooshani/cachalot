"""
A replacement for mx.quantize at 2 bits, and the packing it needs.

MLX's own affine fit is max-abs symmetric: for `bits` bits it sets
`scale = +-max|w| / 2**(bits-1)` and `bias = -2**(bits-1) * scale`, so the
representable levels are `scale * (q - 2**(bits-1))`. At 2 bits that is
{-2, -1, 0, +1} x scale, which spends one of four levels on a value the data
never takes and clips one side of the distribution at half its range. The cost
is measurable: see benchmarks/expert_requant_error.py.

The stored format is just `w ~= scale * q + bias` per group, and mx.dequantize
and mx.quantized_matmul read `scales` and `biases` as plain arrays, so a better
fit needs no runtime change at all - only a packer. At 2 bits the packing is
16 values per uint32, least-significant field first, which pack_2bit reproduces
and test_quant_affine.py verifies against mx.quantize.

Two fits are provided:

  fit_minmax   scale = (max - min) / levels, bias = min. The textbook affine
               fit; no clipping, uses all levels.
  fit_search   a grid over shrunk ranges around the group's own min and max,
               keeping the (scale, bias) pair with the lowest squared error per
               group. Trades a little clipping for a finer step, which is what
               imatrix-calibrated banks buy by other means.
"""
from __future__ import annotations

import mlx.core as mx

BITS = 2
PER_WORD = 32 // BITS
LEVELS = (1 << BITS) - 1


def pack_2bit(q: mx.array) -> mx.array:
    """[..., group] uint values in 0..3 -> [..., group/16] uint32, low field first."""
    if q.shape[-1] % PER_WORD:
        raise ValueError(f"group {q.shape[-1]} is not a multiple of {PER_WORD}")
    fields = q.astype(mx.uint32).reshape(*q.shape[:-1], -1, PER_WORD)
    shifts = (mx.arange(PER_WORD, dtype=mx.uint32) * BITS).reshape(1, PER_WORD)
    return mx.sum(fields << shifts, axis=-1).astype(mx.uint32)


def _quantize(w: mx.array, scale: mx.array, bias: mx.array) -> mx.array:
    """Nearest level per weight, clamped to the 2-bit range."""
    q = mx.round((w - bias) / scale)
    return mx.clip(q, 0, LEVELS)


def _error(w: mx.array, scale: mx.array, bias: mx.array) -> mx.array:
    q = _quantize(w, scale, bias)
    return mx.sum((scale * q + bias - w) ** 2, axis=-1)


def fit_minmax(groups: mx.array) -> tuple[mx.array, mx.array]:
    """Affine min/max fit per group. groups: [n, group] fp32."""
    lo = mx.min(groups, axis=-1, keepdims=True)
    hi = mx.max(groups, axis=-1, keepdims=True)
    scale = mx.maximum((hi - lo) / LEVELS, mx.array(1e-8, dtype=groups.dtype))
    return scale, lo


def fit_search(groups: mx.array, shrinks: tuple[float, ...] = tuple(
        1.0 - 0.025 * i for i in range(13))) -> tuple[mx.array, mx.array]:
    """Lowest-squared-error affine fit per group over shrunk ranges (asymmetric grid)."""
    lo = mx.min(groups, axis=-1, keepdims=True)
    hi = mx.max(groups, axis=-1, keepdims=True)
    mid = (lo + hi) / 2
    best_scale, best_bias = fit_minmax(groups)
    best = _error(groups, best_scale, best_bias)

    for s_lo in shrinks:
        for s_hi in shrinks:
            if s_lo == 1.0 and s_hi == 1.0:
                continue
            new_lo = mid + (lo - mid) * s_lo
            new_hi = mid + (hi - mid) * s_hi
            scale = mx.maximum((new_hi - new_lo) / LEVELS, mx.array(1e-8, dtype=groups.dtype))
            err = _error(groups, scale, new_lo)
            take = (err < best).reshape(-1, 1)
            best_scale = mx.where(take, scale, best_scale)
            best_bias = mx.where(take, new_lo, best_bias)
            best = mx.minimum(err, best)
    return best_scale, best_bias


def quantize_2bit(w: mx.array, group_size: int = 64, fit=fit_search,
                  ) -> tuple[mx.array, mx.array, mx.array]:
    """mx.quantize's signature and output format, with a better 2-bit fit.

    Returns (packed uint32, scales bf16, biases bf16) shaped exactly as
    mx.quantize(w, group_size=group_size, bits=2) does, so mx.dequantize and
    mx.quantized_matmul take them unchanged.
    """
    rows = w.shape[0]
    groups = w.astype(mx.float32).reshape(-1, group_size)
    scale, bias = fit(groups)
    # The format stores scales and biases as bf16, so quantize against the
    # values dequantization will actually use, not the fp32 fit.
    scale = scale.astype(mx.bfloat16).astype(mx.float32)
    bias = bias.astype(mx.bfloat16).astype(mx.float32)
    q = _quantize(groups, scale, bias)
    packed = pack_2bit(q).reshape(rows, -1)
    return (packed,
            scale.reshape(rows, -1).astype(mx.bfloat16),
            bias.reshape(rows, -1).astype(mx.bfloat16))
