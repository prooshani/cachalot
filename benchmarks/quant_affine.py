"""
A replacement for mx.quantize's affine fit at any width, and the packing it needs.

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

Above 2 bits the packing no longer aligns to a word: MLX writes one contiguous
little-endian bit stream per row, so a 3-bit level can straddle a word boundary
and 32 levels occupy exactly 3 words. pack_bits reproduces that stream at any
width, pack_3bit is the width this project wants, and
test_quant_affine.py pins both against mx.quantize's own output at every group
size the kernels accept.

Two fits are provided:

  fit_minmax   scale = (max - min) / levels, bias = min. The textbook affine
               fit; no clipping, uses all levels.
  fit_search   a grid over shrunk ranges around the group's own min and max,
               keeping the (scale, bias) pair with the lowest squared error per
               group. Trades a little clipping for a finer step, which is what
               imatrix-calibrated banks buy by other means.
"""
from __future__ import annotations

from math import gcd

import mlx.core as mx

BITS = 2
PER_WORD = 32 // BITS
LEVELS = (1 << BITS) - 1


def levels_for(bits: int) -> int:
    """Highest representable level index at `bits` bits: 3 at 2 bits, 7 at 3, 15 at 4."""
    return (1 << bits) - 1


def pack_2bit(q: mx.array) -> mx.array:
    """[..., group] uint values in 0..3 -> [..., group/16] uint32, low field first."""
    if q.shape[-1] % PER_WORD:
        raise ValueError(f"group {q.shape[-1]} is not a multiple of {PER_WORD}")
    fields = q.astype(mx.uint32).reshape(*q.shape[:-1], -1, PER_WORD)
    shifts = (mx.arange(PER_WORD, dtype=mx.uint32) * BITS).reshape(1, PER_WORD)
    return mx.sum(fields << shifts, axis=-1).astype(mx.uint32)


def pack_bits(q: mx.array, bits: int) -> mx.array:
    """[..., group] uint levels -> [..., group*bits/32] uint32 in MLX's own layout.

    MLX stores the levels of a row as one contiguous little-endian bit stream:
    level i occupies bits [i*bits, (i+1)*bits) of the stream, words are filled
    least-significant bit first, and nothing is padded. At 2, 4 and 8 bits that
    happens to align to a word and pack_2bit's single shift-and-sum is enough;
    at 3, 5 and 6 bits a level straddles a word boundary, so each level
    contributes a low part to one word and, when it spans, a high part to the
    next.

    The bit pattern repeats every `32 // gcd(bits, 32)` levels -- 32 at 3 bits,
    16 at 2 -- so the packing is built chunk-wise and each level contributes to
    at most two words. The loop below is over that fixed pattern, not over the
    data: every operation inside it is one vectorized pass over all the groups
    at once.
    """
    group = q.shape[-1]
    if (group * bits) % 32:
        raise ValueError(f"group {group} at {bits} bits is not a whole number of uint32 words")
    per_chunk = 32 // gcd(bits, 32)
    if group % per_chunk:
        raise ValueError(f"group {group} is not a multiple of {per_chunk} at {bits} bits")
    words_per_chunk = per_chunk * bits // 32

    fields = q.astype(mx.uint32).reshape(*q.shape[:-1], -1, per_chunk)
    words = []
    for word in range(words_per_chunk):
        acc = mx.zeros(fields.shape[:-1], dtype=mx.uint32)
        for i in range(per_chunk):
            lo_word, shift = divmod(i * bits, 32)
            if lo_word == word:
                acc = acc + (fields[..., i] << mx.array(shift, dtype=mx.uint32))
            elif lo_word + 1 == word and shift + bits > 32:
                acc = acc + (fields[..., i] >> mx.array(32 - shift, dtype=mx.uint32))
        words.append(acc)
    return mx.stack(words, axis=-1).reshape(*q.shape[:-1], -1)


def pack_3bit(q: mx.array) -> mx.array:
    """[..., group] uint values in 0..7 -> [..., group*3/32] uint32, MLX's 3-bit layout."""
    return pack_bits(q, 3)


def _quantize(w: mx.array, scale: mx.array, bias: mx.array, bits: int = BITS) -> mx.array:
    """Nearest level per weight, clamped to the range `bits` bits can represent."""
    q = mx.round((w - bias) / scale)
    return mx.clip(q, 0, levels_for(bits))


def _error(w: mx.array, scale: mx.array, bias: mx.array, bits: int = BITS) -> mx.array:
    q = _quantize(w, scale, bias, bits)
    return mx.sum((scale * q + bias - w) ** 2, axis=-1)


def fit_minmax(groups: mx.array, bits: int = BITS) -> tuple[mx.array, mx.array]:
    """Affine min/max fit per group. groups: [n, group] fp32."""
    lo = mx.min(groups, axis=-1, keepdims=True)
    hi = mx.max(groups, axis=-1, keepdims=True)
    scale = mx.maximum((hi - lo) / levels_for(bits), mx.array(1e-8, dtype=groups.dtype))
    return scale, lo


# A 5x5 grid over the two range ends. Finer grids do not pay: 13 shrinks per end
# cost 241 ms per expert tensor against 37 ms for 5, and buy 0.0005 of relative
# weight error (0.3363 against 0.3368 on gaussian weights). The NLL gate quantizes
# some 20,000 experts per run, so the difference is hours.
SHRINKS = (1.0, 0.925, 0.85, 0.775, 0.7)


def fit_search(groups: mx.array, shrinks: tuple[float, ...] = SHRINKS,
               bits: int = BITS) -> tuple[mx.array, mx.array]:
    """Lowest-squared-error affine fit per group over shrunk ranges (asymmetric grid).

    Generalised beyond 2 bits on 2026-09-18. MLX's own fit is max-abs symmetric
    and wastes one level at every width (section 8.1 of docs/HANDOFF.md); at
    2 bits that cost 16 % of routed-expert output error, and whether it costs
    anything like as much at 3 bits is what decides if a 3-bit bank can be made
    to behave like FP4.
    """
    lo = mx.min(groups, axis=-1, keepdims=True)
    hi = mx.max(groups, axis=-1, keepdims=True)
    mid = (lo + hi) / 2
    best_scale, best_bias = fit_minmax(groups, bits)
    best = _error(groups, best_scale, best_bias, bits)

    for s_lo in shrinks:
        for s_hi in shrinks:
            if s_lo == 1.0 and s_hi == 1.0:
                continue
            new_lo = mid + (lo - mid) * s_lo
            new_hi = mid + (hi - mid) * s_hi
            scale = mx.maximum((new_hi - new_lo) / levels_for(bits),
                               mx.array(1e-8, dtype=groups.dtype))
            err = _error(groups, scale, new_lo, bits)
            take = (err < best).reshape(-1, 1)
            best_scale = mx.where(take, scale, best_scale)
            best_bias = mx.where(take, new_lo, best_bias)
            best = mx.minimum(err, best)
    return best_scale, best_bias


# --- Least-squares refinement -------------------------------------------------
#
# A grid search picks the best (scale, bias) among a fixed set of shrunk ranges.
# Once the level assignment q is fixed, however, the best (scale, bias) for that
# assignment is not a grid point at all: it is the ordinary least-squares fit of
# w against q, which has a closed form per group. Alternating the two -- assign
# levels, re-fit the line, assign again -- is Lloyd's algorithm restricted to a
# uniform grid, and it converges in a handful of iterations. It costs one extra
# pass per iteration and no search, so it is far cheaper per unit of error than
# refining the grid.
#
# The refit rounds scale and bias to bf16 on every iteration, because that is
# the precision the bank stores and therefore the line dequantization will
# actually use; refining in fp32 and rounding once at the end optimizes a
# function that is not the one being evaluated.


def _lsq_step(groups: mx.array, scale: mx.array, bias: mx.array, bits: int = BITS,
              ) -> tuple[mx.array, mx.array]:
    """One assign-then-refit iteration. Groups whose levels collapse keep the old fit."""
    q = _quantize(groups, scale, bias, bits)
    n = float(groups.shape[-1])
    sq = mx.sum(q, axis=-1, keepdims=True)
    sqq = mx.sum(q * q, axis=-1, keepdims=True)
    sw = mx.sum(groups, axis=-1, keepdims=True)
    sqw = mx.sum(q * groups, axis=-1, keepdims=True)
    den = n * sqq - sq * sq
    new_scale = (n * sqw - sq * sw) / mx.where(mx.abs(den) < 1e-12, mx.ones_like(den), den)
    new_bias = (sw - new_scale * sq) / n
    new_scale = new_scale.astype(mx.bfloat16).astype(mx.float32)
    new_bias = new_bias.astype(mx.bfloat16).astype(mx.float32)
    keep = (mx.abs(den) < 1e-12) | (new_scale <= 0)
    return mx.where(keep, scale, new_scale), mx.where(keep, bias, new_bias)


def refine_lsq(groups: mx.array, scale: mx.array, bias: mx.array, iters: int = 4,
               bits: int = BITS) -> tuple[mx.array, mx.array]:
    """Alternate level assignment and least-squares refit, keeping the best iterate per group."""
    scale = scale.astype(mx.bfloat16).astype(mx.float32)
    bias = bias.astype(mx.bfloat16).astype(mx.float32)
    best_scale, best_bias = scale, bias
    best = _error(groups, best_scale, best_bias, bits)
    for _ in range(iters):
        scale, bias = _lsq_step(groups, scale, bias, bits)
        err = _error(groups, scale, bias, bits)
        take = (err < best).reshape(-1, 1)
        best_scale = mx.where(take, scale, best_scale)
        best_bias = mx.where(take, bias, best_bias)
        best = mx.minimum(err, best)
    return best_scale, best_bias


def fit_lsq(groups: mx.array, bits: int = BITS) -> tuple[mx.array, mx.array]:
    """Min/max start, then least-squares refinement. No grid at all."""
    scale, bias = fit_minmax(groups, bits)
    return refine_lsq(groups, scale, bias, bits=bits)


def fit_search_lsq(groups: mx.array, shrinks: tuple[float, ...] = SHRINKS,
                   bits: int = BITS) -> tuple[mx.array, mx.array]:
    """The production grid search, then least-squares refinement of its winner."""
    scale, bias = fit_search(groups, shrinks, bits)
    return refine_lsq(groups, scale, bias, bits=bits)


# A 9x9 grid reaching further in: what a bank can afford that a gate arm cannot.
SHRINKS_FINE = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.625, 0.55)


def fit_search_fine(groups: mx.array, bits: int = BITS) -> tuple[mx.array, mx.array]:
    return fit_search(groups, SHRINKS_FINE, bits)


def fit_search_fine_lsq(groups: mx.array, bits: int = BITS) -> tuple[mx.array, mx.array]:
    scale, bias = fit_search(groups, SHRINKS_FINE, bits)
    return refine_lsq(groups, scale, bias, bits=bits)


# A wider grid still, reaching to 0.4 of the group's own range. Clipping harder
# than this stops paying: with four levels the step is already coarse, and a
# range shrunk past ~0.5 throws away outliers the levels could have covered.
SHRINKS_WIDE = (1.0, 0.925, 0.85, 0.775, 0.7, 0.625, 0.55, 0.475, 0.4)


def fit_search_wide_lsq(groups: mx.array, bits: int = BITS) -> tuple[mx.array, mx.array]:
    scale, bias = fit_search(groups, SHRINKS_WIDE, bits)
    return refine_lsq(groups, scale, bias, bits=bits)


def fit_lsq_long(groups: mx.array, bits: int = BITS) -> tuple[mx.array, mx.array]:
    """Min/max start, twelve refinement iterations. The cheap end of the trade."""
    scale, bias = fit_minmax(groups, bits)
    return refine_lsq(groups, scale, bias, iters=12, bits=bits)


def fit_fine_lsq_long(groups: mx.array, bits: int = BITS) -> tuple[mx.array, mx.array]:
    scale, bias = fit_search(groups, SHRINKS_FINE, bits)
    return refine_lsq(groups, scale, bias, iters=12, bits=bits)


def quantize_affine(w: mx.array, group_size: int = 64, bits: int = BITS, fit=fit_search,
                    ) -> tuple[mx.array, mx.array, mx.array]:
    """mx.quantize's signature and output format, with a better affine fit.

    Returns (packed uint32, scales bf16, biases bf16) shaped exactly as
    mx.quantize(w, group_size=group_size, bits=bits) does, so mx.dequantize and
    mx.quantized_matmul take them unchanged. Every fit in this module takes
    `bits` as a keyword, so the width is the caller's choice and not the fit's.
    """
    rows = w.shape[0]
    groups = w.astype(mx.float32).reshape(-1, group_size)
    scale, bias = fit(groups, bits=bits)
    # The format stores scales and biases as bf16, so quantize against the
    # values dequantization will actually use, not the fp32 fit.
    scale = scale.astype(mx.bfloat16).astype(mx.float32)
    bias = bias.astype(mx.bfloat16).astype(mx.float32)
    q = _quantize(groups, scale, bias, bits)
    packed = (pack_2bit(q) if bits == BITS else pack_bits(q, bits)).reshape(rows, -1)
    return (packed,
            scale.reshape(rows, -1).astype(mx.bfloat16),
            bias.reshape(rows, -1).astype(mx.bfloat16))


def quantize_2bit(w: mx.array, group_size: int = 64, fit=fit_search,
                  ) -> tuple[mx.array, mx.array, mx.array]:
    """quantize_affine at 2 bits, which is the width most of this project's callers want."""
    return quantize_affine(w, group_size=group_size, bits=BITS, fit=fit)


# Convergence probe only: 17 shrinks per end, 289 grid points. Used by
# benchmarks/quant_fit_screen.py to show that the 9x9 wide grid has already
# converged; far too slow to build a bank with.
SHRINKS_MAX = tuple(1.0 - 0.0375 * i for i in range(17))


def fit_search_max_lsq(groups: mx.array, bits: int = BITS) -> tuple[mx.array, mx.array]:
    scale, bias = fit_search(groups, SHRINKS_MAX, bits)
    return refine_lsq(groups, scale, bias, bits=bits)


def dequantized(groups: mx.array, bits: int, fit=fit_search,
                shrinks: tuple[float, ...] = SHRINKS_WIDE,
                refine: bool = True) -> mx.array:
    """Weights as they would come back from a bank at `bits` bits, without packing.

    Screening a fit needs only the values, not the packing, which is what this
    returns -- scales and biases rounded to bf16 first, because that is what the
    bank stores and therefore what dequantization would actually use. Writing a
    bank needs pack_bits as well.
    """
    scale, bias = fit(groups, shrinks, bits) if fit is fit_search else fit(groups, bits)
    if refine:
        scale, bias = refine_lsq(groups, scale, bias, bits=bits)
    scale = scale.astype(mx.bfloat16).astype(mx.float32)
    bias = bias.astype(mx.bfloat16).astype(mx.float32)
    q = _quantize(groups, scale, bias, bits)
    return scale * q + bias
