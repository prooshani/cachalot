"""The fused w1/w3 shared-expert GEMV must be bit-identical to the shipped
two-GEMV path, and it must survive being called from inside `mx.compile`.

HANDOFF section 9.27: w1 and w3 read the same quantized activation and
neither depends on the other, so concatenating them into one [2I, H] GEMV is
bit-identical by construction -- same bytes, same lanes-per-row split, same
summation order per output row.

`shared_expert_forward_fused` takes an already-concatenated `w13`/`w13_scales`
built by `_fused_w13`, which calls `mx.eval` to materialize the concatenation
once per layer. The first version of this fusion built and evaluated `w13`
*inside* `shared_expert_forward` itself, which is called from inside
`_compiled_moe_block`'s `mx.compile`d trace on the shipped 2-bit affine bank
-- `mx.eval` inside a trace raises `ValueError: [eval] Attempting to eval an
array during function transformations like compile or vmap is not allowed`,
caught live by Hamed's own chat session, not by this test suite, because
every test up to that point only called the fused path eagerly. The
`test_..._inside_mx_compile` case below reproduces that call shape.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from cachalot.model import shared_expert_metal as sem

HIDDEN = 5120
INTERMEDIATE = 2304
BLOCK = 32


def _fp8_weight(out_features: int, in_features: int):
    w = mx.random.randint(0, 255, shape=(out_features, in_features)).astype(mx.uint8)
    scales = mx.random.randint(
        120, 131, shape=((out_features + BLOCK - 1) // BLOCK, in_features // BLOCK)
    ).astype(mx.uint8)
    mx.eval(w, scales)
    return w, scales


def _layer():
    w1, s1 = _fp8_weight(INTERMEDIATE, HIDDEN)
    w3, s3 = _fp8_weight(INTERMEDIATE, HIDDEN)
    w2, s2 = _fp8_weight(HIDDEN, INTERMEDIATE)
    return w1, s1, w2, s2, w3, s3


def test_fused_and_shipped_paths_are_bit_identical():
    mx.random.seed(3)
    x = mx.random.normal(shape=(HIDDEN,)).astype(mx.bfloat16)
    w1, s1, w2, s2, w3, s3 = _layer()

    shipped = sem.shared_expert_forward(
        x, w1=w1, w1_scales=s1, w2=w2, w2_scales=s2, w3=w3, w3_scales=s3
    )
    mx.eval(shipped)

    w13, w13_scales = sem._fused_w13(w1, s1, w3, s3)
    fused = sem.shared_expert_forward_fused(
        x, w13=w13, w13_scales=w13_scales, w2=w2, w2_scales=s2
    )
    mx.eval(fused)

    assert np.array_equal(
        np.array(fused.astype(mx.float32)),
        np.array(shipped.astype(mx.float32)),
    )


def test_fused_path_caches_the_concatenation_by_weight_identity():
    sem._fused_w13_cache.clear()

    w1, s1, w2, s2, w3, s3 = _layer()

    w13_first, scales_first = sem._fused_w13(w1, s1, w3, s3)
    assert len(sem._fused_w13_cache) == 1

    w13_second, scales_second = sem._fused_w13(w1, s1, w3, s3)
    assert len(sem._fused_w13_cache) == 1
    assert w13_first is w13_second
    assert scales_first is scales_second


def test_two_layers_do_not_collide_in_the_cache():
    sem._fused_w13_cache.clear()

    layer_a = _layer()
    layer_b = _layer()

    w13_a, _ = sem._fused_w13(layer_a[0], layer_a[1], layer_a[4], layer_a[5])
    w13_b, _ = sem._fused_w13(layer_b[0], layer_b[1], layer_b[4], layer_b[5])

    assert len(sem._fused_w13_cache) == 2
    assert not np.array_equal(np.array(w13_a), np.array(w13_b))


def test_fused_forward_survives_being_called_inside_mx_compile():
    """Reproduces the crash a live chat session hit: `shared_expert_forward`
    called from inside `moe_layer_metal._compiled_moe_block`'s `mx.compile`d
    trace on the shipped 2-bit affine bank. `w13`/`w13_scales` must be built
    *before* entering the trace; building them inside it calls `mx.eval` mid
    trace and MLX raises `ValueError`."""
    mx.random.seed(7)
    x = mx.random.normal(shape=(HIDDEN,)).astype(mx.bfloat16)
    w1, s1, w2, s2, w3, s3 = _layer()

    # w13/w13_scales built eagerly, outside the trace -- the fix.
    w13, w13_scales = sem._fused_w13(w1, s1, w3, s3)

    eager = sem.shared_expert_forward_fused(
        x, w13=w13, w13_scales=w13_scales, w2=w2, w2_scales=s2,
    )
    mx.eval(eager)

    @mx.compile
    def traced(x, w13, w13_scales, w2, w2_scales):
        return sem.shared_expert_forward_fused(
            x, w13=w13, w13_scales=w13_scales, w2=w2, w2_scales=w2_scales
        )

    compiled = traced(x, w13, w13_scales, w2, s2)
    mx.eval(compiled)

    assert np.array_equal(
        np.array(compiled.astype(mx.float32)),
        np.array(eager.astype(mx.float32)),
    )


def test_building_w13_inside_a_trace_is_the_bug_this_guards_against():
    """The regression itself: calling `_fused_w13` (which calls `mx.eval`)
    from inside an `mx.compile`d function must raise, pinning why the
    concatenation has to happen before the trace, not inside it."""
    mx.random.seed(9)
    x = mx.random.normal(shape=(HIDDEN,)).astype(mx.bfloat16)
    w1, s1, w2, s2, w3, s3 = _layer()

    @mx.compile
    def traced_wrong(x, w1, s1, w2, s2, w3, s3):
        w13, w13_scales = sem._fused_w13(w1, s1, w3, s3)
        return sem.shared_expert_forward_fused(
            x, w13=w13, w13_scales=w13_scales, w2=w2, w2_scales=s2
        )

    try:
        out = traced_wrong(x, w1, s1, w2, s2, w3, s3)
        mx.eval(out)
    except ValueError as exc:
        assert "eval" in str(exc).lower()
    else:
        raise AssertionError(
            "expected mx.compile to reject mx.eval inside the trace; "
            "if this now passes, MLX's tracing rules changed and the "
            "eager-precompute requirement in _shared_args may no longer hold"
        )
