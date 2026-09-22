"""wq_a and wkv, fused into one FP8 GEMV, must be bit-identical to the two
independent `fp8_linear` calls attention_compressed.py/attention_layer0.py/
attention_sliding_window.py used to issue.

HANDOFF section 7.1.10 found wq_a [1280, 5120] and wkv [512, 5120] the two
worst-throughput shapes in the FP8 GEMV family, occupancy-bound: 512 and 1280
output rows launch too few simdgroups to fill the GPU. Both read the same x
(the layer's hidden state) and neither depends on the other, exactly the
shape shared_expert_metal.py's w1/w3 fusion exploited (HANDOFF section 9.27),
so they concatenate into one [1792, 5120] GEMV instead of two, and the
redundant second activation quantization goes away with it. Measured
0.35 ms/token, bit-identical, benchmarks/micro_qkv_fusion_roofline.py
(HANDOFF section 9.34).

Attention is never called from inside an `mx.compile`d trace (only the MoE
block is, HANDOFF section 9.21/9.27), so unlike the shared expert's fusion
this one carries no eager-precompute-before-a-trace hazard -- no test for it
is needed here.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from cachalot.model import attention_qkv_fusion as qkv
from cachalot.model.fp8_linear_metal import fp8_linear

HIDDEN = 5120
WQ_A_ROWS = 1280
WKV_ROWS = 512
BLOCK = 32


def _fp8_weight(out_features: int, in_features: int):
    w = mx.random.randint(0, 255, shape=(out_features, in_features)).astype(mx.uint8)
    scales = mx.random.randint(
        120, 131, shape=((out_features + BLOCK - 1) // BLOCK, in_features // BLOCK)
    ).astype(mx.uint8)
    mx.eval(w, scales)
    return w, scales


def test_fused_and_shipped_paths_are_bit_identical():
    mx.random.seed(11)
    x = mx.random.normal(shape=(HIDDEN,)).astype(mx.bfloat16)
    wq_a, sq_a = _fp8_weight(WQ_A_ROWS, HIDDEN)
    wkv, skv = _fp8_weight(WKV_ROWS, HIDDEN)

    shipped_qr = fp8_linear(x, wq_a, sq_a)
    shipped_kv = fp8_linear(x, wkv, skv)
    mx.eval(shipped_qr, shipped_kv)

    fused_qr, fused_kv = qkv.fused_qr_kv_linear(x, wq_a, sq_a, wkv, skv)
    mx.eval(fused_qr, fused_kv)

    assert np.array_equal(
        np.array(fused_qr.astype(mx.float32)), np.array(shipped_qr.astype(mx.float32))
    )
    assert np.array_equal(
        np.array(fused_kv.astype(mx.float32)), np.array(shipped_kv.astype(mx.float32))
    )


def test_fused_path_caches_the_concatenation_by_weight_identity():
    qkv._fused_wqkv_cache.clear()
    wq_a, sq_a = _fp8_weight(WQ_A_ROWS, HIDDEN)
    wkv, skv = _fp8_weight(WKV_ROWS, HIDDEN)

    wqkv_first, scales_first = qkv._fused_wqkv(wq_a, sq_a, wkv, skv)
    assert len(qkv._fused_wqkv_cache) == 1

    wqkv_second, scales_second = qkv._fused_wqkv(wq_a, sq_a, wkv, skv)
    assert len(qkv._fused_wqkv_cache) == 1
    assert wqkv_first is wqkv_second
    assert scales_first is scales_second


def test_two_layers_do_not_collide_in_the_cache():
    qkv._fused_wqkv_cache.clear()

    wq_a_1, sq_a_1 = _fp8_weight(WQ_A_ROWS, HIDDEN)
    wkv_1, skv_1 = _fp8_weight(WKV_ROWS, HIDDEN)
    wq_a_2, sq_a_2 = _fp8_weight(WQ_A_ROWS, HIDDEN)
    wkv_2, skv_2 = _fp8_weight(WKV_ROWS, HIDDEN)

    wqkv_1, _ = qkv._fused_wqkv(wq_a_1, sq_a_1, wkv_1, skv_1)
    wqkv_2, _ = qkv._fused_wqkv(wq_a_2, sq_a_2, wkv_2, skv_2)

    assert len(qkv._fused_wqkv_cache) == 2
    assert not np.array_equal(np.array(wqkv_1), np.array(wqkv_2))


def test_fused_rows_split_back_to_the_shipped_shapes():
    mx.random.seed(13)
    x = mx.random.normal(shape=(HIDDEN,)).astype(mx.bfloat16)
    wq_a, sq_a = _fp8_weight(WQ_A_ROWS, HIDDEN)
    wkv, skv = _fp8_weight(WKV_ROWS, HIDDEN)

    qr, kv = qkv.fused_qr_kv_linear(x, wq_a, sq_a, wkv, skv)
    assert qr.shape == (WQ_A_ROWS,)
    assert kv.shape == (WKV_ROWS,)
