"""Per-expert cost inside the batched MoE loop: gather, 3 qmm, activation ops, scatter-add."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, load_expert_standalone  # noqa: E402
from cachalot.model.fp4_affine8_metal import fp4_affine8_matmul, fp4_to_affine8  # noqa: E402
from cachalot.model.moe_prefill_batched import routed_expert_forward_batched  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader  # noqa: E402


def t(fn, n=40):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    index = build_expert_index(MODEL_PATH)
    e = load_expert_standalone(ExpertReader(), index[(9, 42)])
    T, M = 256, 13
    x = mx.random.normal((T, 5120)).astype(mx.bfloat16)
    slot_buffer = mx.zeros((T, 6, 5120), dtype=mx.float32)
    tok = mx.array(list(range(0, T, T // M))[:M], dtype=mx.int32)
    slot = mx.zeros((M,), dtype=mx.int32)
    w = mx.ones((M,))
    mx.eval(x, slot_buffer, tok, slot, w)
    print(f"gather x[tok] (M={M})           {t(lambda: x[tok]):.3f} ms")
    print(f"repack one matrix               {t(lambda: fp4_to_affine8(e.w1_weight, e.w1_scale, 2304, 5120)[0]):.3f} ms")
    xb = x[tok]
    mx.eval(xb)
    print(f"repack+qmm one matrix           {t(lambda: fp4_affine8_matmul(xb, e.w1_weight, e.w1_scale, out_features=2304, in_features=5120)):.3f} ms")
    print(f"full expert forward (3 matrices) {t(lambda: routed_expert_forward_batched(xb, w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight, w2_scales=e.w2_scale, w3_packed=e.w3_weight, w3_scales=e.w3_scale, weights=w)):.3f} ms")
    y = routed_expert_forward_batched(xb, w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight, w2_scales=e.w2_scale, w3_packed=e.w3_weight, w3_scales=e.w3_scale, weights=w)
    mx.eval(y)
    print(f"scatter-add into [256,6,5120]    {t(lambda: slot_buffer.at[tok, slot].add(y)):.3f} ms")
    print(f"mx.array from 13-int list x3     {t(lambda: mx.array([1] * 13, dtype=mx.int32)):.3f} ms")
    print(f"eval overhead (tiny op)          {t(lambda: w + 1):.3f} ms")


if __name__ == "__main__":
    main()
