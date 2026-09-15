"""Correctness + speed of the fused 6-expert kernels vs the unfused reference."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, load_expert_standalone  # noqa: E402
from cachalot.model.expert_metal import routed_expert_forward  # noqa: E402
from cachalot.model.moe_fused_metal import fused_routed_experts  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader  # noqa: E402


def reference(x, experts, weights):
    routed = mx.zeros((5120,), dtype=mx.float32)
    for e, w in zip(experts, weights.tolist(), strict=True):
        y = routed_expert_forward(x, w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight,
                                  w2_scales=e.w2_scale, w3_packed=e.w3_weight, w3_scales=e.w3_scale,
                                  weight=float(w), swiglu_limit=10.0)
        routed = routed + y.astype(mx.float32)
    return routed


def main():
    index = build_expert_index(MODEL_PATH)
    reader = ExpertReader()
    experts = [load_expert_standalone(reader, index[(7, i * 37)]) for i in range(6)]
    mx.random.seed(0)
    x = (mx.random.normal((5120,)) * 0.5).astype(mx.bfloat16)
    weights = mx.softmax(mx.random.normal((6,))) * 1.5
    mx.eval(x, weights)

    ref = reference(x, experts, weights)
    fused = fused_routed_experts(x, experts, weights)
    mx.eval(ref, fused)
    diff = mx.abs(ref - fused)
    print(f"max|diff| {float(diff.max()):.3e}  mean|diff| {float(diff.mean()):.3e}  "
          f"ref max|.| {float(mx.abs(ref).max()):.3e}  exact-equal fraction {float(mx.mean(ref == fused)):.4f}")

    def bench(fn, n=50):
        mx.eval(fn())
        mx.synchronize()
        t0 = perf_counter()
        for _ in range(n):
            mx.eval(fn())
        mx.synchronize()
        return (perf_counter() - t0) / n * 1e3

    print(f"unfused: {bench(lambda: reference(x, experts, weights)):.3f} ms per layer-MoE")
    print(f"fused  : {bench(lambda: fused_routed_experts(x, experts, weights)):.3f} ms per layer-MoE")


if __name__ == "__main__":
    main()
