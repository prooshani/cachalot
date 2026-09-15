"""Fused multi-expert prefill kernel: exactness vs per-token decode GEMV path, speed vs per-expert affine8 path."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, load_expert_standalone  # noqa: E402
from cachalot.model.expert_metal import routed_expert_forward  # noqa: E402
from cachalot.model.moe_prefill_batched import routed_expert_forward_batched  # noqa: E402
from cachalot.model.moe_prefill_fused_metal import fused_experts_chunk  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader  # noqa: E402


def t(fn, n=20):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    index = build_expert_index(MODEL_PATH)
    reader = ExpertReader()
    C = 4
    experts = [load_expert_standalone(reader, index[(11, 7 * i + 3)]) for i in range(C)]
    rng = np.random.default_rng(0)
    T = 256
    counts = rng.integers(1, 40, C)
    x = (mx.random.normal((T, 5120)) * 0.7).astype(mx.bfloat16)
    row_tok = np.concatenate([rng.choice(T, c, replace=False) for c in counts]).astype(np.int32)
    row_w = rng.uniform(0.1, 0.6, row_tok.size).astype(np.float32)
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    mx.eval(x)
    rt_, rw_, st_ = mx.array(row_tok), mx.array(row_w), mx.array(starts)

    out = fused_experts_chunk(x, experts, rt_, rw_, st_)
    mx.eval(out)
    # reference 1: decode GEMV path per (token, expert)
    ref_rows = []
    for e_i, e in enumerate(experts):
        for r in range(starts[e_i], starts[e_i + 1]):
            y = routed_expert_forward(x[int(row_tok[r])], w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight,
                                      w2_scales=e.w2_scale, w3_packed=e.w3_weight, w3_scales=e.w3_scale, weight=float(row_w[r]))
            ref_rows.append(y)
    ref = mx.stack(ref_rows)
    mx.eval(ref)
    d = mx.abs(out - ref)
    print(f"rows {row_tok.size}: fused vs decode GEMV path  max|diff| {float(d.max()):.3e}  exact fraction {float(mx.mean(out == ref)):.4f}  ref max {float(mx.abs(ref).max()):.3f}")
    # reference 2: current per-expert affine8 batched path (bf16 gate/up)
    ref2 = []
    for e_i, e in enumerate(experts):
        rows = slice(int(starts[e_i]), int(starts[e_i + 1]))
        ref2.append(routed_expert_forward_batched(x[rt_[rows]], w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight,
                                                  w2_scales=e.w2_scale, w3_packed=e.w3_weight, w3_scales=e.w3_scale, weights=rw_[rows]))
    ref2 = mx.concatenate(ref2)
    mx.eval(ref2)
    print(f"fused vs affine8 batched path       max|diff| {float(mx.abs(out - ref2).max()):.3e}")

    def per_expert_affine8():
        outs = []
        for e_i, e in enumerate(experts):
            rows = slice(int(starts[e_i]), int(starts[e_i + 1]))
            outs.append(routed_expert_forward_batched(x[rt_[rows]], w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight,
                                                      w2_scales=e.w2_scale, w3_packed=e.w3_weight, w3_scales=e.w3_scale, weights=rw_[rows]))
        return mx.concatenate(outs)

    print(f"per-expert affine8 path x{C}: {t(per_expert_affine8):.3f} ms")
    print(f"fused chunk (2 launches):    {t(lambda: fused_experts_chunk(x, experts, rt_, rw_, st_)):.3f} ms")


if __name__ == "__main__":
    main()
