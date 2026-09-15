"""gather_qmm over a chunk of experts vs per-expert quantized_matmul: semantics, exactness, speed."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, load_expert_standalone  # noqa: E402
from cachalot.model.fp4_affine8_metal import fp4_to_affine8  # noqa: E402
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
    C = 16
    experts = [load_expert_standalone(reader, index[(11, 7 * i + 3)]) for i in range(C)]
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 40, C)              # tokens per expert (realistic skew)
    R = int(counts.sum())
    x = mx.random.normal((R, 5120)).astype(mx.bfloat16)
    rhs = mx.array(np.repeat(np.arange(C), counts).astype(np.uint32))
    mx.eval(x, rhs)
    print(f"chunk: {C} experts, {R} rows")

    def stack_w1():
        parts = [fp4_to_affine8(e.w1_weight, e.w1_scale, 2304, 5120) for e in experts]
        return (mx.stack([p[0] for p in parts]), mx.stack([p[1] for p in parts]), mx.stack([p[2] for p in parts]))

    w_q, s, b = stack_w1()
    mx.eval(w_q, s, b)
    print("stack shapes", w_q.shape, s.shape, b.shape)

    # semantics: x [R, 1, K] batch=R rows, rhs_indices [R] -> [R, 1, N]
    out = mx.gather_qmm(x[:, None, :], w_q, s, b, rhs_indices=rhs, transpose=True, group_size=32, bits=8, sorted_indices=True)
    mx.eval(out)
    print("gather_qmm out shape", out.shape)
    # reference: per-expert qmm
    ref_parts = []
    start = 0
    for i, e in enumerate(experts):
        n = int(counts[i])
        wq_i, s_i, b_i = fp4_to_affine8(e.w1_weight, e.w1_scale, 2304, 5120)
        ref_parts.append(mx.quantized_matmul(x[start : start + n], wq_i, s_i, b_i, transpose=True, group_size=32, bits=8))
        start += n
    ref = mx.concatenate(ref_parts)
    mx.eval(ref)
    d = mx.abs(out[:, 0, :].astype(mx.float32) - ref.astype(mx.float32))
    print(f"max|diff| gather_qmm vs per-expert qmm: {float(d.max()):.3e} (ref max {float(mx.abs(ref.astype(mx.float32)).max()):.2f})")

    def per_expert():
        outs = []
        start = 0
        for i, e in enumerate(experts):
            n = int(counts[i])
            wq_i, s_i, b_i = fp4_to_affine8(e.w1_weight, e.w1_scale, 2304, 5120)
            outs.append(mx.quantized_matmul(x[start : start + n], wq_i, s_i, b_i, transpose=True, group_size=32, bits=8))
            start += n
        return mx.concatenate(outs)

    def chunked():
        w_q, s, b = stack_w1()
        return mx.gather_qmm(x[:, None, :], w_q, s, b, rhs_indices=rhs, transpose=True, group_size=32, bits=8, sorted_indices=True)

    def chunked_prestacked():
        return mx.gather_qmm(x[:, None, :], w_q, s, b, rhs_indices=rhs, transpose=True, group_size=32, bits=8, sorted_indices=True)

    print(f"per-expert repack+qmm x{C}:      {t(per_expert):.3f} ms")
    print(f"chunk repack+stack+gather_qmm:   {t(chunked):.3f} ms")
    print(f"  of which repack+stack:         {t(lambda: stack_w1()[0]):.3f} ms")
    print(f"  gather_qmm alone (prestacked): {t(chunked_prestacked):.3f} ms")


if __name__ == "__main__":
    main()


def padded_variant():
    index = build_expert_index(MODEL_PATH)
    reader = ExpertReader()
    C = 16
    experts = [load_expert_standalone(reader, index[(11, 7 * i + 3)]) for i in range(C)]
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 40, C)
    mmax = int(counts.max())
    x_pad = mx.random.normal((C, mmax, 5120)).astype(mx.bfloat16)
    parts = [fp4_to_affine8(e.w1_weight, e.w1_scale, 2304, 5120) for e in experts]
    w_q, s, b = (mx.stack([p[0] for p in parts]), mx.stack([p[1] for p in parts]), mx.stack([p[2] for p in parts]))
    mx.eval(x_pad, w_q, s, b)
    idx = mx.arange(C, dtype=mx.uint32)
    out = mx.gather_qmm(x_pad, w_q, s, b, lhs_indices=idx, rhs_indices=idx, transpose=True, group_size=32, bits=8, sorted_indices=True)
    mx.eval(out)
    print("padded gather_qmm out", out.shape)
    ref = mx.stack([mx.quantized_matmul(x_pad[i], w_q[i], s[i], b[i], transpose=True, group_size=32, bits=8) for i in range(C)])
    mx.eval(ref)
    print(f"padded max|diff| vs per-expert qmm: {float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()):.3e}")
    print(f"padded gather_qmm (prestacked, {C}x{mmax} rows): "
          f"{t(lambda: mx.gather_qmm(x_pad, w_q, s, b, lhs_indices=idx, rhs_indices=idx, transpose=True, group_size=32, bits=8, sorted_indices=True)):.3f} ms")
    print(f"batched quantized_matmul (prestacked):            "
          f"{t(lambda: mx.quantized_matmul(x_pad, w_q, s, b, transpose=True, group_size=32, bits=8)):.3f} ms")
    print(f"per-expert qmm on padded rows x{C}:               "
          f"{t(lambda: mx.stack([mx.quantized_matmul(x_pad[i], w_q[i], s[i], b[i], transpose=True, group_size=32, bits=8) for i in range(C)])):.3f} ms")


if __name__ == "__main__":
    padded_variant()
