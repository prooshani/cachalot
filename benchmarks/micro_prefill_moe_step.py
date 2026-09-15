"""
Per-expert GPU cost of the prefill MoE step at a given prompt length, in isolation
(no SSD): gather rows, expert GEMMs (sgmm or affine8 qmm), scatter-add into the
[T, topk, HIDDEN] slot buffer, eval every 8 experts as moe_prefill_grouped does.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, load_expert_standalone  # noqa: E402
from cachalot.model.fp4_sgmm_metal import routed_expert_forward_sgmm  # noqa: E402
from cachalot.model.moe_prefill_batched import routed_expert_forward_batched  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader  # noqa: E402


def run(t_tokens, rows_per_expert, fn, *, scatter=True, n_experts=64, release=8):
    x = mx.random.normal((t_tokens, 5120)).astype(mx.bfloat16)
    buf = mx.zeros((t_tokens, 6, 5120), dtype=mx.float32)
    mx.eval(x, buf)
    e = run.expert
    kw = dict(w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight, w2_scales=e.w2_scale,
              w3_packed=e.w3_weight, w3_scales=e.w3_scale)
    mx.synchronize()
    t0 = perf_counter()
    pending = []
    for _i in range(n_experts):
        tok = mx.random.randint(0, t_tokens, (rows_per_expert,)).astype(mx.int32)
        slot = mx.random.randint(0, 6, (rows_per_expert,)).astype(mx.int32)
        w = mx.random.uniform(shape=(rows_per_expert,))
        y = fn(x[tok], weights=w, **kw)
        if scatter:
            buf = buf.at[tok, slot].add(y)
        pending.append(y)
        if len(pending) == release:
            mx.eval(buf, *pending)
            pending = []
    mx.eval(buf, *pending)
    mx.synchronize()
    return (perf_counter() - t0) / n_experts * 1e3


def main():
    index = build_expert_index(MODEL_PATH)
    run.expert = load_expert_standalone(ExpertReader(), index[(9, 42)])
    spin = mx.random.normal((4096, 4096))
    for _ in range(30):
        spin = spin @ spin * 1e-3
    mx.eval(spin)
    for t_tokens, rows in ((512, 8), (2048, 32), (2048, 32)):
        for name, fn in (("sgmm", routed_expert_forward_sgmm), ("affine8", routed_expert_forward_batched)):
            full = run(t_tokens, rows, fn)
            no_scatter = run(t_tokens, rows, fn, scatter=False)
            print(f"T={t_tokens} rows={rows} {name:8s}: per expert {full:6.3f} ms | without scatter-add {no_scatter:6.3f} ms",
                  flush=True)
        # scatter-add alone
        t0 = perf_counter()
        buf = mx.zeros((t_tokens, 6, 5120), dtype=mx.float32)
        y = mx.ones((rows, 5120), dtype=mx.float32)
        mx.eval(buf, y)
        mx.synchronize()
        t0 = perf_counter()
        for i in range(64):
            tok = mx.random.randint(0, t_tokens, (rows,)).astype(mx.int32)
            slot = mx.random.randint(0, 6, (rows,)).astype(mx.int32)
            buf = buf.at[tok, slot].add(y)
            if i % 8 == 7:
                mx.eval(buf)
        mx.eval(buf)
        mx.synchronize()
        print(f"T={t_tokens} scatter-add alone: {(perf_counter() - t0) / 64 * 1e3:6.3f} ms per expert "
              f"(buffer {buf.nbytes / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
