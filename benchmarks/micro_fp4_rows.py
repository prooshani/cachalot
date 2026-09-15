"""Row-tiled FP4 expert kernels vs the affine-8 quantized_matmul path: accuracy and GPU time per expert by row count."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, load_expert_standalone  # noqa: E402
from cachalot.model.fp4_sgmm_metal import (  # noqa: E402
    fp4_sgmm,
    fp4_sgmm_gate_up,
    routed_expert_forward_sgmm,
)
from cachalot.model.moe_prefill_batched import routed_expert_forward_batched  # noqa: E402
from cachalot.model.moe_prefill_fp4_metal import routed_expert_forward_fp4_rows  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader  # noqa: E402


def chained(fn, n=50):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    outs = [fn() for _ in range(n)]
    mx.eval(*outs)
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    index = build_expert_index(MODEL_PATH)
    e = load_expert_standalone(ExpertReader(), index[(9, 42)])
    kw = dict(w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight, w2_scales=e.w2_scale,
              w3_packed=e.w3_weight, w3_scales=e.w3_scale)
    mx.random.seed(0)
    spin = mx.random.normal((4096, 4096))
    for _ in range(30):
        spin = spin @ spin * 1e-3   # bring the GPU clock up before timing
    mx.eval(spin)
    tiles = [int(t) for t in sys.argv[1].split(",")] if len(sys.argv) > 1 else [None]
    for m in (1, 4, 8, 16, 32, 64, 128, 1):
        x = mx.random.normal((m, 5120)).astype(mx.bfloat16)
        w = mx.random.uniform(shape=(m,)) * 0.5
        mx.eval(x, w)
        ref = routed_expert_forward_batched(x, weights=w, **kw)
        line = f"M={m:4d} | affine8 qmm {chained(lambda x=x, w=w: routed_expert_forward_batched(x, weights=w, **kw)):6.3f} ms"
        for tile in tiles:
            out = routed_expert_forward_fp4_rows(x, weights=w, row_tile=tile, **kw)
            mx.eval(ref, out)
            rel = float(mx.abs(ref - out).max() / (mx.abs(ref).max() + 1e-30))
            same = bool(mx.array_equal(ref.astype(mx.bfloat16), out.astype(mx.bfloat16)))
            t = chained(lambda x=x, w=w, tile=tile: routed_expert_forward_fp4_rows(x, weights=w, row_tile=tile, **kw))
            line += f" | fp4 rows{'' if tile is None else f' mt{tile}'} {t:6.3f} ms (rel {rel:.1e}, bf16-eq {same})"
        out = routed_expert_forward_sgmm(x, weights=w, **kw)
        mx.eval(out)
        rel = float(mx.abs(ref - out).max() / (mx.abs(ref).max() + 1e-30))
        same = bool(mx.array_equal(ref.astype(mx.bfloat16), out.astype(mx.bfloat16)))
        line += f" | sgmm {chained(lambda x=x, w=w: routed_expert_forward_sgmm(x, weights=w, **kw)):6.3f} ms (rel {rel:.1e}, bf16-eq {same})"
        print(line, flush=True)
    x = mx.random.normal((8, 5120)).astype(mx.bfloat16)
    w = mx.ones((8,))
    h = mx.random.normal((8, 2304)).astype(mx.bfloat16)
    mx.eval(x, w, h)
    print(f"M=8 single launches: gate_up {chained(lambda: fp4_sgmm_gate_up(x, e.w1_weight, e.w1_scale, e.w3_weight, e.w3_scale, w, swiglu_limit=10.0)):.3f} ms"
          f" | down {chained(lambda: fp4_sgmm(h, e.w2_weight, e.w2_scale, out_features=5120, in_features=2304)):.3f} ms")


if __name__ == "__main__":
    main()
