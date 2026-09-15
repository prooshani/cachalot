"""Fused router vs MLX route_topk on real gate weights; attention kernel threadgroup sweep (GPU time, chained)."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model import decode_fused_metal as dfm  # noqa: E402
from cachalot.model.router_fused_metal import route_topk_fused  # noqa: E402
from cachalot.model.router_mlx import route_topk  # noqa: E402
from cachalot.model.sparse_attn_mlx import get_window_topk_idxs, sparse_attention  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402


def chained(fn, n=40):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    outs = [fn() for _ in range(n)]
    mx.eval(*outs)
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        mx.random.seed(1)
        same = 0
        total = 0
        wmax = 0.0
        for layer in range(40):
            kw = rt._common_block_kwargs(layer, compressed=layer >= 2)
            w, b = kw["gate_weight"], kw["gate_bias"]
            if layer == 0:
                print("gate weight dtype/shape", w.dtype, w.shape, "bias", b.dtype)
            for _ in range(5):
                x = (mx.random.normal((5120,)) * 2).astype(mx.bfloat16)
                ref = route_topk(x, w, b)
                out = route_topk_fused(x, w, b)
                mx.eval(ref.indices, ref.weights, out.indices, out.weights)
                total += 1
                if ref.indices.tolist() == out.indices.tolist():
                    same += 1
                    wmax = max(wmax, float(mx.abs(ref.weights - out.weights).max()))
                else:
                    print(f"  layer {layer}: indices differ {ref.indices.tolist()} vs {out.indices.tolist()}; "
                          f"sel top ref {[float(v) for v in (ref.scores + b.astype(mx.float32))[ref.indices]]}")
        print(f"router indices identical {same}/{total}; max |weight diff| {wmax:.2e}")
        kw = rt._common_block_kwargs(5, compressed=True)
        x = (mx.random.normal((5120,)) * 2).astype(mx.bfloat16)
        mx.eval(x)
        print(f"router GPU: mlx {chained(lambda: route_topk(x, kw['gate_weight'], kw['gate_bias']).weights):.3f} ms | "
              f"fused {chained(lambda: route_topk_fused(x, kw['gate_weight'], kw['gate_bias']).weights):.3f} ms")

        q = (mx.random.normal((64, 512)) * 0.5).astype(mx.bfloat16)
        sink = mx.random.normal((64,)).astype(mx.float32)
        kv = (mx.random.normal((128 + 2048, 512)) * 0.5).astype(mx.bfloat16)
        win = get_window_topk_idxs(128, 1, 300)
        comp = mx.concatenate([win, (mx.sort(mx.random.permutation(2048)[:512]) + 128).astype(mx.int32)[None]], axis=1)
        mx.eval(q, kv, win, comp)
        for label, idxs in (("window 128", win), ("window+512", comp)):
            line = f"attention {label:11s} GPU: mlx {chained(lambda idxs=idxs: sparse_attention(q[None], kv, sink, idxs, 512 ** -0.5)):.3f} ms"
            for nt in (256, 512):
                out = dfm.sparse_attention_decode_1d(q, kv, sink, idxs[0], 512 ** -0.5, n_threads=nt)
                ref = sparse_attention(q[None], kv, sink, idxs, 512 ** -0.5)[0]
                mx.eval(out, ref)
                diff = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
                line += f" | t{nt} {chained(lambda idxs=idxs, nt=nt: dfm.sparse_attention_decode_1d(q, kv, sink, idxs[0], 512 ** -0.5, n_threads=nt)):.3f} ms (diff {diff:.1e})"
            print(line)


if __name__ == "__main__":
    main()
