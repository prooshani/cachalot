"""
MiniMax-M3 decode attention: `cachalot.minimax.gqa_decode` against `mx.fast.scaled_dot_product_attention`
(HANDOFF 18.3). 60 layers of [1, 64, 1, 128] queries over [1, 4, L, 128] keys and values held in cache-shaped
buffers (capacity rounded up to 256), chained in one eval per token as the model runs them; median ms per token
and the largest difference against a float32 reference.

    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_gqa_decode.py [2048 32768 65536]
"""

import sys
import time

import mlx.core as mx
import numpy as np

from cachalot.minimax.gqa_decode import gqa_decode_attention

H, KVH, D, LAYERS = 64, 4, 128, 60
SCALE = D**-0.5


def main():
    lengths = [int(a) for a in sys.argv[1:]] or [2048, 32768, 65536]
    for L in lengths:
        cap = (L + 255) // 256 * 256
        mx.random.seed(L)
        ks = [mx.random.normal((1, KVH, cap, D)).astype(mx.bfloat16) for _ in range(LAYERS)]
        vs = [mx.random.normal((1, KVH, cap, D)).astype(mx.bfloat16) for _ in range(LAYERS)]
        q0 = (mx.random.normal((1, H, 1, D)) * 3).astype(mx.bfloat16)
        mx.eval(ks, vs, q0)

        # correctness on layer 0 against float32
        k, v = ks[0][..., :L, :], vs[0][..., :L, :]
        ref = mx.fast.scaled_dot_product_attention(q0.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=SCALE)
        stock = mx.fast.scaled_dot_product_attention(q0, k, v, scale=SCALE)
        mine = gqa_decode_attention(q0, ks[0], vs[0], L, SCALE)
        err_stock = mx.abs(stock.astype(mx.float32) - ref).max().item()
        err_mine = mx.abs(mine.astype(mx.float32) - ref).max().item()

        def token(fn):
            q = q0
            for i in range(LAYERS):
                o = fn(q, i)
                q = (o * 0.5 + q0 * 0.5).astype(mx.bfloat16)  # a dependency chain, as layer to layer
            return q

        arms = {
            "stock": lambda q, i: mx.fast.scaled_dot_product_attention(q, ks[i][..., :L, :], vs[i][..., :L, :], scale=SCALE),
            "gqa": lambda q, i: gqa_decode_attention(q, ks[i], vs[i], L, SCALE),
        }
        times = {name: [] for name in arms}
        for _ in range(3):
            mx.eval(token(arms["stock"]), token(arms["gqa"]))
        for _ in range(15):
            for name, fn in arms.items():
                t0 = time.perf_counter()
                mx.eval(token(fn))
                times[name].append((time.perf_counter() - t0) * 1000)
        med = {n: float(np.median(t)) for n, t in times.items()}
        print(f"L={L:6d}  stock {med['stock']:6.2f} ms  gqa {med['gqa']:6.2f} ms  "
              f"max|err| vs fp32: stock {err_stock:.2e} gqa {err_mine:.2e}", flush=True)
        del ks, vs
        mx.clear_cache()


if __name__ == "__main__":
    main()
