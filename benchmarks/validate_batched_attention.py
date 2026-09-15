"""Unit parity: batched attention vs the per-token decode attention on real layer weights and random inputs."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.attention_compressed import compressed_attention_decode_reuse  # noqa: E402
from cachalot.model.attention_prefill_batched import attention_prefill_batched  # noqa: E402
from cachalot.model.attention_sliding_window import sliding_window_attention_decode  # noqa: E402
from cachalot.model.shared_attention import SharedAttentionRuntime  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

ATTN_KEYS = ("rope_cos", "rope_sin", "attn_sink", "q_norm_weight", "kv_norm_weight", "wq_a", "wq_a_scales", "wq_b",
             "wq_b_scales", "wkv", "wkv_scales", "wo_a_bf16", "wo_b", "wo_b_scales")


def compare(name, a, b):
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    print(f"  {name:12s} max|diff| {float(d.max()):.3e}  mean {float(d.mean()):.3e}  ref max {float(mx.abs(b.astype(mx.float32)).max()):.3e}", flush=True)


def main():
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        mx.random.seed(0)
        for start_pos, T in ((0, 40), (200, 64), (300, 300)):
            print(f"sliding layer 1, start_pos={start_pos}, T={T}")
            kw = {k: v for k, v in rt._common_block_kwargs(1, compressed=False).items() if k in ATTN_KEYS}
            x = (mx.random.normal((T, 5120)) * 0.7).astype(mx.bfloat16)
            ring = (mx.random.normal((128, 512)) * 0.3).astype(mx.bfloat16) if start_pos else mx.zeros((128, 512), dtype=mx.bfloat16)
            mx.eval(x, ring)
            t0 = perf_counter()
            outs, cache = [], ring
            for t in range(T):
                o, cache = sliding_window_attention_decode(x[t], start_pos=start_pos + t, window_cache=cache, **kw)
                outs.append(o)
            ref = mx.stack(outs)
            mx.eval(ref, cache)
            t_seq = perf_counter() - t0
            t0 = perf_counter()
            out, new_ring = attention_prefill_batched(x, start_pos=start_pos, window_cache=ring, **kw)
            mx.eval(out, new_ring)
            t_bat = perf_counter() - t0
            compare("output", out, ref)
            compare("ring", new_ring, cache)
            print(f"  sequential {t_seq * 1e3:.0f} ms, batched {t_bat * 1e3:.0f} ms")

        # reuse layer with compressed keys
        start_pos, T, ratio = 200, 64, 2
        print(f"reuse layer 5, start_pos={start_pos}, T={T}")
        kw = {k: v for k, v in rt._common_block_kwargs(5, compressed=True).items() if k in ATTN_KEYS}
        x = (mx.random.normal((T, 5120)) * 0.7).astype(mx.bfloat16)
        ring = (mx.random.normal((128, 512)) * 0.3).astype(mx.bfloat16)
        comp = (mx.random.normal((2048, 512)) * 0.3).astype(mx.bfloat16)
        mx.eval(x, ring, comp)
        topk_by_token = []
        for t in range(T):
            clen = (start_pos + t + 1) // ratio
            k = min(37, clen)
            sel = mx.sort(mx.random.permutation(clen)[:k]).astype(mx.int32)
            topk_by_token.append(sel)
        mx.eval(*topk_by_token)
        outs, cache = [], ring
        t0 = perf_counter()
        for t in range(T):
            sh = SharedAttentionRuntime()
            sh.compress_kv = comp
            sh.topk_idxs = topk_by_token[t]
            o, cache = compressed_attention_decode_reuse(x[t], start_pos=start_pos + t, compress_ratio=ratio, window_cache=cache, shared_attn=sh, **kw)
            outs.append(o)
        ref = mx.stack(outs)
        mx.eval(ref, cache)
        t_seq = perf_counter() - t0
        t0 = perf_counter()
        out, new_ring = attention_prefill_batched(x, start_pos=start_pos, window_cache=ring, compressed_cache=comp,
                                                  compressed_idxs_by_token=tuple(topk_by_token), **kw)
        mx.eval(out, new_ring)
        t_bat = perf_counter() - t0
        compare("output", out, ref)
        compare("ring", new_ring, cache)
        print(f"  sequential {t_seq * 1e3:.0f} ms, batched {t_bat * 1e3:.0f} ms")


if __name__ == "__main__":
    main()
