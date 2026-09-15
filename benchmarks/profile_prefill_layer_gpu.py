"""GPU time of one reuse layer's non-MoE prefill work (attention, shared expert) at T tokens, fp32 vs bf16 GEMM operands."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model import moe_prefill_batched as mpb  # noqa: E402
from cachalot.model.attention_prefill_batched import attention_prefill_batched  # noqa: E402
from cachalot.model.moe_prefill_batched import (  # noqa: E402
    fp8_linear_rows,
    shared_expert_forward_batched,
)
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402


def best(fn, n=3):
    mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(n):
        t0 = perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append(perf_counter() - t0)
    return min(ts) * 1e3


def main():
    t_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        kw = rt._common_block_kwargs(5, compressed=True)
        attn_kw = {k: kw[k] for k in ("rope_cos", "rope_sin", "attn_sink", "q_norm_weight", "kv_norm_weight", "wq_a", "wq_a_scales",
                                      "wq_b", "wq_b_scales", "wkv", "wkv_scales", "wo_a_bf16", "wo_b", "wo_b_scales")}
        mx.random.seed(0)
        x = mx.random.normal((t_tokens, 5120)).astype(mx.bfloat16)
        window = mx.zeros((128, 512), dtype=mx.bfloat16)
        comp = mx.random.normal((t_tokens // 2 + 1, 512)).astype(mx.bfloat16)
        rng = np.random.default_rng(0)
        idxs = []
        for t in range(t_tokens):
            avail = (t + 1) // 2
            k = min(512, avail)
            idxs.append(mx.array(np.sort(rng.choice(avail, k, replace=False)).astype(np.int32)) if k else mx.zeros((0,), dtype=mx.int32))
        mx.eval(x, comp, *idxs)
        # other per-layer pieces that land in the router eval
        from cachalot.model.hc_prefill_exact import hc_mixes_prefill_exact
        xh = mx.random.normal((t_tokens, 4, 5120)).astype(mx.bfloat16)
        mx.eval(xh)
        t_hc = best(lambda: hc_mixes_prefill_exact(xh, kw["hc_attn_fn"], kw["hc_attn_scale"], kw["hc_attn_base"],
                                                    norm_eps=1e-20, hc_mult=4, sinkhorn_iters=20, hc_eps=1e-6)[0])
        t_slot = best(lambda: mx.zeros((t_tokens, 6, 5120), dtype=mx.float32).sum(axis=1))
        buf = mx.zeros((t_tokens, 6, 5120), dtype=mx.float32)
        mx.eval(buf)
        t_sum = best(lambda: buf.sum(axis=1))
        t_alloc = best(lambda: mx.zeros((t_tokens, 6, 5120), dtype=mx.float32) + 1)
        print(f"T={t_tokens} hc_mixes {t_hc:.0f} ms | slot buffer zeros+sum {t_slot:.0f} ms | sum only {t_sum:.0f} ms | "
              f"fresh 252 MB alloc + add {t_alloc:.0f} ms", flush=True)
        for flag in (False, True):
            mpb.PREFILL_BF16_GEMM = flag
            label = "bf16 operands" if flag else "fp32 operands"
            t_attn = best(lambda: attention_prefill_batched(x, start_pos=0, window_cache=window, compressed_cache=comp,
                                                            compressed_idxs_by_token=tuple(idxs), **attn_kw)[0])
            t_win = best(lambda: attention_prefill_batched(x, start_pos=0, window_cache=window, **attn_kw)[0])
            t_shared = best(lambda: shared_expert_forward_batched(
                x, w1=kw["shared_w1"], w1_scales=kw["shared_w1_scales"], w2=kw["shared_w2"], w2_scales=kw["shared_w2_scales"],
                w3=kw["shared_w3"], w3_scales=kw["shared_w3_scales"]))
            t_wqb = best(lambda: fp8_linear_rows(mx.random.normal((t_tokens, 1280)).astype(mx.bfloat16), kw["wq_b"], kw["wq_b_scales"]))
            print(f"T={t_tokens} {label}: attention (window+512 compressed) {t_attn:.0f} ms | attention (window only) {t_win:.0f} ms | "
                  f"shared expert {t_shared:.0f} ms | wq_b linear {t_wqb:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
