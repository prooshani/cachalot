"""
GPU-time breakdown of an all-resident decode token.

Each component is launched N times inside one lazy graph and evaluated once,
so per-launch time excludes the ~0.15 ms mx.eval floor that the isolated
profile (profile_decode_components.py) includes. Also reports min/median
whole-token time over many repeats with the async MoE dispatch on and off.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model import decode_fused_metal as dfm  # noqa: E402
from cachalot.model import moe_layer_metal as mlm  # noqa: E402
from cachalot.model.attention_compressed import compressed_attention_decode_reuse  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.moe_fused_metal import fused_routed_experts  # noqa: E402
from cachalot.model.router_mlx import route_topk  # noqa: E402
from cachalot.model.shared_expert_metal import shared_expert_forward  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402


def chained(fn, n=40):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    outs = [fn() for _ in range(n)]
    t_build = perf_counter() - t0
    mx.eval(*outs)
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3, t_build / n * 1e3


def token_times(rt, snap, tok, n=15):
    ts = []
    for _ in range(n):
        rt.restore(snap)
        t0 = perf_counter()
        r = rt.decode_token(tok)
        mx.eval(r.logits)
        ts.append((perf_counter() - t0) * 1e3)
    return min(ts), statistics.median(ts)


def main():
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = list(rt.tokenizer.encode(enc.encode_messages(
            [{"role": "user", "content": "Describe the deep scattering layer of the ocean in two sentences."}],
            thinking_mode="chat", reasoning_effort=None)))
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        rt.decode_token(tok)
        snap = rt.snapshot()
        for flag in (False, True, False, True):
            mlm.ASYNC_MOE = flag
            lo, med = token_times(rt, snap, tok)
            print(f"whole token async_moe={flag}: min {lo:.1f} ms median {med:.1f} ms", flush=True)
        mlm.ASYNC_MOE = True

        rt.restore(snap)
        pos = rt.position
        kw = rt._common_block_kwargs(5, compressed=True)
        x = mx.random.normal((4, 5120)).astype(mx.bfloat16)
        pre_mix = mx.array([1.0, 0, 0, 0]).astype(mx.float32)
        mx.eval(x, pre_mix)
        xin = dfm.hc_pre_norm_1d(x, pre_mix, kw["attn_norm_weight"], eps=1e-20)
        mx.eval(xin)
        attn_kw = {k: kw[k] for k in ("rope_cos", "rope_sin", "attn_sink", "q_norm_weight", "kv_norm_weight", "wq_a", "wq_a_scales",
                                      "wq_b", "wq_b_scales", "wkv", "wkv_scales", "wo_a_bf16", "wo_b", "wo_b_scales")}
        experts = [rt.expert_store._items[k] for k in list(rt.expert_store._items)[:6]]
        w6 = mx.ones((6,)) / 6
        rows = []

        def rec(name, count, fn):
            ms, build = chained(fn)
            rows.append((name, ms, count))
            print(f"{name:44s} {ms:6.3f} ms (build {build:5.3f}) x{count:3d} = {ms * count:5.1f} ms", flush=True)

        rec("hc_mixes_1d (sinkhorn)", 80, lambda: dfm.hc_mixes_1d(x, kw["hc_attn_fn"], kw["hc_attn_scale"], kw["hc_attn_base"],
                                                                  norm_eps=1e-20, hc_mult=4, sinkhorn_iters=20, hc_eps=1e-6)[0])
        rec("hc_pre_norm_1d", 80, lambda: dfm.hc_pre_norm_1d(x, pre_mix, kw["attn_norm_weight"], eps=1e-20))
        rec("hc_post_1d", 80, lambda: dfm.hc_post_1d(xin, x, mx.ones((4,)), mx.ones((4, 4)) / 4))
        rec("compressed reuse attention", 30, lambda: compressed_attention_decode_reuse(
            xin, start_pos=pos, compress_ratio=2, window_cache=rt.windows[5], shared_attn=rt.shared_attn, **attn_kw)[0])
        rec("router route_topk", 40, lambda: route_topk(xin, kw["gate_weight"], kw["gate_bias"]).indices)
        rec("shared expert (fp8, 3 gemv)", 40, lambda: shared_expert_forward(
            xin, w1=kw["shared_w1"], w1_scales=kw["shared_w1_scales"], w2=kw["shared_w2"], w2_scales=kw["shared_w2_scales"],
            w3=kw["shared_w3"], w3_scales=kw["shared_w3_scales"]))
        rec("fused routed experts (6)", 40, lambda: fused_routed_experts(xin, experts, w6))
        total = sum(ms * c for _, ms, c in rows)
        print(f"\nsum of chained GPU pieces: {total:.1f} ms (+ head, engram, 40 router syncs)")


if __name__ == "__main__":
    main()
