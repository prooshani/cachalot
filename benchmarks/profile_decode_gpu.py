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
from trace_routing import build_prompt, prompt_sources  # noqa: E402


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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="0 keeps the original short prompt; the attention pieces scale with this")
    args = ap.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        if args.prompt_tokens:
            _, text = prompt_sources()[0]
            ids = build_prompt(rt, enc, text, args.prompt_tokens)
        else:
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
        # Half the reuse layers run at compress_ratio 1 and see twice the keys
        # (COMPRESS_RATIO in text_decode_runtime: ratio 2 below layer 20, 1 above),
        # so one row for a ratio-2 layer understates the token. 15 layers each.
        rec("compressed reuse attention, ratio 2", 15, lambda: compressed_attention_decode_reuse(
            xin, start_pos=pos, compress_ratio=2, window_cache=rt.windows[5], shared_attn=rt.shared_attn, **attn_kw)[0])
        kw25 = rt._common_block_kwargs(25, compressed=True)
        attn_kw25 = {k: kw25[k] for k in attn_kw}
        xin25 = dfm.hc_pre_norm_1d(x, pre_mix, kw25["attn_norm_weight"], eps=1e-20)
        mx.eval(xin25)
        rec("compressed reuse attention, ratio 1", 15, lambda: compressed_attention_decode_reuse(
            xin25, start_pos=pos, compress_ratio=1, window_cache=rt.windows[25], shared_attn=rt.shared_attn, **attn_kw25)[0])
        # The runtime takes route_topk_fused whenever the fused decode path is on
        # (moe_layer_metal), and calls it twice per layer: once for this layer's
        # routing and once for the predictor's look at L+1. route_topk is kept as
        # a second row because it is the path an affine bank took before the
        # fused router shipped, and because the gap between them is large.
        from cachalot.model.router_fused_metal import route_topk_fused

        rec("router route_topk_fused (shipped)", 40,
            lambda: route_topk_fused(xin, kw["gate_weight"], kw["gate_bias"]).indices)
        rec("router route_topk (retired path)", 40,
            lambda: route_topk(xin, kw["gate_weight"], kw["gate_bias"]).indices)
        rec("shared expert (fp8, 3 gemv)", 40, lambda: shared_expert_forward(
            xin, w1=kw["shared_w1"], w1_scales=kw["shared_w1_scales"], w2=kw["shared_w2"], w2_scales=kw["shared_w2_scales"],
            w3=kw["shared_w3"], w3_scales=kw["shared_w3_scales"]))
        fmt = getattr(rt.expert_store, "format", None)
        if fmt is not None and fmt.kind == "affine":
            # What the runtime issues since 2026-09-21 is the traced block
            # (HANDOFF section 9.16): one compiled graph for the six experts,
            # replayed on all forty layers. The per-expert loop below it is the
            # path moe_layer_metal took before that, kept as a second row --
            # reading the loop's number as the runtime's cost overstates the
            # routed experts by a factor of three.
            from cachalot.model import expert_affine as _ea
            from cachalot.model.expert_affine import affine_expert_forward

            w6 = mx.array([0.25] * len(experts), dtype=mx.float32)
            mx.eval(w6)

            def routed_six_traced():
                return _ea.affine_routed_experts(xin, experts, fmt, w6, 10.0)

            def routed_six_loop():
                out = affine_expert_forward(xin, experts[0].as_model_dict(), fmt, 0.25, 10.0)
                for expert in experts[1:]:
                    out = out + affine_expert_forward(xin, expert.as_model_dict(), fmt, 0.25, 10.0)
                return out

            rec(f"routed experts, {fmt.bits}-bit affine g{fmt.group_size} (6), traced",
                40, routed_six_traced)
            rec(f"routed experts, {fmt.bits}-bit affine g{fmt.group_size} (6), per-expert loop",
                40, routed_six_loop)
        else:
            rec("fused routed experts (6)", 40, lambda: fused_routed_experts(xin, experts, w6))
        total = sum(ms * c for _, ms, c in rows)
        print(f"\nsum of chained GPU pieces: {total:.1f} ms (+ head, engram, 40 router syncs)")
        print(f"context at measurement: {pos} positions")


if __name__ == "__main__":
    main()
