"""
Where do the ~100 ms of an all-resident decode token go?

Runs one layer's pieces in isolation (attention variants, HC, router, shared
expert, fused experts, head, engram) with eval after each, many repetitions,
and reports per-call time and the per-token total (x layer counts).
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.hyper_connection_mlx import hc_mixes, hc_post, hc_pre  # noqa: E402
from cachalot.model.model_boundary_mlx import final_logits_decode  # noqa: E402
from cachalot.model.moe_fused_metal import fused_routed_experts  # noqa: E402
from cachalot.model.norm_rope_mlx import rms_norm  # noqa: E402
from cachalot.model.router_mlx import route_topk  # noqa: E402
from cachalot.model.shared_expert_metal import shared_expert_forward  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402


def timeit(fn, n=30):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = list(rt.tokenizer.encode(enc.encode_messages(
            [{"role": "user", "content": "List three facts about the deep ocean."}], thinking_mode="chat", reasoning_effort=None)))
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        snap = rt.snapshot()
        rt.decode_token(tok)          # warm everything
        rt.restore(snap)
        t0 = perf_counter()
        r = rt.decode_token(tok)
        mx.eval(r.logits)
        whole = (perf_counter() - t0) * 1e3
        rt.restore(snap)
        print(f"whole all-resident token: {whole:.1f} ms", flush=True)

        pos = rt.position
        x = mx.random.normal((4, 5120)).astype(mx.bfloat16)
        mx.eval(x)
        kw = rt._common_block_kwargs(5, compressed=True)
        kw0 = rt._common_block_kwargs(1, compressed=False)
        xin = rms_norm(hc_pre(x, mx.array([1.0, 0, 0, 0])), kw["attn_norm_weight"], eps=1e-20)
        mx.eval(xin)

        rows = []

        def rec(name, ms, count):
            rows.append((name, ms, count))
            print(f"{name:52s} {ms:7.3f} ms x{count:3d} = {ms * count:6.1f} ms", flush=True)

        rec("hc_mixes (attention HC, sinkhorn kernel)", timeit(lambda: hc_mixes(x, kw["hc_attn_fn"], kw["hc_attn_scale"], kw["hc_attn_base"], norm_eps=1e-20, hc_mult=4, sinkhorn_iters=20, hc_eps=1e-6)[0]), 80)
        rec("hc_pre + rms_norm", timeit(lambda: rms_norm(hc_pre(x, mx.array([1.0, 0, 0, 0])), kw["attn_norm_weight"], eps=1e-20)), 80)
        rec("hc_post", timeit(lambda: hc_post(xin, x, mx.ones((4,)), mx.ones((4, 4)) / 4)), 80)

        from cachalot.model.attention_sliding_window import sliding_window_attention_decode
        attn_kw = {k: kw0[k] for k in ("rope_cos", "rope_sin", "attn_sink", "q_norm_weight", "kv_norm_weight", "wq_a", "wq_a_scales", "wq_b", "wq_b_scales", "wkv", "wkv_scales", "wo_a_bf16", "wo_b", "wo_b_scales")}
        rec("sliding-window attention (layers 0,1)", timeit(lambda: sliding_window_attention_decode(xin, start_pos=pos, window_cache=rt.windows[1], **attn_kw)[0]), 2)

        from cachalot.model.attention_compressed import compressed_attention_decode_reuse
        rkw = {k: kw[k] for k in attn_kw}
        rec("compressed reuse attention (30 layers)", timeit(lambda: compressed_attention_decode_reuse(xin, start_pos=pos, compress_ratio=2, window_cache=rt.windows[5], shared_attn=rt.shared_attn, **rkw)[0]), 30)

        rec("router route_topk + eval", timeit(lambda: route_topk(xin, kw["gate_weight"], kw["gate_bias"]).indices), 40)
        rec("shared expert (fp8, 3 gemv)", timeit(lambda: shared_expert_forward(xin, w1=kw["shared_w1"], w1_scales=kw["shared_w1_scales"], w2=kw["shared_w2"], w2_scales=kw["shared_w2_scales"], w3=kw["shared_w3"], w3_scales=kw["shared_w3_scales"])), 40)

        experts = [rt.expert_store._items[k] for k in list(rt.expert_store._items)[:6]]
        w = mx.ones((6,)) / 4
        rec("fused routed experts (6)", timeit(lambda: fused_routed_experts(xin, experts, w)), 40)
        rec("final head (bf16 gemv) + norm", timeit(lambda: final_logits_decode(x, mx.array([1.0, 0, 0, 0]), rt._global("norm.weight"), rt._global("head.weight"))[1]), 1)

        total = sum(ms * c for _, ms, c in rows)
        print(f"\nsum of isolated pieces (each with its own eval): {total:.1f} ms; whole token {whole:.1f} ms")


if __name__ == "__main__":
    main()
