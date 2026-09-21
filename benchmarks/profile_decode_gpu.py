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

        # ------------------------------------------------------------------
        # The ten layers the two reuse-attention rows above do not cover.
        #
        # _decode_token_impl runs layer 0 and layer 1 as sliding-window layers,
        # 2/8/14/20 as compressed sources, 24/28/32/36 as index-only sources
        # and the other thirty as compressed reuse. Only the last class had
        # ever been timed, which is why HANDOFF section 7.1.4 could only say
        # "ten unprofiled attention layers" about part of its 20 ms gap.
        # ------------------------------------------------------------------
        import inspect

        from cachalot.model.attention_compressed import (
            compressed_attention_decode_index_source,
            compressed_attention_decode_source,
        )
        from cachalot.model.attention_layer0 import layer0_attention_decode
        from cachalot.model.attention_sliding_window import (
            sliding_window_attention_decode,
        )
        from cachalot.model.engram_mlx import engram_forward_decode
        from cachalot.model.engram_rows import load_engram_rows
        from cachalot.model.model_boundary_mlx import final_logits_decode
        from cachalot.model.text_decode_runtime import (
            ENGRAM_LAYER_IDS,
            INDEX_TOPK,
            SOURCE_LAYERS,
        )

        def pick(fn, pool):
            names = set(inspect.signature(fn).parameters)
            return {k: v for k, v in pool.items() if k in names}

        kw0 = rt._common_block_kwargs(0, compressed=False)
        xin0 = dfm.hc_pre_norm_1d(x, pre_mix, kw0["attn_norm_weight"], eps=1e-20)
        mx.eval(xin0)
        pool0 = dict(kw0, start_pos=pos, window_cache=rt.windows[0])
        rec("layer-0 attention (sliding window)", 1,
            lambda: layer0_attention_decode(xin0, **pick(layer0_attention_decode, pool0))[0])

        kw1 = rt._common_block_kwargs(1, compressed=False)
        xin1 = dfm.hc_pre_norm_1d(x, pre_mix, kw1["attn_norm_weight"], eps=1e-20)
        mx.eval(xin1)
        pool1 = dict(kw1, start_pos=pos, window_cache=rt.windows[1])
        rec("layer-1 attention (sliding window)", 1,
            lambda: sliding_window_attention_decode(xin1, **pick(sliding_window_attention_decode, pool1))[0])

        def source_pool(layer_id):
            ratio = SOURCE_LAYERS[layer_id]
            common = rt._common_block_kwargs(layer_id, compressed=True)
            return dict(
                common,
                start_pos=pos,
                compress_ratio=ratio,
                window_cache=rt.windows[layer_id],
                compressed_cache=rt.compressed_caches[layer_id],
                # _decode_source passes no compressor at ratio 1 (layer 20)
                compressor_state=None if ratio == 1 else rt.compressor_states[layer_id],
                compressor_wgate_weight=None if ratio == 1 else rt._t(layer_id, "attn.compressor.wgate.weight"),
                indexer_state=rt.indexer_states[layer_id],
                shared_attn=rt.shared_attn,
                compressor_norm_weight=rt._t(layer_id, "attn.compressor.norm.weight"),
                compressor_wkv_weight=rt._t(layer_id, "attn.compressor.wkv.weight"),
                indexer_weights_proj_weight=rt._t(layer_id, "attn.indexer.weights_proj.weight"),
                indexer_wq_b_weight=rt._t(layer_id, "attn.indexer.wq_b.weight"),
                indexer_wq_b_scales=rt._t(layer_id, "attn.indexer.wq_b.scale"),
                indexer_wk_weight=rt._t(layer_id, "attn.indexer.wk.weight"),
                indexer_k_norm_weight=rt._t(layer_id, "attn.indexer.k_norm.weight"),
                index_topk=INDEX_TOPK,
            )

        # Layers 2, 8 and 14 are ratio-2 sources; layer 20 is the ratio-1 one.
        for src_layer, src_count in ((2, 3), (20, 1)):
            sp = source_pool(src_layer)
            xsrc = dfm.hc_pre_norm_1d(x, pre_mix, sp["attn_norm_weight"], eps=1e-20)
            mx.eval(xsrc)
            rec(f"compressed attention, source ratio {sp['compress_ratio']}", src_count,
                (lambda _x=xsrc, _p=sp: compressed_attention_decode_source(
                    _x, **pick(compressed_attention_decode_source, _p))[0]))

        # An index-only source consumes the candidate mask layer 20 published
        # while the *last* token was decoded, so it is timed at that token's
        # position; at pos the mask is one entry short of compress_len.
        isp = dict(
            rt._common_block_kwargs(24, compressed=True),
            start_pos=pos - 1,
            compress_ratio=1,
            window_cache=rt.windows[24],
            shared_attn=rt.shared_attn,
            indexer_weights_proj_weight=rt._t(24, "attn.indexer.weights_proj.weight"),
            indexer_wq_b_weight=rt._t(24, "attn.indexer.wq_b.weight"),
            indexer_wq_b_scales=rt._t(24, "attn.indexer.wq_b.scale"),
            index_topk=INDEX_TOPK,
        )
        xidx = dfm.hc_pre_norm_1d(x, pre_mix, isp["attn_norm_weight"], eps=1e-20)
        mx.eval(xidx)
        rec("compressed attention, index-only source", 4,
            lambda: compressed_attention_decode_index_source(
                xidx, **pick(compressed_attention_decode_index_source, isp))[0])

        # The head: one bf16 gemv against a 130k-row vocabulary, once a token.
        norm_w = rt._global("norm.weight")
        head_w = rt._global("head.weight")
        rec("final norm + head logits", 1, lambda: final_logits_decode(
            x, pre_mix, norm_w, head_w, head_chunk_size=rt.head_chunk_size)[1])

        # Engram, layers 1 and 14. The row read is I/O and is excluded: this
        # times the forward the GPU runs on rows already in memory.
        hash_rows = rt.engram_hash.push(tok)
        for eng_layer in ENGRAM_LAYER_IDS:
            eng_rows = load_engram_rows(
                rt.engram_reader,
                rt.engram_layouts[eng_layer],
                hash_rows[ENGRAM_LAYER_IDS.index(eng_layer)],
            )
            mx.eval(eng_rows)
            rec(f"engram forward, layer {eng_layer}", 1,
                (lambda _r=eng_rows, _l=eng_layer: engram_forward_decode(x, _r, rt.layers[_l])))
        # ------------------------------------------------------------------
        # What a source layer pays over a reuse layer.
        #
        # HANDOFF section 7.1.5 measured a source layer's attention at 1.483 ms
        # against a reuse layer's 0.441 on the same context -- 5.5 ms per token
        # across the eight source layers, and nothing in this project had ever
        # decomposed it. The extra work is three pieces: the compressor's
        # partial-group pooling, the indexer (its own FP4 index-K cache and a
        # top-k over the compressed positions at INDEX_TOPK) and the
        # compressed-KV write. The first two are timed here; the write is what
        # is left over.
        #
        # These rows carry count 0 so they do not double-count against the
        # source-attention rows above; their per-token cost is derived below.
        #
        # A ratio-2 compressor returns a latent only on the closing token of
        # each group, so both phases are timed: a token pays one of each per
        # pair of positions.
        # ------------------------------------------------------------------
        from cachalot.model.compressor_mlx import compressor_forward
        from cachalot.model.decode_fused_metal import rms_norm_decode
        from cachalot.model.fp8_linear_metal import fp8_linear
        from cachalot.model.indexer_mlx import indexer_decode_base

        sp2 = source_pool(2)
        xs2 = dfm.hc_pre_norm_1d(x, pre_mix, sp2["attn_norm_weight"], eps=1e-20)
        mx.eval(xs2)

        def compress(at_pos):
            return compressor_forward(
                xs2.reshape(1, 1, xs2.shape[0]),
                start_pos=at_pos,
                compress_ratio=2,
                norm_weight=sp2["compressor_norm_weight"],
                wkv_weight=sp2["compressor_wkv_weight"],
                wgate_weight=sp2["compressor_wgate_weight"],
                state=sp2["compressor_state"],
                eps=1e-20,
            )

        closing = pos if compress(pos) is not None else pos + 1
        # The open-group call returns no latent and only writes the partial
        # state, so the state array is what has to be evaluated to force it.
        rec("source: compressor, closing token (latent out)", 0, lambda: compress(closing))
        def compress_open():
            out = compress(closing + 1)
            return out if out is not None else sp2["compressor_state"].kv_state

        rec("source: compressor, open group (no latent)", 0, compress_open)

        latent = compress(closing)
        mx.eval(latent)
        latent_1d = latent.reshape(latent.shape[-1])
        qr2 = rms_norm_decode(
            fp8_linear(xs2, sp2["wq_a"], sp2["wq_a_scales"]),
            sp2["q_norm_weight"],
            eps=1e-20,
        )
        mx.eval(qr2, latent_1d)

        def index_at(width):
            return indexer_decode_base(
                x=xs2,
                qr=qr2,
                latent=latent_1d,
                start_pos=closing,
                compress_ratio=2,
                state=sp2["indexer_state"],
                rope_cos=sp2["rope_cos"],
                rope_sin=sp2["rope_sin"],
                weights_proj_weight=sp2["indexer_weights_proj_weight"],
                wq_b_weight=sp2["indexer_wq_b_weight"],
                wq_b_scales=sp2["indexer_wq_b_scales"],
                wk_weight=sp2["indexer_wk_weight"],
                k_norm_weight=sp2["indexer_k_norm_weight"],
                norm_eps=1e-20,
                index_topk=width,
            ).topk_idxs

        idx_rows = {}
        for width in (INDEX_TOPK, 128, 256, 1024):
            if width in idx_rows:
                continue
            label = f"source: indexer, index_topk {width}"
            if width == INDEX_TOPK:
                label += " (shipped)"
            before_n = len(rows)
            rec(label, 0, lambda w=width: index_at(w))
            idx_rows[width] = rows[before_n][1]

        comp_close = next(ms for name, ms, _ in rows if name.startswith("source: compressor, closing"))
        comp_open = next(ms for name, ms, _ in rows if name.startswith("source: compressor, open"))
        comp_mean = (comp_close + comp_open) / 2
        idx_ship = idx_rows[INDEX_TOPK]
        print(
            f"\n  source-layer extra, derived: compressor {comp_mean:.3f} ms x 3 ratio-2 layers"
            f" = {comp_mean * 3:.1f} ms/token; indexer {idx_ship:.3f} ms x 4 base-indexer layers"
            f" = {idx_ship * 4:.1f} ms/token",
            flush=True,
        )
        print(
            "  index_topk: "
            + ", ".join(f"{w} -> {idx_rows[w]:.3f} ms" for w in sorted(idx_rows)),
            flush=True,
        )

        rt.restore(snap)
        total = sum(ms * c for _, ms, c in rows)
        print(f"\nsum of chained GPU pieces: {total:.1f} ms (all forty layers, head and Engram included)")
        print(f"context at measurement: {pos} positions")

        # ------------------------------------------------------------------
        # What an eval costs to drain.
        #
        # Every row above is chained: forty launches into one graph, one eval.
        # A token does not pay for its layers that way -- moe_layer_metal.py:180
        # evaluates this layer's routing before the next layer's graph can be
        # built, 40 times, and the tail adds four more. If an eval has a fixed
        # cost the chained rows exclude it by construction, and the sum of the
        # pieces is not allowed to close on the token. This prices it on two
        # real pieces of the decode graph.
        # ------------------------------------------------------------------
        def drained(fn, n=40):
            mx.eval(fn())
            mx.synchronize()
            t0 = perf_counter()
            outs = [fn() for _ in range(n)]
            mx.eval(*outs)
            mx.synchronize()
            chain = (perf_counter() - t0) / n * 1e3
            t0 = perf_counter()
            for _ in range(n):
                mx.eval(fn())
            mx.synchronize()
            each = (perf_counter() - t0) / n * 1e3
            return chain, each

        print()
        for name, fn in (
            ("compressed reuse attention, ratio 1", lambda: compressed_attention_decode_reuse(
                xin25, start_pos=pos, compress_ratio=1, window_cache=rt.windows[25],
                shared_attn=rt.shared_attn, **attn_kw25)[0]),
            ("router route_topk_fused", lambda: route_topk_fused(xin, kw["gate_weight"], kw["gate_bias"]).indices),
        ):
            chain, each = drained(fn)
            print(f"eval drain, {name:38s}: chained {chain:6.3f} ms  one eval each {each:6.3f} ms  "
                  f"delta {each - chain:6.3f} ms x 44 = {(each - chain) * 44:5.1f} ms/token", flush=True)


if __name__ == "__main__":
    main()
