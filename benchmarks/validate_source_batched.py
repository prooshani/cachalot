"""Unit parity: batched source / index-source layer path vs the per-token sequential path on real weights."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.attention_compressed import (  # noqa: E402
    compressed_attention_decode_index_source,
    compressed_attention_decode_source,
)
from cachalot.model.compressor_mlx import CompressorState  # noqa: E402
from cachalot.model.indexer_mlx import IndexerState  # noqa: E402
from cachalot.model.shared_attention import SharedAttentionRuntime  # noqa: E402
from cachalot.model.source_prefill_batched import (  # noqa: E402
    compressed_source_chunk,
    index_source_chunk,
)
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

ATTN = ("rope_cos", "rope_sin", "attn_sink", "q_norm_weight", "kv_norm_weight", "wq_a", "wq_a_scales", "wq_b",
        "wq_b_scales", "wkv", "wkv_scales", "wo_a_bf16", "wo_b", "wo_b_scales")


def cmp(name, a, b):
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    d = mx.abs(a32 - b32)
    print(f"    {name:18s} max|diff| {float(d.max()):.3e}  mean {float(d.mean()):.3e}  ref max {float(mx.abs(b32).max()):.3e}")


def topk_equal(a_list, b_list):
    same = sum(1 for a, b in zip(a_list, b_list, strict=True) if a.shape == b.shape and bool(mx.array_equal(a, b)))
    return f"{same}/{len(a_list)} identical top-k sets"


def source_case(rt, layer, ratio, start_pos, T):
    kw = rt._common_block_kwargs(layer, compressed=True)
    attn = {k: kw[k] for k in ATTN}
    comp = dict(
        compressor_norm_weight=rt._t(layer, "attn.compressor.norm.weight"),
        compressor_wkv_weight=rt._t(layer, "attn.compressor.wkv.weight"),
        compressor_wgate_weight=rt._t(layer, "attn.compressor.wgate.weight") if ratio > 1 else None,
        indexer_weights_proj_weight=rt._t(layer, "attn.indexer.weights_proj.weight"),
        indexer_wq_b_weight=rt._t(layer, "attn.indexer.wq_b.weight"),
        indexer_wq_b_scales=rt._t(layer, "attn.indexer.wq_b.scale"),
        indexer_wk_weight=rt._t(layer, "attn.indexer.wk.weight"),
        indexer_k_norm_weight=rt._t(layer, "attn.indexer.k_norm.weight"),
    )
    mx.random.seed(layer * 100 + start_pos)
    x = (mx.random.normal((T, 5120)) * 0.7).astype(mx.bfloat16)
    ring0 = (mx.random.normal((128, 512)) * 0.3).astype(mx.bfloat16) if start_pos else mx.zeros((128, 512), dtype=mx.bfloat16)
    cache_len = 4096 // ratio
    ccache0 = mx.zeros((cache_len, 512), dtype=mx.bfloat16)
    kcache0 = mx.zeros((cache_len, 128), dtype=mx.bfloat16)
    pre = start_pos // ratio
    if pre:
        # fabricate history: random compressed KV / index K for positions before start_pos
        ccache0[:pre] = (mx.random.normal((pre, 512)) * 0.3).astype(mx.bfloat16)
        kcache0[:pre] = (mx.random.normal((pre, 128)) * 0.3).astype(mx.bfloat16)
    mx.eval(x, ring0, ccache0, kcache0)

    def fresh_states():
        cs = CompressorState.create(max_batch_size=1, compress_ratio=2, head_dim=512) if ratio > 1 else None
        if cs is not None and start_pos % ratio:
            cs.kv_state = (mx.random.normal((1, 2, 512)) * 0.5).astype(mx.float32)
            cs.score_state = mx.concatenate([mx.random.normal((1, 1, 512)), mx.full((1, 1, 512), -mx.inf)], axis=1)
            mx.eval(cs.kv_state, cs.score_state)
        ist = IndexerState(k_cache=mx.array(kcache0))
        return cs, ist

    print(f"source layer {layer} (ratio {ratio}), start_pos={start_pos}, T={T}")
    cs_seq, ist_seq = fresh_states()
    seq_kv_state = mx.array(cs_seq.kv_state) if cs_seq is not None else None
    cs_bat, ist_bat = fresh_states()
    if cs_bat is not None:
        cs_bat.kv_state = mx.array(seq_kv_state)
        cs_bat.score_state = mx.array(cs_seq.score_state)

    t0 = perf_counter()
    outs, cache, comp_c = [], ring0, ccache0
    res_seq = []
    sh = SharedAttentionRuntime()
    for t in range(T):
        o, cache, comp_c, r = compressed_attention_decode_source(
            x[t], start_pos=start_pos + t, compress_ratio=ratio, window_cache=cache, compressed_cache=comp_c,
            compressor_state=cs_seq, indexer_state=ist_seq, shared_attn=sh, candidate_source=(layer == 20), **attn, **comp)
        outs.append(o)
        res_seq.append(r)
    ref = mx.stack(outs)
    mx.eval(ref, cache, comp_c, ist_seq.k_cache)
    t_seq = perf_counter() - t0

    t0 = perf_counter()
    out, ring_b, comp_b, res_bat = compressed_source_chunk(
        x, start_pos=start_pos, compress_ratio=ratio, window_cache=ring0, compressed_cache=ccache0,
        compressor_state=cs_bat, indexer_state=ist_bat, candidate_source=(layer == 20), **attn, **comp)
    mx.eval(out, ring_b, comp_b, ist_bat.k_cache)
    t_bat = perf_counter() - t0
    cmp("output", out, ref)
    cmp("window ring", ring_b, cache)
    cmp("compressed cache", comp_b, comp_c)
    cmp("index k cache", ist_bat.k_cache, ist_seq.k_cache)
    if cs_seq is not None:
        cmp("compressor kv", cs_bat.kv_state, cs_seq.kv_state)
    print("    top-k:", topk_equal([r.topk_idxs for r in res_bat], [r.topk_idxs for r in res_seq]),
          "| compress_len equal:", all(a.compress_len == b.compress_len for a, b in zip(res_bat, res_seq, strict=True)))
    if layer == 20:
        same = sum(1 for a, b in zip(res_bat, res_seq, strict=True)
                   if a.candidates.shape == b.candidates.shape and bool(mx.array_equal(a.candidates, b.candidates)))
        print(f"    candidates: {same}/{T} identical")
    print(f"    sequential {t_seq * 1e3:.0f} ms, batched {t_bat * 1e3:.0f} ms")
    return res_bat, comp_b, ist_bat.k_cache


def index_source_case(rt, layer, start_pos, T, index_k, compressed, cands):
    kw = rt._common_block_kwargs(layer, compressed=True)
    attn = {k: kw[k] for k in ATTN}
    idx = dict(
        indexer_weights_proj_weight=rt._t(layer, "attn.indexer.weights_proj.weight"),
        indexer_wq_b_weight=rt._t(layer, "attn.indexer.wq_b.weight"),
        indexer_wq_b_scales=rt._t(layer, "attn.indexer.wq_b.scale"),
    )
    mx.random.seed(layer)
    x = (mx.random.normal((T, 5120)) * 0.7).astype(mx.bfloat16)
    ring0 = (mx.random.normal((128, 512)) * 0.3).astype(mx.bfloat16)
    mx.eval(x, ring0)
    print(f"index-source layer {layer}, start_pos={start_pos}, T={T}")
    t0 = perf_counter()
    outs, cache, res_seq = [], ring0, []
    for t in range(T):
        sh = SharedAttentionRuntime()
        sh.compress_kv = compressed
        sh.index_k = index_k
        sh.candidates = cands[t]
        sh.topk_idxs = mx.zeros((0,), dtype=mx.int32)
        o, cache, r = compressed_attention_decode_index_source(
            x[t], start_pos=start_pos + t, compress_ratio=1, window_cache=cache, shared_attn=sh, **attn, **idx)
        outs.append(o)
        res_seq.append(r)
    ref = mx.stack(outs)
    mx.eval(ref, cache)
    t_seq = perf_counter() - t0
    t0 = perf_counter()
    out, ring_b, res_bat = index_source_chunk(
        x, start_pos=start_pos, compress_ratio=1, window_cache=ring0, compressed_cache=compressed, index_k=index_k,
        candidates_by_token=tuple(cands), **attn, **idx)
    mx.eval(out, ring_b)
    t_bat = perf_counter() - t0
    cmp("output", out, ref)
    cmp("window ring", ring_b, cache)
    print("    top-k:", topk_equal([r.topk_idxs for r in res_bat], [r.topk_idxs for r in res_seq]))
    print(f"    sequential {t_seq * 1e3:.0f} ms, batched {t_bat * 1e3:.0f} ms")


def main():
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        source_case(rt, 2, 2, 0, 40)
        source_case(rt, 2, 2, 201, 64)     # odd start: pending partial group
        source_case(rt, 8, 2, 300, 128)
        res20, comp20, k20 = source_case(rt, 20, 1, 150, 96)
        cands = [r.candidates for r in res20]
        index_source_case(rt, 24, 150, 96, k20, comp20, cands)
        _ = np


if __name__ == "__main__":
    main()
