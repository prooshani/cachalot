"""GPU-time (chained launches) of each piece of compressed_attention_decode_reuse on real layer weights."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model import decode_fused_metal as dfm  # noqa: E402
from cachalot.model.attention_compressed import compressed_attention_decode_reuse  # noqa: E402
from cachalot.model.fp8_fused_metal import fp8_roundtrip_fused  # noqa: E402
from cachalot.model.fp8_linear_metal import (  # noqa: E402
    fp8_linear,
    fp8_linear_quantized,
    quantize_fp8_activation,
)
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.sparse_attn_mlx import get_window_topk_idxs  # noqa: E402
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
        enc = load_official_encoding(MODEL_PATH)
        ids = list(rt.tokenizer.encode(enc.encode_messages(
            [{"role": "user", "content": "Describe the deep scattering layer of the ocean in two sentences."}],
            thinking_mode="chat", reasoning_effort=None)))
        rt.reset()
        res = rt.prefill_tokens(ids)
        rt.decode_token(int(res.logits.argmax().item()))
        pos = rt.position
        kw = rt._common_block_kwargs(5, compressed=True)
        x = mx.random.normal((5120,)).astype(mx.bfloat16)
        mx.eval(x)
        for name in ("wq_a", "wq_b", "wkv", "wo_a_bf16", "wo_b"):
            print(f"  {name:10s} {kw[name].shape} {kw[name].dtype} {kw[name].nbytes / 1e6:.1f} MB")
        print(f"compressed cache {rt.shared_attn.compress_kv.shape}, topk idxs {rt.shared_attn.topk_idxs.shape}, pos {pos}")

        def rec(name, fn):
            print(f"{name:40s} {chained(fn):6.3f} ms", flush=True)

        rec("whole reuse attention", lambda: compressed_attention_decode_reuse(
            x, start_pos=pos, compress_ratio=2, window_cache=rt.windows[5], shared_attn=rt.shared_attn,
            **{k: kw[k] for k in ("rope_cos", "rope_sin", "attn_sink", "q_norm_weight", "kv_norm_weight", "wq_a", "wq_a_scales",
                                  "wq_b", "wq_b_scales", "wkv", "wkv_scales", "wo_a_bf16", "wo_b", "wo_b_scales")})[0])
        qr = fp8_linear(x, kw["wq_a"], kw["wq_a_scales"])
        qx = quantize_fp8_activation(x)
        mx.eval(qr, qx.values, qx.scales)
        rec("quantize x (fused)", lambda: quantize_fp8_activation(x).values)
        rec("wq_a gemv (fp8, pre-quantized)", lambda: fp8_linear_quantized(qx, kw["wq_a"], kw["wq_a_scales"]))
        rec("rms_norm q (fused)", lambda: dfm.rms_norm_decode(qr, kw["q_norm_weight"], eps=1e-20))
        qq = quantize_fp8_activation(qr)
        mx.eval(qq.values)
        rec("wq_b gemv (fp8)", lambda: fp8_linear_quantized(qq, kw["wq_b"], kw["wq_b_scales"]))
        rec("wkv gemv (fp8)", lambda: fp8_linear_quantized(qx, kw["wkv"], kw["wkv_scales"]))
        q = fp8_linear_quantized(qq, kw["wq_b"], kw["wq_b_scales"]).reshape(64, 512)
        mx.eval(q)
        rec("rope q (fused)", lambda: dfm.rope_decode(q[:, -64:], kw["rope_cos"][pos], kw["rope_sin"][pos]))
        rec("concat q nope|rope", lambda: mx.concatenate([q[:, :-64], q[:, -64:]], axis=-1))
        kv = fp8_linear_quantized(qx, kw["wkv"], kw["wkv_scales"])[:512]
        mx.eval(kv)
        rec("fp8 roundtrip kv (fused)", lambda: fp8_roundtrip_fused(kv))
        rec("window cache concat (slot write)", lambda: mx.concatenate([rt.windows[5][:5], kv[None], rt.windows[5][6:]], axis=0))
        comp = rt.shared_attn.compress_kv[: (pos + 1) // 2]
        rec("attention_kv concat (window+compressed)", lambda: mx.concatenate([rt.windows[5], comp], axis=0))
        akv = mx.concatenate([rt.windows[5], comp], axis=0)
        widx = get_window_topk_idxs(128, seqlen=1, start_pos=pos)
        cidx = (rt.shared_attn.topk_idxs + 128).astype(mx.int32)
        idxs = mx.concatenate([widx, cidx[None]], axis=1)
        mx.eval(akv, idxs)
        rec("idx build (window idxs + concat)", lambda: mx.concatenate([get_window_topk_idxs(128, seqlen=1, start_pos=pos), cidx[None]], axis=1))
        rec("sparse attention (fused)", lambda: dfm.sparse_attention_decode(q[None], akv, kw["attn_sink"], idxs, 512 ** -0.5))
        o = dfm.sparse_attention_decode(q[None], akv, kw["attn_sink"], idxs, 512 ** -0.5)[0]
        mx.eval(o)
        wo_a = kw["wo_a_bf16"].reshape(8, 1024, 4096)
        rec("wo_a grouped bf16 matmul", lambda: mx.matmul(wo_a, o.reshape(8, 4096)[..., None])[..., 0].reshape(-1))
        lr = mx.matmul(wo_a, o.reshape(8, 4096)[..., None])[..., 0].reshape(-1)
        mx.eval(lr)
        rec("wo_b fp8 linear (quant + gemv)", lambda: fp8_linear(lr, kw["wo_b"], kw["wo_b_scales"]))


if __name__ == "__main__":
    main()
