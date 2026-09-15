"""Per-layer fused vs unfused routed-expert output on a real decode token."""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model import moe_fused_metal, moe_layer_metal  # noqa: E402
from cachalot.model.expert_metal import routed_expert_forward  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

orig = moe_fused_metal.fused_routed_experts
layer_no = [0]


def checked(x, experts, router_weights, *, hidden_size=5120, intermediate=2304, swiglu_limit=10.0):
    fused = orig(x, experts, router_weights, hidden_size=hidden_size, intermediate=intermediate, swiglu_limit=swiglu_limit)
    ref = mx.zeros((hidden_size,), dtype=mx.float32)
    for e, w in zip(experts, router_weights.tolist(), strict=True):
        y = routed_expert_forward(x, w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight, w2_scales=e.w2_scale,
                                  w3_packed=e.w3_weight, w3_scales=e.w3_scale, weight=float(w), swiglu_limit=swiglu_limit)
        ref = ref + y.astype(mx.float32)
    mx.eval(fused, ref)
    d = mx.abs(fused - ref)
    print(f"  layer-call {layer_no[0]:2d}: k={len(experts)} max|diff| {float(d.max()):.3e} ref max {float(mx.abs(ref).max()):.3e} "
          f"x dtype {x.dtype} x max {float(mx.abs(x).max()):.2f} weights {[round(w, 3) for w in router_weights.tolist()]}", flush=True)
    layer_no[0] += 1
    return fused


moe_layer_metal.fused_routed_experts = checked


def main():
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = list(rt.tokenizer.encode(enc.encode_messages([{"role": "user", "content": "Explain why a mixture-of-experts model streamed from SSD is bandwidth-bound."}], thinking_mode="chat", reasoning_effort=None)))
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        for _ in range(3):
            r = rt.decode_token(tok)
            tok = int(r.logits.argmax().item())
            print("token", tok, repr(rt.tokenizer.decode([tok])), flush=True)


if __name__ == "__main__":
    main()
