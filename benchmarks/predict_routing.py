"""
Can layer L+1's (and L+2's) routing be predicted early enough to be useful?

Two prediction sources are scored against the same actual routing.

"late" is what the runtime does today (moe_layer_metal.PREDICT_AHEAD): apply
layer L+k's router to the post-attention normalized activation that layer L's
own router sees. It is the most accurate activation available, and it is
produced after attention L has already run.

"early" applies layer L+k's router to layer L's block input, the residual
stream before attention L, normalized with layer L+k's FFN norm weight and the
attention pre-mix the block was handed. It is a worse activation, because it
skips attention L as well as the MoE, but it is available one attention
earlier.

Lead time is why this matters. Decode compute is about 3.3 ms per layer and a
15.5 MiB expert read takes about 5.3 ms (docs/HANDOFF-2026-09-17.md section 1
and the read-pool split in benchmarks/decode_anatomy.py), so a prefetch issued
from the "late" activation lands roughly 2 ms after the layer that needs it has
already blocked. Issuing from the "early" activation buys attention L, about
2.9 ms, which is the difference between arriving in time and not.

Reports top-k recall against the actual routing and, separately, recall on the
experts that actually missed the cache, which are the only ones a prefetch has
to load.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model import moe_layer_metal as mlm  # noqa: E402
from cachalot.model.decode_fused_metal import hc_pre_norm_decode  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.router_fused_metal import route_topk_fused  # noqa: E402
from cachalot.model.text_decode_runtime import NORM_EPS, TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402

STATE = {
    "rt": None,
    "pred": {},
    "stats": defaultdict(lambda: [0, 0, 0, 0]),
    "top": (3, 6, 12, 24),
    # layer_id -> (block input residual, attention pre-mix), captured before
    # attention runs, which is where an earlier prefetch would be issued
    "block_in": {},
}
BLOCKS = {}


def capture_block_inputs() -> None:
    """Record every decode block's input activation before attention runs."""

    def wrap(name: str, takes_layer_id: bool) -> None:
        original = getattr(TextDecodeRuntime, name)

        if takes_layer_id:
            def patched(self, layer_id, x, pre_mix, start_pos, *a, **kw):
                STATE["block_in"][int(layer_id)] = (x, pre_mix)
                return original(self, layer_id, x, pre_mix, start_pos, *a, **kw)
        else:
            def patched(self, x, pre_mix, start_pos, *a, **kw):
                STATE["block_in"][0] = (x, pre_mix)
                return original(self, x, pre_mix, start_pos, *a, **kw)

        setattr(TextDecodeRuntime, name, patched)

    wrap("_decode_layer0", False)
    for name in ("_decode_sliding", "_decode_source", "_decode_reuse", "_decode_index_source"):
        wrap(name, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--decode-tokens", type=int, default=48)
    args = ap.parse_args()
    orig = mlm.moe_layer_forward

    def patched(x, *, layer_id, gate_weight, gate_bias, expert_index, expert_store, **kw):
        rt = STATE["rt"]
        # 1) score this layer's actual routing against earlier predictions
        route_now = route_topk_fused(x, gate_weight, gate_bias)
        actual = set(int(i) for i in route_now.indices.tolist())
        resident = {e for e in actual if expert_store.is_resident((layer_id, e))}
        misses = actual - resident
        for source in ("late", "early"):
            for ahead in (1, 2):
                pred = STATE["pred"].pop((source, layer_id, ahead), None)
                if pred is None:
                    continue
                for top in STATE["top"]:
                    s = STATE["stats"][(source, ahead, top)]
                    chosen = set(pred[:top])
                    s[0] += len(actual & chosen)
                    s[1] += len(actual)
                    s[2] += len(misses & chosen)
                    s[3] += len(misses)

        # 2) predict the next layers' routing from two activations of this
        #    layer: the post-attention one the router just used ("late", what
        #    the runtime does today) and this block's input ("early", one
        #    attention earlier and therefore issuable sooner)
        block_in = STATE["block_in"].get(layer_id)
        for ahead in (1, 2):
            nxt = layer_id + ahead
            if nxt >= 40:
                continue
            kwn = BLOCKS.get(nxt)
            if kwn is None:
                kwn = BLOCKS[nxt] = rt._common_block_kwargs(nxt, compressed=nxt >= 2)

            r = route_topk_fused(x, kwn["gate_weight"], kwn["gate_bias"], topk=max(STATE["top"]))
            # route_topk returns ascending selection order; strongest last
            STATE["pred"][("late", nxt, ahead)] = list(reversed(r.indices.tolist()))

            if block_in is not None:
                x_block, pre_mix = block_in
                early_input = hc_pre_norm_decode(
                    x_block, pre_mix, kwn["ffn_norm_weight"], eps=NORM_EPS
                )
                r_early = route_topk_fused(
                    early_input, kwn["gate_weight"], kwn["gate_bias"], topk=max(STATE["top"])
                )
                STATE["pred"][("early", nxt, ahead)] = list(reversed(r_early.indices.tolist()))
        return orig(x, layer_id=layer_id, gate_weight=gate_weight, gate_bias=gate_bias,
                    expert_index=expert_index, expert_store=expert_store, **kw)

    for mod_name in ("block_sliding_window", "block_layer0", "block_compressed_source",
                     "block_compressed_reuse", "block_compressed_index_source"):
        mod = __import__(f"cachalot.model.{mod_name}", fromlist=["moe_layer_forward"])
        mod.moe_layer_forward = patched

    capture_block_inputs()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        STATE["rt"] = rt
        enc = load_official_encoding(MODEL_PATH)
        ids = build_prompt(rt, enc, prompt_sources()[0][1], args.prompt_tokens)
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        STATE["stats"].clear()
        for _ in range(args.decode_tokens):
            r = rt.decode_token(tok)
            tok = int(r.logits.argmax().item())
        print(f"{args.decode_tokens} decoded tokens after a {len(ids)}-token prompt")
        where = {
            "late": "layer L's post-attention router input (what the runtime uses today)",
            "early": "layer L's block input, one attention earlier",
        }
        for source in ("late", "early"):
            print(f"\n  from {where[source]}:")
            for (src_name, ahead, top), (hit, n, mhit, mn) in sorted(STATE["stats"].items()):
                if src_name != source or n == 0:
                    continue
                print(f"    predict layer L+{ahead}, top-{top:2d}: "
                      f"recall {hit / n:5.1%} of routed experts | {mhit / max(mn, 1):5.1%} of cache misses "
                      f"({mn / args.decode_tokens:.1f} misses/token in scored layers)")


if __name__ == "__main__":
    main()
