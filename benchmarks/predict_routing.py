"""
Can layer L+1's (and L+2's) routing be predicted one layer early?

Proxy: apply layer L+k's router to the normalized activation that layer L's
router sees (available before layer L's experts are loaded). Reports top-k
recall against the actual routing and, separately, recall on the experts
that actually missed the cache (the ones a prefetch would have to load).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model import moe_layer_metal as mlm  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.router_fused_metal import route_topk_fused  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402

STATE = {"rt": None, "pred": {}, "stats": defaultdict(lambda: [0, 0, 0, 0]), "top": (6, 12, 24)}
BLOCKS = {}


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
        for ahead in (1, 2):
            pred = STATE["pred"].pop((layer_id, ahead), None)
            if pred is None:
                continue
            for top in STATE["top"]:
                s = STATE["stats"][(ahead, top)]
                p = set(pred[:top])
                s[0] += len(actual & p)
                s[1] += len(actual)
                s[2] += len(misses & p)
                s[3] += len(misses)
        # 2) predict the next layers' routing from this layer's router input
        for ahead in (1, 2):
            nxt = layer_id + ahead
            if nxt < 40:
                kwn = BLOCKS.get(nxt)
                if kwn is None:
                    kwn = BLOCKS[nxt] = rt._common_block_kwargs(nxt, compressed=nxt >= 2)
                r = route_topk_fused(x, kwn["gate_weight"], kwn["gate_bias"], topk=max(STATE["top"]))
                # route_topk returns ascending selection order; strongest last
                STATE["pred"][(nxt, ahead)] = list(reversed(r.indices.tolist()))
        return orig(x, layer_id=layer_id, gate_weight=gate_weight, gate_bias=gate_bias,
                    expert_index=expert_index, expert_store=expert_store, **kw)

    for mod_name in ("block_sliding_window", "block_layer0", "block_compressed_source",
                     "block_compressed_reuse", "block_compressed_index_source"):
        mod = __import__(f"cachalot.model.{mod_name}", fromlist=["moe_layer_forward"])
        mod.moe_layer_forward = patched

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
        for (ahead, top), (hit, n, mhit, mn) in sorted(STATE["stats"].items()):
            print(f"  predict layer L+{ahead} from layer L's router input, top-{top:2d}: "
                  f"recall {hit / n:5.1%} of routed experts | {mhit / mn:5.1%} of cache misses "
                  f"({mn / args.decode_tokens:.1f} misses/token in scored layers)")


if __name__ == "__main__":
    main()
