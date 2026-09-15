"""
Parity + speed: batched MoE prefill vs the token-sequential grouped path.

Runs the same prompt through both (fresh reset each), compares the final
logits, greedy argmax, and reports wall time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, StoreSnapshot  # noqa: E402
from cachalot.model import moe_prefill_grouped as mpg  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=128)
    args = ap.parse_args()
    orig = mpg.moe_prefill_grouped

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = build_prompt(rt, enc, prompt_sources()[1][1], args.prompt_tokens)
        results = {}
        import os

        for mode in ("sequential", "batched", "batched"):
            flag = "0" if mode == "sequential" else "1"
            for env in ("CACHALOT_PREFILL_BATCHED_HC", "CACHALOT_PREFILL_BATCHED_ENGRAM", "CACHALOT_PREFILL_BATCHED_ATTN"):
                os.environ[env] = flag

            def patched(*a, _mode=mode, **kw):
                kw["batched"] = _mode == "batched"
                return orig(*a, **kw)

            for mod_name in ("block_layer0_prefill", "block_sliding_window_prefill", "block_compressed_source_prefill",
                             "block_compressed_reuse_prefill", "block_compressed_index_source_prefill"):
                mod = sys.modules[f"cachalot.model.{mod_name}"]
                mod.moe_prefill_grouped = patched
            rt.reset()
            before = StoreSnapshot.take(rt)
            t0 = perf_counter()
            res = rt.prefill_tokens(ids)
            mx.eval(res.logits)
            dt = perf_counter() - t0
            d = before.delta(StoreSnapshot.take(rt))
            results.setdefault(mode, []).append((mx.array(res.logits), dt, d.cache_misses))
            print(f"{mode:10s}: {dt:6.1f}s ({len(ids) / dt:.1f} tok/s), misses {d.cache_misses}, argmax {int(res.logits.argmax())}", flush=True)

        seq_logits = results["sequential"][0][0]
        bat_logits = results["batched"][1][0]   # second batched run is warm for experts -> speed
        diff = mx.abs(seq_logits - bat_logits)
        print(f"logits max|diff| {float(diff.max()):.3e}, mean {float(diff.mean()):.3e}, "
              f"argmax equal {int(seq_logits.argmax()) == int(bat_logits.argmax())}; "
              f"top-5 seq {mx.argsort(-seq_logits)[:5].tolist()} bat {mx.argsort(-bat_logits)[:5].tolist()}")


if __name__ == "__main__":
    main()
