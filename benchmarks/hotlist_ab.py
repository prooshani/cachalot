"""
Does the startup hotlist actually help the turn it is meant to help?

benchmarks/hotlist_coverage.py says a hot set generalizes -- 5.6 % of the bank
covers about 30 % of an unseen prompt's requests. That is coverage of requests,
which is an upper bound: in a real first turn the cache also fills as it goes.
This measures the thing itself, cold prefill and first-turn decode, with
CACHALOT_HOTLIST set and unset.

One process per arm is the wrong shape here -- the hotlist is a startup effect,
so the runtime has to be built fresh for every arm, and the OS page cache has to
be treated as the confound it is. Each arm therefore runs in its own process
(the caller interleaves them) and the script reports one arm only.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh \
      --budget-gib 36 --max-seconds 1800 --tag hotlist-on -- env \
      CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 \
      CACHALOT_PAGE_CACHE=1 CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json \
      CACHALOT_HOTLIST_GIB=8 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
      benchmarks/hotlist_ab.py --prompt-tokens 512 --decode-tokens 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR, StoreSnapshot  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--prompt", type=int, default=0, help="index into trace_routing.prompt_sources")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    hotlist = os.environ.get("CACHALOT_HOTLIST")
    arm = "on" if hotlist else "off"

    t0 = perf_counter()
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        ready = perf_counter() - t0
        print(
            f"runtime ready in {ready:.1f} s (expert budget "
            f"{rt.expert_cache_budget_bytes / 2**30:.1f} GiB, hotlist {arm}, "
            f"{len(rt.expert_store)} experts resident)",
            flush=True,
        )
        enc = load_official_encoding(MODEL_PATH)
        label, text = prompt_sources()[args.prompt]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)

        # build_prompt decodes nothing but does prefill while searching for the
        # token budget, so reset and measure the turn from a clean sequence.
        rt.reset()
        before = StoreSnapshot.take(rt)

        t0 = perf_counter()
        result = rt.prefill_tokens(ids)
        mx.eval(result.logits)
        prefill = perf_counter() - t0

        token = int(result.logits.argmax().item())
        t0 = perf_counter()
        for _ in range(args.decode_tokens):
            step = rt.decode_token(token)
            token = int(step.logits.argmax().item())
        decode = perf_counter() - t0

        delta = before.delta(StoreSnapshot.take(rt))
        requests = delta.cache_hits + delta.cache_misses
        row = {
            "arm": arm,
            "hotlist": hotlist,
            "hotlist_gib": os.environ.get("CACHALOT_HOTLIST_GIB"),
            "prompt": label,
            "ready_seconds": ready,
            "preloaded_experts": rt.expert_store.preloaded_experts,
            "preload_seconds": rt.expert_store.preload_seconds,
            "prefill_seconds": prefill,
            "decode_seconds": decode,
            "ms_per_token": decode / args.decode_tokens * 1e3,
            "turn_seconds": prefill + decode,
            "hit_rate": delta.cache_hits / requests if requests else 0.0,
            "misses": delta.cache_misses,
            "gib_read": delta.ssd_bytes_read / 1024**3,
        }
        print(
            f"hotlist {arm}: ready {ready:.1f} s (preload {rt.expert_store.preloaded_experts} experts in "
            f"{rt.expert_store.preload_seconds:.1f} s) | prefill {prefill:.1f} s | "
            f"decode {row['ms_per_token']:.1f} ms/token | turn {row['turn_seconds']:.1f} s | "
            f"first-turn hit {row['hit_rate']:.1%} | {delta.cache_misses} misses | "
            f"{row['gib_read']:.1f} GiB read",
            flush=True,
        )

        out_path = Path(args.out) if args.out else RESULTS_DIR / f"hotlist_ab_{arm}.json"
        existing = []
        if out_path.exists():
            existing = json.loads(out_path.read_text()).get("runs", [])
        existing.append(row)
        out_path.write_text(json.dumps({"runs": existing}, indent=2))
        print(f"wrote {out_path} ({len(existing)} runs)")


if __name__ == "__main__":
    main()
