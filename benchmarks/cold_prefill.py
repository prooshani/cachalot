"""Time one cold prefill of N tokens and report SSD bytes vs. wall (pipeline efficiency)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, StoreSnapshot, Timer, phase_report  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--expert-budget-gib", type=float, default=0.0)
    args = ap.parse_args()
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096,
                           expert_cache_budget_bytes=int(args.expert_budget_gib * 2**30)) as rt:
        enc = load_official_encoding(MODEL_PATH)
        name, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)
        rt.reset()
        before = StoreSnapshot.take(rt)
        with Timer() as t:
            rt.prefill_tokens(ids)
        rep = phase_report(f"{name}:cold_prefill", t.seconds, len(ids), before.delta(StoreSnapshot.take(rt)), rt)
        floor = rep["ssd_gib"] * 1.073741824
        print(f"SSD floor at 1.0 GB/s: {floor:.0f}s; wall {t.seconds:.0f}s; SSD busy {floor / t.seconds:.0%}", flush=True)


if __name__ == "__main__":
    main()
