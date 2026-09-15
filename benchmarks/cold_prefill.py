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
    ap.add_argument("--disk-gbps", type=float, default=None, help="sequential read GB/s (default: measured)")
    ap.add_argument("--repeat", type=int, default=1, help="prefill the same prompt N times (later runs are warm)")
    args = ap.parse_args()
    if args.disk_gbps is None:
        from cachalot.cli import _read_speed

        shard = sorted(Path(MODEL_PATH).glob("model-*.safetensors"))[24]
        args.disk_gbps = _read_speed(shard) / 1000
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096,
                           expert_cache_budget_bytes=int(args.expert_budget_gib * 2**30)) as rt:
        enc = load_official_encoding(MODEL_PATH)
        name, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)
        for run in range(args.repeat):
            rt.reset()
            before = StoreSnapshot.take(rt)
            with Timer() as t:
                rt.prefill_tokens(ids)
            rep = phase_report(f"{name}:prefill_run{run}", t.seconds, len(ids), before.delta(StoreSnapshot.take(rt)), rt)
            floor = rep["ssd_gib"] * 1.073741824 / args.disk_gbps
            print(f"run {run}: SSD floor at {args.disk_gbps:.1f} GB/s: {floor:.0f}s; wall {t.seconds:.0f}s; "
                  f"SSD share {floor / t.seconds:.0%}, compute/overhead ~{t.seconds - floor:.0f}s", flush=True)


if __name__ == "__main__":
    main()
