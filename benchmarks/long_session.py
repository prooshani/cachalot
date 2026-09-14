"""
Long-session stability: alternate several short conversations for N turns,
recording MLX active/cache/peak memory, resident experts, prefix-cache size,
process RSS and vm compressor after every turn.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/long_session.py --turns 20 --decode-tokens 8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR, write_json  # noqa: E402
from cachalot.model.api import V41Model  # noqa: E402

TOPICS = [
    "Explain how a sperm whale's spermaceti organ might help it dive.",
    "Write a Python function that merges overlapping intervals.",
    "Summarize the causes of the 2008 financial crisis in three bullets.",
    "What is the difference between a Sinkhorn iteration and softmax?",
]


def compressor_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Pages occupied by compressor"):
            return int(line.split(":")[1].strip().rstrip(".")) * 16384 / 1e9
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--decode-tokens", type=int, default=8)
    args = ap.parse_args()
    proc = psutil.Process()
    rows = []
    convos = [[] for _ in TOPICS]
    with V41Model.from_pretrained(MODEL_PATH, max_seq_len=8192) as model:
        for turn in range(args.turns):
            i = turn % len(TOPICS)
            convos[i].append({"role": "user", "content": TOPICS[i] if not convos[i] else "Continue with one more sentence."})
            t0 = perf_counter()
            resp = model.chat(convos[i], max_new_tokens=args.decode_tokens, temperature=0.0)
            dt = perf_counter() - t0
            convos[i].append({"role": "assistant", "content": resp.content})
            st = model.stats()
            row = {
                "turn": turn, "conversation": i, "seconds": dt,
                "prompt_tokens": len(resp.prompt_tokens), "reused": resp.reused_prefix_tokens,
                "completion_tokens": len(resp.completion_tokens),
                "mlx_active_gib": mx.get_active_memory() / 2**30, "mlx_cache_gib": mx.get_cache_memory() / 2**30,
                "mlx_peak_gib": mx.get_peak_memory() / 2**30, "resident_experts": st["resident_experts"],
                "prefix_entries": st["prefix_cache_entries"], "prefix_mib": st["prefix_cache_bytes"] / 2**20,
                "rss_gib": proc.memory_info().rss / 2**30, "compressor_gb": compressor_gb(),
            }
            rows.append(row)
            print(f"turn {turn:2d} conv {i} {dt:6.1f}s prompt {row['prompt_tokens']:4d} reused {row['reused']:4d} "
                  f"| active {row['mlx_active_gib']:.2f} cache {row['mlx_cache_gib']:.2f} peak {row['mlx_peak_gib']:.2f} GiB "
                  f"| rss {row['rss_gib']:.1f} GiB | prefix {row['prefix_entries']} entries {row['prefix_mib']:.0f} MiB "
                  f"| compressor {row['compressor_gb']:.1f} GB", flush=True)
    write_json(RESULTS_DIR / "long_session.json", rows)


if __name__ == "__main__":
    main()
