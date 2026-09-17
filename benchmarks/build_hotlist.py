"""
Rank routed experts by how often past sessions used them, for startup preload.

benchmarks/hotlist_coverage.py establishes that a hot set generalizes: ranked on
other prompts, 5.6 % of the bank covers about 30 % of an unseen prompt's
requests. This writes that ranking out in the form the runtime reads.

The output is bank-independent -- it names (layer, expert) pairs, not bytes -- so
one hotlist serves any bank of the same model, and CACHALOT_HOTLIST_GIB decides
how much of it a given run takes.

Rank on as many real sessions as exist. A hotlist built from one prompt is a
hotlist for that prompt.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/build_hotlist.py \
      benchmarks/results/trace_routing_v7.trace.npz \
      --out /Users/hamedprooshani/cachalot-hotlist.json

    CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json \
    CACHALOT_HOTLIST_GIB=8 ... python -m cachalot.cli chat ...
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cachalot.metrics.routing_trace import PHASE_PREFILL, load_trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=4096,
                    help="experts to write; the runtime takes a prefix of these")
    ap.add_argument("--phase", default="both", choices=("both", "prefill", "decode"))
    args = ap.parse_args()

    counts: Counter = Counter()
    sources = []

    for path in args.traces:
        arrays, segments = load_trace(path)
        layer, experts, phase = arrays["layer"], arrays["experts"], arrays["phase"]
        for row in range(len(layer)):
            if args.phase == "prefill" and phase[row] != PHASE_PREFILL:
                continue
            if args.phase == "decode" and phase[row] == PHASE_PREFILL:
                continue
            for expert in experts[row]:
                counts[(int(layer[row]), int(expert))] += 1
        sources.append({"trace": path, "segments": [s["label"] for s in segments]})

    if not counts:
        raise SystemExit("no routed-expert records in the traces given")

    ranked = [list(key) for key, _ in counts.most_common(args.limit)]
    total = sum(counts.values())
    covered = sum(count for _, count in counts.most_common(args.limit))

    out = Path(args.out)
    out.write_text(
        json.dumps(
            {
                "experts": ranked,
                "sources": sources,
                "phase": args.phase,
                "distinct_experts_seen": len(counts),
                "requests": total,
            },
            indent=2,
        )
    )
    print(
        f"{len(ranked)} experts written to {out}; "
        f"{len(counts)} distinct experts seen over {total:,} requests, "
        f"and this prefix accounts for {covered / total:.1%} of them "
        f"(in-sample, which is why hotlist_coverage.py exists)"
    )


if __name__ == "__main__":
    main()
