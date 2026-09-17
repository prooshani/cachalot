"""
Is a startup hotlist worth preloading, and how big should it be?

Lever 5 of docs/HANDOFF.md: a session is 16.4 s to ready and its first turn pays
full miss cost, while later turns run at an 87.3 % hit rate because they reuse
what the first turn dragged in. Preloading a recorded hot set would help exactly
one turn per session -- but only if the experts a *new* prompt wants are the same
ones old prompts wanted, and that is a property of the model's routing, not an
assumption anyone should make.

This measures it honestly, leave-one-prompt-out: rank experts by how often the
*other* prompts in the trace used them, then score the coverage of the held-out
prompt's requests. A hot set that only covers prompts it was built from would
show up here as coverage that collapses when the prompt is unseen.

Coverage is of requests, not of misses, so it is an upper bound on what a
preload buys: in a real first turn the cache also fills as it goes, and the hot
set stops mattering once natural residency takes over. It is also worth noting
what a preload costs, which is almost nothing -- the experts are read once at
6.7 GB/s and LRU evicts whatever the session turns out not to want.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/hotlist_coverage.py \
      benchmarks/results/trace_routing_v7.trace.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS_DIR  # noqa: E402
from cachalot.metrics.routing_trace import PHASE_PREFILL, load_trace  # noqa: E402

GIB = 1024**3
Q2G128_EXPERT_BYTES = 9_953_280
N_EXPERTS_TOTAL = 15_360


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--sizes-gib", default="1,2,4,8,16")
    ap.add_argument("--expert-bytes", type=int, default=Q2G128_EXPERT_BYTES)
    ap.add_argument("--drive-gbps", type=float, default=6.7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arrays, segments = load_trace(args.trace)
    layer, experts, phase = arrays["layer"], arrays["experts"], arrays["phase"]
    bounds = [s["at"] for s in segments] + [len(layer)]

    def prompt_of(index: int) -> str:
        return segments[index]["label"].split(":")[0]

    prompts = sorted({prompt_of(i) for i in range(len(segments))})
    if len(prompts) < 2:
        raise SystemExit("leave-one-out needs a trace with at least two prompts")

    counts: dict[str, Counter] = {p: Counter() for p in prompts}
    for index in range(len(segments)):
        owner = counts[prompt_of(index)]
        for row in range(bounds[index], bounds[index + 1]):
            for expert in experts[row]:
                owner[(int(layer[row]), int(expert))] += 1

    print(f"{len(prompts)} prompts, leave-one-out\n")
    print("| hot set | experts | of the bank | load time | unseen prefill | unseen decode |")
    print("|---:|---:|---:|---:|---:|---:|")

    rows = []
    for raw in args.sizes_gib.split(","):
        gib = float(raw)
        slots = int(gib * GIB) // args.expert_bytes
        prefill_cov, decode_cov = [], []

        for held in prompts:
            others = Counter()
            for other, counter in counts.items():
                if other != held:
                    others.update(counter)
            hot = {key for key, _ in others.most_common(slots)}

            pre_hit = pre_n = dec_hit = dec_n = 0
            for index in range(len(segments)):
                if prompt_of(index) != held:
                    continue
                for row in range(bounds[index], bounds[index + 1]):
                    for expert in experts[row]:
                        key = (int(layer[row]), int(expert))
                        if phase[row] == PHASE_PREFILL:
                            pre_n += 1
                            pre_hit += key in hot
                        else:
                            dec_n += 1
                            dec_hit += key in hot
            prefill_cov.append(pre_hit / max(1, pre_n))
            decode_cov.append(dec_hit / max(1, dec_n))

        prefill = sum(prefill_cov) / len(prefill_cov)
        decode = sum(decode_cov) / len(decode_cov)
        seconds = slots * args.expert_bytes / 1e9 / args.drive_gbps
        rows.append(
            {
                "gib": gib,
                "slots": slots,
                "load_seconds": seconds,
                "unseen_prefill_coverage": prefill,
                "unseen_decode_coverage": decode,
            }
        )
        print(
            f"| {gib:.0f} GiB | {slots} | {slots / N_EXPERTS_TOTAL:.1%} | {seconds:.1f} s | "
            f"{prefill:.1%} | {decode:.1%} |"
        )

    print(
        "\nCoverage of an unseen prompt is the number that matters: it is the part of the "
        "hot set that generalizes rather than the part that memorized the prompts it was "
        "built from. Against the 16.4 s a session already spends becoming ready, the load "
        "time is small, and LRU evicts whatever the session does not want."
    )

    out_path = Path(args.out) if args.out else RESULTS_DIR / "hotlist_coverage.json"
    out_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
