"""
Would confidence-scheduled verification pay, and at what threshold?

The three measurements this combines all exist:

  * `dspark_acceptance.py` records, per draft block, the confidence the head
    gave each of the five positions and whether that position matched.
  * `speculation_bytes.py` gives misses per forward for a verification width,
    replayed from a routing trace at a chosen budget.
  * `decode_resident.py` and `verify_forward_cost.py` give the compute a
    forward costs: a fixed part per forward and a marginal part per extra
    position.

Put together they answer the question the acceptance curve alone cannot. Deeper
verification accepts more tokens *and* reads more experts, so the depth that pays
is not the deepest one; and because the confidence head separates accepted from
rejected positions, the depth need not be fixed at all. A verifier can extend the
block only while confidence holds, paying for bytes where they are likely to be
used.

Verification is one contiguous forward, so a position can only be verified if
every position before it is: the policy is "extend while confidence >= t", not
"verify the confident ones".

This is a projection, not a measurement. The compute model is linear in width and
the drive model assumes reads do not overlap compute at all, which is
conservative. Treat the ranking as informative and the absolute numbers as an
estimate to be replaced by a real integrated run.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/speculation_policy.py \
      --acceptance benchmarks/results/dspark_acceptance_conf.json \
      --bytes benchmarks/results/speculation_bytes.json --budget-gib 36
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acceptance", default=str(RESULTS_DIR / "dspark_acceptance_conf.json"))
    ap.add_argument("--bytes", default=str(RESULTS_DIR / "speculation_bytes.json"))
    ap.add_argument("--budget-gib", type=float, default=36.0)
    ap.add_argument("--baseline-ms", type=float, default=182.5,
                    help="measured decode ms/token to beat, handoff section 7.1")
    ap.add_argument("--compute-fixed-ms", type=float, default=93.0,
                    help="all-resident decode at 512 context, decode_resident.py")
    ap.add_argument("--compute-marginal-ms", type=float, default=19.8,
                    help="slope of verify_forward_cost.py over width")
    ap.add_argument("--draft-ms", type=float, default=95.0,
                    help="one DSpark draft block, dspark_draft_cost.py")
    ap.add_argument("--miss-scale", type=float, default=0.738,
                    help="measured misses per token (45.7) over the trace replay's "
                         "width-1 figure (61.9): the replay has no prefill quota "
                         "planner and no prefetch, so it over-counts by this much")
    ap.add_argument("--drive-gbps", type=float, default=6.7)
    ap.add_argument("--expert-bytes", type=int, default=9_953_280)
    ap.add_argument("--thresholds", default="-99,0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    blocks = json.loads(Path(args.acceptance).read_text())["blocks"]
    if not blocks:
        raise SystemExit(f"{args.acceptance} holds no per-block confidence; re-run the measurement")

    byte_rows = json.loads(Path(args.bytes).read_text())["rows"]
    misses_by_width = {
        int(row["width"]): row["misses_per_forward"] * args.miss_scale
        for row in byte_rows
        if abs(row["budget_gib"] - args.budget_gib) < 1e-6
    }
    if not misses_by_width:
        raise SystemExit(f"{args.bytes} has no rows at a {args.budget_gib} GiB budget")

    def drive_ms(width: int) -> float:
        mib = misses_by_width[width] * args.expert_bytes / 2**20
        return mib / 2**10 / args.drive_gbps * 1.024**3 * 1e3

    print(f"budget {args.budget_gib:.0f} GiB | baseline {args.baseline_ms:.1f} ms/token | "
          f"draft {args.draft_ms:.0f} ms | compute {args.compute_fixed_ms:.0f} + "
          f"{args.compute_marginal_ms:.1f}/position\n")
    print("| confidence >= | mean width | tokens/forward | compute ms | drive ms | "
          "ms/accepted token | vs baseline |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    rows = []
    for raw in args.thresholds.split(","):
        threshold = float(raw)
        widths = []
        tokens = []
        for block in blocks:
            confidence = block["confidence"]
            hits = block["hits"]
            extend = 0
            while extend < len(confidence) and confidence[extend] >= threshold:
                extend += 1
            accepted = 0
            while accepted < extend and hits[accepted]:
                accepted += 1
            widths.append(1 + extend)
            tokens.append(1 + accepted)

        mean_width = sum(widths) / len(widths)
        mean_tokens = sum(tokens) / len(tokens)

        # The cost model is per-forward; a fractional mean width is priced by
        # mixing the two integer widths it lies between.
        low = max(1, min(misses_by_width, key=lambda w: abs(w - mean_width)))
        floor_w = int(mean_width)
        ceil_w = min(max(misses_by_width), floor_w + 1)
        frac = mean_width - floor_w
        if floor_w in misses_by_width and ceil_w in misses_by_width:
            drive = drive_ms(floor_w) * (1 - frac) + drive_ms(ceil_w) * frac
        else:
            drive = drive_ms(low)

        compute = args.compute_fixed_ms + args.compute_marginal_ms * (mean_width - 1)
        per_token = (compute + drive + args.draft_ms) / mean_tokens
        rows.append(
            {
                "threshold": threshold,
                "mean_width": mean_width,
                "tokens_per_forward": mean_tokens,
                "compute_ms": compute,
                "drive_ms": drive,
                "ms_per_token": per_token,
                "speedup": args.baseline_ms / per_token,
            }
        )
        label = "none" if threshold < -1 else f"{threshold:.2f}"
        print(
            f"| {label} | {mean_width:.2f} | {mean_tokens:.3f} | {compute:.0f} | "
            f"{drive:.0f} | {per_token:.0f} | {args.baseline_ms / per_token:.2f}x |"
        )

    best = max(rows, key=lambda r: r["speedup"])
    print(
        f"\nbest: threshold {best['threshold']:.2f}, width {best['mean_width']:.2f}, "
        f"{best['tokens_per_forward']:.2f} tokens per forward, "
        f"{best['ms_per_token']:.0f} ms per accepted token, {best['speedup']:.2f}x"
    )
    print(
        "The draft cost enters once per forward, so it is the term a lower width "
        "amortizes worst; halving it moves every row."
    )

    out_path = Path(args.out) if args.out else RESULTS_DIR / "speculation_policy.json"
    out_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
