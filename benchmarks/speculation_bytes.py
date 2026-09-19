"""
How many expert reads does a K-position verification forward actually need?

This is the byte half of lever 1, and a routing trace answers it without running
anything. A forward over K positions asks each layer for the *union* of those
positions' top-6 experts, not six times K of them, because adjacent tokens
overlap; and residency is shared, so the union is what the cache is asked for.
Replaying a recorded trace in blocks of K through the same LRU the runtime uses
gives misses per forward directly, at any budget, for free.

Divide by the tokens a block of that depth actually yields -- from
benchmarks/dspark_acceptance.py -- and the result is bytes per *accepted* token,
which is the number that decides whether speculation helps or hurts on a machine
that streams its experts.

Prefill is replayed unchanged ahead of the decode blocks so the cache starts in
a realistic state.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/speculation_bytes.py \
      benchmarks/results/trace_routing_v7.trace.npz --budgets-gib 36,44
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS_DIR  # noqa: E402
from cachalot.metrics.routing_trace import PHASE_PREFILL, load_trace  # noqa: E402

GIB = 1024**3
N_LAYERS = 40
Q2G128_EXPERT_BYTES = 9_953_280

# Tokens produced per main forward when the verifier checks this many drafted
# positions, measured over four prompts in benchmarks/results/dspark_acceptance.json.
TOKENS_PER_FORWARD = {0: 1.0, 1: 1.727, 2: 2.216, 3: 2.534, 4: 2.732, 5: 2.846}


class Cache:
    """The runtime's decode cache: LRU over (layer, expert), one slot per expert."""

    def __init__(self, slots: int):
        self.slots = slots
        self.items: OrderedDict[tuple[int, int], None] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def request(self, keys) -> None:
        for key in keys:
            if key in self.items:
                self.items.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
                self.items[key] = None
                if len(self.items) > self.slots:
                    self.items.popitem(last=False)


def replay(arrays, segments, slots: int, width: int) -> dict:
    """Replay the trace with decode positions grouped into blocks of `width`.

    Segments matter: a trace holds several prompts, and their decode positions
    restart from the same numbers. Grouping by position alone silently merges
    one prompt's token with another's, which inflates every union.
    """
    phase = arrays["phase"]
    layer = arrays["layer"]
    position = arrays["position"]
    experts = arrays["experts"]

    cache = Cache(slots)
    bounds = [s["at"] for s in segments] + [len(layer)]

    decode_hits = decode_misses = forwards = positions_seen = 0

    for index in range(len(segments)):
        a, b = bounds[index], bounds[index + 1]
        if b <= a:
            continue

        if phase[a] == PHASE_PREFILL:
            for row in range(a, b):
                cache.request((int(layer[row]), int(e)) for e in experts[row])
            continue

        rows_by_position: dict[int, list[int]] = {}
        for row in range(a, b):
            rows_by_position.setdefault(int(position[row]), []).append(row)
        ordered = sorted(rows_by_position)
        positions_seen += len(ordered)

        before_hits, before_misses = cache.hits, cache.misses
        for start in range(0, len(ordered), width):
            block = ordered[start : start + width]
            per_layer: dict[int, set[int]] = {}
            for pos in block:
                for row in rows_by_position[pos]:
                    per_layer.setdefault(int(layer[row]), set()).update(
                        int(e) for e in experts[row]
                    )
            for layer_id in sorted(per_layer):
                cache.request((layer_id, e) for e in sorted(per_layer[layer_id]))
            forwards += 1
        decode_hits += cache.hits - before_hits
        decode_misses += cache.misses - before_misses

    if forwards == 0:
        raise SystemExit("trace holds no decode positions")

    requests = decode_hits + decode_misses
    return {
        "width": width,
        "forwards": forwards,
        "positions": positions_seen,
        "requests_per_forward": requests / forwards,
        "misses_per_forward": decode_misses / forwards,
        "hit_rate": decode_hits / requests if requests else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budgets-gib", default="36,44")
    ap.add_argument("--widths", default="1,2,3,4,5,6")
    ap.add_argument("--expert-bytes", type=int, default=Q2G128_EXPERT_BYTES)
    ap.add_argument("--drive-gbps", type=float, default=6.7,
                    help="cold sequential expert bandwidth, handoff section 6")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arrays, segments = load_trace(args.trace)
    widths = [int(w) for w in args.widths.split(",")]
    budgets = [float(b) for b in args.budgets_gib.split(",")]

    rows = []
    print("| budget | positions/forward | hit rate | misses/forward | MiB/forward | "
          "drive ms/forward | tokens/forward | MiB per accepted token |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for budget in budgets:
        slots = int(budget * GIB) // args.expert_bytes
        for width in widths:
            stats = replay(arrays, segments, slots, width)
            mib = stats["misses_per_forward"] * args.expert_bytes / 2**20
            drive_ms = mib / 2**10 / args.drive_gbps * 1.024**3 * 1e3
            tokens = TOKENS_PER_FORWARD.get(width - 1)
            per_token = mib / tokens if tokens else None
            rows.append({"budget_gib": budget, **stats, "mib_per_forward": mib,
                         "drive_ms_per_forward": drive_ms,
                         "tokens_per_forward": tokens,
                         "mib_per_accepted_token": per_token})
            print(
                f"| {budget:.0f} | {width} | {stats['hit_rate']:.1%} | "
                f"{stats['misses_per_forward']:.1f} | {mib:.0f} | {drive_ms:.0f} | "
                f"{tokens:.3f} | {per_token:.0f} |"
            )

    print(
        "\nA width of K verifies K-1 drafted positions plus the token the previous "
        "forward already produced. Bytes per accepted token below the width-1 row "
        "means speculation reads less per token, not more; above it means the "
        "drive pays for the speed and the budget has to absorb it."
    )

    out_path = Path(args.out) if args.out else RESULTS_DIR / "speculation_bytes.json"
    out_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
