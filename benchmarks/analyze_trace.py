"""
Offline analysis of a routing trace produced by trace_routing.py.

Answers the questions that decide cache policy and memory budget:

  1. Coverage curve: what fraction of expert activations is covered by the
     N globally hottest (layer, expert) pairs? This is the upper bound any
     static resident set can reach at a given budget.
  2. Per-layer skew: how unevenly does each layer use its 384 experts?
     Uniform per-layer quotas are only optimal when skew is uniform.
  3. Cache simulation over the actual access sequence for several budgets
     and policies (LRU, LFU-weighted, static-hot + LRU), giving the
     achievable hit rate rather than the static upper bound.
  4. Adjacent-token expert overlap, which bounds the benefit of
     speculative/batched decode (shared expert loads across tokens).

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/analyze_trace.py \
        benchmarks/results/trace_routing.trace.npz
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cachalot.metrics.routing_trace import PHASE_DECODE, PHASE_PREFILL, load_trace  # noqa: E402

N_LAYERS = 40
N_EXPERTS = 384
EXPERT_BYTES = 18_800_640
GIB = 1024**3


def budget_to_slots(gib: float) -> int:
    return int(gib * GIB) // EXPERT_BYTES


def flat_ids(layer: np.ndarray, experts: np.ndarray) -> np.ndarray:
    """(layer, expert) pairs as one int id per activation, shape [records, topk]."""
    return layer[:, None].astype(np.int64) * N_EXPERTS + experts.astype(np.int64)


def coverage_curve(counts: np.ndarray, slots_list: list[int]) -> list[tuple[int, float]]:
    order = np.sort(counts)[::-1]
    total = counts.sum()
    cum = np.cumsum(order)
    return [(n, float(cum[min(n, len(cum)) - 1] / total)) for n in slots_list]


def layer_skew(counts: np.ndarray) -> list[dict]:
    rows = []
    per_layer = counts.reshape(N_LAYERS, N_EXPERTS)
    for layer in range(N_LAYERS):
        c = per_layer[layer]
        total = c.sum()
        if total == 0:
            continue
        p = c / total
        nz = p[p > 0]
        entropy = float(-(nz * np.log2(nz)).sum())
        srt = np.sort(c)[::-1]
        rows.append(
            {
                "layer": layer,
                "used_experts": int((c > 0).sum()),
                "entropy_bits": entropy,
                "top57_coverage": float(srt[:57].sum() / total),
                "top100_coverage": float(srt[:100].sum() / total),
            }
        )
    return rows


def optimal_layer_allocation(counts: np.ndarray, slots: int) -> tuple[list[int], float]:
    """Greedy global top-N: how many slots each layer gets, and coverage."""
    order = np.argsort(counts)[::-1][:slots]
    layers = order // N_EXPERTS
    alloc = np.bincount(layers, minlength=N_LAYERS).tolist()
    return alloc, float(counts[order].sum() / counts.sum())


def simulate(access: np.ndarray, slots: int, policy: str, hot: np.ndarray | None = None) -> float:
    """
    access: 1-D sequence of (layer*384+expert) ids in execution order.
    policy: 'lru' | 'lfu' | 'static' (pin `hot`, LRU for the rest)
    Returns hit rate.
    """
    hits = 0
    if policy == "static":
        pinned = set(int(x) for x in hot[:slots])
        dyn_slots = max(slots - len(pinned), 0)
        lru: OrderedDict[int, None] = OrderedDict()
        for a in access:
            a = int(a)
            if a in pinned:
                hits += 1
                continue
            if a in lru:
                hits += 1
                lru.move_to_end(a)
                continue
            if dyn_slots == 0:
                continue
            if len(lru) >= dyn_slots:
                lru.popitem(last=False)
            lru[a] = None
        return hits / len(access)

    if policy == "lru":
        lru = OrderedDict()
        for a in access:
            a = int(a)
            if a in lru:
                hits += 1
                lru.move_to_end(a)
            else:
                if len(lru) >= slots:
                    lru.popitem(last=False)
                lru[a] = None
        return hits / len(access)

    if policy == "lfu":
        # Frequency-aware: evict the resident with the lowest (count, recency).
        # Counts decay by halving every `window` accesses so old phases fade.
        window = 50_000
        count: dict[int, float] = {}
        resident: dict[int, int] = {}  # id -> last access index
        for i, a in enumerate(access):
            a = int(a)
            count[a] = count.get(a, 0.0) + 1.0
            if i and i % window == 0:
                for k in count:
                    count[k] *= 0.5
            if a in resident:
                hits += 1
                resident[a] = i
                continue
            if len(resident) >= slots:
                victim = min(resident, key=lambda k: (count.get(k, 0.0), resident[k]))
                del resident[victim]
            resident[a] = i
        return hits / len(access)

    raise ValueError(policy)


def adjacent_overlap(layer, position, experts, phase) -> float:
    mask = phase == PHASE_DECODE
    lay, pos, exp = layer[mask], position[mask], experts[mask]
    key = {(int(li), int(pi)): set(row.tolist()) for li, pi, row in zip(lay, pos, exp, strict=True)}
    overlaps = []
    for (li, pi), s in key.items():
        nxt = key.get((li, pi + 1))
        if nxt is not None:
            overlaps.append(len(s & nxt) / len(s))
    return float(np.mean(overlaps)) if overlaps else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budgets-gib", default="30,40,48,56,64,80")
    ap.add_argument("--out", default=None, help="Markdown report path")
    args = ap.parse_args()

    arrays, segments = load_trace(args.trace)
    phase, layer, position, experts = (arrays[k] for k in ("phase", "layer", "position", "experts"))
    ids = flat_ids(layer, experts)
    counts = np.bincount(ids.ravel(), minlength=N_LAYERS * N_EXPERTS)
    budgets = [float(x) for x in args.budgets_gib.split(",")]
    slots_list = [budget_to_slots(b) for b in budgets]

    lines: list[str] = []
    w = lines.append
    w("# Routing trace analysis\n")
    w(f"trace: `{args.trace}`  ")
    w(f"records: {len(layer):,} (token, layer) pairs, activations: {ids.size:,}, "
      f"prefill tokens: {int((phase == PHASE_PREFILL).sum() // N_LAYERS):,}, "
      f"decode tokens: {int((phase == PHASE_DECODE).sum() // N_LAYERS):,}  ")
    w(f"distinct experts touched: {int((counts > 0).sum()):,} of {N_LAYERS * N_EXPERTS:,}\n")

    w("## 1. Static coverage upper bound (global top-N experts)\n")
    w("| budget GiB | slots | share of all experts | activations covered |")
    w("|---:|---:|---:|---:|")
    for b, s, (_, cov) in zip(budgets, slots_list, coverage_curve(counts, slots_list), strict=True):
        w(f"| {b:.0f} | {s:,} | {s / (N_LAYERS * N_EXPERTS):.1%} | {cov:.1%} |")
    w("")

    w("## 2. Per-layer skew\n")
    rows = layer_skew(counts)
    w("| layer | used experts | entropy (bits, max 8.58) | top-57 coverage | top-100 coverage |")
    w("|---:|---:|---:|---:|---:|")
    for r in rows:
        w(f"| {r['layer']} | {r['used_experts']} | {r['entropy_bits']:.2f} | {r['top57_coverage']:.1%} | {r['top100_coverage']:.1%} |")
    w("")
    alloc, cov = optimal_layer_allocation(counts, budget_to_slots(40))
    uniform_cov = float(np.mean([r["top57_coverage"] for r in rows]))
    w(f"Uniform 57/layer quota covers **{uniform_cov:.1%}** of activations; "
      f"global top-{budget_to_slots(40):,} allocation covers **{cov:.1%}** "
      f"(per-layer slots min {min(alloc)}, max {max(alloc)}).\n")

    w("## 3. Cache simulation over actual access order\n")
    # Access order: prefill is expert-major per layer (each distinct expert once per layer),
    # decode is token-major. Reproduce that from the trace.
    access: list[int] = []
    seg_bounds = [s["at"] for s in segments] + [len(layer)]
    for si in range(len(segments)):
        a, b = seg_bounds[si], seg_bounds[si + 1]
        seg_ids, seg_layer, seg_phase = ids[a:b], layer[a:b], phase[a:b]
        if seg_phase.size and seg_phase[0] == PHASE_PREFILL:
            for li in range(N_LAYERS):
                m = seg_layer == li
                if m.any():
                    # one request per distinct expert per layer (grouped prefill)
                    access.extend(np.unique(seg_ids[m]).tolist())
        else:
            access.extend(seg_ids.ravel().tolist())
    access_arr = np.asarray(access, dtype=np.int64)
    w(f"simulated requests: {len(access_arr):,}\n")
    hot = np.argsort(counts)[::-1]
    w("| budget GiB | slots | LRU | LFU-decay | static-hot + LRU |")
    w("|---:|---:|---:|---:|---:|")
    for b, s in zip(budgets, slots_list, strict=True):
        lru = simulate(access_arr, s, "lru")
        lfu = simulate(access_arr, s, "lfu")
        st = simulate(access_arr, s, "static", hot)
        w(f"| {b:.0f} | {s:,} | {lru:.1%} | {lfu:.1%} | {st:.1%} |")
    w("")
    w("static-hot uses the trace's own frequencies (optimistic); it shows the ceiling of an offline hot set.\n")

    w("## 4. Decode locality\n")
    w(f"Mean overlap of a layer's 6 experts between consecutive decode tokens: **{adjacent_overlap(layer, position, experts, phase):.1%}**\n")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report)


if __name__ == "__main__":
    main()
