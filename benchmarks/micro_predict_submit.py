"""
Screen: what the prediction *submission* path costs the decode thread.

Section 9.18 prices routing prediction at 11 ms of a 77 ms token, about half of
it in submitting the prediction rather than computing it. Submission is pure
Python bookkeeping, repeated on all forty layers of every token:

    for e in reversed(p_idx.tolist()):
        entry = expert_index.get((nxt, int(e)))

which is one `.tolist()` off an evaluated 6-element array, six tuple
constructions, six `int()` calls and six dict lookups in a 15,360-entry dict --
plus the layer's own `route.indices.tolist()` and `route.weights.tolist()`.

Four arms, over one token's worth of work (40 layers), on arrays of the shapes
the runtime actually uses. No model, no store: this times the bookkeeping only,
and it is a screen, not a gate (HANDOFF section 8.3).

  shipped     exactly the loop above
  rows        the dict replaced by a per-layer list indexed by expert id, and
              the redundant int() dropped -- `.tolist()` already yields ints
  onelist     rows, plus the layer's routing and the prediction evaluated and
              converted in one array instead of two
  nolookup    rows with the entry lookup removed entirely, as a floor

    PYTHONPATH=src python benchmarks/micro_predict_submit.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

LAYERS = 40
N_EXPERTS = 384
TOPK = 6
REPEATS = 21


class FakeEntry:
    __slots__ = ("layer", "expert")

    def __init__(self, layer: int, expert: int) -> None:
        self.layer = layer
        self.expert = expert


expert_index = {
    (layer, expert): FakeEntry(layer, expert)
    for layer in range(LAYERS)
    for expert in range(N_EXPERTS)
}
rows = [[expert_index[(layer, expert)] for expert in range(N_EXPERTS)] for layer in range(LAYERS)]


def make_routing():
    """One token's evaluated routing: per layer, this layer's top-k and the
    prediction for the next, as the runtime has them after its mx.eval."""
    mx.random.seed(0)
    per_layer = []
    for _ in range(LAYERS):
        idx = mx.argsort(mx.random.uniform(shape=(N_EXPERTS,)))[:TOPK].astype(mx.int32)
        weights = mx.random.uniform(shape=(TOPK,))
        pred = mx.argsort(mx.random.uniform(shape=(N_EXPERTS,)))[:TOPK].astype(mx.int32)
        mx.eval(idx, weights, pred)
        per_layer.append((idx, weights, pred))
    mx.synchronize()
    return per_layer


def arm_shipped(per_layer):
    out = []
    for nxt, (idx, weights, pred) in enumerate(per_layer):
        prefetch = []
        for e in reversed(pred.tolist()):
            entry = expert_index.get((nxt, int(e)))
            if entry is not None:
                prefetch.append(entry)
        expert_ids = idx.tolist()
        router_weights = weights.tolist()
        out.append((prefetch, expert_ids, router_weights))
    return out


def arm_rows(per_layer):
    out = []
    for nxt, (idx, weights, pred) in enumerate(per_layer):
        row = rows[nxt]
        prefetch = [row[e] for e in reversed(pred.tolist())]
        expert_ids = idx.tolist()
        router_weights = weights.tolist()
        out.append((prefetch, expert_ids, router_weights))
    return out


def arm_onelist(per_layer):
    out = []
    for nxt, (idx, weights, pred) in enumerate(per_layer):
        row = rows[nxt]
        both = mx.concatenate([idx, pred]).tolist()
        expert_ids = both[:TOPK]
        prefetch = [row[e] for e in reversed(both[TOPK:])]
        router_weights = weights.tolist()
        out.append((prefetch, expert_ids, router_weights))
    return out


def arm_nolookup(per_layer):
    out = []
    for _nxt, (idx, weights, pred) in enumerate(per_layer):
        prefetch = list(reversed(pred.tolist()))
        expert_ids = idx.tolist()
        router_weights = weights.tolist()
        out.append((prefetch, expert_ids, router_weights))
    return out


def time_arm(fn, per_layer):
    fn(per_layer)
    samples = []
    for _ in range(REPEATS):
        t0 = perf_counter()
        fn(per_layer)
        samples.append((perf_counter() - t0) * 1e3)
    samples.sort()
    return samples


def main() -> None:
    per_layer = make_routing()
    print(f"one token = {LAYERS} layers, top-{TOPK}, {len(expert_index)} entries in the index\n")
    print(f"{'arm':10s} {'min':>8s} {'median':>8s} {'max':>8s}   ms per token of CPU")
    base = None
    for name, fn in (
        ("shipped", arm_shipped),
        ("rows", arm_rows),
        ("onelist", arm_onelist),
        ("nolookup", arm_nolookup),
    ):
        s = time_arm(fn, per_layer)
        median = statistics.median(s)
        if base is None:
            base = median
        print(f"{name:10s} {s[0]:8.3f} {median:8.3f} {s[-1]:8.3f}   {median - base:+.3f} vs shipped")

    # The two paths must agree on what they submit.
    a = arm_shipped(per_layer)
    b = arm_rows(per_layer)
    c = arm_onelist(per_layer)
    for i in range(LAYERS):
        assert [id(x) for x in a[i][0]] == [id(x) for x in b[i][0]] == [id(x) for x in c[i][0]]
        assert a[i][1] == b[i][1] == c[i][1]
    print("\nall arms submit the same entries in the same order")


if __name__ == "__main__":
    main()
