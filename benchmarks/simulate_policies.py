"""
Replay a routing trace through a model of ResidentExpertStore and compare
cache policies. Unlike analyze_trace.py's generic simulation, this mirrors
the runtime's two paths:

  prefill  : per-layer quota (slots // 40); keep needed residents, admit
             misses in a chosen ORDER until the quota is full; hits do not
             touch recency.
  decode   : per token, per layer, 6 requests; eviction policy varies.

Prefill admission orders:
  first-come   : trace/work order (current runtime)
  popularity   : experts used by more prompt tokens first

Decode eviction policies:
  lru          : current runtime
  slru         : segmented LRU (probation -> protected on re-hit)
  lfu          : frequency with periodic decay, ties by recency

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/simulate_policies.py \
        benchmarks/results/trace_routing.trace.npz --budgets-gib 40,50,64
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cachalot.metrics.routing_trace import PHASE_PREFILL, load_trace  # noqa: E402

N_LAYERS = 40
EXPERT_BYTES = 18_800_640
GIB = 1024**3


class Store:
    def __init__(self, slots: int, decode_policy: str):
        self.slots = slots
        self.policy = decode_policy
        self.items: OrderedDict[tuple[int, int], None] = OrderedDict()  # recency order (LRU at front)
        self.protected: OrderedDict[tuple[int, int], None] = OrderedDict()  # slru
        self.freq: Counter = Counter()
        self.hits = 0
        self.misses = 0
        self.prefill_hits = 0
        self.prefill_misses = 0
        self.decode_hits = 0
        self.decode_misses = 0
        self.tick = 0

    # ---- helpers -------------------------------------------------------
    def _resident(self, key) -> bool:
        return key in self.items or key in self.protected

    def _count(self) -> int:
        return len(self.items) + len(self.protected)

    def _evict_one(self, avoid_layer: int | None = None):
        if self.policy == "lru":
            for key in self.items:
                if avoid_layer is None or key[0] != avoid_layer:
                    del self.items[key]
                    return
            self.items.popitem(last=False)
        elif self.policy == "slru":
            # evict from probation first, then protected LRU
            src = self.items if self.items else self.protected
            for key in src:
                if avoid_layer is None or key[0] != avoid_layer:
                    del src[key]
                    return
            src.popitem(last=False)
        elif self.policy == "lfu":
            victim = None
            best = None
            for key in self.items:
                if avoid_layer is not None and key[0] == avoid_layer:
                    continue
                score = (self.freq[key], self.items[key])
                if best is None or score < best:
                    best, victim = score, key
            if victim is None:
                victim = next(iter(self.items))
            del self.items[victim]

    def _touch(self, key):
        self.tick += 1
        if self.policy == "slru":
            if key in self.protected:
                self.protected.move_to_end(key)
            else:
                # second hit promotes to protected (cap 80% of slots)
                del self.items[key]
                self.protected[key] = None
                cap = int(self.slots * 0.8)
                while len(self.protected) > cap:
                    k, _ = self.protected.popitem(last=False)
                    self.items[k] = None
                    self.items.move_to_end(k, last=False)  # demoted -> LRU end of probation
        else:
            self.items[key] = self.tick if self.policy == "lfu" else None
            self.items.move_to_end(key)
        self.freq[key] += 1

    def _insert(self, key):
        while self._count() >= self.slots:
            self._evict_one()
        self.tick += 1
        self.items[key] = self.tick if self.policy == "lfu" else None
        self.freq[key] += 1

    def decay(self):
        for k in list(self.freq):
            self.freq[k] *= 0.5

    # ---- runtime paths ---------------------------------------------------
    def prefill_layer(self, layer: int, ordered_keys: list[tuple[int, int]]):
        quota = self.slots // N_LAYERS + (1 if layer < self.slots % N_LAYERS else 0)
        needed = set(ordered_keys)
        # retain residents needed by this layer (up to quota)
        retained = [k for k in ordered_keys if self._resident(k)][:quota]
        retained_set = set(retained)
        # evict stale residents of this layer
        for k in [k for k in list(self.items) + list(self.protected) if k[0] == layer and k not in retained_set]:
            self.items.pop(k, None)
            self.protected.pop(k, None)
        # admission set in given order until quota
        admit = []
        for k in ordered_keys:
            if len(retained) + len(admit) >= quota:
                break
            if k in retained_set or k in admit:
                continue
            if not self._resident(k):
                admit.append(k)
        admit_set = set(admit)
        for k in ordered_keys:
            if self._resident(k):
                self.hits += 1
                self.prefill_hits += 1
                # prefill hits do not perturb recency in the runtime
            else:
                self.misses += 1
                self.prefill_misses += 1
                if k in admit_set:
                    while self._count() >= self.slots:
                        self._evict_one(avoid_layer=layer)
                    self.tick += 1
                    self.items[k] = self.tick if self.policy == "lfu" else None
                    self.freq[k] += 1
        _ = needed

    def decode_request(self, key):
        if self._resident(key):
            self.hits += 1
            self.decode_hits += 1
            self._touch(key)
        else:
            self.misses += 1
            self.decode_misses += 1
            self._insert(key)


def run(arrays, segments, slots, prefill_order, decode_policy):
    phase, layer, position, experts = (arrays[k] for k in ("phase", "layer", "position", "experts"))
    st = Store(slots, decode_policy)
    bounds = [s["at"] for s in segments] + [len(layer)]
    decode_tokens = 0
    for si in range(len(segments)):
        a, b = bounds[si], bounds[si + 1]
        if b <= a:
            continue
        if phase[a] == PHASE_PREFILL:
            for la in range(N_LAYERS):
                m = (layer[a:b] == la)
                rows = experts[a:b][m]  # [tokens, topk]
                if rows.size == 0:
                    continue
                flat = rows.ravel().tolist()
                if prefill_order == "first-come":
                    seen, order = set(), []
                    for e in flat:
                        if e not in seen:
                            seen.add(e)
                            order.append(e)
                else:
                    cnt = Counter(flat)
                    first = {}
                    for i, e in enumerate(flat):
                        first.setdefault(e, i)
                    order = sorted(cnt, key=lambda e: (-cnt[e], first[e]))
                st.prefill_layer(la, [(la, e) for e in order])
        else:
            for pos in np.unique(position[a:b]):
                m = position[a:b] == pos
                for la, row in zip(layer[a:b][m], experts[a:b][m], strict=True):
                    for e in row.tolist():
                        st.decode_request((int(la), int(e)))
                decode_tokens += 1
                if decode_policy == "lfu" and decode_tokens % 16 == 0:
                    st.decay()
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budgets-gib", default="40,50,64")
    args = ap.parse_args()
    arrays, segments = load_trace(args.trace)
    budgets = [float(x) for x in args.budgets_gib.split(",")]

    print("| budget | prefill order | decode policy | prefill hit | decode hit | overall hit | SSD GiB (trace) |")
    print("|---:|---|---|---:|---:|---:|---:|")
    for b in budgets:
        slots = int(b * GIB) // EXPERT_BYTES
        for order in ("first-come", "popularity"):
            for pol in ("lru", "slru", "lfu"):
                st = run(arrays, segments, slots, order, pol)
                pre = st.prefill_hits / max(1, st.prefill_hits + st.prefill_misses)
                dec = st.decode_hits / max(1, st.decode_hits + st.decode_misses)
                tot = st.hits / max(1, st.hits + st.misses)
                gib = st.misses * EXPERT_BYTES / GIB
                print(f"| {b:.0f} | {order} | {pol} | {pre:.1%} | {dec:.1%} | {tot:.1%} | {gib:,.0f} |", flush=True)


if __name__ == "__main__":
    main()
