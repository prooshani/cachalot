"""
Replay a GLM / MiniMax decode routing trace through cache policies at several budgets (HANDOFF section 18.1).

The trace is what `glm_prefill_timeline.py` writes with ROUTE_TRACE=path.npy: one row per decode request,
(token step, layer, expert), in request order. The replay starts from an empty cache, so the first
WARMUP_TOKENS tokens only fill it and are not scored.

Policies:
  lru     the runtime's default (ResidentExpertStore, CACHALOT_EVICT=lru)
  slru    segmented LRU, protected share 0.8 (CACHALOT_EVICT=slru)
  lfu     frequency with halving every 64 tokens, ties by recency
  belady  the offline optimum (evict the expert whose next use is farthest): an upper bound, not a policy

    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_policy_replay.py TRACE.npy \
        --expert-mib 23.62 --budgets-gib 44,52,60,64 --warmup 64
"""

from __future__ import annotations

import argparse
import heapq
from collections import Counter, OrderedDict

import numpy as np


def replay(trace, slots, policy, warmup):
    items: OrderedDict = OrderedDict()
    protected: OrderedDict = OrderedDict()
    freq: Counter = Counter()
    hits = misses = 0
    last_decay = 0
    keys = [(int(layer), int(expert)) for _, layer, expert in trace]
    steps = trace[:, 0]
    if policy == "belady":
        nxt = [0] * len(keys)
        seen: dict = {}
        for i in range(len(keys) - 1, -1, -1):
            nxt[i] = seen.get(keys[i], 1 << 60)
            seen[keys[i]] = i
        heap: list = []  # (-next_use, key)
        next_use: dict = {}
    for i, key in enumerate(keys):
        scored = steps[i] > warmup
        if policy == "lfu" and steps[i] - last_decay >= 64:
            last_decay = steps[i]
            for k in list(freq):
                freq[k] //= 2
        if key in items:
            hits += scored
            items.move_to_end(key)
            if policy == "slru":
                protected[key] = None
                protected.move_to_end(key)
                limit = int(0.8 * slots)
                while len(protected) > limit:
                    protected.popitem(last=False)
            freq[key] += 1
            if policy == "belady":
                next_use[key] = nxt[i]
                heapq.heappush(heap, (-nxt[i], key))
            continue
        misses += scored
        if len(items) >= slots:
            if policy == "lru":
                items.popitem(last=False)
            elif policy == "slru":
                victim = next((k for k in items if k not in protected), None)
                if victim is None:
                    victim = next(iter(items))
                    protected.pop(victim, None)
                del items[victim]
            elif policy == "lfu":
                # sample the 256 least recent, evict the least frequent among them
                cand = [k for _, k in zip(range(256), items)]
                victim = min(cand, key=lambda k: freq[k])
                del items[victim]
            elif policy == "belady":
                while True:
                    neg, k = heapq.heappop(heap)
                    if k in items and next_use.get(k) == -neg:
                        break
                del items[k]
        items[key] = None
        freq[key] += 1
        if policy == "belady":
            next_use[key] = nxt[i]
            heapq.heappush(heap, (-nxt[i], key))
    return hits, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--expert-mib", type=float, default=23.62)
    ap.add_argument("--budgets-gib", default="44,52,60,64")
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--policies", default="lru,slru,lfu,belady")
    args = ap.parse_args()
    trace = np.load(args.trace)
    tokens = int(trace[:, 0].max())
    scored_tokens = tokens - args.warmup
    print(f"trace: {len(trace)} requests, {tokens} tokens, {len(set(map(tuple, trace[:, 1:].tolist())))} distinct experts; "
          f"scored after token {args.warmup}")
    for gib in [float(g) for g in args.budgets_gib.split(",")]:
        slots = int(gib * 1024 / args.expert_mib)
        for policy in args.policies.split(","):
            h, m = replay(trace, slots, policy, args.warmup)
            print(f"budget {gib:5.1f} GiB  slots {slots:5d}  {policy:6s}  hit {h / max(1, h + m):.3f}  "
                  f"misses/token {m / max(1, scored_tokens):5.1f}", flush=True)


if __name__ == "__main__":
    main()
