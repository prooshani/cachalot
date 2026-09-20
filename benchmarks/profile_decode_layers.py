"""
Per-layer attribution of an all-resident decode token, with no barrier added.

Section 6.3 of the handoff left roughly 30 ms of a 90 ms token unaccounted: the
four SOURCE_LAYERS, the four INDEX_ONLY_SOURCE_LAYERS, the two sliding-window
layers, Engram, the final head. Nothing had timed them, and the instrument that
had priced the other pieces evaluates each one behind its own barrier, which is
exactly the mistake that cost three sessions.

This adds no synchronisation of its own. It wraps the three per-layer entry
points on the runtime instance and records wall time around each call, and it
wraps mx.eval/mx.synchronize so the time a layer spends waiting inside its own
router eval can be separated from the time it spends on the CPU building
operations. The token's natural 40 drains are the only barriers.

One caveat on reading the output: a layer's routed-expert work is queued after
its router eval and is therefore drained by the *next* layer's eval, so about
one stage of GPU time is shifted forward. Every layer queues the same top-6
expert work, so the shift is uniform across classes and the difference between
a source layer and a reuse layer is still attributed to the layer that caused
it. Engram and the head are measured against their own eval sites.

    PYTHONPATH=src python benchmarks/profile_decode_layers.py --prompt-tokens 512
"""
from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import (  # noqa: E402
    COMPRESS_RATIO,
    INDEX_ONLY_SOURCE_LAYERS,
    SOURCE_LAYERS,
    TextDecodeRuntime,
)
from trace_routing import build_prompt, prompt_sources  # noqa: E402

_real_eval = mx.eval
_real_sync = mx.synchronize

CURRENT = {"tag": "outside"}
EVAL_T: dict[str, float] = defaultdict(float)
EVAL_N: dict[str, int] = defaultdict(int)
WALL: dict[str, list[float]] = defaultdict(list)


def _timed_eval(*args, **kwargs):
    t0 = perf_counter()
    _real_eval(*args, **kwargs)
    EVAL_T[CURRENT["tag"]] += perf_counter() - t0
    EVAL_N[CURRENT["tag"]] += 1


def _timed_sync(*args, **kwargs):
    t0 = perf_counter()
    _real_sync(*args, **kwargs)
    EVAL_T[CURRENT["tag"]] += perf_counter() - t0
    EVAL_N[CURRENT["tag"]] += 1


def _wrap_layer(fn, kind):
    def inner(layer_id, *args, **kwargs):
        tag = f"{kind}:{layer_id}"
        prev = CURRENT["tag"]
        CURRENT["tag"] = tag
        t0 = perf_counter()
        try:
            return fn(layer_id, *args, **kwargs)
        finally:
            WALL[tag].append(perf_counter() - t0)
            CURRENT["tag"] = prev

    return inner


def _wrap_engram(fn):
    def inner(x, layer_id, *args, **kwargs):
        tag = f"engram:{layer_id}"
        prev = CURRENT["tag"]
        CURRENT["tag"] = tag
        t0 = perf_counter()
        try:
            return fn(x, layer_id, *args, **kwargs)
        finally:
            WALL[tag].append(perf_counter() - t0)
            CURRENT["tag"] = prev

    return inner


def layer_class(layer_id: int) -> str:
    if layer_id in SOURCE_LAYERS:
        return f"source (ratio {SOURCE_LAYERS[layer_id]})"
    if layer_id in INDEX_ONLY_SOURCE_LAYERS:
        return "index-only source"
    ratio = COMPRESS_RATIO.get(layer_id, 0)
    if ratio == 0:
        return "sliding window"
    return f"reuse (ratio {ratio})"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=12)
    args = ap.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        snap = rt.snapshot()
        rt.decode_token(tok)
        rt.restore(snap)

        rt._decode_source = _wrap_layer(rt._decode_source, "source")
        rt._decode_index_source = _wrap_layer(rt._decode_index_source, "index")
        rt._decode_reuse = _wrap_layer(rt._decode_reuse, "reuse")
        rt._apply_engram = _wrap_engram(rt._apply_engram)

        totals = []
        mx.eval = _timed_eval
        mx.synchronize = _timed_sync
        try:
            for _ in range(args.repeats):
                rt.restore(snap)
                CURRENT["tag"] = "outside"
                t0 = perf_counter()
                r = rt.decode_token(tok)
                mx.eval(r.logits)
                totals.append((perf_counter() - t0) * 1e3)
        finally:
            mx.eval = _real_eval
            mx.synchronize = _real_sync

        n = args.repeats
        print(f"all-resident token, {args.prompt_tokens}-token context, {n} repeats")
        print(f"  whole token   min {min(totals):6.1f} ms   median {statistics.median(totals):6.1f} ms")
        wall_sum = sum(statistics.median(v) for v in WALL.values()) * 1e3
        eval_sum = sum(EVAL_T.values()) / n * 1e3
        print(f"  attributed    {wall_sum:6.1f} ms in wrapped layer calls, of which {eval_sum:6.1f} ms inside eval")

        print("\n  per layer, median over repeats (wall = CPU build + wait inside this layer's evals):")
        print(f"    {'layer':>5s} {'class':20s} {'wall':>8s} {'in eval':>9s} {'cpu':>8s}")
        by_class: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for layer_id in range(40):
            tags = [t for t in WALL if t.split(":")[1] == str(layer_id) and not t.startswith("engram")]
            if not tags:
                continue
            tag = tags[0]
            wall = statistics.median(WALL[tag]) * 1e3
            ev = EVAL_T[tag] / n * 1e3
            cls = layer_class(layer_id)
            by_class[cls].append((wall, ev))
            print(f"    {layer_id:5d} {cls:20s} {wall:7.2f} ms {ev:8.2f} ms {wall - ev:7.2f} ms")

        print("\n  by class:")
        print(f"    {'class':20s} {'layers':>6s} {'wall/layer':>11s} {'total wall':>11s} {'in eval':>9s} {'cpu':>8s}")
        grand = 0.0
        for cls, rows in sorted(by_class.items(), key=lambda kv: -sum(w for w, _ in kv[1])):
            walls = [w for w, _ in rows]
            evs = [e for _, e in rows]
            grand += sum(walls)
            print(f"    {cls:20s} {len(rows):6d} {statistics.mean(walls):10.2f} ms "
                  f"{sum(walls):10.1f} ms {sum(evs):8.1f} ms {sum(walls) - sum(evs):7.1f} ms")
        for tag, vals in sorted(WALL.items()):
            if tag.startswith("engram"):
                w = statistics.median(vals) * 1e3
                grand += w
                print(f"    {tag:20s} {1:6d} {w:10.2f} ms {w:10.1f} ms {EVAL_T[tag] / n * 1e3:8.1f} ms "
                      f"{w - EVAL_T[tag] / n * 1e3:7.1f} ms")
        outside_ev = EVAL_T['outside'] / n * 1e3
        print(f"    {'outside layers':20s} {'':6s} {'':10s}    {statistics.median(totals) - grand:7.1f} ms "
              f"{outside_ev:8.1f} ms  (head, embedding, final HC, sampling)")


if __name__ == "__main__":
    main()
