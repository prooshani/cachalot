"""
Per-layer breakdown of a cold decode step: wall time inside get_many vs the
SSD bytes it moved, read and promotion durations per miss, plus the pure
compute time of the same token when every expert is resident.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/profile_decode_loader.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.cache import resident_store as rs  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

EXPERT_BYTES = 18_800_640


GPU_TOUCH = [False]


def main():
    events = []
    loads = []
    orig_get_many = rs.ResidentExpertStore.get_many
    orig_load = rs.ResidentExpertStore._load_expert

    def get_many(self, entries):
        t0 = perf_counter()
        n0 = len(loads)
        out = orig_get_many(self, entries)
        events.append((perf_counter() - t0, len(loads) - n0))
        return out

    def load(self, entry):
        t0 = perf_counter()
        res = orig_load(self, entry)
        if GPU_TOUCH[0]:
            r = res[0]
            mx.eval(r.w1_weight.sum(), r.w2_weight.sum(), r.w3_weight.sum(), r.w1_scale.sum(), r.w2_scale.sum(), r.w3_scale.sum())
        loads.append((perf_counter() - t0, res[2], res[3], threading.current_thread().name))
        return res

    rs.ResidentExpertStore.get_many = get_many
    rs.ResidentExpertStore._load_expert = load

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-gib', type=float, default=2.0)
    ap.add_argument('--defer-free', action='store_true')
    ap.add_argument('--gpu-touch', action='store_true')
    ap.add_argument('--gc', default='on', choices=['on', 'freeze', 'off'])
    args = ap.parse_args()
    GPU_TOUCH[0] = args.gpu_touch
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096, mlx_cache_limit_bytes=int(args.cache_gib * 1024**3)) as rt:
        evict_time = [0.0]
        graveyard = []
        import collections

        class TimedDict(collections.OrderedDict):
            def popitem(self, last=True):
                t0 = perf_counter()
                item = super().popitem(last)
                if args.defer_free:
                    graveyard.append(item)
                else:
                    del item
                evict_time[0] += perf_counter() - t0
                return (None, type('E', (), {'size': EXPERT_BYTES})())

        td = TimedDict(rt.expert_store._items)
        rt.expert_store._items = td
        print('mlx cache limit GiB', args.cache_gib, 'defer_free', args.defer_free, 'gpu_touch', args.gpu_touch, 'gc', args.gc, flush=True)
        import gc
        gc_stats = {'n': [0, 0, 0], 't': [0.0, 0.0, 0.0], 'start': 0.0}

        def gc_cb(phase, info):
            if phase == 'start':
                gc_stats['start'] = perf_counter()
            else:
                g = info['generation']
                gc_stats['n'][g] += 1
                gc_stats['t'][g] += perf_counter() - gc_stats['start']

        gc.callbacks.append(gc_cb)
        print('tracked objects before decode:', len(gc.get_objects()), flush=True)
        if args.gc == 'freeze':
            gc.collect()
            gc.freeze()
        elif args.gc == 'off':
            gc.disable()
        enc = load_official_encoding(MODEL_PATH)
        ids = list(rt.tokenizer.encode(enc.encode_messages(
            [{"role": "user", "content": "List three facts about the deep ocean."}],
            thinking_mode="chat", reasoning_effort=None)))
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())

        for step in range(4):
            events.clear()
            loads.clear()
            evict_time[0] = 0.0
            gc_stats['n'] = [0, 0, 0]
            gc_stats['t'] = [0.0, 0.0, 0.0]
            if args.defer_free:
                import threading as _t
                dead = list(graveyard)
                graveyard.clear()
                _t.Thread(target=lambda d=dead: d.clear(), daemon=True).start()
            t0 = perf_counter()
            res = rt.decode_token(tok)
            mx.eval(res.logits)
            wall = perf_counter() - t0
            misses = sum(n for _, n in events)
            gm_wall = sum(w for w, _ in events)
            read_sum = sum(r for _, r, _, _ in loads)
            promo_sum = sum(p for _, _, p, _ in loads)
            ssd_floor = misses * EXPERT_BYTES / 1.0e9
            print(f"token {step}: wall {wall:.2f}s | misses {misses:3d} | get_many wall {gm_wall:.2f}s "
                  f"| SSD floor {ssd_floor:.2f}s | read sum {read_sum:.2f}s | promo sum {promo_sum:.2f}s "
                  f"| outside get_many {wall - gm_wall:.2f}s | evict(free) {evict_time[0]:.2f}s | mlx cache {mx.get_cache_memory() / 2**30:.2f} GiB | gc n={gc_stats['n']} t={[round(t, 3) for t in gc_stats['t']]}", flush=True)
            if step == 0:
                per_layer = [(round(w, 3), n) for w, n in events]
                print("  per-layer (get_many wall, misses):", per_layer, flush=True)
                slow = sorted(loads, key=lambda x: -x[0])[:5]
                print("  slowest loads (total, read, promote, thread):",
                      [(round(a, 3), round(b, 3), round(c, 3), d[-1]) for a, b, c, d in slow], flush=True)
                if loads:
                    print(f"  mean read {read_sum / len(loads) * 1e3:.1f} ms, mean promote {promo_sum / len(loads) * 1e3:.1f} ms",
                          flush=True)
            tok = int(res.logits.argmax().item())


if __name__ == "__main__":
    main()
