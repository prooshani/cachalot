"""Timeline of SSD reads vs main-thread phases during a cold prefill: where does the disk idle?"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cachalot.model.moe_prefill_grouped as mpg  # noqa: E402
from _common import MODEL_PATH  # noqa: E402
from cachalot.cache import resident_store as rs  # noqa: E402
from cachalot.model import (  # noqa: E402
    block_compressed_index_source_prefill as bcis,
)
from cachalot.model import (
    block_compressed_reuse_prefill as bcr,
)
from cachalot.model import (
    block_compressed_source_prefill as bcs,
)
from cachalot.model import (
    block_layer0_prefill as bl0,
)
from cachalot.model import (
    block_sliding_window_prefill as bsw,
)
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402

READS = []      # (start, end)
MARKS = []      # (time, label)

orig_read = rs.ResidentExpertStore._read_into


def read_into(self, entry, slot):
    t0 = perf_counter()
    r = orig_read(self, entry, slot)
    READS.append((t0, perf_counter()))
    return r


rs.ResidentExpertStore._read_into = read_into
orig_moe = mpg.moe_prefill_grouped


def moe(*a, **k):
    MARKS.append((perf_counter(), f"moe_start L{k.get('layer_id')}"))
    r = orig_moe(*a, **k)
    MARKS.append((perf_counter(), f"moe_end L{k.get('layer_id')}"))
    return r


for mod in (bl0, bsw, bcr, bcs, bcis):
    mod.moe_prefill_grouped = moe

import cachalot.model.text_decode_runtime as tdr  # noqa: E402

for name in ("layer0_block_prefill", "sliding_window_block_prefill", "compressed_source_block_prefill",
             "compressed_reuse_block_prefill", "compressed_index_source_block_prefill"):
    orig_fn = getattr(tdr, name)

    def make(fn, label):
        def wrapped(*a, **k):
            MARKS.append((perf_counter(), f"block_start {label}"))
            r = fn(*a, **k)
            MARKS.append((perf_counter(), f"block_end {label}"))
            return r
        return wrapped
    setattr(tdr, name, make(orig_fn, name))

orig_eval = mpg.mx.eval


def eval_marked(*a, **k):
    MARKS.append((perf_counter(), "eval_start"))
    r = orig_eval(*a, **k)
    MARKS.append((perf_counter(), "eval_end"))
    return r


mpg.mx.eval = eval_marked


def main():
    n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = build_prompt(rt, enc, prompt_sources()[0][1], n_tokens)
        rt.reset()
        t0 = perf_counter()
        rt.prefill_tokens(ids)
        wall = perf_counter() - t0
    reads = np.array(READS) - t0
    marks = [(t - t0, lab) for t, lab in MARKS]
    # union of busy intervals
    order = reads[np.argsort(reads[:, 0])]
    busy = 0.0
    gaps = []
    cur_s, cur_e = order[0]
    for s, e in order[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            gaps.append((cur_e, s))
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    print(f"wall {wall:.2f}s, SSD busy {busy:.2f}s ({busy / wall:.0%}), first read at {order[0, 0]:.2f}s, last read end {cur_e:.2f}s")
    gaps = [(a, b) for a, b in gaps if b - a > 0.02]
    total_gap = sum(b - a for a, b in gaps)
    print(f"idle gaps > 20 ms: {len(gaps)}, total {total_gap:.2f}s")
    # classify gaps by phase: inside a MoE call or between (attention/route)
    inside = 0.0
    for a, b in gaps:
        # find last mark before gap start
        prev = [lab for t, lab in marks if t <= a]
        lab = prev[-1] if prev else "pre"
        if lab.startswith("moe_start"):
            inside += b - a
    print(f"  idle inside MoE loops: {inside:.2f}s; idle between layers (attention/route/HC): {total_gap - inside:.2f}s")
    print("  largest gaps:", [(round(a, 2), round(b - a, 3)) for a, b in sorted(gaps, key=lambda g: g[0] - g[1])[:8]])
    # per-layer MoE durations
    starts = {lab.split()[1]: t for t, lab in marks if lab.startswith("moe_start")}
    ends = {lab.split()[1]: t for t, lab in marks if lab.startswith("moe_end")}
    durs = [ends[k] - starts[k] for k in starts]
    between = [starts[f"L{i + 1}"] - ends[f"L{i}"] for i in range(39)]
    # first eval inside each MoE = route eval; time from moe_start to that eval end
    route_eval = []
    pre_moe = []
    post_moe = []
    for i in range(40):
        ms = starts[f"L{i}"]
        evs = [(t, lab) for t, lab in marks if t > ms]
        es = next((t for t, lab in evs if lab == "eval_start"), None)
        ee = next((t for t, lab in evs if lab == "eval_end"), None)
        if es and ee:
            route_eval.append(ee - es)
        bs = max((t for t, lab in marks if lab.startswith("block_start") and t <= ms), default=ms)
        pre_moe.append(ms - bs)
        me = ends[f"L{i}"]
        be = min((t for t, lab in marks if lab.startswith("block_end") and t >= me), default=me)
        post_moe.append(be - me)
    print(f"  route eval (GPU sync at MoE start): mean {np.mean(route_eval):.3f}s sum {np.sum(route_eval):.2f}s")
    print(f"  block_start->moe_start (attention+HC issue): mean {np.mean(pre_moe):.3f}s sum {np.sum(pre_moe):.2f}s")
    print(f"  moe_end->block_end (hc_post issue): mean {np.mean(post_moe):.3f}s sum {np.sum(post_moe):.2f}s")
    print(f"  MoE per layer: mean {np.mean(durs):.3f}s, sum {np.sum(durs):.2f}s; between-layer (attention etc): mean {np.mean(between):.3f}s, sum {np.sum(between):.2f}s")


if __name__ == "__main__":
    main()
