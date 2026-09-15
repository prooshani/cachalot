"""Timeline of SSD reads vs main-thread phases during a cold prefill: where does the disk idle?"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
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

# finer marks for the pre-read phase of each layer
from cachalot.cache import resident_store as _rs  # noqa: E402

_orig_route_rows = mpg.route_topk_rows
_orig_prepare = _rs.ResidentExpertStore.prepare_prefill_layer


def _route_rows_marked(*a, **k):
    MARKS.append((perf_counter(), "route_start"))
    r = _orig_route_rows(*a, **k)
    MARKS.append((perf_counter(), "route_built"))
    return r


def _prepare_marked(self, *a, **k):
    MARKS.append((perf_counter(), "prepare_start"))
    r = _orig_prepare(self, *a, **k)
    MARKS.append((perf_counter(), "prepare_end"))
    return r


mpg.route_topk_rows = _route_rows_marked
_rs.ResidentExpertStore.prepare_prefill_layer = _prepare_marked

from cachalot.model.text_decode_runtime import TextDecodeRuntime as _TDR  # noqa: E402

_orig_engram = _TDR._prefill_apply_engram


def _engram_marked(self, *a, **k):
    MARKS.append((perf_counter(), "engram_start"))
    r = _orig_engram(self, *a, **k)
    mx.eval(r)
    MARKS.append((perf_counter(), "engram_end"))
    return r


_TDR._prefill_apply_engram = _engram_marked

# CACHALOT_PROFILE_SYNC=1: evaluate + synchronize around the main non-MoE phases so
# their GPU time is attributed per phase (changes overlap; use for attribution only)
PHASE = {}
if os.environ.get("CACHALOT_PROFILE_SYNC") == "1":
    import cachalot.model.block_compressed_index_source_prefill as _bis
    import cachalot.model.block_compressed_reuse_prefill as _brp
    import cachalot.model.block_compressed_source_prefill as _bsp
    import cachalot.model.block_layer0_prefill as _bl0
    import cachalot.model.block_sliding_window_prefill as _bsw

    def _synced(name, fn):
        def wrapper(*a, **k):
            # first: evaluate the inputs (upstream lazy work), then time the phase alone,
            # then time a second identical call (GPU clock already up)
            ins = [v for v in list(a) + list(k.values()) if isinstance(v, mx.array)]
            mx.synchronize()
            t0 = perf_counter()
            mx.eval(*ins)
            mx.synchronize()
            PHASE[name + " (upstream lazy)"] = PHASE.get(name + " (upstream lazy)", 0.0) + perf_counter() - t0
            t0 = perf_counter()
            r = fn(*a, **k)
            flat = [v for v in (r if isinstance(r, tuple) else (r,)) if isinstance(v, mx.array)]
            mx.eval(*flat)
            mx.synchronize()
            PHASE[name] = PHASE.get(name, 0.0) + perf_counter() - t0
            t0 = perf_counter()
            r2 = fn(*a, **k)
            flat2 = [v for v in (r2 if isinstance(r2, tuple) else (r2,)) if isinstance(v, mx.array)]
            mx.eval(*flat2)
            mx.synchronize()
            PHASE[name + " (2nd call)"] = PHASE.get(name + " (2nd call)", 0.0) + perf_counter() - t0
            return r
        return wrapper

    for mod in (_brp, _bsw, _bl0, _bsp, _bis):
        for fname in ("attention_prefill_batched", "hc_mixes_prefill_exact", "shared_expert_forward_batched",
                      "compressed_source_chunk", "index_source_chunk"):
            if hasattr(mod, fname):
                setattr(mod, fname, _synced(fname, getattr(mod, fname)))
    mpg.shared_expert_forward_batched = _synced("shared_expert_forward_batched", mpg.shared_expert_forward_batched)


def analyze(t0, wall, label):
    reads = np.array(READS) - t0
    marks = [(t - t0, lab) for t, lab in MARKS]
    order = reads[np.argsort(reads[:, 0])]
    busy = 0.0
    gaps = []
    cur_s, cur_e = order[0]
    for s_, e_ in order[1:]:
        if s_ > cur_e:
            busy += cur_e - cur_s
            gaps.append((cur_e, s_))
            cur_s, cur_e = s_, e_
        else:
            cur_e = max(cur_e, e_)
    busy += cur_e - cur_s
    gaps = [(a, b) for a, b in gaps if b - a > 0.02]
    starts = {lab.split()[1]: t for t, lab in marks if lab.startswith("moe_start")}
    ends = {lab.split()[1]: t for t, lab in marks if lab.startswith("moe_end")}
    durs = [ends[k] - starts[k] for k in starts]
    between = [starts[f"L{i + 1}"] - ends[f"L{i}"] for i in range(39)]
    evals = [(t, lab) for t, lab in marks if lab.startswith("eval")]
    ev_time = sum(evals[i + 1][0] - evals[i][0] for i in range(0, len(evals) - 1, 2) if evals[i][1] == "eval_start")
    print(f"{label}: wall {wall:.2f}s | SSD busy {busy:.2f}s ({busy / wall:.0%}) | {len(READS)} reads | "
          f"idle gaps>20ms {len(gaps)} total {sum(b - a for a, b in gaps):.2f}s | MoE sum {np.sum(durs):.2f}s | "
          f"between-layer sum {np.sum(between):.2f}s | eval sum {ev_time:.2f}s ({len(evals) // 2} evals)")
    print("   largest gaps:", [(round(a, 1), round(b - a, 2)) for a, b in sorted(gaps, key=lambda g: g[0] - g[1])[:6]])
    # where do the gaps fall relative to layer boundaries?
    bounds = sorted([(t, f"start {k}") for k, t in starts.items()] + [(t, f"end {k}") for k, t in ends.items()])
    tail_total = 0.0
    for a, b in sorted(gaps, key=lambda g: g[0] - g[1])[:12]:
        before = [lab for t, lab in bounds if t <= a]
        inside = [f"{lab}@{t - a:+.2f}" for t, lab in bounds if a < t < b]
        print(f"   gap {a:6.1f}s +{b - a:.2f}s | last boundary before: {before[-1] if before else '-'} ({a - [t for t, lab in bounds if t <= a][-1]:+.2f}s) "
              f"| boundaries inside: {inside}")
    # loader tail per layer: last read end of layer L vs moe_end L
    for k in sorted(starts, key=lambda x: int(x[1:])):
        i = int(k[1:])
        s_, e_ = starts[k], ends[k]
        layer_reads = order[(order[:, 0] >= s_) & (order[:, 0] < e_)]
        if len(layer_reads):
            first_read = layer_reads[:, 0].min() - s_
            last_read_end = layer_reads[:, 1].max()
            tail = e_ - last_read_end
            tail_total += max(tail, 0)
            if i < 6 or i % 8 == 0:
                print(f"   L{i:2d}: moe {e_ - s_:5.2f}s, {len(layer_reads):4d} reads, first read +{first_read:.2f}s, compute tail after last read {tail:.2f}s")
    print(f"   sum of per-layer compute tails after the last read: {tail_total:.2f}s")
    # pre-read phase breakdown for a few layers
    for k in ("L2", "L3", "L20", "L33"):
        s_ = starts[k]
        seq = [(t - s_, lab) for t, lab in marks if s_ <= t <= s_ + 1.5 and not lab.startswith("moe")]
        first_read = order[order[:, 0] >= s_][0, 0] - s_
        head = [f"{lab}@{t:.3f}" for t, lab in seq if t < first_read + 0.01][:12]
        print(f"   {k} pre-read sequence (first read at +{first_read:.3f}s): {head}")


def mem_report() -> str:
    import resource
    import subprocess
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    comp = free = 0
    for line in vm.splitlines():
        if "occupied by compressor" in line:
            comp = int(line.split()[-1].rstrip(".")) * 16384 / 2**30
        if line.startswith("Pages free"):
            free = int(line.split()[-1].rstrip(".")) * 16384 / 2**30
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    return (f"mlx active {mx.get_active_memory() / 2**30:.1f} GiB peak {mx.get_peak_memory() / 2**30:.1f} GiB "
            f"cache {mx.get_cache_memory() / 2**30:.1f} GiB | max rss {rss:.1f} GiB | free {free:.1f} GiB "
            f"compressor {comp:.1f} GiB | {swap}")


def main():
    n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = build_prompt(rt, enc, prompt_sources()[0][1], n_tokens)
        for run in range(repeat):
            READS.clear()
            MARKS.clear()
            rt.reset()
            t0 = perf_counter()
            rt.prefill_tokens(ids)
            wall = perf_counter() - t0
            analyze(t0, wall, f"run {run} ({'cold' if run == 0 else 'warm'})")
            eng = [(t - t0, lab) for t, lab in MARKS if lab.startswith("engram")]
            if eng:
                durs = [eng[i + 1][0] - eng[i][0] for i in range(0, len(eng) - 1, 2)]
                print(f"   engram phases: {[round(d, 2) for d in durs]} s")
            print("   memory:", mem_report())
            if PHASE:
                print("   synced phase totals:", {k: round(v, 2) for k, v in sorted(PHASE.items(), key=lambda kv: -kv[1])})
                PHASE.clear()


if __name__ == "__main__":
    main()
