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


if __name__ == "__main__":
    main()
