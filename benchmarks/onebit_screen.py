"""Is 1 bit per weight even in the room? MLX cannot express it, so this asks the
cheaper question first: how much output error does a 1-bit affine fit add on real
experts, against the 2-bit fit that is now shipping and passes the gate."""
import sys
from pathlib import Path
sys.path.insert(0, "benchmarks")
import mlx.core as mx
import numpy as np
from expert_requant_error import build_expert_index, read_fp4_expert, swiglu_expert, relative
from _common import MODEL_PATH

def affine_quant_np(w, bits, group):
    """min/max affine fit, the shape mx.quantize uses, done in numpy so bits=1 works."""
    rows, cols = w.shape
    g = w.reshape(rows, cols // group, group)
    lo = g.min(axis=-1, keepdims=True)
    hi = g.max(axis=-1, keepdims=True)
    levels = (1 << bits) - 1
    scale = np.where(hi > lo, (hi - lo) / levels, 1.0)
    q = np.clip(np.rint((g - lo) / scale), 0, levels)
    return (q * scale + lo).reshape(rows, cols)

def main():
    index = build_expert_index(MODEL_PATH)
    keys = sorted(index)[:: max(1, len(index) // 24)][:24]
    fds = {}
    rng = np.random.default_rng(20260921)
    rows = {b: [] for b in (1, 2, 3)}
    for key in keys:
        e = read_fp4_expert(index[key], fds)
        ref = {k: np.array(v.astype(mx.float32)) for k, v in e.items()}
        x = rng.normal(size=(ref["w1"].shape[1], 8)).astype(np.float32)
        xm = mx.array(x)
        y_ref = swiglu_expert(xm, mx.array(ref["w1"]), mx.array(ref["w2"]), mx.array(ref["w3"]))
        for b in (1, 2, 3):
            q = {k: affine_quant_np(v, b, 128) for k, v in ref.items()}
            y = swiglu_expert(xm, mx.array(q["w1"]), mx.array(q["w2"]), mx.array(q["w3"]))
            werr = float(np.mean([np.abs(q[k]-ref[k]).mean()/np.abs(ref[k]).mean() for k in ref]))
            rows[b].append((werr, relative(y, y_ref)))
    print(f"{len(keys)} real FP4 experts, group 128, min/max affine fit, 8 probes each\n")
    print(f"{'bits':>5} {'MiB/expert':>11} {'mean w-err':>11} {'mean y-err':>11} {'vs 2-bit':>10}")
    base = float(np.mean([r[1] for r in rows[2]]))
    for b in (1, 2, 3):
        w = float(np.mean([r[0] for r in rows[b]])); y = float(np.mean([r[1] for r in rows[b]]))
        mib = 9.49 * b / 2
        print(f"{b:>5} {mib:>11.2f} {w:>11.4f} {y:>11.4f} {y/base:>9.2f}x")
main()
