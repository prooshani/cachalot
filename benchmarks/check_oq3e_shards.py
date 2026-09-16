"""
Compare oQ3e 3-bit experts against the shipped FP4 experts, weight by weight,
for every layer whose oQ3e shards have finished downloading. No model load:
reads both banks directly. Reports per-layer relative RMS error of the
dequantized matrices and the relative output error of the expert function on
random activations. Validates the stacked-tensor offsets before the full
download completes, and gives a first quality hint ahead of the NLL gate.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/check_oq3e_shards.py --experts-per-layer 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from nll_expert_precision import HIDDEN, INTER, OQ3E_DEFAULT, OQ3EDense, dense_expert  # noqa: E402
from cachalot.cache.resident_store import tensor_sizes_from_entry  # noqa: E402
from cachalot.model.fp4_mlx import dequantize_fp4_weight  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader  # noqa: E402


def fp4_dense(reader, entry):
    sizes = tensor_sizes_from_entry(entry)
    views = {n: bytearray(sz) for n, sz in sizes.items()}
    reader.read_expert_into(entry, views)
    a = {n: mx.array(np.frombuffer(v, dtype=np.uint8)) for n, v in views.items()}
    w1 = dequantize_fp4_weight(a["w1.weight"].reshape(INTER, HIDDEN // 2), a["w1.scale"].reshape(INTER, HIDDEN // 32))
    w3 = dequantize_fp4_weight(a["w3.weight"].reshape(INTER, HIDDEN // 2), a["w3.scale"].reshape(INTER, HIDDEN // 32))
    w2 = dequantize_fp4_weight(a["w2.weight"].reshape(HIDDEN, INTER // 2), a["w2.scale"].reshape(HIDDEN, INTER // 32))
    return w1, w2, w3


def rel_rms(a, b):
    return float(mx.sqrt(mx.mean((a - b) ** 2)) / mx.sqrt(mx.mean(b ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oq3e-path", default=OQ3E_DEFAULT)
    ap.add_argument("--experts-per-layer", type=int, default=4)
    ap.add_argument("--layers", default=None, help="comma-separated subset")
    args = ap.parse_args()
    oq = OQ3EDense(args.oq3e_path, cache_bytes=2 * 1024**3)
    layers = oq.available_layers()
    if args.layers:
        layers = [l for l in layers if l in {int(v) for v in args.layers.split(",")}]
    print(f"oQ3e layers complete: {len(layers)}/40 -> {layers}")
    if not layers:
        return
    fp4_index = build_expert_index(MODEL_PATH)
    reader = ExpertReader(bypass_page_cache=False)
    mx.random.seed(0)
    xs = mx.random.normal((4, HIDDEN)).astype(mx.bfloat16) * 0.5
    try:
        for layer in layers:
            errs = {"w1": [], "w2": [], "w3": [], "out": []}
            for e in np.linspace(0, 383, args.experts_per_layer, dtype=int):
                (o1, o2, o3), = oq.weights(layer, [int(e)])
                f1, f2, f3 = fp4_dense(reader, fp4_index[(layer, int(e))])
                mx.eval(o1, o2, o3, f1, f2, f3)
                errs["w1"].append(rel_rms(o1, f1))
                errs["w2"].append(rel_rms(o2, f2))
                errs["w3"].append(rel_rms(o3, f3))
                for x in xs:
                    yo = dense_expert(x, o1, o2, o3, 1.0, 10.0)
                    yf = dense_expert(x, f1, f2, f3, 1.0, 10.0)
                    errs["out"].append(rel_rms(yo, yf))
            print(f"layer {layer:2d}: rel RMS error w1 {np.mean(errs['w1']):.4f} w2 {np.mean(errs['w2']):.4f} "
                  f"w3 {np.mean(errs['w3']):.4f} | expert output {np.mean(errs['out']):.4f} (max {np.max(errs['out']):.4f}) "
                  f"| {args.experts_per_layer} experts", flush=True)
        print(f"oq3e reads {oq.reads} slabs, {oq.read_bytes / 1e9:.1f} GB at {oq.read_bytes / 1e9 / max(1e-9, oq.read_seconds):.2f} GB/s")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
