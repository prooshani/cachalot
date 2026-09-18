"""
How much precision is lost by re-quantizing the shipped FP4 routed experts to a
coarser MLX affine format, measured per layer without running the model.

This is the cheap screen in front of benchmarks/nll_expert_precision.py, which
remains the quality gate. It answers two questions the NLL run is too expensive
to answer for every candidate format:

  * which affine (bits, group_size) pairs are even worth an NLL run, and
  * which layers are the sensitive ones, if a mixed-precision bank is needed.

For a sample of experts per layer it reads the FP4 weights straight from the
shipped checkpoint's shards, dequantizes them to fp32, re-quantizes them with
mx.quantize at each candidate format, and reports:

  w-err    relative Frobenius error of the weights themselves
  y-err    relative error of the routed-expert output on random unit inputs,
           through the official SwiGLU math, which is what the model sees

Both are relative to the FP4 weights, not to the original bf16 checkpoint,
because FP4 is the highest precision this machine holds.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/expert_requant_error.py \
      --experts-per-layer 2 --probes 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR  # noqa: E402
from cachalot.model.fp4_mlx import dequantize_fp4_weight  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from nll_expert_precision import OQ3E_DEFAULT, OQ3EDense  # noqa: E402
from quant_affine import fit_minmax, fit_search, quantize_2bit  # noqa: E402

HIDDEN, INTER, N_EXPERTS, N_LAYERS = 5120, 2304, 384, 40
SHAPES = {
    "w1": ((INTER, HIDDEN // 2), (INTER, HIDDEN // 32)),
    "w3": ((INTER, HIDDEN // 2), (INTER, HIDDEN // 32)),
    "w2": ((HIDDEN, INTER // 2), (HIDDEN, INTER // 32)),
}
# (bits, group) pairs scored with mx.quantize -- which is exactly what
# build_affine_bank.py --fit mlx writes, so a 3-bit row here is the bank that
# would actually be built. Settable, because the interesting pair changes.
CANDIDATES = ((3, 64), (3, 128), (2, 64), (2, 128))
# (label, group_size, fit): our own 2-bit fits, which MLX's max-abs fit loses to.
OWN_FITS = (("2/64 minmax", 64, fit_minmax), ("2/64 search", 64, fit_search),
            ("2/128 search", 128, fit_search))


def expert_bytes(bits: int, group: int) -> int:
    """Slot bytes for one affine-quantized expert: packed weights + bf16 scales and biases."""
    weights = 3 * INTER * HIDDEN
    return weights * bits // 8 + 2 * 2 * (weights // group)


def read_fp4_expert(entry, fds: dict[Path, int]) -> dict[str, mx.array]:
    """Dense fp32 w1, w2, w3 for one expert, read from the shipped shards."""
    raw: dict[str, bytes] = {}
    for t in entry.tensors:
        fd = fds.get(t.shard)
        if fd is None:
            fd = fds[t.shard] = os.open(t.shard, os.O_RDONLY)
        raw[t.name.rsplit(".", 2)[-2] + "." + t.name.rsplit(".", 1)[-1]] = os.pread(
            fd, t.end - t.start, t.start
        )

    dense = {}
    for proj, (wshape, sshape) in SHAPES.items():
        w = mx.array(bytearray(raw[f"{proj}.weight"]), dtype=mx.uint8).reshape(wshape)
        s = mx.array(bytearray(raw[f"{proj}.scale"]), dtype=mx.uint8).reshape(sshape)
        dense[proj] = dequantize_fp4_weight(w, s).astype(mx.float32)
    return dense


def swiglu_expert(x: mx.array, w1: mx.array, w2: mx.array, w3: mx.array, limit: float = 10.0) -> mx.array:
    """Official routed-expert math, unit routing weight (see nll_expert_precision.dense_expert)."""
    xb = x.astype(mx.bfloat16).astype(mx.float32)
    gate = (w1 @ xb).astype(mx.bfloat16).astype(mx.float32)
    up = (w3 @ xb).astype(mx.bfloat16).astype(mx.float32)
    up = mx.clip(up, -limit, limit)
    gate = mx.minimum(gate, mx.array(limit, dtype=mx.float32))
    hidden = (nn.silu(gate) * up).astype(mx.bfloat16).astype(mx.float32)
    return (w2 @ hidden).astype(mx.bfloat16).astype(mx.float32)


def relative(a: mx.array, b: mx.array) -> float:
    num = mx.sum((a - b) ** 2)
    den = mx.sum(b**2)
    return float(mx.sqrt(num / den))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--experts-per-layer", type=int, default=2)
    ap.add_argument("--probes", type=int, default=8, help="random input vectors per expert")
    ap.add_argument("--layers", default="all", help="'all' or a comma-separated list")
    ap.add_argument("--candidates", default="",
                    help="comma-separated bits/group pairs, e.g. '3/64,3/128'; "
                         "default scores both 3-bit group sizes against both 2-bit ones")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--include-oq3e", action="store_true",
                    help="also score the calibrated 3-bit download against the same FP4 reference, "
                         "which measures what imatrix calibration is worth")
    ap.add_argument("--oq3e-path", default=OQ3E_DEFAULT)
    args = ap.parse_args()

    global CANDIDATES
    if args.candidates:
        CANDIDATES = tuple(
            (int(part.split("/")[0]), int(part.split("/")[1]))
            for part in args.candidates.split(",")
            if part
        )

    layers = (
        list(range(N_LAYERS))
        if args.layers == "all"
        else [int(v) for v in args.layers.split(",")]
    )
    index = build_expert_index(args.model_path)
    if not index:
        raise SystemExit(f"no FP4 experts under {args.model_path}")

    oq3e = OQ3EDense(args.oq3e_path) if args.include_oq3e else None
    oq3e_layers = set(oq3e.available_layers()) if oq3e else set()

    rng = random.Random(args.seed)
    mx.random.seed(args.seed)
    fds: dict[Path, int] = {}
    per_layer: dict[int, dict[str, dict[str, float]]] = {}
    t_start = perf_counter()

    for layer in layers:
        ids = rng.sample(range(N_EXPERTS), args.experts_per_layer)
        acc: dict[str, list[tuple[float, float]]] = {f"{b}/{g}": [] for b, g in CANDIDATES}
        for label, _, _ in OWN_FITS:
            acc[label] = []
        if oq3e is not None and layer in oq3e_layers:
            acc["oq3e 3/64"] = []
        for expert in ids:
            entry = index.get((layer, expert))
            if entry is None:
                continue
            dense = read_fp4_expert(entry, fds)
            x = mx.random.normal((HIDDEN, args.probes), dtype=mx.float32)
            x = x / mx.sqrt(mx.sum(x**2, axis=0, keepdims=True)) * mx.sqrt(mx.array(float(HIDDEN)))
            reference = swiglu_expert(x, dense["w1"], dense["w2"], dense["w3"])
            mx.eval(reference)

            for bits, group in CANDIDATES:
                requant = {}
                werr_num = werr_den = 0.0
                for proj, w in dense.items():
                    q, s, b = mx.quantize(w, group_size=group, bits=bits)
                    d = mx.dequantize(q, s, b, group_size=group, bits=bits).astype(mx.float32)
                    requant[proj] = d
                    werr_num += float(mx.sum((d - w) ** 2))
                    werr_den += float(mx.sum(w**2))
                y = swiglu_expert(x, requant["w1"], requant["w2"], requant["w3"])
                acc[f"{bits}/{group}"].append(
                    ((werr_num / werr_den) ** 0.5, relative(y, reference))
                )

            for label, group, fit in OWN_FITS:
                requant = {}
                werr_num = werr_den = 0.0
                for proj, w in dense.items():
                    q, sc, bi = quantize_2bit(w, group_size=group, fit=fit)
                    d = mx.dequantize(q, sc, bi, group_size=group, bits=2).astype(mx.float32)
                    requant[proj] = d
                    werr_num += float(mx.sum((d - w) ** 2))
                    werr_den += float(mx.sum(w**2))
                y = swiglu_expert(x, requant["w1"], requant["w2"], requant["w3"])
                acc[label].append(((werr_num / werr_den) ** 0.5, relative(y, reference)))

            if "oq3e 3/64" in acc:
                cw1, cw2, cw3 = oq3e.weights(layer, [expert])[0]
                werr_num = werr_den = 0.0
                for proj, d in (("w1", cw1), ("w2", cw2), ("w3", cw3)):
                    werr_num += float(mx.sum((d - dense[proj]) ** 2))
                    werr_den += float(mx.sum(dense[proj] ** 2))
                y = swiglu_expert(x, cw1, cw2, cw3)
                acc["oq3e 3/64"].append(((werr_num / werr_den) ** 0.5, relative(y, reference)))

        per_layer[layer] = {
            key: {
                "w_err": sum(v[0] for v in vals) / len(vals),
                "y_err": sum(v[1] for v in vals) / len(vals),
            }
            for key, vals in acc.items()
            if vals
        }
        row = "  ".join(
            f"{key}: w {m['w_err']:.4f} y {m['y_err']:.4f}"
            for key, m in per_layer[layer].items()
        )
        print(f"layer {layer:2d}  {row}", flush=True)

    for fd in fds.values():
        os.close(fd)

    print(f"\n{len(per_layer)} layers, {args.experts_per_layer} experts each, "
          f"{perf_counter() - t_start:.1f} s")
    print("\nformat        MiB/expert   vs oQ3e   mean w-err   mean y-err   worst layer (y)")
    baseline = expert_bytes(3, 64)
    summary = {}
    formats = [(f"{b}/{g}", b, g) for b, g in CANDIDATES]
    formats += [(label, 2, group) for label, group, _ in OWN_FITS]
    if oq3e is not None:
        formats.append(("oq3e 3/64", 3, 64))
    for key, bits, group in formats:
        rows = [m[key] for m in per_layer.values() if key in m]
        if not rows:
            continue
        w = sum(r["w_err"] for r in rows) / len(rows)
        y = sum(r["y_err"] for r in rows) / len(rows)
        worst = max(per_layer, key=lambda layer: per_layer[layer][key]["y_err"])
        size = expert_bytes(bits, group)
        summary[key] = {"bytes": size, "w_err": w, "y_err": y, "worst_layer": worst}
        label = ("oQ3e, calibrated" if key.startswith("oq3e")
                 else key if " " in key else f"{bits}-bit g{group}")
        print(f"{label:<16}   {size / 2**20:8.2f}   {size / baseline * 100:6.1f} %   "
              f"{w:10.4f}   {y:10.4f}   {worst:2d} ({per_layer[worst][key]['y_err']:.4f})")

    out = RESULTS_DIR / "expert_requant_error.json"
    out.write_text(json.dumps({"args": vars(args) | {"model_path": str(args.model_path)},
                               "per_layer": per_layer, "summary": summary}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
