"""
Compare candidate affine fits at one format on real FP4 routed experts, in minutes.

benchmarks/expert_requant_error.py screens *formats* -- bits and group size --
against a fixed pair of fits. This screens *fits* at a fixed format, which is
the question lever 0 of docs/HANDOFF.md asks: mx.quantize's affine fit is
max-abs symmetric and wastes one level at every width, so a better fit buys
accuracy without changing a byte of runtime code or a byte of the bank's size.

At 2 bits that was worth about 17 % of routed-expert output error and the
searched fit is what every 2-bit bank was built with. At 3 bits it is worth
more, not less, which is the whole case for lever 0: the 3-bit bank that was
built, adopted and retired on 2026-09-18 used mx.quantize's fit.

It reports, per fit, both error the screen can see:

  w-err    relative Frobenius error of the weights
  y-err    relative error of the routed-expert output on random unit inputs,
           through the official SwiGLU math

Neither is the quality gate. Section 8.3 of the handoff records that this screen
ranks correctly *within* one quantizer and wrongly *across* quantizers, so its
job is to decide which fit deserves an NLL run, never to decide adoption.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/quant_fit_screen.py \
      --experts 24 --probes 8 --groups 128

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/quant_fit_screen.py \
      --bits 3 --experts 16 --probes 8 --groups 64,128 \
      --fits mlx,search,search+lsq,wide+lsq
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR  # noqa: E402
from expert_requant_error import HIDDEN, N_EXPERTS, N_LAYERS, read_fp4_expert, relative, swiglu_expert  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
import quant_affine as qa  # noqa: E402

FITS = {
    "mlx": None,  # mx.quantize's own max-abs fit, for reference
    "minmax": qa.fit_minmax,
    "search": qa.fit_search,              # the fit the bank in use was built with
    "lsq": qa.fit_lsq,
    "search+lsq": qa.fit_search_lsq,
    "fine": qa.fit_search_fine,
    "fine+lsq": qa.fit_search_fine_lsq,
    "wide+lsq": qa.fit_search_wide_lsq,
    "lsq12": qa.fit_lsq_long,
    "fine+lsq12": qa.fit_fine_lsq_long,
    "max+lsq": qa.fit_search_max_lsq,
}


def quantized(w: mx.array, group: int, bits: int, name: str, scale_dtype) -> mx.array:
    """Dequantized weights for one fit, at the scale precision the bank would store."""
    if name == "mlx":
        q, s, b = mx.quantize(w, group_size=group, bits=bits)
        return mx.dequantize(q, s, b, group_size=group, bits=bits).astype(mx.float32)
    groups = w.astype(mx.float32).reshape(-1, group)
    scale, bias = FITS[name](groups, bits=bits)
    scale = scale.astype(scale_dtype).astype(mx.float32)
    bias = bias.astype(scale_dtype).astype(mx.float32)
    q = qa._quantize(groups, scale, bias, bits)
    return (scale * q + bias).reshape(w.shape).astype(mx.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--experts", type=int, default=24, help="experts sampled across all layers")
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--groups", default="128", help="comma-separated group sizes")
    ap.add_argument("--bits", type=int, default=2, choices=(2, 3, 4),
                    help="width to screen the fits at; the fits themselves are width-agnostic")
    ap.add_argument("--fits", default=",".join(FITS))
    ap.add_argument("--scale-dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"),
                    help="precision the scales and biases are stored at; bfloat16 is the bank format")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    scale_dtype = getattr(mx, args.scale_dtype)
    groups = [int(g) for g in args.groups.split(",")]
    names = [f for f in args.fits.split(",") if f]
    for name in names:
        if name not in FITS:
            raise SystemExit(f"unknown fit {name!r}; known: {', '.join(FITS)}")

    index = build_expert_index(args.model_path)
    if not index:
        raise SystemExit(f"no FP4 experts under {args.model_path}")

    rng = random.Random(args.seed)
    mx.random.seed(args.seed)
    picks = [(rng.randrange(N_LAYERS), rng.randrange(N_EXPERTS)) for _ in range(args.experts)]

    fds: dict[Path, int] = {}
    acc: dict[tuple[int, str], list[tuple[float, float]]] = {}
    t_start = perf_counter()

    for n, (layer, expert) in enumerate(picks, 1):
        entry = index.get((layer, expert))
        if entry is None:
            continue
        dense = read_fp4_expert(entry, fds)
        x = mx.random.normal((HIDDEN, args.probes), dtype=mx.float32)
        x = x / mx.sqrt(mx.sum(x**2, axis=0, keepdims=True)) * mx.sqrt(mx.array(float(HIDDEN)))
        reference = swiglu_expert(x, dense["w1"], dense["w2"], dense["w3"])
        mx.eval(reference)

        for group in groups:
            for name in names:
                requant = {}
                num = den = 0.0
                for proj, w in dense.items():
                    d = quantized(w, group, args.bits, name, scale_dtype)
                    requant[proj] = d
                    num += float(mx.sum((d - w) ** 2))
                    den += float(mx.sum(w**2))
                y = swiglu_expert(x, requant["w1"], requant["w2"], requant["w3"])
                acc.setdefault((group, name), []).append(((num / den) ** 0.5, relative(y, reference)))
        print(f"  expert {n}/{len(picks)}  layer {layer:2d} id {expert:3d}", flush=True)

    for fd in fds.values():
        os.close(fd)

    print(f"\n{len(picks)} experts, {args.probes} probes, {args.bits}-bit, scales stored as "
          f"{args.scale_dtype}, {perf_counter() - t_start:.1f} s\n")
    print("| group | fit | mean w-err | mean y-err | vs search (y) |")
    print("|---:|---|---:|---:|---:|")
    summary = {}
    for group in groups:
        base = acc.get((group, "search"))
        base_y = sum(v[1] for v in base) / len(base) if base else None
        for name in names:
            vals = acc.get((group, name))
            if not vals:
                continue
            w = sum(v[0] for v in vals) / len(vals)
            y = sum(v[1] for v in vals) / len(vals)
            rel = f"{(y / base_y - 1) * 100:+.2f} %" if base_y else "-"
            summary[f"{group}/{name}"] = {"w_err": w, "y_err": y}
            print(f"| {group} | {name} | {w:.4f} | {y:.4f} | {rel} |")

    out = (Path(args.out) if args.out else
           RESULTS_DIR / f"quant_fit_screen_{args.bits}bit_{args.scale_dtype}.json")
    out.write_text(json.dumps({"args": vars(args) | {"model_path": str(args.model_path)},
                               "summary": summary}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
