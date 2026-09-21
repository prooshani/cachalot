"""
Screen: is the shared FP8 expert at the memory wall, and is anything left in
its launch count?

Section 9.25 prices the shared expert at 0.121 ms per layer, 4.8 ms per token
across forty layers, and calls it "never screened". It is three FP8 GEMVs per
layer over 33.75 MiB of E4M3 weights -- w1 and w3 from the same quantized
activation, then w2 from the SwiGLU hidden -- so it is a pure streaming read
the same way the routed experts are.

Four arms, all on the real shapes, all chained across forty distinct layers
inside one mx.eval so the 0.20 ms per-eval floor is excluded and no launch
reads what the previous launch left in cache:

  shipped     shared_expert_forward exactly as moe_layer_metal calls it
  fused w1w3  w1 and w3 stacked into one [2*I, H] weight and issued as one
              GEMV, which is the only structural idea the shape allows: both
              read the same activation and neither depends on the other.
              Two launches per layer instead of three, the same bytes.
  sum         mx.sum over the same uint8 buffers: no arithmetic, no
              dequantization, just a full read. This is what the machine will
              give for these buffers.
  quantize    the two quantize_fp8_activation calls alone, which move 5120 and
              2304 bytes and should be launch cost only.

Reported as GB/s over the bytes each arm must read, so the arms are comparable.

    PYTHONPATH=src python benchmarks/micro_shared_expert_roofline.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cachalot.model.fp8_linear_metal import (  # noqa: E402
    fp8_linear_quantized,
    quantize_fp8_activation,
)
from cachalot.model.shared_expert_metal import shared_expert_forward  # noqa: E402

HIDDEN = 5120
INTERMEDIATE = 2304
BLOCK = 32
LAYERS = 40
REPEATS = 9


def fp8_weight(out_features: int, in_features: int):
    """Raw E4M3 bytes and their E8M0 block scales, of the checkpoint's shape.

    The kernel reads bytes; it does not care what they decode to. The scale
    bytes are kept in a narrow band around 127 (2^0) so the products stay
    finite and the arithmetic the GPU does is the arithmetic it does in the
    runtime.
    """
    w = mx.random.randint(0, 255, shape=(out_features, in_features)).astype(mx.uint8)
    scales = mx.random.randint(
        120, 131, shape=((out_features + BLOCK - 1) // BLOCK, in_features // BLOCK)
    ).astype(mx.uint8)
    mx.eval(w, scales)
    return w, scales


def nbytes(*arrays):
    return sum(a.nbytes for a in arrays)


def chained(fn, n, repeats=REPEATS):
    mx.eval(fn())
    mx.synchronize()
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        times.append((perf_counter() - t0) / n * 1e3)
    times.sort()
    return times[0], statistics.median(times)


def main() -> None:
    x = mx.random.normal(shape=(HIDDEN,)).astype(mx.bfloat16)

    layers = []
    for _ in range(LAYERS):
        w1, s1 = fp8_weight(INTERMEDIATE, HIDDEN)
        w3, s3 = fp8_weight(INTERMEDIATE, HIDDEN)
        w2, s2 = fp8_weight(HIDDEN, INTERMEDIATE)
        layers.append((w1, s1, w3, s3, w2, s2))

    layer_bytes = nbytes(*layers[0])
    print(
        f"one layer {layer_bytes / 2**20:.2f} MiB of E4M3 weights and scales, "
        f"{LAYERS} distinct layers {LAYERS * layer_bytes / 2**30:.2f} GiB, "
        f"hidden {HIDDEN}, intermediate {INTERMEDIATE}\n"
    )

    # ---- arm 1: the shipped path -------------------------------------------
    def arm_shipped():
        out = None
        for w1, s1, w3, s3, w2, s2 in layers:
            y = shared_expert_forward(
                x, w1=w1, w1_scales=s1, w2=w2, w2_scales=s2, w3=w3, w3_scales=s3
            )
            out = y if out is None else out + y
        return out

    # ---- arm 2: w1 and w3 as one GEMV --------------------------------------
    # Both consume the same quantized activation and neither depends on the
    # other, so the two [I, H] weights concatenate into one [2I, H] and the
    # block scales concatenate with them (I/BLOCK is exact at 2304/32 = 72).
    fused = []
    for w1, s1, w3, s3, w2, s2 in layers:
        w13 = mx.concatenate([w1, w3], axis=0)
        s13 = mx.concatenate([s1, s3], axis=0)
        mx.eval(w13, s13)
        fused.append((w13, s13, w2, s2))

    import mlx.nn as nn

    def arm_fused():
        out = None
        for w13, s13, w2, s2 in fused:
            qx = quantize_fp8_activation(x)
            gu = fp8_linear_quantized(qx, w13, s13).astype(mx.float32)
            gate = mx.minimum(gu[:INTERMEDIATE], 10.0)
            up = mx.clip(gu[INTERMEDIATE:], -10.0, 10.0)
            hidden = (nn.silu(gate) * up).astype(x.dtype)
            y = fp8_linear_quantized(quantize_fp8_activation(hidden), w2, s2)
            out = y if out is None else out + y
        return out

    # ---- arm 3: what the machine gives for these buffers --------------------
    def arm_sum():
        out = None
        for w1, s1, w3, s3, w2, s2 in layers:
            t = w1.sum() + w3.sum() + w2.sum()
            out = t if out is None else out + t
        return out

    # ---- arm 4: the two activation quantizations alone ----------------------
    xi = mx.random.normal(shape=(INTERMEDIATE,)).astype(mx.bfloat16)

    def arm_quantize():
        outs = []
        for _ in range(LAYERS):
            a = quantize_fp8_activation(x)
            b = quantize_fp8_activation(xi)
            outs.append(a.values.sum() + b.values.sum())
        return outs

    weight_bytes = float(layer_bytes)
    act_bytes = float((HIDDEN + INTERMEDIATE) * 2)

    rows = [
        ("shipped, 3 GEMVs", arm_shipped, weight_bytes, 3),
        ("fused w1w3, 2 GEMVs", arm_fused, weight_bytes, 2),
        ("sum over the same bytes", arm_sum, weight_bytes, 3),
        ("quantize activations only", arm_quantize, act_bytes, 2),
    ]

    print(f"{'arm':<28}{'min/layer':>11}{'median':>10}{'GB/s':>9}{'x40':>9}{'launches':>10}")
    for name, fn, per_layer_bytes, launches in rows:
        lo, med = chained(fn, LAYERS)
        gbs = per_layer_bytes / (lo * 1e-3) / 1e9
        print(
            f"{name:<28}{lo:>9.3f} ms{med:>8.3f} ms{gbs:>9.0f}"
            f"{lo * LAYERS:>8.1f} ms{launches * LAYERS:>10d}"
        )


if __name__ == "__main__":
    main()
