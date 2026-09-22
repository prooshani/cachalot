"""
Screen: can wq_a and wkv fuse into one FP8 GEMV the same way the shared
expert's w1/w3 did (HANDOFF section 9.27)?

HANDOFF section 7.1.10 found wq_a [1280, 5120] and wkv [512, 5120] the two
worst-throughput shapes in the FP8 GEMV family -- 171 and 87 GB/s against
wo_b's 383 and shared w1/w3's 243 -- and pinned it on occupancy: 512 and 1280
output rows launch too few simdgroups (64 and 160 threadgroups of 256 threads)
to fill eighty GPU cores. Together they cost 2.74 ms/token against a 2.31 ms
mx.sum ceiling on the same bytes -- 0.43 ms/token sitting in exactly the two
shapes the occupancy theory predicts, out of the family's 3.48 ms total gap.

Both are read straight off `x`, the layer's hidden state (attention_compressed.py:
`qr = fp8_linear(x, wq_a, wq_a_scales)` then `window_kv = fp8_linear(x, wkv,
wkv_scales)`), independently -- neither depends on the other's output, exactly
the shape shared_expert_metal.py's w1/w3 fusion exploited. The shipped path
also quantizes x to FP8 twice, once per call.

Four arms, chained across forty distinct layers inside one mx.eval:

  shipped   two fp8_linear calls exactly as attention_compressed.py issues them
            (two activation quantizations, two GEMV launches, 512+1280 rows)
  fused     wq_a and wkv concatenated into one [1792, 5120] weight, one
            activation quantization, one GEMV, split back into two outputs
  sum       mx.sum over the same uint8 buffers -- the machine's ceiling
  quantize  the second (redundant) activation quantization alone

    PYTHONPATH=src python benchmarks/micro_qkv_fusion_roofline.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cachalot.model.fp8_linear_metal import (  # noqa: E402
    fp8_linear,
    fp8_linear_quantized,
    quantize_fp8_activation,
)

HIDDEN = 5120
WQ_A_ROWS = 1280
WKV_ROWS = 512
BLOCK = 32
LAYERS = 40
REPEATS = 9


def fp8_weight(out_features: int, in_features: int):
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
        wq_a, sq_a = fp8_weight(WQ_A_ROWS, HIDDEN)
        wkv, skv = fp8_weight(WKV_ROWS, HIDDEN)
        layers.append((wq_a, sq_a, wkv, skv))

    layer_bytes = nbytes(*layers[0])
    print(
        f"one layer {layer_bytes / 2**20:.2f} MiB of E4M3 weights and scales, "
        f"{LAYERS} distinct layers {LAYERS * layer_bytes / 2**30:.2f} GiB, "
        f"K={HIDDEN}, rows {WQ_A_ROWS}+{WKV_ROWS}={WQ_A_ROWS + WKV_ROWS}\n"
    )

    # ---- correctness: fused output must equal shipped, bit-identical ------
    wq_a, sq_a, wkv, skv = layers[0]
    ref_q = fp8_linear(x, wq_a, sq_a)
    ref_kv = fp8_linear(x, wkv, skv)
    mx.eval(ref_q, ref_kv)

    wqkv = mx.concatenate([wq_a, wkv], axis=0)
    sqkv = mx.concatenate([sq_a, skv], axis=0)
    mx.eval(wqkv, sqkv)
    qx = quantize_fp8_activation(x)
    fused_out = fp8_linear_quantized(qx, wqkv, sqkv)
    got_q = fused_out[:WQ_A_ROWS]
    got_kv = fused_out[WQ_A_ROWS:]
    mx.eval(got_q, got_kv)
    exact = bool(mx.all(got_q == ref_q).item()) and bool(mx.all(got_kv == ref_kv).item())
    print(f"fused output bit-identical to shipped: {'yes' if exact else 'NO'}\n")

    # ---- arm 1: the shipped path -------------------------------------------
    def arm_shipped():
        out = None
        for wq_a, sq_a, wkv, skv in layers:
            qr = fp8_linear(x, wq_a, sq_a)
            window_kv = fp8_linear(x, wkv, skv)
            y = qr.sum() + window_kv.sum()
            out = y if out is None else out + y
        return out

    # ---- arm 2: wq_a and wkv as one GEMV -----------------------------------
    fused = []
    for wq_a, sq_a, wkv, skv in layers:
        wqkv = mx.concatenate([wq_a, wkv], axis=0)
        sqkv = mx.concatenate([sq_a, skv], axis=0)
        mx.eval(wqkv, sqkv)
        fused.append((wqkv, sqkv))

    def arm_fused():
        out = None
        for wqkv, sqkv in fused:
            qx = quantize_fp8_activation(x)
            gu = fp8_linear_quantized(qx, wqkv, sqkv)
            qr = gu[:WQ_A_ROWS]
            window_kv = gu[WQ_A_ROWS:]
            y = qr.sum() + window_kv.sum()
            out = y if out is None else out + y
        return out

    # ---- arm 3: what the machine gives for these buffers --------------------
    def arm_sum():
        out = None
        for wq_a, sq_a, wkv, skv in layers:
            t = wq_a.sum() + wkv.sum()
            out = t if out is None else out + t
        return out

    # ---- arm 4: the redundant activation quantization alone -----------------
    def arm_quantize():
        outs = []
        for _ in range(LAYERS):
            outs.append(quantize_fp8_activation(x).values.sum())
        return outs

    weight_bytes = float(layer_bytes)
    act_bytes = float(HIDDEN * 2)

    rows = [
        ("shipped, 2 GEMVs", arm_shipped, weight_bytes, 2),
        ("fused wq_a+wkv, 1 GEMV", arm_fused, weight_bytes, 1),
        ("sum over the same bytes", arm_sum, weight_bytes, 2),
        ("redundant quantize alone", arm_quantize, act_bytes, 1),
    ]

    print(f"{'arm':<28}{'min/layer':>11}{'median':>10}{'GB/s':>9}{'x40':>9}{'launches':>10}")
    for name, fn, per_layer_bytes, launches in rows:
        lo, med = chained(fn, LAYERS)
        gbs = per_layer_bytes / (lo * 1e-3) / 1e9
        print(
            f"{name:<28}{lo:>9.3f} ms{med:>8.3f} ms{gbs:>9.0f}"
            f"{lo * LAYERS:>8.2f} ms{launches * LAYERS:>10d}"
        )


if __name__ == "__main__":
    main()
