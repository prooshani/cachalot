"""
Screen: can attention's wq_b and the indexer's wq_b fuse into one FP8 GEMV,
the same way wq_a/wkv did (HANDOFF section 9.34) and the shared expert's
w1/w3 did before that (section 9.27)?

attention_compressed.py itself flags the shared activation, in a comment
above `qr = fused_qr_kv_linear(...)`:

    # qr is shared conceptually between:
    #   attention wq_b
    #   indexer wq_b

Both read `qr` -- the post-rms_norm, 1280-dim low-rank query -- and neither
depends on the other's output:

    q       = fp8_linear(qr, wq_b, wq_b_scales)              [32768, 1280]
    index_q = fp8_linear(qr, indexer_wq_b_weight, ...)        [4096, 1280]

(indexer_mlx.py: INDEX_N_HEADS=32 * INDEX_HEAD_DIM=128 = 4096 rows, in all
three of indexer_decode_base/candidate_source/candidate_consumer.)

Unlike wq_a/wkv, attention's wq_b is NOT occupancy-bound -- HANDOFF 7.1.10
puts it at 456 GB/s, 8 lanes/row, already near the 483 GB/s mx.sum ceiling
implied by wo_b's family. This screen is therefore about the other cost the
qr/wkv fusion also cut: one activation quantization and one kernel launch
instead of two, not occupancy. And unlike wq_a/wkv (all 40 layers), the
indexer only runs on the layers that compute topk_idxs -- text_decode_runtime.py's
SOURCE_LAYERS = {2, 8, 14, 20} (indexer_decode_base at ratio 2, or
indexer_decode_candidate_source at layer 20's ratio 1) and
INDEX_ONLY_SOURCE_LAYERS = {24, 28, 32, 36} (indexer_decode_candidate_consumer)
-- 8 of 40 layers, not 40.

Four arms, chained across 8 layers inside one mx.eval:

  shipped   two fp8_linear calls exactly as issued today
  fused     wq_b and indexer_wq_b concatenated into one [36864, 1280]
            weight, one activation quantization, one GEMV, split back
  sum       mx.sum over the same uint8 buffers -- the machine's ceiling
  quantize  the second (redundant) activation quantization alone

    PYTHONPATH=src python benchmarks/micro_qb_indexer_fusion_roofline.py
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

QR_DIM = 1280
WQ_B_ROWS = 32768
INDEXER_WQ_B_ROWS = 4096
BLOCK = 32
INDEXER_LAYERS = 8
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
    qr = mx.random.normal(shape=(QR_DIM,)).astype(mx.bfloat16)

    layers = []
    for _ in range(INDEXER_LAYERS):
        wq_b, sq_b = fp8_weight(WQ_B_ROWS, QR_DIM)
        wix, six = fp8_weight(INDEXER_WQ_B_ROWS, QR_DIM)
        layers.append((wq_b, sq_b, wix, six))

    layer_bytes = nbytes(*layers[0])
    print(
        f"one layer {layer_bytes / 2**20:.2f} MiB of E4M3 weights and scales, "
        f"{INDEXER_LAYERS} indexer-bearing layers of 40, "
        f"K={QR_DIM}, rows {WQ_B_ROWS}+{INDEXER_WQ_B_ROWS}={WQ_B_ROWS + INDEXER_WQ_B_ROWS}\n"
    )

    # ---- correctness: fused output must equal shipped, bit-identical ------
    wq_b, sq_b, wix, six = layers[0]
    ref_q = fp8_linear(qr, wq_b, sq_b)
    ref_ix = fp8_linear(qr, wix, six)
    mx.eval(ref_q, ref_ix)

    wfused = mx.concatenate([wq_b, wix], axis=0)
    sfused = mx.concatenate([sq_b, six], axis=0)
    mx.eval(wfused, sfused)
    qx = quantize_fp8_activation(qr)
    fused_out = fp8_linear_quantized(qx, wfused, sfused)
    got_q = fused_out[:WQ_B_ROWS]
    got_ix = fused_out[WQ_B_ROWS:]
    mx.eval(got_q, got_ix)
    exact = bool(mx.all(got_q == ref_q).item()) and bool(mx.all(got_ix == ref_ix).item())
    print(f"fused output bit-identical to shipped: {'yes' if exact else 'NO'}\n")

    # ---- arm 1: the shipped path -------------------------------------------
    def arm_shipped():
        out = None
        for wq_b, sq_b, wix, six in layers:
            q = fp8_linear(qr, wq_b, sq_b)
            index_q = fp8_linear(qr, wix, six)
            y = q.sum() + index_q.sum()
            out = y if out is None else out + y
        return out

    # ---- arm 2: wq_b and indexer wq_b as one GEMV ---------------------------
    fused = []
    for wq_b, sq_b, wix, six in layers:
        wfused = mx.concatenate([wq_b, wix], axis=0)
        sfused = mx.concatenate([sq_b, six], axis=0)
        mx.eval(wfused, sfused)
        fused.append((wfused, sfused))

    def arm_fused():
        out = None
        for wfused, sfused in fused:
            qx = quantize_fp8_activation(qr)
            gu = fp8_linear_quantized(qx, wfused, sfused)
            q = gu[:WQ_B_ROWS]
            index_q = gu[WQ_B_ROWS:]
            y = q.sum() + index_q.sum()
            out = y if out is None else out + y
        return out

    # ---- arm 3: what the machine gives for these buffers --------------------
    def arm_sum():
        out = None
        for wq_b, sq_b, wix, six in layers:
            t = wq_b.sum() + wix.sum()
            out = t if out is None else out + t
        return out

    # ---- arm 4: the redundant activation quantization alone -----------------
    def arm_quantize():
        outs = []
        for _ in range(INDEXER_LAYERS):
            outs.append(quantize_fp8_activation(qr).values.sum())
        return outs

    weight_bytes = float(layer_bytes)
    act_bytes = float(QR_DIM * 2)

    rows = [
        ("shipped, 2 GEMVs", arm_shipped, weight_bytes, 2),
        ("fused wq_b+indexer, 1 GEMV", arm_fused, weight_bytes, 1),
        ("sum over the same bytes", arm_sum, weight_bytes, 2),
        ("redundant quantize alone", arm_quantize, act_bytes, 1),
    ]

    print(f"{'arm':<30}{'min/layer':>11}{'median':>10}{'GB/s':>9}{'x8':>9}{'launches':>10}")
    for name, fn, per_layer_bytes, launches in rows:
        lo, med = chained(fn, INDEXER_LAYERS)
        gbs = per_layer_bytes / (lo * 1e-3) / 1e9
        print(
            f"{name:<30}{lo:>9.3f} ms{med:>8.3f} ms{gbs:>9.0f}"
            f"{lo * INDEXER_LAYERS:>8.3f} ms{launches * INDEXER_LAYERS:>10d}"
        )


if __name__ == "__main__":
    main()
