"""
Screen: how far is the routed-expert GEMV from the memory wall?

`profile_decode_gpu.py` prices the six routed experts of one decode layer at
0.494 ms, which is 19.8 ms of a token and the largest single piece of its GPU
side. Those six experts are 6 x 9.49 MiB = 56.9 MiB of weights, and a
batch-1 quantized matmul is a pure streaming read: every weight is touched
once and almost nothing is reused. 56.9 MiB in 0.494 ms is **115 GB/s** on a
machine whose unified memory does several hundred.

That ratio is either the kernel or the machine, and this says which. Four arms
on the real shapes, all chained inside one eval so the per-launch eval floor is
excluded:

  quantized_matmul   what the runtime issues: three calls per expert, six
                     experts, exactly as expert_affine builds them
  gather_qmm         the same arithmetic over pre-stacked weights, which
                     section 11 screened at 20-25 % better and which bounds
                     what any fusion of the shipped path can return
  sum                mx.sum over the same packed buffers: no arithmetic, no
                     dequantization, just a full read. This is what the
                     machine will give for these buffers
  bf16 matvec        a dense matvec of the same logical shape, four times the
                     bytes, as a second reference point for a streaming kernel

Reported as GB/s over the bytes each arm must read, so the arms are comparable
although they move different amounts.

    PYTHONPATH=src python benchmarks/micro_expert_roofline.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

HIDDEN = 5120
INTERMEDIATE = 2304
EXPERTS = 6
LAYERS = 40
BITS = 2
GROUP = 128
REPEATS = 9

# A token routes 240 distinct experts, 2.3 GiB of weights, and touches each one
# once. An arm that replays the same six experts on all forty layers reads
# 57 MiB forty times and can serve most of it from the system-level cache, which
# flatters it against the runtime. --distinct builds one set of six per layer.


def quantized(shape):
    w = mx.random.normal(shape=shape).astype(mx.bfloat16)
    q, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
    mx.eval(q, scales, biases)
    return q, scales, biases


def packed(shape):
    """Packed weights of the right size and dtype, without the bf16 original.

    The kernel reads bytes; it does not care what they decode to, and building
    240 experts through mx.quantize would need 9 GiB of bf16 first.
    """
    rows, cols = shape
    q = mx.random.randint(0, 2**31 - 1, shape=(rows, cols * BITS // 32)).astype(mx.uint32)
    scales = mx.random.normal(shape=(rows, cols // GROUP)).astype(mx.bfloat16)
    biases = mx.random.normal(shape=(rows, cols // GROUP)).astype(mx.bfloat16)
    mx.eval(q, scales, biases)
    return q, scales, biases


def nbytes(*arrays):
    return sum(a.nbytes for a in arrays)


def chained(fn, n, repeats=REPEATS):
    mx.eval(fn())
    mx.synchronize()
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        outs = [fn() for _ in range(n)]
        mx.eval(outs)
        mx.synchronize()
        times.append((perf_counter() - t0) / n * 1e3)
    times.sort()
    return times[0], statistics.median(times)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--distinct", action="store_true",
                    help="one distinct set of six experts per layer (2.3 GiB), "
                         "so no launch reads what a previous launch left in cache")
    args = ap.parse_args()

    x = mx.random.normal(shape=(HIDDEN,)).astype(mx.bfloat16)
    xi = mx.random.normal(shape=(INTERMEDIATE,)).astype(mx.bfloat16)

    # One expert: w1, w3 are (intermediate, hidden); w2 is (hidden, intermediate).
    experts = []
    for _ in range(EXPERTS):
        w1 = quantized((INTERMEDIATE, HIDDEN))
        w3 = quantized((INTERMEDIATE, HIDDEN))
        w2 = quantized((HIDDEN, INTERMEDIATE))
        experts.append((w1, w3, w2))

    expert_bytes = sum(nbytes(*m) for m in experts[0])
    six_bytes = expert_bytes * EXPERTS
    print(f"one expert {expert_bytes / 2**20:.2f} MiB, six {six_bytes / 2**20:.2f} MiB, "
          f"{BITS}-bit g{GROUP}, hidden {HIDDEN}, intermediate {INTERMEDIATE}\n")

    def arm_qmm():
        out = None
        for w1, w3, w2 in experts:
            a = mx.quantized_matmul(x, w1[0], w1[1], w1[2], transpose=True,
                                    group_size=GROUP, bits=BITS)
            b = mx.quantized_matmul(x, w3[0], w3[1], w3[2], transpose=True,
                                    group_size=GROUP, bits=BITS)
            h = (a * mx.sigmoid(a)) * b
            y = mx.quantized_matmul(h.astype(mx.bfloat16), w2[0], w2[1], w2[2],
                                    transpose=True, group_size=GROUP, bits=BITS)
            out = y if out is None else out + y
        return out

    # Stacked weights: the upper bound a fused path could reach, ignoring what
    # it would cost to make six LRU slots contiguous.
    s1 = (mx.stack([m[0][0] for m in experts]), mx.stack([m[0][1] for m in experts]),
          mx.stack([m[0][2] for m in experts]))
    s3 = (mx.stack([m[1][0] for m in experts]), mx.stack([m[1][1] for m in experts]),
          mx.stack([m[1][2] for m in experts]))
    s2 = (mx.stack([m[2][0] for m in experts]), mx.stack([m[2][1] for m in experts]),
          mx.stack([m[2][2] for m in experts]))
    idx = mx.arange(EXPERTS, dtype=mx.uint32).reshape(1, EXPERTS)
    xb = x.reshape(1, 1, HIDDEN)
    mx.eval(s1, s3, s2, idx, xb)

    def arm_gather():
        a = mx.gather_qmm(xb, s1[0], s1[1], s1[2], rhs_indices=idx, transpose=True,
                          group_size=GROUP, bits=BITS)
        b = mx.gather_qmm(xb, s3[0], s3[1], s3[2], rhs_indices=idx, transpose=True,
                          group_size=GROUP, bits=BITS)
        h = ((a * mx.sigmoid(a)) * b).astype(mx.bfloat16)
        return mx.gather_qmm(h, s2[0], s2[1], s2[2], rhs_indices=idx, transpose=True,
                             group_size=GROUP, bits=BITS)

    def arm_sum():
        total = None
        for w1, w3, w2 in experts:
            for q, scales, biases in (w1, w3, w2):
                s = mx.sum(q.view(mx.uint32).astype(mx.uint32) & 0xFFFF)
                total = s if total is None else total + s
        return total

    dw1 = [mx.random.normal(shape=(INTERMEDIATE, HIDDEN)).astype(mx.bfloat16) for _ in range(EXPERTS)]
    mx.eval(dw1)
    dense_bytes = sum(w.nbytes for w in dw1)

    def arm_dense():
        out = None
        for w in dw1:
            y = w @ x
            out = y if out is None else out + y
        return out

    if args.distinct:
        layers = []
        for _ in range(LAYERS):
            layers.append([(packed((INTERMEDIATE, HIDDEN)),
                            packed((INTERMEDIATE, HIDDEN)),
                            packed((HIDDEN, INTERMEDIATE))) for _ in range(EXPERTS)])
        counter = {"i": 0}

        def arm_qmm_distinct():
            group = layers[counter["i"] % LAYERS]
            counter["i"] += 1
            out = None
            for w1, w3, w2 in group:
                a = mx.quantized_matmul(x, w1[0], w1[1], w1[2], transpose=True,
                                        group_size=GROUP, bits=BITS)
                b = mx.quantized_matmul(x, w3[0], w3[1], w3[2], transpose=True,
                                        group_size=GROUP, bits=BITS)
                h = ((a * mx.sigmoid(a)) * b).astype(mx.bfloat16)
                y = mx.quantized_matmul(h, w2[0], w2[1], w2[2], transpose=True,
                                        group_size=GROUP, bits=BITS)
                out = y if out is None else out + y
            return out

        lo, med = chained(arm_qmm_distinct, LAYERS)
        print(f"{'qmm, 240 distinct':20s} {lo:8.3f} {med:8.3f} "
              f"{six_bytes / (lo / 1e3) / 1e9:8.0f} {lo * LAYERS:9.1f}"
              if False else "")
        distinct_row = ("qmm, 240 distinct experts", arm_qmm_distinct, six_bytes)
    else:
        distinct_row = None

    rows = [
        ("quantized_matmul", arm_qmm, six_bytes),
        ("gather_qmm", arm_gather, six_bytes),
        ("sum (read only)", arm_sum, six_bytes),
        ("bf16 matvec x6", arm_dense, dense_bytes),
    ]
    if distinct_row is not None:
        rows.insert(1, distinct_row)
    print(f"{'arm':26s} {'min ms':>8s} {'median':>8s} {'GB/s':>8s} {'ms/token':>9s}")
    for name, fn, byts in rows:
        try:
            lo, med = chained(fn, LAYERS)
        except Exception as exc:  # a build without gather_qmm, say
            print(f"{name:26s}   unavailable: {type(exc).__name__}: {exc}")
            continue
        print(f"{name:26s} {lo:8.3f} {med:8.3f} {byts / (lo / 1e3) / 1e9:8.0f} {lo * LAYERS:9.1f}")

    print("\nms/token is this arm repeated on all 40 layers. The runtime pays 0.494 ms per layer,")
    print("19.8 ms per token, for the quantized_matmul row (HANDOFF section 7.1.4).")


if __name__ == "__main__":
    main()
