"""
Screen: `fp8_gemv_decoded`'s lanes-per-row policy is chosen by divisibility,
not by speed.

This kernel is the largest GPU path in the runtime. It serves every attention
projection except wo_a -- wq_a, wq_b, wkv, wo_b, 93 MB per layer -- and all
three of the shared expert's GEMVs, 33.8 MB per layer: about 5.1 GB of weights
read per decoded token, against 2.3 GB for the routed experts.

`_lanes_per_row(n_blocks)` returns the first of 32, 16, 8 that divides the
block count, so the split is picked for load balance alone:

    K = 5120 -> 160 blocks -> 32 lanes per row, 1 row per simdgroup
    K = 8192 -> 256 blocks -> 32 lanes per row, 1 row per simdgroup
    K = 1280 ->  40 blocks ->  8 lanes per row, 4 rows per simdgroup
    K = 2304 ->  72 blocks ->  8 lanes per row, 4 rows per simdgroup

Fewer lanes per row means more output rows in flight per simdgroup and more
blocks per lane; more lanes means the opposite. Which side wins depends on N
as well as K, and N is not an input to the policy at all. A 512-row projection
at 32 lanes per row launches 512 simdgroups -- 64 threadgroups of 256 threads
on a machine with eighty cores.

Every legal split is screened on every shape the runtime issues, against
`mx.sum` over the same bytes as the machine's ceiling. The split changes the
order in which a row's blocks are summed, so each arm is also checked against
the shipped split for bit-identical output; a split that is faster and not
bit-identical is a quality question, not a free win.

    PYTHONPATH=src python benchmarks/micro_fp8_gemv_kernel.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cachalot.model.fp8_fused_metal import (  # noqa: E402
    BLOCK_SIZE,
    _gemv_v2_kernel,
    _lanes_per_row,
    quantize_activation_fp8_fused,
    u32,
)

REPEATS = 9
CHAIN = 20
LANE_ARMS = (8, 16, 32)

# The shapes the runtime issues and how many launches a token makes of each:
# forty layers of attention, forty of shared expert (w1 and w3 are the same
# shape, hence eighty).
SHAPES = [
    ("wq_a  [1280, 5120]", 1280, 5120, 40),
    ("wq_b  [32768, 1280]", 32768, 1280, 40),
    ("wkv   [512, 5120]", 512, 5120, 40),
    ("wo_b  [5120, 8192]", 5120, 8192, 40),
    ("shared w1/w3 [2304, 5120]", 2304, 5120, 80),
    ("shared w2 [5120, 2304]", 5120, 2304, 40),
]


def launch(kernel, act, act_scales, weight, weight_scales, n, lanes):
    rows_per_sg = 32 // lanes
    n_sg = (n + rows_per_sg - 1) // rows_per_sg
    return kernel(
        inputs=[act, act_scales, weight, weight_scales, u32(n)],
        template=[],
        grid=(n_sg * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]


def timed(fn):
    mx.eval(fn())
    mx.synchronize()
    times = []
    for _ in range(REPEATS):
        t0 = perf_counter()
        mx.eval(fn())
        mx.synchronize()
        times.append((perf_counter() - t0) / CHAIN * 1e3)
    return min(times), statistics.median(times)


def main() -> None:
    totals = {lanes: 0.0 for lanes in LANE_ARMS}
    totals["shipped"] = 0.0
    totals["best"] = 0.0
    totals["ceiling"] = 0.0
    best_choice = {}

    print(f"{'shape':<27}{'lanes':>6}{'rows/sg':>8}{'min':>10}{'GB/s':>8}"
          f"{'x/token':>10}{'exact':>7}")
    for label, n, k, launches in SHAPES:
        w = mx.random.randint(0, 255, shape=(n, k)).astype(mx.uint8)
        s = mx.random.randint(120, 131, shape=((n + 31) // 32, k // 32)).astype(mx.uint8)
        x = mx.random.normal(shape=(k,)).astype(mx.bfloat16)
        _, act_scales, act = quantize_activation_fp8_fused(x)
        mx.eval(w, s, act, act_scales)
        nbytes = w.nbytes + s.nbytes
        shipped_lanes = _lanes_per_row(k // BLOCK_SIZE)

        ref = launch(_gemv_v2_kernel(k, shipped_lanes), act, act_scales, w, s, n,
                     shipped_lanes)
        mx.eval(ref)

        best = None
        for lanes in LANE_ARMS:
            kern = _gemv_v2_kernel(k, lanes)
            got = launch(kern, act, act_scales, w, s, n, lanes)
            mx.eval(got)
            exact = bool(mx.all(got == ref).item())

            def once(kern=kern, lanes=lanes):
                acc = None
                for _ in range(CHAIN):
                    y = launch(kern, act, act_scales, w, s, n, lanes)
                    acc = y if acc is None else acc + y
                return acc

            lo, _med = timed(once)
            if lanes == shipped_lanes:
                totals["shipped"] += lo * launches
            for arm in LANE_ARMS:
                if arm == lanes:
                    totals[arm] += lo * launches
            if best is None or lo < best[1]:
                best = (lanes, lo, exact)
            mark = "  <- ships" if lanes == shipped_lanes else ""
            print(f"{label if lanes == LANE_ARMS[0] else '':<27}{lanes:>6}"
                  f"{32 // lanes:>8}{lo:>8.3f} ms"
                  f"{nbytes / (lo * 1e-3) / 1e9:>8.0f}{lo * launches:>8.2f} ms"
                  f"{('yes' if exact else 'NO'):>7}{mark}")

        def ceiling():
            acc = None
            for _ in range(CHAIN):
                t = w.sum()
                acc = t if acc is None else acc + t
            return acc

        clo, _ = timed(ceiling)
        totals["ceiling"] += clo * launches
        totals["best"] += best[1] * launches
        best_choice[label] = best
        print(f"{'':<27}{'sum':>6}{'':>8}{clo:>8.3f} ms"
              f"{nbytes / (clo * 1e-3) / 1e9:>8.0f}{clo * launches:>8.2f} ms{'':>7}")
        print()

    print("per token, every FP8 GEMV launch in the model:")
    print(f"  shipped policy        {totals['shipped']:>6.2f} ms")
    for lanes in LANE_ARMS:
        print(f"  {lanes:2d} lanes everywhere   {totals[lanes]:>6.2f} ms   "
              f"({totals[lanes] - totals['shipped']:+.2f})")
    print(f"  best split per shape  {totals['best']:>6.2f} ms   "
          f"({totals['best'] - totals['shipped']:+.2f})")
    print(f"  mx.sum over the bytes {totals['ceiling']:>6.2f} ms   "
          f"({totals['ceiling'] - totals['shipped']:+.2f})  <- the machine")
    print("\n  best split per shape, and whether it is bit-identical to what ships:")
    for label, (lanes, lo, exact) in best_choice.items():
        print(f"    {label:<28}{lanes:>3} lanes  {lo:.3f} ms  "
              f"{'bit-identical' if exact else 'DIFFERENT last bits'}")


if __name__ == "__main__":
    main()
