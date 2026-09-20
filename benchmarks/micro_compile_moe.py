"""
Screen: does mx.compile take the CPU cost out of a decode layer's expert math?

Section 9.15 measured 35 % of an all-resident token being spent on the CPU
building the next layer's operations while the GPU has nothing queued, and the
one piece taken so far (caching the affine slot views) was worth 2.5 ms by
removing op constructions rather than GPU work. mx.compile is the obvious next
instrument: it traces a function once and replays the traced graph, so the
Python and nanobind construction cost of everything inside it is paid once
instead of forty times per token.

This times the post-routing half of a decode layer -- six routed 2-bit affine
experts summed, the shape the runtime actually issues -- three ways:

  shipped        the loop as moe_layer_forward runs it, views already cached
  compiled       the same function under mx.compile, router weights passed as
                 an array so a new token does not retrace
  compiled+cap   the same, with mx.compile's shapeless tracing disabled

Construction time is measured with no eval inside the loop (the CPU side), and
chained time with one eval over many launches (CPU + GPU the way a token pays).

It is a screen: it decides whether to wire compile into the runtime, not
whether anything ships.

    PYTHONPATH=src python benchmarks/micro_compile_moe.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cachalot.model.expert_affine import affine_expert_forward, affine_views  # noqa: E402
from cachalot.storage.index import detect_expert_bank  # noqa: E402

BANK = "/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128"
TOPK = 6
LAYERS = 40
HIDDEN = 5120


def build_slot(fmt) -> dict[str, mx.array]:
    arrays = {}
    item = {"uint32": 4, "float16": 2, "bfloat16": 2, "float32": 4, "uint8": 1}
    for short in fmt.tensor_names:
        rows, cols = fmt.shapes[short]
        arrays[short] = mx.zeros((rows * cols * item[fmt.dtypes[short]],), dtype=mx.uint8)
    mx.eval(*arrays.values())
    return arrays


def build_time(fn, n=200):
    out = fn()
    mx.eval(out)
    mx.synchronize()
    t0 = perf_counter()
    outs = [fn() for _ in range(n)]
    build = (perf_counter() - t0) / n * 1e3
    mx.eval(*outs)
    mx.synchronize()
    return build


def chained(fn, n=40, repeats=7):
    mx.eval(fn())
    mx.synchronize()
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        outs = [fn() for _ in range(n)]
        mx.eval(*outs)
        mx.synchronize()
        times.append((perf_counter() - t0) / n * 1e3)
    return min(times), statistics.median(times)


def main():
    fmt, _ = detect_expert_bank(BANK)
    print(f"bank {fmt.bits}-bit g{fmt.group_size}, {TOPK} experts per layer, {LAYERS} layers\n")
    slots = [build_slot(fmt) for _ in range(TOPK)]
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    weights = mx.array([0.25] * TOPK, dtype=mx.float32)
    mx.eval(x, weights)

    caches = [{} for _ in slots]
    for slot, cache in zip(slots, caches, strict=True):
        for proj in ("w1", "w2", "w3"):
            affine_views(slot, fmt, proj, cache)
        mx.eval(*[a for proj in ("w1", "w2", "w3") for a in cache[proj]])

    def shipped():
        routed = mx.zeros(x.shape, dtype=mx.float32)
        for slot, cache in zip(slots, caches, strict=True):
            routed = routed + affine_expert_forward(x, slot, fmt, 0.25, 10.0, cache)
        return routed

    flat = [a for cache in caches for proj in ("w1", "w2", "w3") for a in cache[proj]]

    def core(xin, w_vec, views):
        """Same math as affine_expert_forward, six experts, router weights as an array."""
        routed = mx.zeros((HIDDEN,), dtype=mx.float32)
        xb = xin.astype(mx.bfloat16)
        for i in range(TOPK):
            w1, s1, b1 = views[9 * i + 0], views[9 * i + 1], views[9 * i + 2]
            w2, s2, b2 = views[9 * i + 3], views[9 * i + 4], views[9 * i + 5]
            w3, s3, b3 = views[9 * i + 6], views[9 * i + 7], views[9 * i + 8]
            kw = {"transpose": True, "group_size": fmt.group_size, "bits": fmt.bits}
            gate = mx.quantized_matmul(xb, w1, s1, b1, **kw).astype(mx.bfloat16).astype(mx.float32)
            up = mx.quantized_matmul(xb, w3, s3, b3, **kw).astype(mx.bfloat16).astype(mx.float32)
            up = mx.clip(up, -10.0, 10.0)
            gate = mx.minimum(gate, 10.0)
            hidden = (nn.silu(gate) * up * w_vec[i]).astype(mx.bfloat16)
            routed = routed + mx.quantized_matmul(
                hidden, w2, s2, b2, **kw).astype(mx.bfloat16).astype(mx.float32)
        return routed

    compiled = mx.compile(core)
    compiled_shapeless = mx.compile(core, shapeless=True)

    def run_uncompiled():
        return core(x, weights, flat)

    def run_compiled():
        return compiled(x, weights, flat)

    def run_shapeless():
        return compiled_shapeless(x, weights, flat)

    # warm the traces
    for fn in (run_uncompiled, run_compiled, run_shapeless):
        mx.eval(fn())
    mx.synchronize()

    print("graph construction only, one layer of six experts (CPU side):")
    rows = []
    for name, fn in (("shipped loop (cached views)", shipped),
                     ("same math, weights as array", run_uncompiled),
                     ("mx.compile", run_compiled),
                     ("mx.compile shapeless", run_shapeless)):
        b = build_time(fn)
        rows.append((name, b))
        print(f"  {name:32s} {b:7.4f} ms -> {b * LAYERS:6.2f} ms/token")

    print("\nchained, 40 launches in one graph (CPU + GPU, the way a token pays):")
    for name, fn in (("shipped loop (cached views)", shipped),
                     ("mx.compile", run_compiled)):
        lo, med = chained(fn)
        print(f"  {name:32s} min {lo:7.4f} ms  median {med:7.4f} ms -> {lo * LAYERS:6.2f} ms/token")

    a, b = shipped(), run_compiled()
    mx.eval(a, b)
    same = bool(mx.array_equal(a, b))
    diff = float(mx.max(mx.abs(a - b)).item())
    print(f"\nidentical output: {same}  (max abs difference {diff:.3e})")

    saved = rows[0][1] - rows[2][1]
    print(f"CPU construction saved by compiling: {saved:.4f} ms per layer -> "
          f"{saved * LAYERS:.2f} ms per token")


if __name__ == "__main__":
    main()
