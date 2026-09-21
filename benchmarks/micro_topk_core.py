"""
Screen: where do 11 ms of the routed-expert term go?

`profile_decode_gpu.py` prices the shipped six-expert block at 0.494 ms per
layer, 19.8 ms per token. `micro_expert_roofline.py` runs the same three
quantized matmuls per expert, on the same shapes, over 240 distinct experts so
nothing is served from cache, in **0.217 ms** -- 8.7 ms per token. The
arithmetic is the same; the difference is everything `topk_core` does around
it.

This adds that back one layer at a time, so the gap lands on a line:

  qmm only          three quantized_matmul per expert, bf16 throughout
  + fp32 casts      each matmul result .astype(bf16).astype(fp32), as shipped
  + swiglu clamp    mx.clip on up, mx.minimum on gate
  + weighted sum    silu * up * weights[i], fp32 accumulate: the shipped body
  topk_core         the shipped function itself, uncompiled
  topk_core compiled  what the runtime calls, through mx.compile

All arms take the same view list, built to the slot layout the runtime uses:
uint8 buffers viewed as packed weights and bf16 scales, one allocation per
tensor, exactly as cache/slots.py allocates them.

    PYTHONPATH=src python benchmarks/micro_topk_core.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cachalot.model.expert_affine import _compiled_topk, topk_core  # noqa: E402

HIDDEN = 5120
INTERMEDIATE = 2304
EXPERTS = 6
LAYERS = 40
BITS = 2
GROUP = 128
LIMIT = 10.0
REPEATS = 9


def slot_tensor(nbytes: int) -> mx.array:
    """A slot's uint8 buffer, as ExpertSlotPool allocates it."""
    a = mx.random.randint(0, 255, shape=(nbytes,)).astype(mx.uint8)
    mx.eval(a)
    return a


def views_for(out_features: int, in_features: int) -> tuple[mx.array, mx.array, mx.array]:
    """(weight, scales, biases) viewed out of slot bytes, as expert_affine does."""
    words = out_features * in_features * BITS // 32
    groups = out_features * (in_features // GROUP)
    w = slot_tensor(words * 4).view(mx.uint32).reshape(out_features, in_features * BITS // 32)
    s = slot_tensor(groups * 2).view(mx.bfloat16).reshape(out_features, in_features // GROUP)
    b = slot_tensor(groups * 2).view(mx.bfloat16).reshape(out_features, in_features // GROUP)
    mx.eval(w, s, b)
    return w, s, b


def build_views() -> list[mx.array]:
    """The flat (w1, w2, w3) x (weight, scales, biases) list, expert order."""
    flat: list[mx.array] = []
    for _ in range(EXPERTS):
        for out_f, in_f in ((INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE), (INTERMEDIATE, HIDDEN)):
            flat.extend(views_for(out_f, in_f))
    return flat


def chained(fn, n=LAYERS, repeats=REPEATS):
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
    views = build_views()
    x = mx.random.normal(shape=(HIDDEN,)).astype(mx.float32)
    weights = mx.random.uniform(shape=(EXPERTS,)).astype(mx.float32)
    mx.eval(x, weights)
    kw = {"transpose": True, "group_size": GROUP, "bits": BITS}
    xb = x.astype(mx.bfloat16)
    mx.eval(xb)

    def arm_qmm():
        out = None
        for i in range(EXPERTS):
            base = 9 * i
            gate = mx.quantized_matmul(xb, views[base], views[base + 1], views[base + 2], **kw)
            up = mx.quantized_matmul(xb, views[base + 6], views[base + 7], views[base + 8], **kw)
            h = (nn.silu(gate) * up).astype(mx.bfloat16)
            y = mx.quantized_matmul(h, views[base + 3], views[base + 4], views[base + 5], **kw)
            out = y if out is None else out + y
        return out

    def arm_casts():
        out = None
        for i in range(EXPERTS):
            base = 9 * i
            gate = mx.quantized_matmul(xb, views[base], views[base + 1], views[base + 2],
                                       **kw).astype(mx.bfloat16).astype(mx.float32)
            up = mx.quantized_matmul(xb, views[base + 6], views[base + 7], views[base + 8],
                                     **kw).astype(mx.bfloat16).astype(mx.float32)
            h = (nn.silu(gate) * up).astype(mx.bfloat16)
            y = mx.quantized_matmul(h, views[base + 3], views[base + 4], views[base + 5],
                                    **kw).astype(mx.bfloat16).astype(mx.float32)
            out = y if out is None else out + y
        return out

    def arm_clamp():
        out = None
        for i in range(EXPERTS):
            base = 9 * i
            gate = mx.quantized_matmul(xb, views[base], views[base + 1], views[base + 2],
                                       **kw).astype(mx.bfloat16).astype(mx.float32)
            up = mx.quantized_matmul(xb, views[base + 6], views[base + 7], views[base + 8],
                                     **kw).astype(mx.bfloat16).astype(mx.float32)
            up = mx.clip(up, -LIMIT, LIMIT)
            gate = mx.minimum(gate, LIMIT)
            h = (nn.silu(gate) * up).astype(mx.bfloat16)
            y = mx.quantized_matmul(h, views[base + 3], views[base + 4], views[base + 5],
                                    **kw).astype(mx.bfloat16).astype(mx.float32)
            out = y if out is None else out + y
        return out

    def arm_body():
        return topk_core(x, weights, views, n_experts=EXPERTS, bits=BITS,
                         group_size=GROUP, hidden=HIDDEN, swiglu_limit=LIMIT)

    compiled = _compiled_topk(EXPERTS, BITS, GROUP, HIDDEN, LIMIT)

    def arm_compiled():
        return compiled(x, weights, views)

    print(f"six experts, {BITS}-bit g{GROUP}, hidden {HIDDEN}, intermediate {INTERMEDIATE}, "
          f"chained over {LAYERS} launches\n")
    print(f"{'arm':22s} {'min ms':>8s} {'median':>8s} {'ms/token':>9s}")
    base_ms = None
    for name, fn in (
        ("qmm only", arm_qmm),
        ("+ fp32 casts", arm_casts),
        ("+ swiglu clamp", arm_clamp),
        ("topk_core", arm_body),
        ("topk_core compiled", arm_compiled),
    ):
        lo, med = chained(fn)
        if base_ms is None:
            base_ms = lo
        print(f"{name:22s} {lo:8.3f} {med:8.3f} {lo * LAYERS:9.1f}   {(lo - base_ms) * LAYERS:+.1f} ms/token")
    print("\nThe runtime pays 0.494 ms per layer, 19.8 ms per token (profile_decode_gpu.py).")


if __name__ == "__main__":
    main()
