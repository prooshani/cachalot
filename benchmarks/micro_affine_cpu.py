"""
What does the affine routed-expert path cost the CPU, and how much of that is
re-deriving the slot views?

On the 2-bit bank moe_layer_forward evaluates the six routed experts in a
Python loop, and every call re-runs affine_views(), which builds a .view() and
a .reshape() for each of the nine slot tensors of each expert. That is 108 MLX
op constructions per layer before any arithmetic is queued, 4,320 per token.

This times graph construction alone (no eval, no GPU) against a variant that
reuses views built once, so the two numbers differ only in the work the CPU
does between dispatches.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cachalot.model.expert_affine import _qmm, affine_views  # noqa: E402
from cachalot.storage.index import detect_expert_bank  # noqa: E402

BANK = "/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128"
TOPK = 6
LAYERS = 40


def build_slot(fmt) -> dict[str, mx.array]:
    """A slot holding one expert's bytes, exactly as the pool serves them: raw uint8."""
    arrays = {}
    item = {"uint32": 4, "float16": 2, "bfloat16": 2, "float32": 4, "uint8": 1}
    for short in fmt.tensor_names:
        rows, cols = fmt.shapes[short]
        arrays[short] = mx.zeros((rows * cols * item[fmt.dtypes[short]],), dtype=mx.uint8)
    mx.eval(*arrays.values())
    return arrays


def forward_from_views(x, views, weight, swiglu_limit=10.0):
    """affine_expert_forward with the views already built."""
    import mlx.nn as nn

    xb = x.astype(mx.bfloat16)
    w, s, b = views["w1"]
    gate = mx.quantized_matmul(xb, w, s, b, transpose=True, group_size=views["g"], bits=views["b"]).astype(mx.bfloat16).astype(mx.float32)
    w, s, b = views["w3"]
    up = mx.quantized_matmul(xb, w, s, b, transpose=True, group_size=views["g"], bits=views["b"]).astype(mx.bfloat16).astype(mx.float32)
    up = mx.clip(up, -swiglu_limit, swiglu_limit)
    gate = mx.minimum(gate, mx.array(swiglu_limit, dtype=mx.float32))
    hidden = (nn.silu(gate) * up * weight).astype(mx.bfloat16)
    w, s, b = views["w2"]
    return mx.quantized_matmul(hidden, w, s, b, transpose=True, group_size=views["g"], bits=views["b"]).astype(mx.bfloat16).astype(mx.float32)


def build_time(fn, n=200):
    """Wall time to CONSTRUCT the graph, with a single eval outside the timed loop."""
    out = fn()
    mx.eval(out)
    mx.synchronize()
    t0 = perf_counter()
    outs = [fn() for _ in range(n)]
    build = (perf_counter() - t0) / n * 1e3
    mx.eval(*outs)
    mx.synchronize()
    return build


def main():
    fmt, _ = detect_expert_bank(BANK)
    print(f"bank {fmt.bits}-bit g{fmt.group_size}, tensors {fmt.tensor_names}")
    slots = [build_slot(fmt) for _ in range(TOPK)]
    x = mx.random.normal((5120,)).astype(mx.bfloat16)
    mx.eval(x)

    from cachalot.model.expert_affine import affine_expert_forward

    def shipped():
        routed = mx.zeros(x.shape, dtype=mx.float32)
        for slot in slots:
            routed = routed + affine_expert_forward(x, slot, fmt, 0.25, 10.0)
        return routed

    cached = []
    for slot in slots:
        v = {"g": fmt.group_size, "b": fmt.bits}
        for proj in ("w1", "w2", "w3"):
            v[proj] = affine_views(slot, fmt, proj)
        mx.eval(*[a for proj in ("w1", "w2", "w3") for a in v[proj]])
        cached.append(v)

    def with_cached_views():
        routed = mx.zeros(x.shape, dtype=mx.float32)
        for v in cached:
            routed = routed + forward_from_views(x, v, 0.25)
        return routed

    def views_only():
        out = []
        for slot in slots:
            for proj in ("w1", "w2", "w3"):
                out.extend(affine_views(slot, fmt, proj))
        return out[0]

    ship = build_time(shipped)
    cache = build_time(with_cached_views)
    views = build_time(views_only)
    print(f"graph construction, six experts, one layer:")
    print(f"  shipped (views rebuilt)   {ship:7.4f} ms  -> {ship * LAYERS:6.2f} ms/token of CPU")
    print(f"  views rebuilt alone       {views:7.4f} ms  -> {views * LAYERS:6.2f} ms/token")
    print(f"  cached views              {cache:7.4f} ms  -> {cache * LAYERS:6.2f} ms/token")
    print(f"  saved by caching views    {ship - cache:7.4f} ms  -> {(ship - cache) * LAYERS:6.2f} ms/token")

    # the numbers only mean something if both paths compute the same thing
    a, b = shipped(), with_cached_views()
    mx.eval(a, b)
    print(f"  identical output: {bool(mx.array_equal(a, b))}")


if __name__ == "__main__":
    main()
