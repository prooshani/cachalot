"""
Screen: can the six routed experts of one 2-bit layer be read faster than the
shipped per-expert loop reads them?

moe_layer_forward calls affine_expert_forward once per routed expert, so one
layer is 18 mx.quantized_matmul dispatches. The chained GPU profile prices that
loop at 0.35 ms per layer, 13.9 ms per token, while moving 57 MiB -- about
164 GB/s on a machine whose unified memory does far more. This compares it
against two upper bounds that read exactly the same bytes in fewer dispatches:

  concat   one quantized_matmul per projection over the six experts' rows
           concatenated (the fusion a custom kernel would aim at)
  gather   mx.gather_qmm over a stacked [6, rows, cols] tensor

Both bounds ignore what it would cost to get the slot bytes contiguous, which
is the reason this is a screen and not a gate: it decides whether a fused
affine expert kernel is worth a session, not whether one ships.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cachalot.storage.index import detect_expert_bank  # noqa: E402

BANK = "/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128"
TOPK = 6
LAYERS = 40


def chained(fn, n=60):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    outs = [fn() for _ in range(n)]
    mx.eval(*outs)
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    fmt, _ = detect_expert_bank(BANK)
    g, bits = fmt.group_size, fmt.bits
    print(f"bank {bits}-bit g{g}")

    def rand_q(rows, cols):
        w = mx.random.normal((rows, cols)).astype(mx.bfloat16)
        return mx.quantize(w, group_size=g, bits=bits)

    per = {proj: [rand_q(*fmt.shapes[f"{proj}.weight"][:1] + (5120,)) if False else None for _ in range(TOPK)]
           for proj in ("w1",)}
    del per

    # real projection shapes: w1/w3 are [2304, 5120], w2 is [5120, 2304]
    shapes = {"w1": (2304, 5120), "w3": (2304, 5120), "w2": (5120, 2304)}
    experts = []
    for _ in range(TOPK):
        experts.append({proj: rand_q(*shapes[proj]) for proj in shapes})
    x = mx.random.normal((5120,)).astype(mx.bfloat16)
    h = mx.random.normal((2304,)).astype(mx.bfloat16)
    mx.eval(x, h, *[a for e in experts for t in e.values() for a in t])

    def per_expert():
        out = None
        for e in experts:
            w, s, b = e["w1"]
            gate = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=g, bits=bits)
            w, s, b = e["w3"]
            up = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=g, bits=bits)
            hid = (gate * up).astype(mx.bfloat16)
            w, s, b = e["w2"]
            y = mx.quantized_matmul(hid, w, s, b, transpose=True, group_size=g, bits=bits)
            out = y if out is None else out + y
        return out

    cat = {}
    for proj in shapes:
        cat[proj] = tuple(mx.concatenate([e[proj][i] for e in experts], axis=0) for i in range(3))
    stack = {}
    for proj in shapes:
        stack[proj] = tuple(mx.stack([e[proj][i] for e in experts], axis=0) for i in range(3))
    mx.eval(*[a for t in cat.values() for a in t], *[a for t in stack.values() for a in t])

    def concat_fused():
        w, s, b = cat["w1"]
        gate = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=g, bits=bits)
        w, s, b = cat["w3"]
        up = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=g, bits=bits)
        hid = (gate * up).astype(mx.bfloat16).reshape(TOPK, 2304)
        w, s, b = cat["w2"]
        # w2 rows are the hidden dim, so the concat trick does not apply to it;
        # this runs the six down projections from the stacked tensor instead.
        ws, ss, bs = stack["w2"]
        y = mx.gather_qmm(hid[:, None, :], ws, ss, bs, transpose=True, group_size=g, bits=bits)
        return mx.sum(y.reshape(TOPK, 5120), axis=0)

    idx = mx.arange(TOPK, dtype=mx.uint32)
    xb = mx.broadcast_to(x, (TOPK, 5120)).reshape(TOPK, 1, 5120)
    mx.eval(idx, xb)

    def gather_fused():
        w, s, b = stack["w1"]
        gate = mx.gather_qmm(xb, w, s, b, transpose=True, group_size=g, bits=bits)
        w, s, b = stack["w3"]
        up = mx.gather_qmm(xb, w, s, b, transpose=True, group_size=g, bits=bits)
        hid = (gate * up).astype(mx.bfloat16)
        w, s, b = stack["w2"]
        y = mx.gather_qmm(hid, w, s, b, transpose=True, group_size=g, bits=bits)
        return mx.sum(y.reshape(TOPK, 5120), axis=0)

    def per_expert_interleaved():
        """Same 18 dispatches and the same accumulation order, issued projection
        by projection across the six experts instead of expert by expert."""
        gates, ups = [], []
        for e in experts:
            w, s, b = e["w1"]
            gates.append(mx.quantized_matmul(x, w, s, b, transpose=True, group_size=g, bits=bits))
        for e in experts:
            w, s, b = e["w3"]
            ups.append(mx.quantized_matmul(x, w, s, b, transpose=True, group_size=g, bits=bits))
        ys = []
        for e, gate, up in zip(experts, gates, ups, strict=True):
            hid = (gate * up).astype(mx.bfloat16)
            w, s, b = e["w2"]
            ys.append(mx.quantized_matmul(hid, w, s, b, transpose=True, group_size=g, bits=bits))
        out = ys[0]
        for y in ys[1:]:
            out = out + y
        return out

    a, b_ = per_expert(), per_expert_interleaved()
    mx.eval(a, b_)
    print(f"  interleaved output identical to shipped order: {bool(mx.array_equal(a, b_))}")

    mib = TOPK * sum(r * c for r, c in shapes.values()) * bits / 8 / 2**20
    for name, fn in (("per-expert loop (shipped)", per_expert),
                     ("per-expert, interleaved", per_expert_interleaved),
                     ("concat w1/w3 + gather w2", concat_fused),
                     ("gather_qmm, all three", gather_fused)):
        ms = chained(fn)
        print(f"  {name:28s} {ms:6.3f} ms  {mib / ms / 1024 * 1000:6.0f} GB/s  x{LAYERS} = {ms * LAYERS:5.1f} ms/token")
    print(f"  ({mib:.1f} MiB of expert weights per layer)")


if __name__ == "__main__":
    main()
