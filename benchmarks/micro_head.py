"""Head GEMV: chunked fp32 conversion (current) vs bf16-reading kernel."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cachalot.model.bf16_gemv_metal import bf16_gemv_f32  # noqa: E402
from cachalot.model.model_boundary_mlx import parallel_head_logits_decode  # noqa: E402


def main():
    mx.random.seed(0)
    w = mx.random.normal((129280, 5120)).astype(mx.bfloat16)
    h = mx.random.normal((5120,)).astype(mx.float32)
    mx.eval(w, h)
    ref = parallel_head_logits_decode(h, w)
    new = bf16_gemv_f32(h, w)
    mx.eval(ref, new)
    d = mx.abs(ref - new)
    print(f"max|diff| {float(d.max()):.3e} rel {float(d.max() / mx.abs(ref).max()):.3e} argmax equal {int(ref.argmax()) == int(new.argmax())}")

    def bench(fn, n=20):
        mx.eval(fn())
        mx.synchronize()
        t0 = perf_counter()
        for _ in range(n):
            mx.eval(fn())
        mx.synchronize()
        return (perf_counter() - t0) / n * 1e3

    print(f"chunked fp32 convert: {bench(lambda: parallel_head_logits_decode(h, w)):.2f} ms")
    print(f"bf16 gemv kernel    : {bench(lambda: bf16_gemv_f32(h, w)):.2f} ms")


if __name__ == "__main__":
    main()
