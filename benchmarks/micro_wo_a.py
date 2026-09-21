"""
Screen: attention's `wo_a` is the only projection in the token that is held
dequantized, and it is the largest one.

The checkpoint ships `attn.wo_a.weight` as F8_E4M3 [8192, 4096] with E8M0
block scales -- 33.55 MB per layer. `text_decode_runtime._get_wo_a`
dequantizes it to BF16 at startup for all forty layers, so what the GPU reads
every token is 67.11 MB per layer, 2.68 GiB across the model. Every other
projection in the attention block (wq_a, wq_b, wkv, wo_b) is read as FP8 bytes
through `fp8_gemv_quantized`.

The reason is the shape, not the precision: `wo_a` is applied as eight
independent [1024, 4096] matvecs, one per output group, and the FP8 GEMV
kernel takes a plain 2D [N, K] weight, so the grouped form goes through
`mx.matmul` on a BF16 [8, 1024, 4096] view instead.

Section 7.1.5 prices a reuse layer's attention at 0.441 ms. With wo_a in BF16
that layer's attention weights are 160.2 MB, which is 363 GB/s -- at the wall
this machine gives for a streaming read. Halving wo_a takes the layer to
126.6 MB. This asks what that is worth before anyone writes a kernel.

Four arms, forty distinct layers so nothing is served out of cache, all
chained inside one eval so the per-eval floor is excluded:

  bf16 grouped   what ships: mx.matmul over BF16 [8, 1024, 4096]
  fp8 x8         eight fp8_gemv_quantized calls, one per group. Same bytes a
                 grouped FP8 kernel would read, but eight launches instead of
                 one, so it bounds the naive version from below.
  fp8 x1         one fp8_gemv_quantized over the flat [8192, 4096]. It computes
                 the wrong thing -- every row sees the same 4096 inputs instead
                 of its own group's -- but it reads exactly the bytes a grouped
                 kernel would read in exactly one launch, so it is the upper
                 bound on what such a kernel could return.
  sum            mx.sum over the same FP8 bytes: the machine's ceiling.

    PYTHONPATH=src python benchmarks/micro_wo_a.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cachalot.model.fp8_gemv_metal import fp8_gemv_quantized  # noqa: E402
from cachalot.model.fp8_linear_metal import quantize_fp8_activation  # noqa: E402

GROUPS = 8
O_LORA_RANK = 1024
GROUP_INPUT = 4096  # n_heads * head_dim // n_groups = 64 * 512 / 8
N = GROUPS * O_LORA_RANK  # 8192
BLOCK = 32
LAYERS = 40
REPEATS = 9


def main() -> None:
    o = mx.random.normal(shape=(GROUPS * GROUP_INPUT,)).astype(mx.bfloat16)
    og = o.reshape(GROUPS, GROUP_INPUT)

    fp8, bf16 = [], []
    for _ in range(LAYERS):
        w = mx.random.randint(0, 255, shape=(N, GROUP_INPUT)).astype(mx.uint8)
        s = mx.random.randint(120, 131, shape=(N // BLOCK, GROUP_INPUT // BLOCK)).astype(mx.uint8)
        d = mx.random.normal(shape=(GROUPS, O_LORA_RANK, GROUP_INPUT)).astype(mx.bfloat16)
        mx.eval(w, s, d)
        fp8.append((w, s))
        bf16.append(d)

    fp8_bytes = fp8[0][0].nbytes + fp8[0][1].nbytes
    bf16_bytes = bf16[0].nbytes
    print(
        f"wo_a per layer: FP8 {fp8_bytes / 1e6:.2f} MB, BF16 {bf16_bytes / 1e6:.2f} MB; "
        f"across {LAYERS} layers {LAYERS * bf16_bytes / 2**30:.2f} GiB resident as BF16 "
        f"against {LAYERS * fp8_bytes / 2**30:.2f} GiB as FP8\n"
    )

    def arm_bf16():
        out = None
        for d in bf16:
            y = mx.matmul(d, og[..., None])[..., 0].reshape(-1)
            out = y if out is None else out + y
        return out

    def arm_fp8_x8():
        out = None
        for w, s in fp8:
            parts = []
            for g in range(GROUPS):
                qx = quantize_fp8_activation(og[g])
                parts.append(
                    fp8_gemv_quantized(
                        qx.values, qx.scales,
                        w[g * O_LORA_RANK:(g + 1) * O_LORA_RANK],
                        s[g * O_LORA_RANK // BLOCK:(g + 1) * O_LORA_RANK // BLOCK],
                        out_features=O_LORA_RANK, in_features=GROUP_INPUT,
                    )
                )
            y = mx.concatenate(parts)
            out = y if out is None else out + y
        return out

    def arm_fp8_x1():
        out = None
        qx = quantize_fp8_activation(og[0])
        for w, s in fp8:
            y = fp8_gemv_quantized(
                qx.values, qx.scales, w, s,
                out_features=N, in_features=GROUP_INPUT,
            )
            out = y if out is None else out + y
        return out

    def arm_sum():
        out = None
        for w, s in fp8:
            t = w.sum()
            out = t if out is None else out + t
        return out

    def chained(fn):
        mx.eval(fn())
        mx.synchronize()
        times = []
        for _ in range(REPEATS):
            t0 = perf_counter()
            mx.eval(fn())
            mx.synchronize()
            times.append((perf_counter() - t0) / LAYERS * 1e3)
        times.sort()
        return times[0], statistics.median(times)

    rows = [
        ("bf16 grouped matmul (ships)", arm_bf16, bf16_bytes, GROUPS),
        ("fp8, one GEMV per group", arm_fp8_x8, fp8_bytes, GROUPS),
        ("fp8, one GEMV (bound)", arm_fp8_x1, fp8_bytes, 1),
        ("sum over the fp8 bytes", arm_sum, fp8_bytes, 1),
    ]
    print(f"{'arm':<30}{'min/layer':>11}{'median':>10}{'GB/s':>9}{'x40':>10}{'launches':>10}")
    for name, fn, nbytes, launches in rows:
        lo, med = chained(fn)
        print(
            f"{name:<30}{lo:>9.3f} ms{med:>8.3f} ms"
            f"{nbytes / (lo * 1e-3) / 1e9:>9.0f}{lo * LAYERS:>8.1f} ms{launches * LAYERS:>10d}"
        )


if __name__ == "__main__":
    main()
