"""
M13b feasibility (HANDOFF 18.6): MLX's own 3-bit `qmv_fast` kernel, copied from the installed MLX headers into an
`mx.fast.metal_kernel`, with one change: the group bias is computed from a 4-bit code and the group scale,

    bias = bf16_rne(float32(code - 7) * float32(scale))        (the bank's rule, HANDOFF 18.4; k in -7..-3)

instead of read from a bf16 bias array. Everything else is MLX's code, so if the copy with real biases matches
`mx.quantized_matmul` bit for bit, the codes version does too, and an expert slot can hold 4-bit codes (0.42 MiB)
instead of bf16 biases (1.69 MiB): ~5 % more slots in the same budget, no extra kernel launch.

Checks, on MiniMax's expert shapes (w1/w3: 6144 -> 3072, w2: 3072 -> 6144), random weights, bf16 activations:
  1. the copy reading biases  == mx.quantized_matmul   (bit-identical?)
  2. the copy computing biases from codes == mx.quantized_matmul with the rebuilt biases
  3. time per call, chained like a decode token (57 layers x 4 experts x 3 projections)

    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_codes_qmv.py
"""

from __future__ import annotations

import os
import statistics
import time

import mlx.core as mx
import numpy as np

GROUP, BITS = 64, 3
K_BASE = -7  # code c in 0..4 means k = c - 7


def _mlx_header() -> str:
    path = os.path.join(os.path.dirname(mx.__file__), "include/mlx/backend/metal/kernels/quantized.h")
    lines = open(path).read().split("\n")
    # SIMD_SIZE, get_pack_factor, get_bytes_per_pack, load_vector, qdot: MLX's text, unchanged
    start = next(i for i, s in enumerate(lines) if s.startswith("using namespace metal"))
    end = next(i for i, s in enumerate(lines) if s.startswith("inline U qdot(")) + 1
    while lines[end] != "}":
        end += 1
    return "\n".join(lines[start:end + 1]) + "\n"


IMPL = r"""
// MLX's qmv_fast_impl (quantized.h), bias source switchable: CODES reads a 4-bit code per group and rebuilds the
// bias exactly as the bank does; otherwise the bf16 bias array, as MLX.
template <typename T, int group_size, int bits, bool CODES>
METAL_FUNC void qmv_codes_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device uint8_t* codes,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  int goff = out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  scales += goff;
  biases += goff;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;

      U s = sl[0];
      U b;
      if (CODES) {
        int g = goff + row * in_vec_size_g;
        int c = (codes[g >> 1] >> ((g & 1) * 4)) & 0xF;
        float p = float(c + K_BASE) * float(sl[0]);
        uint u = as_type<uint>(p);
        ushort hb = ushort((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
        b = static_cast<U>(as_type<T>(hb));
      } else {
        const device T* bl = biases + row * in_vec_size_g;
        b = bl[0];
      }
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    goff += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}
"""

BODY = r"""
  qmv_codes_impl<bfloat16_t, 64, 3, {codes}>(
      w, scales, biases, codes, x, y, IN_SIZE, OUT_SIZE,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""


def _kernel(codes: bool):
    return mx.fast.metal_kernel(
        name=f"minimax_qmv_{'codes' if codes else 'biases'}",
        input_names=["w", "scales", "biases", "codes", "x"],
        output_names=["y"],
        source=BODY.replace("{codes}", "true" if codes else "false"),
        header=_mlx_header() + f"constant constexpr int K_BASE = {K_BASE};\n" + IMPL,
        ensure_row_contiguous=True,
    )


_KERNELS: dict[bool, object] = {}


def qmv(x, w, scales, biases, codes, *, use_codes: bool):
    """y = x @ dequant(w).T for one row x [1, in] (the decode shape), MLX's qmv_fast dispatch."""
    k = _KERNELS.get(use_codes)
    if k is None:
        k = _KERNELS[use_codes] = _kernel(use_codes)
    out = w.shape[0]
    inp = x.shape[-1]
    return k(
        inputs=[w, scales, biases, codes, x],
        template=[("IN_SIZE", inp), ("OUT_SIZE", out)],
        # MLX: threadgroup (32, 2, 1), grid of threadgroups (M, out / 8, 1)
        grid=(32 * x.shape[0], 2 * (out // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(x.shape[0], out)],
        output_dtypes=[x.dtype],
    )[0]


def bf16_bits_rne(v: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def make_expert(out: int, inp: int, rng):
    """Random 3-bit weights, bf16 scales like the checkpoint's, biases = bf16(k * scale), k in -7..-3."""
    w = mx.array(rng.integers(0, 2**32, size=(out, inp * BITS // 32), dtype=np.uint32))
    g = inp // GROUP
    scale_f = (np.abs(rng.normal(0, 0.004, size=(out, g))) + 1e-4).astype(np.float32)
    scale_bits = bf16_bits_rne(scale_f)
    scale_f = (scale_bits.astype(np.uint32) << 16).view(np.float32)
    code = rng.integers(0, 5, size=(out, g)).astype(np.uint8)
    bias_bits = bf16_bits_rne(scale_f * (code.astype(np.float32) + K_BASE))
    scales = mx.array(scale_bits.view(np.uint16)).view(mx.bfloat16)
    biases = mx.array(bias_bits).view(mx.bfloat16)
    flat = code.reshape(-1)
    packed = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    return w, scales, biases, mx.array(packed)


def main() -> None:
    rng = np.random.default_rng(0)
    shapes = {"w1": (3072, 6144), "w2": (6144, 3072)}
    ok = True
    for name, (out, inp) in shapes.items():
        w, s, b, c = make_expert(out, inp, rng)
        for trial in range(4):
            x = mx.array(rng.normal(0, 1, size=(1, inp)).astype(np.float32)).astype(mx.bfloat16)
            ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GROUP, bits=BITS)
            got_b = qmv(x, w, s, b, c, use_codes=False)
            got_c = qmv(x, w, s, b, c, use_codes=True)
            mx.eval(ref, got_b, got_c)
            eq_b = bool(mx.array_equal(ref, got_b).item())
            eq_c = bool(mx.array_equal(ref, got_c).item())
            diff = float(mx.max(mx.abs(ref.astype(mx.float32) - got_c.astype(mx.float32))).item())
            ok &= eq_b and eq_c
            print(f"{name} trial {trial}: biases-copy identical={eq_b} codes identical={eq_c} max|diff|={diff}",
                  flush=True)
    print("BIT-IDENTICAL" if ok else "DIFFERS", flush=True)

    # time: a decode token's routed experts, 57 layers x 4 experts x (w1, w3, w2), one eval per layer
    experts = [(make_expert(3072, 6144, rng), make_expert(3072, 6144, rng), make_expert(6144, 3072, rng))
               for _ in range(8)]
    x0 = mx.array(rng.normal(0, 1, size=(1, 6144)).astype(np.float32)).astype(mx.bfloat16)

    def token(arm):
        h = x0
        for layer in range(57):
            outs = []
            for e in range(4):
                p1, p3, p2 = experts[(layer * 4 + e) % len(experts)]
                if arm == "mlx":
                    g = mx.quantized_matmul(h, *p1[:3], transpose=True, group_size=GROUP, bits=BITS)
                    u = mx.quantized_matmul(h, *p3[:3], transpose=True, group_size=GROUP, bits=BITS)
                    a = (g * mx.sigmoid(g) * u).astype(mx.bfloat16)
                    outs.append(mx.quantized_matmul(a, *p2[:3], transpose=True, group_size=GROUP, bits=BITS))
                else:
                    uc = arm == "codes"
                    g = qmv(h, *p1, use_codes=uc)
                    u = qmv(h, *p3, use_codes=uc)
                    a = (g * mx.sigmoid(g) * u).astype(mx.bfloat16)
                    outs.append(qmv(a, *p2, use_codes=uc))
            h = (sum(outs) * 0.25).astype(mx.bfloat16)
            mx.eval(h)
        return h

    for arm in ("mlx", "copy", "codes"):
        token(arm)
    times = {"mlx": [], "copy": [], "codes": []}
    for _ in range(30):
        for arm in ("mlx", "copy", "codes"):
            t0 = time.perf_counter()
            token(arm)
            times[arm].append(time.perf_counter() - t0)
    for arm, ts in times.items():
        print(f"CHAIN {arm} median_ms={1000 * statistics.median(ts):.2f} min_ms={1000 * min(ts):.2f}", flush=True)


if __name__ == "__main__":
    main()
