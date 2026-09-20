"""
Is hc_mixes_1d limited by its own occupancy?

The shipped kernel computes 24 dot products of length hc_mult*hidden (20,480)
plus a sum of squares in a SINGLE threadgroup, so one GPU core reads the whole
983 KiB hc_fn matrix. This script compares it against a two-stage version that
splits the reduction across many threadgroups and finishes the split + Sinkhorn
in a second, tiny dispatch.

Both are timed chained (many launches in one lazy graph, one eval) so the
~0.15 ms mx.eval round trip is excluded, which is how a decode token pays for
them. Outputs are compared for bitwise-close agreement.
"""
from __future__ import annotations

import sys
from functools import cache
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cachalot.model.decode_fused_metal import HEADER, hc_mixes_1d  # noqa: E402

HC = 4
MIX_HC = (2 + HC) * HC          # 24
HIDDEN = 5120
N = HC * HIDDEN                 # 20480
ITERS = 20


@cache
def _partial_kernel(mix_hc: int, groups: int):
    """Stage 1: threadgroup g reduces its slice of n into partial[g, 0..mix_hc]."""
    source = f"""
        uint g = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint n = dims[0];
        uint per = (n + {groups} - 1) / {groups};
        uint lo = g * per;
        uint hi = metal::min(lo + per, n);
        threadgroup float red[256 * {mix_hc + 1}];
        float acc[{mix_hc}];
        for (uint j = 0; j < {mix_hc}; ++j) acc[j] = 0.0f;
        float sq = 0.0f;
        for (uint i = lo + tid; i < hi; i += 256) {{
            float xv = float(x[i]);
            sq += xv * xv;
            for (uint j = 0; j < {mix_hc}; ++j) acc[j] += xv * float(hc_fn[j * n + i]);
        }}
        for (uint j = 0; j < {mix_hc}; ++j) red[j * 256 + tid] = acc[j];
        red[{mix_hc} * 256 + tid] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128; s > 0; s >>= 1) {{
            if (tid < s) {{
                for (uint j = 0; j <= {mix_hc}; ++j) red[j * 256 + tid] += red[j * 256 + tid + s];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0) {{
            for (uint j = 0; j <= {mix_hc}; ++j) partial[g * {mix_hc + 1} + j] = red[j * 256];
        }}
    """
    return mx.fast.metal_kernel(
        name=f"hc_mixes_partial_m{mix_hc}_g{groups}",
        input_names=["x", "hc_fn", "dims"],
        output_names=["partial"],
        source=source,
        header=HEADER,
    )


@cache
def _finish_kernel(hc_mult: int, sinkhorn_iters: int, groups: int):
    """Stage 2: reduce the group partials, then the original split + Sinkhorn."""
    mix_hc = (2 + hc_mult) * hc_mult
    source = f"""
        uint tid = thread_position_in_grid.x;
        if (tid != 0) return;
        uint n = dims[0];
        float sums[{mix_hc + 1}];
        for (uint j = 0; j <= {mix_hc}; ++j) sums[j] = 0.0f;
        for (uint g = 0; g < {groups}; ++g) {{
            for (uint j = 0; j <= {mix_hc}; ++j) sums[j] += partial[g * {mix_hc + 1} + j];
        }}
        float rs = metal::rsqrt(sums[{mix_hc}] / float(n) + norm_eps[0]);
        float mixes[{mix_hc}];
        for (uint j = 0; j < {mix_hc}; ++j) mixes[j] = sums[j] * rs;
        float eps = hc_eps[0];
        for (uint j = 0; j < {hc_mult}; ++j) {{
            float z = mixes[j] * hc_scale[0] + hc_base[j];
            pre[j] = 1.0f / (1.0f + metal::exp(-z)) + eps;
            uint idx = {hc_mult} + j;
            float z2 = mixes[idx] * hc_scale[1] + hc_base[idx];
            post[j] = 2.0f / (1.0f + metal::exp(-z2));
        }}
        float c[{hc_mult * hc_mult}];
        for (uint j = 0; j < {hc_mult * hc_mult}; ++j) {{
            uint idx = 2 * {hc_mult} + j;
            c[j] = mixes[idx] * hc_scale[2] + hc_base[idx];
        }}
        for (uint r = 0; r < {hc_mult}; ++r) {{
            float rm = -INFINITY;
            for (uint k = 0; k < {hc_mult}; ++k) rm = metal::max(rm, c[r * {hc_mult} + k]);
            float sum = 0.0f;
            for (uint k = 0; k < {hc_mult}; ++k) {{ float v = metal::exp(c[r * {hc_mult} + k] - rm); c[r * {hc_mult} + k] = v; sum += v; }}
            for (uint k = 0; k < {hc_mult}; ++k) c[r * {hc_mult} + k] = c[r * {hc_mult} + k] / sum + eps;
        }}
        for (uint k = 0; k < {hc_mult}; ++k) {{
            float cs = 0.0f;
            for (uint r = 0; r < {hc_mult}; ++r) cs += c[r * {hc_mult} + k];
            for (uint r = 0; r < {hc_mult}; ++r) c[r * {hc_mult} + k] = c[r * {hc_mult} + k] / (cs + eps);
        }}
        for (uint it = 1; it < {sinkhorn_iters}; ++it) {{
            for (uint r = 0; r < {hc_mult}; ++r) {{
                float rsum = 0.0f;
                for (uint k = 0; k < {hc_mult}; ++k) rsum += c[r * {hc_mult} + k];
                for (uint k = 0; k < {hc_mult}; ++k) c[r * {hc_mult} + k] = c[r * {hc_mult} + k] / (rsum + eps);
            }}
            for (uint k = 0; k < {hc_mult}; ++k) {{
                float cs = 0.0f;
                for (uint r = 0; r < {hc_mult}; ++r) cs += c[r * {hc_mult} + k];
                for (uint r = 0; r < {hc_mult}; ++r) c[r * {hc_mult} + k] = c[r * {hc_mult} + k] / (cs + eps);
            }}
        }}
        for (uint j = 0; j < {hc_mult * hc_mult}; ++j) comb[j] = c[j];
    """
    return mx.fast.metal_kernel(
        name=f"hc_mixes_finish_h{hc_mult}_i{sinkhorn_iters}_g{groups}",
        input_names=["partial", "hc_scale", "hc_base", "dims", "norm_eps", "hc_eps"],
        output_names=["pre", "post", "comb"],
        source=source,
        header=HEADER,
    )


def hc_mixes_split(x, hc_fn, hc_scale, hc_base, *, norm_eps, hc_mult, sinkhorn_iters, hc_eps, groups):
    n = x.size
    mix_hc = (2 + hc_mult) * hc_mult
    dims = mx.array([n], dtype=mx.uint32)
    partial = _partial_kernel(mix_hc, groups)(
        inputs=[x.reshape(-1), hc_fn, dims],
        grid=(groups * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(groups * (mix_hc + 1),)],
        output_dtypes=[mx.float32],
    )[0]
    return _finish_kernel(hc_mult, sinkhorn_iters, groups)(
        inputs=[partial, hc_scale.astype(mx.float32), hc_base.astype(mx.float32), dims,
                mx.array([norm_eps], dtype=mx.float32), mx.array([hc_eps], dtype=mx.float32)],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(hc_mult,), (hc_mult,), (hc_mult, hc_mult)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


def chained(fn, n=40):
    mx.eval(fn())
    mx.synchronize()
    t0 = perf_counter()
    outs = [fn() for _ in range(n)]
    mx.eval(*outs)
    mx.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    mx.random.seed(0)
    x = mx.random.normal((HC, HIDDEN)).astype(mx.bfloat16)
    hc_fn = (mx.random.normal((MIX_HC, N)) * 0.01).astype(mx.bfloat16)
    hc_scale = mx.array([0.1, 0.1, 0.1])
    hc_base = mx.random.normal((MIX_HC,))
    mx.eval(x, hc_fn, hc_scale, hc_base)

    kw = dict(norm_eps=1e-20, hc_mult=HC, sinkhorn_iters=ITERS, hc_eps=1e-6)
    ref = hc_mixes_1d(x, hc_fn, hc_scale, hc_base, **kw)
    mx.eval(*ref)

    bytes_read = hc_fn.size * 2 + x.size * 2
    base_ms = chained(lambda: hc_mixes_1d(x, hc_fn, hc_scale, hc_base, **kw)[0])
    print(f"hc_fn {bytes_read / 2**20:.2f} MiB per call")
    print(f"shipped, 1 threadgroup      {base_ms:6.3f} ms   {bytes_read / base_ms / 1e6:7.1f} GB/s   x80 = {base_ms * 80:5.1f} ms/token")

    for groups in (4, 8, 16, 32, 64, 128):
        got = hc_mixes_split(x, hc_fn, hc_scale, hc_base, groups=groups, **kw)
        mx.eval(*got)
        err = max(float(mx.max(mx.abs(a - b))) for a, b in zip(ref, got, strict=True))
        ms = chained(lambda g=groups: hc_mixes_split(x, hc_fn, hc_scale, hc_base, groups=g, **kw)[0])
        print(f"split, {groups:3d} threadgroups   {ms:6.3f} ms   {bytes_read / ms / 1e6:7.1f} GB/s   "
              f"x80 = {ms * 80:5.1f} ms/token   max|delta| {err:.3e}")


if __name__ == "__main__":
    main()
