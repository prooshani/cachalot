from __future__ import annotations

import mlx.core as mx

from cachalot.model.hc_sinkhorn_metal import (
    hc_split_sinkhorn_metal,
)


def hc_split_sinkhorn(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    MLX equivalent of DeepSeek hc_split_sinkhorn().

    mixes:
        [..., (2 + hc_mult) * hc_mult] FP32

    Returns:
        pre:  [..., hc_mult] FP32
        post: [..., hc_mult] FP32
        comb: [..., hc_mult, hc_mult] FP32
    """
    mix_hc = (2 + hc_mult) * hc_mult

    if mixes.shape[-1] != mix_hc:
        raise ValueError(
            f"mixes last dim {mixes.shape[-1]} != {mix_hc}"
        )

    if hc_scale.shape != (3,):
        raise ValueError(
            f"hc_scale shape {hc_scale.shape} != (3,)"
        )

    if hc_base.shape != (mix_hc,):
        raise ValueError(
            f"hc_base shape {hc_base.shape} != ({mix_hc},)"
        )

    mixes = mixes.astype(mx.float32)
    hc_scale = hc_scale.astype(mx.float32)
    hc_base = hc_base.astype(mx.float32)

    pre_logits = (
        mixes[..., :hc_mult] * hc_scale[0]
        + hc_base[:hc_mult]
    )

    pre = mx.sigmoid(pre_logits) + eps

    post_logits = (
        mixes[..., hc_mult:2 * hc_mult] * hc_scale[1]
        + hc_base[hc_mult:2 * hc_mult]
    )

    post = 2.0 * mx.sigmoid(post_logits)

    comb = (
        mixes[..., 2 * hc_mult:] * hc_scale[2]
        + hc_base[2 * hc_mult:]
    ).reshape(
        *mixes.shape[:-1],
        hc_mult,
        hc_mult,
    )

    # Official first step:
    #
    # comb = softmax(comb, -1) + eps
    row_max = mx.max(
        comb,
        axis=-1,
        keepdims=True,
    )

    comb = mx.exp(
        comb - row_max
    )

    comb = (
        comb
        / mx.sum(
            comb,
            axis=-1,
            keepdims=True,
        )
        + eps
    )

    # First column normalization.
    comb = comb / (
        mx.sum(
            comb,
            axis=-2,
            keepdims=True,
        )
        + eps
    )

    # Remaining Sinkhorn iterations.
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (
            mx.sum(
                comb,
                axis=-1,
                keepdims=True,
            )
            + eps
        )

        comb = comb / (
            mx.sum(
                comb,
                axis=-2,
                keepdims=True,
            )
            + eps
        )

    return pre, post, comb


def hc_mixes(
    x: mx.array,
    hc_fn: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    *,
    norm_eps: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Equivalent of Block.hc_mixes() for arbitrary leading dimensions.

    x:
        [..., hc_mult, dim]

    hc_fn:
        [(2 + hc_mult) * hc_mult, hc_mult * dim]
    """
    flat = x.reshape(
        *x.shape[:-2],
        -1,
    ).astype(mx.float32)

    rsqrt = mx.rsqrt(
        mx.mean(
            flat * flat,
            axis=-1,
            keepdims=True,
        )
        + norm_eps
    )

    mixes = mx.matmul(
        flat,
        mx.transpose(
            hc_fn.astype(mx.float32)
        ),
    ) * rsqrt

    # Decode path: one token produces a single 24-value
    # mix vector. The custom Metal kernel performs the
    # complete split + Sinkhorn in one dispatch.
    #
    # Keep the MLX implementation as the generic fallback
    # for prefill/batched shapes and as our reference path.
    if mixes.ndim == 1:
        return hc_split_sinkhorn_metal(
            mixes,
            hc_scale,
            hc_base,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=hc_eps,
        )

    return hc_split_sinkhorn(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=hc_eps,
    )


def hc_pre(
    x: mx.array,
    pre_mix: mx.array,
) -> mx.array:
    """
    [..., hc, dim] x [..., hc] -> [..., dim]
    """
    dtype = x.dtype

    y = mx.sum(
        pre_mix[..., :, None]
        * x.astype(mx.float32),
        axis=-2,
    )

    return y.astype(dtype)


def hc_post(
    x: mx.array,
    residual: mx.array,
    post: mx.array,
    comb: mx.array,
) -> mx.array:
    """
    x:
        [..., dim]

    residual:
        [..., hc, dim]

    post:
        [..., hc]

    comb:
        [..., hc, hc]

    Returns:
        [..., hc, dim]
    """
    dtype = x.dtype

    sublayer = (
        post[..., :, None]
        * x.astype(mx.float32)[..., None, :]
    )

    # Official (model.py, Block.hc_post):
    #
    #     torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    #
    # which broadcasts to elem[i, j, :] = comb[i, j] * residual[i, :] and sums
    # over i, so output stream j is sum_i comb[i, j] * residual[i]. comb is
    # contracted over its FIRST index: the mixing matrix is applied transposed.
    #
    # Contracting the second index instead -- comb @ residual rather than
    # comb.T @ residual -- is very hard to see, because comb comes out of a
    # sinkhorn normalization and is close to doubly stochastic, so each stream
    # still receives about the right total weight and the model stays fluent.
    # What it destroys is which stream a given piece of information lands in.
    mixed_residual = mx.sum(
        comb[..., :, :, None]
        * residual.astype(mx.float32)[..., :, None, :],
        axis=-3,
    )

    return (
        sublayer + mixed_residual
    ).astype(dtype)
