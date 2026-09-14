from __future__ import annotations

import mlx.core as mx

from v41runtime.model.hyper_connection_mlx import hc_mixes


def hc_mixes_prefill_exact(
    x: mx.array,
    hc_fn: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    *,
    norm_eps: float = 1e-20,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
]:
    """
    Decode-numerically-exact HC coefficients for prompt prefill.

    x:
        [tokens, hc_mult, hidden]

    Each token is intentionally passed through the existing validated
    one-token HC path. This therefore uses the same custom Metal
    Sinkhorn implementation as decode rather than the generic batched
    MLX Sinkhorn.

    This is a correctness bridge. Once full prefill parity is locked,
    HC can be replaced by a genuinely batched Metal kernel.
    """
    if x.ndim != 3:
        raise ValueError(
            f"x must be [tokens, hc, dim], got {x.shape}"
        )

    n_tokens = x.shape[0]

    if n_tokens == 0:
        raise ValueError(
            "x must contain at least one token"
        )

    pre = []
    post = []
    comb = []

    for token_index in range(
        n_tokens
    ):
        (
            token_pre,
            token_post,
            token_comb,
        ) = hc_mixes(
            x[token_index],
            hc_fn,
            hc_scale,
            hc_base,
            norm_eps=norm_eps,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=hc_eps,
        )

        pre.append(
            token_pre
        )

        post.append(
            token_post
        )

        comb.append(
            token_comb
        )

    return (
        mx.stack(
            pre,
            axis=0,
        ),
        mx.stack(
            post,
            axis=0,
        ),
        mx.stack(
            comb,
            axis=0,
        ),
    )
