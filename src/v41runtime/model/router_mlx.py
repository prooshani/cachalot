from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class RouterResult:
    indices: mx.array
    weights: mx.array
    scores: mx.array


def route_topk(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    *,
    topk: int = 6,
    gate_temp: float = 1.0,
    route_scale: float = 1.5,
    norm_topk_prob: bool = True,
) -> RouterResult:
    """
    DeepSeek V4.1 text-token MoE router for single-token decode.

    Official semantics:
      scores = linear(x.float(), weight.float()) / gate_temp
      scores = sqrt(softplus(scores))
      indices = topk(scores + bias)
      weights = scores[indices]
      weights /= sum(weights) + 1e-20
      weights *= route_scale

    The correction bias affects selection only.
    """

    if x.ndim != 1:
        raise ValueError(
            f"x must be 1D, got {x.shape}"
        )

    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2D, got {weight.shape}"
        )

    n_experts, hidden_size = weight.shape

    if x.size != hidden_size:
        raise ValueError(
            f"x has {x.size} elements, "
            f"gate expects {hidden_size}"
        )

    if bias.shape != (n_experts,):
        raise ValueError(
            f"bias shape {bias.shape} != ({n_experts},)"
        )

    if not 1 <= topk <= n_experts:
        raise ValueError(
            f"Invalid topk={topk} for {n_experts} experts"
        )

    xf = x.astype(mx.float32)
    wf = weight.astype(mx.float32)
    bf = bias.astype(mx.float32)

    scores = (
        mx.matmul(
            wf,
            xf,
        )
        / gate_temp
    )

    scores = mx.sqrt(
        nn.softplus(scores)
    )

    selection_scores = (
        scores + bf
    )

    # Stable full sort is simple and inexpensive for 384 experts.
    order = mx.argsort(
        selection_scores
    )

    indices = order[-topk:]

    weights = mx.take(
        scores,
        indices,
        axis=0,
    )

    if norm_topk_prob and topk > 1:
        weights = (
            weights
            / (
                mx.sum(weights)
                + 1e-20
            )
        )

    weights = (
        weights * route_scale
    )

    return RouterResult(
        indices=indices,
        weights=weights,
        scores=scores,
    )


def route_topk_batch(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    *,
    topk: int = 6,
    gate_temp: float = 1.0,
    route_scale: float = 1.5,
    norm_topk_prob: bool = True,
) -> RouterResult:
    """
    Decode-exact batched DeepSeek V4.1 router.

    Routing is intentionally evaluated token-by-token through the
    validated route_topk() implementation.

    This preserves the exact numerical path used during decode while
    still allowing the subsequent MoE execution to be scheduled
    expert-major across all prompt tokens.

    x:
        [tokens, hidden]

    Returns:
        indices:
            [tokens, topk]

        weights:
            [tokens, topk]

        scores:
            [tokens, n_experts]
    """
    if x.ndim != 2:
        raise ValueError(
            f"x must be 2D [tokens, hidden], got {x.shape}"
        )

    if x.shape[0] == 0:
        raise ValueError(
            "x must contain at least one token"
        )

    results = []

    for token_index in range(
        x.shape[0]
    ):
        result = route_topk(
            x[token_index],
            weight,
            bias,
            topk=topk,
            gate_temp=gate_temp,
            route_scale=route_scale,
            norm_topk_prob=norm_topk_prob,
        )

        results.append(
            result
        )

    indices = mx.stack(
        [
            result.indices
            for result in results
        ],
        axis=0,
    )

    weights = mx.stack(
        [
            result.weights
            for result in results
        ],
        axis=0,
    )

    scores = mx.stack(
        [
            result.scores
            for result in results
        ],
        axis=0,
    )

    return RouterResult(
        indices=indices,
        weights=weights,
        scores=scores,
    )

