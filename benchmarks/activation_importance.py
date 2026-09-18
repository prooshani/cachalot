"""
Turning recorded activations into the per-column importance a fit is weighted by.

benchmarks/capture_activations.py records what the routed experts are multiplied
by. This module is the small amount of arithmetic between that recording and
benchmarks/quant_affine.py's `weights` argument, shared by the screen and the
bank builder so the two cannot drift.

The weighting is per *column* of a weight matrix, because that is the axis the
input is contracted over and the axis the groups run along:

  w1, w3   multiply the MoE input x, so their importance is the recorded mean
           square of x, which is a property of the layer
  w2       multiplies the SwiGLU hidden that x produces *through this expert*,
           so its importance has to be computed per expert from the expert's
           own weights

Importance is normalized to mean 1 per matrix. The fit's objective is scale
invariant, so this changes nothing but the size of the numbers in it.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

N_LAYERS = 40


def mean_square(v: mx.array) -> mx.array:
    """Per-component mean square over probes, normalized to mean 1."""
    m = mx.mean(v * v, axis=-1)
    return m / mx.maximum(mx.mean(m), mx.array(1e-30, dtype=mx.float32))


def swiglu_hidden(x: mx.array, w1: mx.array, w3: mx.array, limit: float = 10.0) -> mx.array:
    """The vector w2 is multiplied by, in the arithmetic swiglu_expert uses for it."""
    xb = x.astype(mx.bfloat16).astype(mx.float32)
    gate = (w1 @ xb).astype(mx.bfloat16).astype(mx.float32)
    up = mx.clip((w3 @ xb).astype(mx.bfloat16).astype(mx.float32), -limit, limit)
    gate = mx.minimum(gate, mx.array(limit, dtype=mx.float32))
    return (nn.silu(gate) * up).astype(mx.bfloat16).astype(mx.float32)


def load_activations(path: str | Path, layers: int = N_LAYERS) -> dict[int, dict[str, mx.array]]:
    """Per layer: `importance` [HIDDEN] normalized to mean 1, and `x` [HIDDEN, probes]."""
    data = np.load(path)
    out: dict[int, dict[str, mx.array]] = {}
    for layer in range(layers):
        if f"sumsq_{layer}" not in data or f"samples_{layer}" not in data:
            continue
        tokens = max(1, int(data[f"tokens_{layer}"]))
        acc = mx.array(data[f"sumsq_{layer}"].astype(np.float32)) / tokens
        out[layer] = {
            "importance": acc / mx.maximum(mx.mean(acc), mx.array(1e-30, dtype=mx.float32)),
            # [HIDDEN, probes], the orientation the expert math wants.
            "x": mx.array(data[f"samples_{layer}"].astype(np.float32)).T,
        }
    return out


def expert_importance(dense: dict[str, mx.array], layer_acts: dict[str, mx.array],
                      ) -> dict[str, mx.array]:
    """Per-projection importance for one expert: recorded for w1/w3, derived for w2."""
    x = layer_acts["x"]
    return {
        "w1": layer_acts["importance"],
        "w3": layer_acts["importance"],
        "w2": mean_square(swiglu_hidden(x, dense["w1"], dense["w3"])),
    }
