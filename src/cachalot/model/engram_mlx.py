from __future__ import annotations

import mlx.core as mx

from cachalot.model.fp8_linear_metal import fp8_linear
from cachalot.model.resident_layer import ResidentLayer


def _tensor(
    layer: ResidentLayer,
    suffix: str,
) -> mx.array:
    name = f"layers.{layer.layer_id}.engram.{suffix}"

    try:
        return layer.tensors[name].data
    except KeyError as exc:
        raise KeyError(
            f"Missing Engram tensor {name!r}"
        ) from exc


def engram_forward_decode(
    x: mx.array,
    embed_rows: mx.array,
    layer: ResidentLayer,
    *,
    token_enabled: bool = True,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
    return_debug: bool = False,
):
    """
    Decode-time Engram forward for one token.

    x:
        [hc_mult, dim]

    embed_rows:
        [n_hash_cols, head_dim]

    For DeepSeek-V4.1-Flash:
        x           = [4, 5120]
        embed_rows  = [24, 256]
        flattened   = [6144]
        WKV output  = [25600]
        key         = [4, 5120]
        value       = [5120]
    """
    if x.ndim != 2:
        raise ValueError(
            f"x must be [hc_mult, dim], got {x.shape}"
        )

    if embed_rows.ndim != 2:
        raise ValueError(
            "embed_rows must be [n_hash_cols, head_dim], "
            f"got {embed_rows.shape}"
        )

    hc_mult, dim = x.shape

    q_weight = _tensor(layer, "q_weight")
    k_weight = _tensor(layer, "k_weight")
    wkv_weight = _tensor(layer, "wkv.weight")
    wkv_scale = _tensor(layer, "wkv.scale")

    if q_weight.shape != (hc_mult, dim):
        raise ValueError(
            f"q_weight shape {q_weight.shape} "
            f"does not match x {x.shape}"
        )

    if k_weight.shape != (hc_mult, dim):
        raise ValueError(
            f"k_weight shape {k_weight.shape} "
            f"does not match x {x.shape}"
        )

    flat_dim = embed_rows.size

    expected_wkv_shape = (
        (hc_mult + 1) * dim,
        flat_dim,
    )

    if wkv_weight.shape != expected_wkv_shape:
        raise ValueError(
            f"wkv.weight shape {wkv_weight.shape}, "
            f"expected {expected_wkv_shape}"
        )

    # Official ParallelEngramEmbedding.forward():
    #
    #   values = ...dequant...
    #   values = values.flatten(-2).to(torch.bfloat16)
    #
    # Therefore the WKV activation starts from BF16.
    embed_flat = (
        embed_rows
        .reshape(-1)
        .astype(mx.bfloat16)
    )

    kv = fp8_linear(
        embed_flat,
        wkv_weight,
        wkv_scale,
    )

    expected_kv = (hc_mult + 1) * dim

    if kv.size != expected_kv:
        raise ValueError(
            f"WKV produced {kv.size} values, "
            f"expected {expected_kv}"
        )

    key = (
        kv[: hc_mult * dim]
        .reshape(hc_mult, dim)
        .astype(mx.float32)
    )

    value = (
        kv[hc_mult * dim :]
        .astype(mx.float32)
    )

    # Official implementation performs all gating math in FP32.
    h = x.astype(mx.float32)

    weight = (
        q_weight.astype(mx.float32)
        * k_weight.astype(mx.float32)
    )

    h_rstd = mx.rsqrt(
        mx.mean(h * h, axis=-1) + eps
    )

    key_rstd = mx.rsqrt(
        mx.mean(key * key, axis=-1) + eps
    )

    rstd = h_rstd * key_rstd

    dot = mx.sum(
        h * weight * key,
        axis=-1,
    )

    dot = (
        dot
        * rstd
        * (dim ** -0.5)
    )

    # torch.copysign(
    #     dot.abs().clamp_min(1e-6).sqrt(),
    #     dot,
    # )
    magnitude = mx.sqrt(
        mx.maximum(
            mx.abs(dot),
            mx.array(
                clamp_value,
                dtype=mx.float32,
            ),
        )
    )

    signed_root = mx.where(
        dot < 0,
        -magnitude,
        magnitude,
    )

    gate = mx.sigmoid(signed_root)

    if not token_enabled:
        gate = mx.zeros_like(gate)

    out = (
        h
        + gate[:, None] * value[None, :]
    )

    out = out.astype(x.dtype)

    if return_debug:
        return out, {
            "key": key,
            "value": value,
            "dot": dot,
            "signed_root": signed_root,
            "gate": gate,
        }

    return out


def engram_forward_batched(
    x: mx.array,
    embed_rows: mx.array,
    layer: ResidentLayer,
    *,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
    token_mask: mx.array | None = None,
) -> mx.array:
    """
    engram_forward_decode for a chunk of tokens.

    x:          [tokens, hc_mult, dim]
    embed_rows: [tokens, n_hash_cols, head_dim] (fp32, dequantized)
    token_mask: [tokens] bool or None. False shuts the gate so that position
                passes through untouched -- the reference's image-span
                handling (inference/model.py Engram.forward). None, the
                text-only case, is bit-identical to before this existed.
    """
    from cachalot.model.moe_prefill_batched import (
        dequantize_fp8_weight,
        quantize_activation_fp8_rows,
    )

    n_tokens, hc_mult, dim = x.shape
    q_weight = _tensor(layer, "q_weight")
    k_weight = _tensor(layer, "k_weight")
    wkv_weight = _tensor(layer, "wkv.weight")
    wkv_scale = _tensor(layer, "wkv.scale")

    embed_flat = embed_rows.reshape(n_tokens, -1).astype(mx.bfloat16)
    qx = quantize_activation_fp8_rows(embed_flat)
    kv = (qx @ dequantize_fp8_weight(wkv_weight, wkv_scale).T).astype(mx.bfloat16)

    key = kv[:, : hc_mult * dim].reshape(n_tokens, hc_mult, dim).astype(mx.float32)
    value = kv[:, hc_mult * dim :].astype(mx.float32)

    h = x.astype(mx.float32)
    weight = q_weight.astype(mx.float32) * k_weight.astype(mx.float32)
    h_rstd = mx.rsqrt(mx.mean(h * h, axis=-1) + eps)
    key_rstd = mx.rsqrt(mx.mean(key * key, axis=-1) + eps)
    dot = mx.sum(h * weight[None] * key, axis=-1) * (h_rstd * key_rstd) * (dim ** -0.5)
    magnitude = mx.sqrt(mx.maximum(mx.abs(dot), mx.array(clamp_value, dtype=mx.float32)))
    gate = mx.sigmoid(mx.where(dot < 0, -magnitude, magnitude))
    if token_mask is not None:
        gate = mx.where(token_mask[:, None], gate, mx.zeros_like(gate))
    out = h + gate[..., None] * value[:, None, :]
    return out.astype(x.dtype)
