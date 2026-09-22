"""
MLX port of the official DeepSeek-V4.1 ViT + Aligner
(`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/vision.py`).

Feasibility-spike piece only (HANDOFF.md section 16, piece 1). Not wired into
TextDecodeRuntime. `resident_trunk.py` still filters every vision.*/aligner.*
tensor out of the resident trunk; this module loads them separately via
`cachalot.storage.tensor_index`/`tensor_loader`, the same loader the trunk
uses, so a bare checkpoint read is the only shared surface.

Dense bidirectional attention over one image, 2D RoPE, no streaming, no
cache, no batching across images.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from cachalot.model.norm_rope_mlx import rms_norm
from cachalot.storage.tensor_loader import ResidentTensor


@dataclass(frozen=True)
class VisionConfig:
    dim: int = 1024
    n_heads: int = 16
    n_layers: int = 32
    inter_dim: int = 2816
    patch_size: int = 14
    rope_theta: float = 10000.0
    downsample_ratio: int = 3

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


@dataclass(frozen=True)
class VisionWeights:
    patch_embed_w: mx.array
    patch_embed_b: mx.array
    blocks: list[dict[str, mx.array]]
    norm_w: mx.array
    aligner_w1: mx.array
    aligner_b1: mx.array
    aligner_w2: mx.array
    aligner_b2: mx.array


def load_vision_weights(
    tensors: dict[str, ResidentTensor],
    cfg: VisionConfig = VisionConfig(),
) -> VisionWeights:
    def t(name: str) -> mx.array:
        return tensors[name].data

    blocks = []
    for i in range(cfg.n_layers):
        p = f"vision.blocks.{i}."
        blocks.append(
            {
                "norm1_w": t(p + "norm1.weight"),
                "wqkv_w": t(p + "attn.wqkv.weight"),
                "wqkv_b": t(p + "attn.wqkv.bias"),
                "wo_w": t(p + "attn.wo.weight"),
                "wo_b": t(p + "attn.wo.bias"),
                "norm2_w": t(p + "norm2.weight"),
                "mlp_w1": t(p + "mlp.w1.weight"),
                "mlp_w2": t(p + "mlp.w2.weight"),
            }
        )

    return VisionWeights(
        patch_embed_w=t("vision.patch_embed.proj.weight"),
        patch_embed_b=t("vision.patch_embed.proj.bias"),
        blocks=blocks,
        norm_w=t("vision.norm.weight"),
        aligner_w1=t("aligner.w1.weight"),
        aligner_b1=t("aligner.w1.bias"),
        aligner_w2=t("aligner.w2.weight"),
        aligner_b2=t("aligner.w2.bias"),
    )


def _linear(x: mx.array, w: mx.array, b: mx.array | None = None) -> mx.array:
    y = x @ w.T
    return y + b if b is not None else y


def vision_rope_cos_sin(
    n_h: int,
    n_w: int,
    dim: int,
    theta: float,
) -> tuple[mx.array, mx.array]:
    """
    dim is head_dim // 2. Each position's `dim`-wide frequency row is the
    concatenation of its row-position frequencies (first dim//2) and its
    column-position frequencies (last dim//2) — matches the reference
    get_vision_cos_sin exactly, including that concatenation order.
    """
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    hpos = mx.broadcast_to(mx.arange(n_h, dtype=mx.float32)[:, None], (n_h, n_w))
    wpos = mx.broadcast_to(mx.arange(n_w, dtype=mx.float32)[None, :], (n_h, n_w))
    pos = mx.stack([hpos, wpos], axis=-1).reshape(n_h * n_w, 2, 1)
    freqs = (pos * inv_freq).reshape(n_h * n_w, dim)
    return mx.cos(freqs)[:, None, :], mx.sin(freqs)[:, None, :]


def apply_vision_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    dtype = x.dtype
    xf = x.astype(mx.float32)
    x1, x2 = mx.split(xf, 2, axis=-1)
    out = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return out.astype(dtype)


def vision_attention(
    x: mx.array,
    w: dict[str, mx.array],
    cos: mx.array,
    sin: mx.array,
    cfg: VisionConfig,
) -> mx.array:
    n = x.shape[0]
    head_dim = cfg.head_dim

    qkv = _linear(x, w["wqkv_w"], w["wqkv_b"])
    q, k, v = mx.split(qkv, 3, axis=-1)
    q = q.reshape(n, cfg.n_heads, head_dim)
    k = k.reshape(n, cfg.n_heads, head_dim)
    v = v.reshape(n, cfg.n_heads, head_dim)

    q = apply_vision_rotary(q, cos, sin)
    k = apply_vision_rotary(k, cos, sin)

    qh = mx.transpose(q, (1, 0, 2))[None]
    kh = mx.transpose(k, (1, 0, 2))[None]
    vh = mx.transpose(v, (1, 0, 2))[None]

    o = mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=head_dim**-0.5, mask=None)
    o = mx.transpose(o[0], (1, 0, 2)).reshape(n, -1)

    return _linear(o, w["wo_w"], w["wo_b"])


def vision_mlp(x: mx.array, w: dict[str, mx.array]) -> mx.array:
    gate, up = mx.split(_linear(x, w["mlp_w1"]), 2, axis=-1)
    return _linear(mx.sigmoid(gate) * gate * up, w["mlp_w2"])


def vision_block(
    x: mx.array,
    w: dict[str, mx.array],
    cos: mx.array,
    sin: mx.array,
    cfg: VisionConfig,
) -> mx.array:
    x = x + vision_attention(rms_norm(x, w["norm1_w"]), w, cos, sin, cfg)
    x = x + vision_mlp(rms_norm(x, w["norm2_w"]), w)
    return x


def vit_forward(
    patches: mx.array,
    n_h: int,
    n_w: int,
    weights: VisionWeights,
    cfg: VisionConfig = VisionConfig(),
) -> mx.array:
    """patches: [n_h * n_w, 3, patch_size, patch_size], row-major over the grid."""
    n = patches.shape[0]
    x = _linear(patches.reshape(n, -1), weights.patch_embed_w, weights.patch_embed_b)

    cos, sin = vision_rope_cos_sin(n_h, n_w, cfg.head_dim // 2, cfg.rope_theta)
    for w in weights.blocks:
        x = vision_block(x, w, cos, sin, cfg)

    return rms_norm(x, weights.norm_w)


def _gelu(x: mx.array) -> mx.array:
    xf = x.astype(mx.float32)
    y = 0.5 * xf * (1.0 + mx.erf(xf / mx.sqrt(mx.array(2.0, dtype=mx.float32))))
    return y.astype(x.dtype)


def aligner_forward(
    x: mx.array,
    n_h: int,
    n_w: int,
    weights: VisionWeights,
    cfg: VisionConfig = VisionConfig(),
) -> mx.array:
    """
    Space-to-depth downsample by `downsample_ratio` then a two-layer MLP into
    the text embedding space. The reshape/transpose below reproduces
    torch.nn.functional.unfold(kernel=r, stride=r)'s (channel, kh, kw)
    flattening and row-major block order exactly — there is no MLX unfold.
    """
    r = cfg.downsample_ratio
    dim = x.shape[-1]

    pad_h = (-n_h) % r
    pad_w = (-n_w) % r

    grid = mx.transpose(x.reshape(n_h, n_w, dim), (2, 0, 1))  # [dim, n_h, n_w]
    if pad_h or pad_w:
        grid = mx.pad(grid, [(0, 0), (0, pad_h), (0, pad_w)])

    hb, wb = (n_h + pad_h) // r, (n_w + pad_w) // r
    blocks = grid.reshape(dim, hb, r, wb, r)
    blocks = mx.transpose(blocks, (1, 3, 0, 2, 4)).reshape(hb * wb, dim * r * r)

    h = _gelu(_linear(blocks, weights.aligner_w1, weights.aligner_b1))
    return _linear(h, weights.aligner_w2, weights.aligner_b2)


def vision_embed(
    patches: mx.array,
    n_h: int,
    n_w: int,
    weights: VisionWeights,
    cfg: VisionConfig = VisionConfig(),
) -> mx.array:
    """ViT + Aligner in one call: image patches -> text-embedding-space rows."""
    return aligner_forward(vit_forward(patches, n_h, n_w, weights, cfg), n_h, n_w, weights, cfg)
