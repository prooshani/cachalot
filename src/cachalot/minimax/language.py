"""
MiniMax-M3 text backbone, adapted from the `minimax_m3.py` that ships with the MLX conversion
(pipenetwork/MiniMax-M3-MLX-3bit, an mlx-lm model file). Same modules and weight names; the
mlx-lm helpers it imported (`BaseModelArgs`, `create_attention_mask`,
`scaled_dot_product_attention`, `SwitchGLU`) are replaced by MLX calls, because mlx-lm is not
installed here and the routed experts are streamed by Cachalot instead (the MoE block's
`switch_mlp` is replaced before any weight is loaded, see `cachalot.minimax.model`).

M3: 60 layers (3 dense, 57 MoE with 128 routed experts top-4, sigmoid routing with a correction
bias, one shared expert, routed scaling 2.0), GQA 64 query / 4 KV heads of 128, per-head Gemma
QK-norm, partial RoPE (64 of 128 dims), Gemma RMSNorm, clamped SwiGLU-OAI. MiniMax Sparse
Attention is run as full causal attention (exact up to 2,048 tokens, the dense form beyond).
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    model_type: str
    hidden_size: int
    intermediate_size: int
    dense_intermediate_size: int
    shared_intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    num_local_experts: int
    num_experts_per_tok: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    vocab_size: int
    head_dim: int = 128
    max_position_embeddings: int = 1048576
    routed_scaling_factor: float = 2.0
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    scoring_func: str = "sigmoid"
    use_qk_norm: bool = True
    tie_word_embeddings: bool = False
    mlp_layer_types: list | None = None

    @classmethod
    def from_dict(cls, params: dict) -> ModelArgs:
        names = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in params.items() if k in names})


# HANDOFF 18.2: fewer GPU kernels per decode layer. FAST_NORM changes rounding only (same-text KL inside the
# model's own chunking noise); the two fusions are bit-identical and drop the originals (no extra memory).
FAST_NORM = os.environ.get("CACHALOT_MINIMAX_FAST_NORM", "1") != "0"
FUSE_QKV = os.environ.get("CACHALOT_MINIMAX_FUSE_QKV", "1") != "0"
FUSE_SHARED = os.environ.get("CACHALOT_MINIMAX_FUSE_SHARED", "1") != "0"


def _stack(parts: list) -> nn.QuantizedLinear | None:
    """Quantized linears on the same input stacked along their output rows into one: one kernel instead of
    len(parts), bit-identical rows (HANDOFF 18.2). None when they are not all quantized alike."""
    if not all(isinstance(p, nn.QuantizedLinear) and "bias" not in p for p in parts):
        return None
    first = parts[0]
    if any((p.group_size, p.bits, p.mode) != (first.group_size, first.bits, first.mode) for p in parts):
        return None
    fused = nn.QuantizedLinear(first.group_size, 1, bias=False, group_size=first.group_size, bits=first.bits,
                               mode=first.mode)
    fused.weight = mx.concatenate([p.weight for p in parts], axis=0)
    fused.scales = mx.concatenate([p.scales for p in parts], axis=0)
    fused.biases = mx.concatenate([p.biases for p in parts], axis=0)
    mx.eval(fused.parameters())
    return fused


class GemmaRMSNorm(nn.Module):
    """Normalize in fp32 and scale by ``weight + 1``."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x):
        ot = x.dtype
        if FAST_NORM:
            # one fused kernel in fp32 with the weight's `+ 1` precomputed once (per loaded weight)
            w = self.__dict__.get("_w1")
            if w is None or w[0] is not self.weight:
                w = (self.weight, (1.0 + self.weight.astype(mx.float32)))
                mx.eval(w[1])
                self.__dict__["_w1"] = w
            return mx.fast.rms_norm(x.astype(mx.float32), w[1], self.eps).astype(ot)
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)
        return (x * (1.0 + self.weight.astype(mx.float32))).astype(ot)


def swiglu_oai(x_gate, x_up, alpha: float, limit: float):
    """(clamp(up)+1) * gate*sigmoid(alpha*gate), gate clamped from above."""
    gate = mx.minimum(x_gate, limit)
    up = mx.clip(x_up, -limit, limit)
    return (up + 1.0) * (gate * mx.sigmoid(gate * alpha))



class SwiGLUOAI(nn.Module):
    """Activation for the routed experts: called as (x_up, x_gate), like mlx-lm's SwitchGLU does."""

    def __init__(self, alpha: float, limit: float):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def __call__(self, x_up, x_gate):
        return swiglu_oai(x_gate, x_up, self.alpha, self.limit)


class MiniMaxM3MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.alpha = args.swiglu_alpha
        self.limit = args.swiglu_limit
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def fuse(self) -> None:
        """After loading: gate and up as one matmul; the originals are dropped (no extra memory)."""
        fused = _stack([self.gate_proj, self.up_proj])
        if fused is not None:
            self.gate_up_proj = fused
            del self["gate_proj"], self["up_proj"]

    def __call__(self, x):
        if "gate_up_proj" in self:
            n = self.down_proj.weight.shape[1] * 32 // self.down_proj.bits
            y = self.gate_up_proj(x)
            gate, up = y[..., :n], y[..., n:]
            return self.down_proj(swiglu_oai(gate, up, self.alpha, self.limit))
        return self.down_proj(swiglu_oai(self.gate_proj(x), self.up_proj(x), self.alpha, self.limit))


class MiniMaxM3Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(args.hidden_size, self.num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * head_dim, args.hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)

    def fuse(self) -> None:
        """After loading: q, k and v as one matmul; the originals are dropped (no extra memory)."""
        fused = _stack([self.q_proj, self.k_proj, self.v_proj])
        if fused is not None:
            self.qkv_proj = fused
            del self["q_proj"], self["k_proj"], self["v_proj"]

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        if "qkv_proj" in self:
            nq, nk = self.num_attention_heads * self.head_dim, self.num_key_value_heads * self.head_dim
            y = self.qkv_proj(x)
            queries = y[..., :nq].reshape(B, L, self.num_attention_heads, self.head_dim)
            keys = y[..., nq:nq + nk].reshape(B, L, self.num_key_value_heads, self.head_dim)
            values = y[..., nq + nk:].reshape(B, L, self.num_key_value_heads, self.head_dim)
        else:
            queries = self.q_proj(x).reshape(B, L, self.num_attention_heads, self.head_dim)
            keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)
            values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MiniMaxM3SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor
        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))
        self.activation = SwiGLUOAI(args.swiglu_alpha, args.swiglu_limit)
        self.switch_mlp = None  # cachalot.glm.experts.StreamingSwitchGLU, set by the loader
        self.shared_experts = MiniMaxM3MLP(args, args.shared_intermediate_size)

    def route_scores(self, x):
        """Selection scores (sigmoid plus the correction bias) and the plain sigmoid used as weights."""
        scores = mx.sigmoid(self.gate(x.astype(mx.float32)))
        return scores + self.e_score_correction_bias, scores

    def __call__(self, x, residual=None):
        scores, orig_scores = self.route_scores(x)
        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(orig_scores, inds, axis=-1)
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = (weights * self.routed_scaling_factor).astype(x.dtype)
        decode_hook = getattr(self, "decode_hook", None)
        if decode_hook is not None and x.shape[1] == 1:
            # one decode token (cachalot.minimax.model): routing, the next layer's predicted routing and
            # the shared expert are queued so the GPU runs them while the routed misses are read
            shared, prefetch = decode_hook(x, residual, inds, weights)
            y = self.switch_mlp(x, inds, prefetch=prefetch)
            y = (y * weights[..., None]).sum(axis=-2)
            return y + shared
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2)
        return y + self.shared_experts(x)


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = MiniMaxM3Attention(args)
        types = args.mlp_layer_types or ["sparse"] * args.num_hidden_layers
        self.is_sparse = types[layer_idx] == "sparse"
        if self.is_sparse:
            self.block_sparse_moe = MiniMaxM3SparseMoeBlock(args)
        else:
            self.mlp = MiniMaxM3MLP(args, args.dense_intermediate_size)
        self.input_layernorm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        if self.is_sparse:
            return r + self.block_sparse_moe(self.post_attention_layernorm(r), residual=r)
        return r + self.mlp(self.post_attention_layernorm(r))


class MiniMaxM3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [MiniMaxM3DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        # one token attends to everything cached; a chunk is causal, aligned to the end of the keys
        mask = "causal" if h.shape[1] > 1 else None
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = MiniMaxM3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers
