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


class GemmaRMSNorm(nn.Module):
    """Normalize in fp32 and scale by ``weight + 1``."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x):
        ot = x.dtype
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

    def __call__(self, x):
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

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
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

    def __call__(self, x):
        gates = self.gate(x.astype(mx.float32))
        scores = mx.sigmoid(gates)
        orig_scores = scores
        scores = scores + self.e_score_correction_bias
        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(orig_scores, inds, axis=-1)
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = (weights * self.routed_scaling_factor).astype(x.dtype)
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
        mlp = self.block_sparse_moe if self.is_sparse else self.mlp
        return r + mlp(self.post_attention_layernorm(r))


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
