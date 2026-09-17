"""
DSpark draft head: the `mtp.*` stage of DeepSeek-V4.1-Flash.

The checkpoint carries three complete draft blocks under the `mtp.N` namespace,
7.39 GiB in total, which the text runtime has never loaded. The model card calls
them "DSpark speculative decoding (semi-autoregressive draft generation with
confidence-scheduled verification)", and `inference/model.py` in the checkpoint
implements their forward pass -- `DSparkBlock`, `DSparkAttention`,
`DSparkMarkovHead`, `DSparkConfidenceHead` and `Transformer.forward_spec` --
while never calling it.

What the draft does, given the main model's forward at position p:

  * `main_hidden` is the concatenated *input* of layers 37, 38 and 39, meaned
    over the hyper-connection copies: [3 * 5120]. The runtime produces it when
    `TextDecodeRuntime.capture_main_hidden` is set.
  * `main_proj` and `main_norm` (on stage 0 only) collapse that to one [5120]
    vector, `main_x`, which is the *only* thing the draft sees of the main model.
  * Each stage keeps its own 128-entry sliding-window KV cache, filled from
    `main_x` -- not from the main model's own KV. One entry per decoded
    position, at slot `p % 128`.
  * The draft input is `block_size` = 5 token embeddings: the token the main
    model just produced, followed by four copies of the noise token 128799.
    Those five positions attend to the whole window *and to each other,
    bidirectionally*: `get_dspark_topk_idxs` hands every query the same index
    set, so this is a block, not a causal suffix.
  * The three stages run in order, then the shared head produces logits for all
    five positions at once. A rank-256 Markov head re-introduces the sequential
    dependency cheaply: position i's logits are corrected by a bigram term from
    the token sampled at position i-1, so the five tokens are sampled in order
    but through one transformer pass rather than five.
  * A confidence head scores each drafted position, which is what
    "confidence-scheduled verification" would consume.

So one main forward yields up to five speculative tokens. This module computes
them; deciding whether they are worth verifying is what
`benchmarks/dspark_acceptance.py` measures.

Numerics follow the runtime's own prefill path exactly: the same FP8 activation
quantization, the same RMSNorm, the same RoPE tables, the same sparse attention
and the same FP4 expert math. Nothing here is approximated for speed, because
the first question this has to answer is a correctness question.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from cachalot.model.fp4_mlx import dequantize_fp4_weight
from cachalot.model.hc_prefill_exact import hc_mixes_prefill_exact
from cachalot.model.hyper_connection_mlx import hc_post, hc_pre
from cachalot.model.moe_prefill_batched import (
    fp8_linear_rows,
    quantize_activation_fp8_rows,
    route_topk_rows,
    routed_expert_forward_batched,
    shared_expert_forward_batched,
)
import mlx.nn as nn

from cachalot.model.norm_rope_mlx import apply_rotary_emb, rms_norm
from cachalot.model.sparse_attn_mlx import sparse_attention
from cachalot.model.wo_a_dequant import dequantize_wo_a
from cachalot.storage.tensor_index import build_tensor_index
from cachalot.storage.tensor_loader import load_resident_tensor

# From text_config in the checkpoint's config.json.
N_MTP_LAYERS = 3
BLOCK_SIZE = 5
NOISE_TOKEN_ID = 128799
N_ROUTED_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 3
MARKOV_RANK = 256
ROUTED_SCALING_FACTOR = 1.5

DIM = 5120
INTERMEDIATE = 2304
HC_MULT = 4
WINDOW_SIZE = 128
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
N_HEADS = 64
N_GROUPS = 8
O_LORA_RANK = 1024
NORM_EPS = 1e-20
HC_EPS = 1e-6
HC_SINKHORN_ITERS = 20
SWIGLU_LIMIT = 10.0

SOFTMAX_SCALE = HEAD_DIM ** -0.5


def dspark_topk_idxs(start_pos: int) -> mx.array:
    """The key set every draft query attends to, as get_dspark_topk_idxs builds it.

    Rows 0 .. min(WINDOW_SIZE, start_pos + 1) - 1 are the filled window slots --
    a position p occupies slot p % WINDOW_SIZE, so while the window is not yet
    full those are slots 0 .. start_pos, and once it is full they are all of
    them. Rows WINDOW_SIZE .. WINDOW_SIZE + BLOCK_SIZE - 1 are the draft block's
    own positions, appended after the window in the attention KV.

    Every draft query gets the same set, itself included: the block is attended
    bidirectionally, not causally. That is what makes the draft semi-
    autoregressive, and it is why the Markov head exists to put the sequential
    dependency back.
    """
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    valid = min(WINDOW_SIZE, start_pos + 1)
    idxs = mx.concatenate(
        [
            mx.arange(valid, dtype=mx.int32),
            WINDOW_SIZE + mx.arange(BLOCK_SIZE, dtype=mx.int32),
        ]
    )
    return mx.broadcast_to(idxs[None, :], (BLOCK_SIZE, valid + BLOCK_SIZE))


def is_mtp_tensor(name: str) -> bool:
    return name.startswith("mtp.")


def is_mtp_expert_tensor(name: str) -> bool:
    return is_mtp_tensor(name) and ".ffn.experts." in name


@dataclass
class DraftResult:
    """One draft block for the position that follows `anchor_token`."""

    tokens: list[int]          # BLOCK_SIZE drafted ids, the first following anchor_token
    logits: mx.array           # [BLOCK_SIZE, vocab] fp32, Markov correction included
    confidence: mx.array       # [BLOCK_SIZE] fp32
    anchor_token: int


@dataclass
class DSparkDraft:
    """The three `mtp.*` stages, their window caches, and the heads on top.

    Weights other than the routed experts are loaded eagerly (about 670 MiB);
    the 384 routed experts are read on first use and kept, which is 6.72 GiB if
    the draft ends up touching all of them.
    """

    model_path: Path
    embed_weight: mx.array
    head_weight: mx.array
    rope_cos: mx.array
    rope_sin: mx.array

    # Optional: a packer with mx.quantize's signature, applied to every routed
    # expert once at first use. The FP4 experts the checkpoint ships are repacked
    # into MLX's 8-bit affine layout on *every* matmul, which measured 0.39 ms per
    # projection on top of a 0.55 ms product; a bank quantized once costs neither.
    # A draft's errors are rejected rather than emitted, so it can afford a
    # coarser format than the model it drafts for -- but whether it still agrees
    # often enough is a measurement, not an assumption.
    expert_quantizer: object | None = None
    expert_bits: int = 2
    expert_group: int = 128

    tensors: dict[str, mx.array] = field(default_factory=dict)
    windows: list[mx.array] = field(default_factory=list)
    _expert_cache: dict[tuple[int, int, str], mx.array] = field(default_factory=dict)
    _quantized: dict[tuple[int, int], dict[str, tuple[mx.array, mx.array, mx.array]]] = field(
        default_factory=dict
    )
    _expert_index: dict = field(default_factory=dict)
    _wo_a: dict[int, mx.array] = field(default_factory=dict)

    # --------------------------------------------------------------- loading
    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        embed_weight: mx.array,
        head_weight: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
        expert_quantizer: object | None = None,
        expert_bits: int = 2,
        expert_group: int = 128,
        verbose: bool = False,
    ) -> "DSparkDraft":
        model_path = Path(model_path)
        index = build_tensor_index(model_path)

        tensors: dict[str, mx.array] = {}
        expert_index: dict = {}

        for name, entry in index.items():
            if not is_mtp_tensor(name):
                continue
            if is_mtp_expert_tensor(name):
                expert_index[name] = entry
                continue
            tensors[name] = load_resident_tensor(entry).data

        if not tensors:
            raise KeyError(f"no mtp.* tensors under {model_path}")

        mx.eval(list(tensors.values()))

        if verbose:
            total = sum(t.nbytes for t in tensors.values())
            print(
                f"DSpark draft: {len(tensors)} dense tensors ({total / 2**20:.0f} MiB), "
                f"{len(expert_index)} expert tensors held on disk"
            )

        draft = cls(
            model_path=model_path,
            embed_weight=embed_weight,
            head_weight=head_weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            tensors=tensors,
            expert_quantizer=expert_quantizer,
            expert_bits=expert_bits,
            expert_group=expert_group,
            _expert_index=expert_index,
        )
        draft.reset()
        return draft

    def reset(self) -> None:
        """Clear the per-stage window caches. Call between prompts."""
        self.windows = [
            mx.zeros((WINDOW_SIZE, HEAD_DIM), dtype=mx.bfloat16)
            for _ in range(N_MTP_LAYERS)
        ]

    # --------------------------------------------------------------- tensors
    def _t(self, stage: int, suffix: str) -> mx.array:
        name = f"mtp.{stage}.{suffix}"
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"missing draft tensor {name!r}") from exc

    def _wo_a_bf16(self, stage: int) -> mx.array:
        value = self._wo_a.get(stage)
        if value is None:
            value = dequantize_wo_a(
                self._t(stage, "attn.wo_a.weight"),
                self._t(stage, "attn.wo_a.scale"),
            )
            mx.eval(value)
            self._wo_a[stage] = value
        return value

    def _expert(self, stage: int, expert_id: int, part: str) -> mx.array:
        key = (stage, expert_id, part)
        value = self._expert_cache.get(key)
        if value is None:
            name = f"mtp.{stage}.ffn.experts.{expert_id}.{part}"
            entry = self._expert_index.get(name)
            if entry is None:
                raise KeyError(f"missing draft expert tensor {name!r}")
            value = load_resident_tensor(entry).data
            mx.eval(value)
            self._expert_cache[key] = value
        return value

    def _quantized_expert(
        self, stage: int, expert_id: int
    ) -> dict[str, tuple[mx.array, mx.array, mx.array]]:
        """Quantize one routed expert once, and keep it in the packer's format."""
        key = (stage, expert_id)
        packed = self._quantized.get(key)
        if packed is None:
            packed = {}
            for proj, (out_features, in_features) in (
                ("w1", (INTERMEDIATE, DIM)),
                ("w3", (INTERMEDIATE, DIM)),
                ("w2", (DIM, INTERMEDIATE)),
            ):
                dense = dequantize_fp4_weight(
                    self._expert(stage, expert_id, f"{proj}.weight"),
                    self._expert(stage, expert_id, f"{proj}.scale"),
                ).astype(mx.float32).reshape(out_features, in_features)
                packed[proj] = self.expert_quantizer(dense, group_size=self.expert_group)
                mx.eval(*packed[proj])
            self._quantized[key] = packed
            # The FP4 source is no longer needed once the expert is packed.
            for proj in ("w1", "w2", "w3"):
                self._expert_cache.pop((stage, expert_id, f"{proj}.weight"), None)
                self._expert_cache.pop((stage, expert_id, f"{proj}.scale"), None)
        return packed

    def _quantized_expert_forward(
        self,
        packed: dict[str, tuple[mx.array, mx.array, mx.array]],
        x: mx.array,
        weights: mx.array,
    ) -> mx.array:
        """expert_affine.affine_expert_forward_batched over arrays held in memory."""
        def qmm(value: mx.array, proj: str) -> mx.array:
            w, scales, biases = packed[proj]
            return mx.quantized_matmul(
                value, w, scales, biases,
                transpose=True, group_size=self.expert_group, bits=self.expert_bits,
            )

        xb = x.astype(mx.bfloat16)
        gate = qmm(xb, "w1").astype(mx.bfloat16).astype(mx.float32)
        up = qmm(xb, "w3").astype(mx.bfloat16).astype(mx.float32)
        up = mx.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        gate = mx.minimum(gate, mx.array(SWIGLU_LIMIT, dtype=mx.float32))
        hidden = (
            nn.silu(gate) * up * weights.astype(mx.float32)[:, None]
        ).astype(mx.bfloat16)
        return qmm(hidden, "w2").astype(mx.bfloat16).astype(mx.float32)

    def resident_expert_bytes(self) -> int:
        total = sum(t.nbytes for t in self._expert_cache.values())
        for packed in self._quantized.values():
            for arrays in packed.values():
                total += sum(a.nbytes for a in arrays)
        return total

    # ------------------------------------------------------------- main path
    def main_x(self, main_hidden: mx.array) -> mx.array:
        """[.., DIM * 3] -> [.., DIM]: stage 0's projection of the main model's hidden state."""
        rows = main_hidden.reshape(-1, main_hidden.shape[-1])
        projected = fp8_linear_rows(
            rows,
            self._t(0, "main_proj.weight"),
            self._t(0, "main_proj.scale"),
        )
        normed = rms_norm(projected, self._t(0, "main_norm.weight"), eps=NORM_EPS)
        return normed.reshape(*main_hidden.shape[:-1], DIM)

    def _window_kv(self, stage: int, main_x_rows: mx.array, positions: mx.array) -> mx.array:
        """KV entries a stage stores for main-model positions. main_x_rows: [n, DIM]."""
        kv = fp8_linear_rows(
            main_x_rows,
            self._t(stage, "attn.wkv.weight"),
            self._t(stage, "attn.wkv.scale"),
        )
        kv = rms_norm(kv, self._t(stage, "attn.kv_norm.weight"), eps=NORM_EPS)
        cos = mx.take(self.rope_cos, positions, axis=0)
        sin = mx.take(self.rope_sin, positions, axis=0)
        kv = mx.concatenate(
            [kv[:, :-ROPE_HEAD_DIM], apply_rotary_emb(kv[:, -ROPE_HEAD_DIM:], cos, sin)],
            axis=-1,
        )
        return quantize_activation_fp8_rows(kv, mx.bfloat16)

    def observe(self, main_hidden: mx.array, start_pos: int) -> None:
        """Record the main model's positions `start_pos ...` in every stage's window.

        `main_hidden` is [DIM * 3] for one position or [n, DIM * 3] for a prefill
        chunk. Only the last WINDOW_SIZE positions can survive, which is exactly
        what the ring cache keeps.
        """
        if main_hidden.ndim == 1:
            main_hidden = main_hidden[None, :]
        n = main_hidden.shape[0]

        keep = min(n, WINDOW_SIZE)
        first = n - keep
        positions = mx.arange(start_pos + first, start_pos + n, dtype=mx.int32)

        rows = self.main_x(main_hidden)[first:]
        slots = [(start_pos + first + i) % WINDOW_SIZE for i in range(keep)]

        for stage in range(N_MTP_LAYERS):
            kv = self._window_kv(stage, rows, positions)
            window = self.windows[stage]
            for i, slot in enumerate(slots):
                window = mx.concatenate(
                    [window[:slot], kv[i][None, :], window[slot + 1:]],
                    axis=0,
                )
            mx.eval(window)
            self.windows[stage] = window

    # -------------------------------------------------------------- the block
    def _attention(
        self,
        stage: int,
        x: mx.array,
        *,
        start_pos: int,
    ) -> mx.array:
        """DSparkAttention for the BLOCK_SIZE draft positions after `start_pos`."""
        wq_a = self._t(stage, "attn.wq_a.weight")
        wq_a_s = self._t(stage, "attn.wq_a.scale")
        wq_b = self._t(stage, "attn.wq_b.weight")
        wq_b_s = self._t(stage, "attn.wq_b.scale")

        qr = fp8_linear_rows(x, wq_a, wq_a_s)
        qr = rms_norm(qr, self._t(stage, "attn.q_norm.weight"), eps=NORM_EPS)
        q = fp8_linear_rows(qr, wq_b, wq_b_s).reshape(BLOCK_SIZE, N_HEADS, HEAD_DIM)

        draft_positions = mx.arange(
            start_pos + 1, start_pos + 1 + BLOCK_SIZE, dtype=mx.int32
        )
        cos = mx.take(self.rope_cos, draft_positions, axis=0)
        sin = mx.take(self.rope_sin, draft_positions, axis=0)

        q_rope = apply_rotary_emb(q[None, :, :, -ROPE_HEAD_DIM:], cos, sin)[0]
        q = mx.concatenate([q[:, :, :-ROPE_HEAD_DIM], q_rope], axis=-1)

        kv = fp8_linear_rows(
            x,
            self._t(stage, "attn.wkv.weight"),
            self._t(stage, "attn.wkv.scale"),
        )
        kv = rms_norm(kv, self._t(stage, "attn.kv_norm.weight"), eps=NORM_EPS)
        kv = mx.concatenate(
            [kv[:, :-ROPE_HEAD_DIM], apply_rotary_emb(kv[:, -ROPE_HEAD_DIM:], cos, sin)],
            axis=-1,
        )
        kv = quantize_activation_fp8_rows(kv, mx.bfloat16)

        topk_idxs = dspark_topk_idxs(start_pos)

        attention_kv = mx.concatenate([self.windows[stage], kv], axis=0)

        o = sparse_attention(
            q.astype(mx.bfloat16),
            attention_kv,
            self._t(stage, "attn.attn_sink"),
            topk_idxs,
            SOFTMAX_SCALE,
        )

        o_rope = apply_rotary_emb(
            o[None, :, :, -ROPE_HEAD_DIM:], cos, sin, inverse=True
        )[0]
        o = mx.concatenate([o[:, :, :-ROPE_HEAD_DIM], o_rope], axis=-1)

        group_input_dim = (N_HEADS * HEAD_DIM) // N_GROUPS
        grouped_o = o.reshape(BLOCK_SIZE, N_GROUPS, group_input_dim)
        grouped_wo_a = self._wo_a_bf16(stage).reshape(
            N_GROUPS, O_LORA_RANK, group_input_dim
        )
        low_rank = mx.einsum("sgd,grd->sgr", grouped_o, grouped_wo_a)

        return fp8_linear_rows(
            low_rank.reshape(BLOCK_SIZE, N_GROUPS * O_LORA_RANK),
            self._t(stage, "attn.wo_b.weight"),
            self._t(stage, "attn.wo_b.scale"),
        )

    def _moe(self, stage: int, x: mx.array) -> mx.array:
        """MoE with the draft's own 128 experts, top-3, plus the shared expert."""
        indices, weights, _ = route_topk_rows(
            x,
            self._t(stage, "ffn.gate.weight"),
            self._t(stage, "ffn.gate.bias"),
            topk=NUM_EXPERTS_PER_TOK,
            route_scale=ROUTED_SCALING_FACTOR,
        )
        mx.eval(indices, weights)

        chosen = indices.tolist()
        chosen_weights = weights.tolist()

        # Expert-major: one call per distinct expert over the rows that chose it,
        # which is how the prefill path schedules routed work.
        by_expert: dict[int, list[tuple[int, float]]] = {}
        for row, (ids, ws) in enumerate(zip(chosen, chosen_weights, strict=True)):
            for expert_id, weight in zip(ids, ws, strict=True):
                by_expert.setdefault(int(expert_id), []).append((row, float(weight)))

        row_terms: list[list[mx.array]] = [[] for _ in range(BLOCK_SIZE)]
        for expert_id, hits in sorted(by_expert.items()):
            rows = mx.array([row for row, _ in hits], dtype=mx.int32)
            expert_weights = mx.array([w for _, w in hits], dtype=mx.float32)
            if self.expert_quantizer is not None:
                y = self._quantized_expert_forward(
                    self._quantized_expert(stage, expert_id),
                    mx.take(x, rows, axis=0),
                    expert_weights,
                )
            else:
                y = routed_expert_forward_batched(
                    mx.take(x, rows, axis=0),
                    w1_packed=self._expert(stage, expert_id, "w1.weight"),
                    w1_scales=self._expert(stage, expert_id, "w1.scale"),
                    w2_packed=self._expert(stage, expert_id, "w2.weight"),
                    w2_scales=self._expert(stage, expert_id, "w2.scale"),
                    w3_packed=self._expert(stage, expert_id, "w3.weight"),
                    w3_scales=self._expert(stage, expert_id, "w3.scale"),
                    weights=expert_weights,
                    swiglu_limit=SWIGLU_LIMIT,
                )
            for i, (row, _) in enumerate(hits):
                row_terms[row].append(y[i])

        routed = mx.stack(
            [
                mx.sum(mx.stack(terms, axis=0), axis=0)
                if terms
                else mx.zeros((DIM,), dtype=mx.float32)
                for terms in row_terms
            ],
            axis=0,
        )

        shared = shared_expert_forward_batched(
            x.astype(mx.bfloat16),
            w1=self._t(stage, "ffn.shared_experts.w1.weight"),
            w1_scales=self._t(stage, "ffn.shared_experts.w1.scale"),
            w2=self._t(stage, "ffn.shared_experts.w2.weight"),
            w2_scales=self._t(stage, "ffn.shared_experts.w2.scale"),
            w3=self._t(stage, "ffn.shared_experts.w3.weight"),
            w3_scales=self._t(stage, "ffn.shared_experts.w3.scale"),
            swiglu_limit=SWIGLU_LIMIT,
        )

        return (routed + shared.astype(mx.float32)).astype(x.dtype)

    def _block(
        self,
        stage: int,
        x: mx.array,
        pre_mix: mx.array,
        *,
        start_pos: int,
    ) -> tuple[mx.array, mx.array]:
        """Block.forward from the reference, with DSpark's attention."""
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes_prefill_exact(
            x,
            self._t(stage, "hc_attn_fn"),
            self._t(stage, "hc_attn_scale"),
            self._t(stage, "hc_attn_base"),
            norm_eps=NORM_EPS,
            hc_mult=HC_MULT,
            sinkhorn_iters=HC_SINKHORN_ITERS,
            hc_eps=HC_EPS,
        )
        y = hc_pre(x, pre_mix)
        y = rms_norm(y, self._t(stage, "attn_norm.weight"), eps=NORM_EPS)
        y = self._attention(stage, y, start_pos=start_pos)
        x = hc_post(y, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes_prefill_exact(
            x,
            self._t(stage, "hc_ffn_fn"),
            self._t(stage, "hc_ffn_scale"),
            self._t(stage, "hc_ffn_base"),
            norm_eps=NORM_EPS,
            hc_mult=HC_MULT,
            sinkhorn_iters=HC_SINKHORN_ITERS,
            hc_eps=HC_EPS,
        )
        y = hc_pre(x, attn_pre)
        y = rms_norm(y, self._t(stage, "ffn_norm.weight"), eps=NORM_EPS)
        y = self._moe(stage, y)
        x = hc_post(y, residual, ffn_post, ffn_comb)

        return x, ffn_pre

    # -------------------------------------------------------------- the head
    def _markov_bias_at(self, token: mx.array) -> tuple[mx.array, mx.array]:
        """Bigram correction for a token held on the GPU, with no round trip.

        The five positions are sampled in order, so each one's correction needs
        the token sampled before it -- but it needs its *embedding*, not its
        value, and mx.take indexes with an array. Keeping the token as a
        zero-dimensional array instead of a Python int removes five
        GPU-to-CPU synchronizations from every draft block, which is most of
        what the block costs once the experts are quantized.
        """
        head = self._t(2, "markov_head.head.weight")
        embed = mx.take(self._t(2, "markov_head.embed.weight"), token, axis=0)
        bias = mx.matmul(head, embed.astype(head.dtype)).astype(mx.float32)
        return bias, embed.astype(mx.float32)

    def _markov_bias(self, token_id: int) -> tuple[mx.array, mx.array]:
        """Bigram correction for one drafted position: a rank-256 lookup and matmul.

        Both matmuls stay in bf16. Casting the [vocab, 256] head to fp32 on every
        call costs 132 MiB of conversion per position and five positions per
        block, which measured as most of the Markov correction's time; MLX
        accumulates a bf16 matmul in fp32 anyway, and this is a draft whose
        wrong guesses are rejected rather than emitted.
        """
        embed = self._t(2, "markov_head.embed.weight")[token_id]
        head = self._t(2, "markov_head.head.weight")
        bias = mx.matmul(head, embed.astype(head.dtype)).astype(mx.float32)
        return bias, embed.astype(mx.float32)

    def draft(
        self,
        main_hidden: mx.array,
        anchor_token: int,
        start_pos: int,
        *,
        temperature: float = 0.0,
    ) -> DraftResult:
        """Draft BLOCK_SIZE tokens following `anchor_token`.

        `main_hidden` and `start_pos` belong to the main model's forward that
        *produced* `anchor_token`, so `anchor_token` sits at `start_pos + 1` and
        the draft covers `start_pos + 2 ... start_pos + 1 + BLOCK_SIZE`.
        """
        if start_pos < 1:
            raise ValueError("DSpark drafts only after at least one main position")

        self.observe(main_hidden, start_pos)

        draft_ids = [anchor_token] + [NOISE_TOKEN_ID] * (BLOCK_SIZE - 1)
        x = mx.take(self.embed_weight, mx.array(draft_ids, dtype=mx.int32), axis=0)
        x = mx.broadcast_to(x[:, None, :], (BLOCK_SIZE, HC_MULT, DIM)).astype(
            self.embed_weight.dtype
        )

        pre_mix = mx.zeros((BLOCK_SIZE, HC_MULT), dtype=mx.float32)
        pre_mix = mx.concatenate(
            [mx.ones((BLOCK_SIZE, 1), dtype=mx.float32), pre_mix[:, 1:]], axis=-1
        )

        for stage in range(N_MTP_LAYERS):
            x, pre_mix = self._block(stage, x, pre_mix, start_pos=start_pos)
            mx.eval(x, pre_mix)

        collapsed = hc_pre(x, pre_mix)
        normed = rms_norm(collapsed, self._t(2, "norm.weight"), eps=NORM_EPS)
        # The shared head is [129280, 5120] bf16. Converting it to fp32 for the
        # product, as the reference module does once at load time, is 2.65 GiB of
        # conversion per draft block here; the product itself is five rows.
        logits = mx.matmul(
            normed.astype(self.head_weight.dtype), self.head_weight.T
        ).astype(mx.float32)
        mx.eval(logits)

        # Confidence-scheduled sampling: one bigram correction per position,
        # applied in order, so the block is sampled sequentially but computed once.
        sampled: list[mx.array] = []
        previous = mx.array(anchor_token, dtype=mx.int32)
        corrected_rows = []
        markov_embeds = []
        for _ in range(BLOCK_SIZE):
            bias, embed = self._markov_bias_at(previous)
            row = logits[len(sampled)] + bias
            if temperature <= 0:
                token = mx.argmax(row).astype(mx.int32)
            else:
                scaled = row / temperature
                gumbel = -mx.log(
                    -mx.log(mx.random.uniform(shape=scaled.shape) + 1e-20) + 1e-20
                )
                token = mx.argmax(scaled + gumbel).astype(mx.int32)
            sampled.append(token)
            corrected_rows.append(row)
            markov_embeds.append(embed)
            previous = token

        markov_embed = mx.stack(markov_embeds, axis=0)
        token_ids = mx.stack(sampled, axis=0)
        mx.eval(token_ids)
        tokens = [int(t) for t in token_ids.tolist()]
        confidence_input = mx.concatenate(
            [collapsed.astype(mx.float32), markov_embed], axis=-1
        )
        # The confidence projection is stored fp32 and is [1, 5376]: small enough
        # that its precision costs nothing, and it is the score a verifier gates on.
        confidence = mx.matmul(
            confidence_input,
            self._t(2, "confidence_head.proj.weight").astype(mx.float32).T,
        ).reshape(-1)
        mx.eval(confidence)

        return DraftResult(
            tokens=tokens,
            logits=mx.stack(corrected_rows, axis=0),
            confidence=confidence,
            anchor_token=anchor_token,
        )
