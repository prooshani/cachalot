"""
MiniMax-M3 on Cachalot: the conversion's model code (`cachalot.minimax.language`) for everything but
the routed experts, which stream from SSD through the wired expert store exactly as GLM's do.

`MiniMaxModel` subclasses `GlmModel` and replaces only what differs: loading (stacked experts, a
plain KV cache per layer, the conversion's 8-bit routers), the forward call, and the chat format
(`<mm:think>` reasoning, `]<]minimax[>[`-namespaced XML tool calls). Prefill (the scan-resistant
expert path), the prefix cache, disk snapshots, `stream` and `generate` are GLM's.

Non-expert weights: 5.3 GiB resident. Routed experts: 57 layers x 128 at 24.8 MB (3-bit), 168 GiB.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.glm.engine import _GlmSplitter
from cachalot.glm.experts import StreamingSwitchGLU, tensor_sizes
from cachalot.glm.model import GlmModel, load_non_expert_weights
from cachalot.minimax.experts import build_minimax_expert_index
from cachalot.minimax.language import Model, ModelArgs, MiniMaxM3SparseMoeBlock
from cachalot.storage.reader import ExpertReader
from cachalot.third_party.mlx_vlm.models.cache import KVCache

NS = "]<]minimax[>["
THINK_BEGIN = "<mm:think>"
THINK_END = "</mm:think>"
TOOL_START = NS + "<tool_call>"


class MiniMaxSplitter(_GlmSplitter):
    THINK_END = THINK_END
    TOOL_START = TOOL_START


_INVOKE_RE = re.compile(r'<invoke name="([^"]*)">(.*?)</invoke>', re.S)
_OPEN_RE = re.compile(r"<([A-Za-z_][\w.\-]*)>")


def _children(xml: str) -> list[tuple[str, str]] | None:
    """Top-level `<tag>...</tag>` elements of `xml`, or None when it is not a sequence of elements."""
    out, pos, xml = [], 0, xml.strip()
    while pos < len(xml):
        m = _OPEN_RE.match(xml, pos)
        if not m:
            return None
        tag, depth, i = m[1], 1, m.end()
        opener, closer = f"<{tag}>", f"</{tag}>"
        while depth:
            a, b = xml.find(opener, i), xml.find(closer, i)
            if b < 0:
                return None
            if 0 <= a < b:
                depth, i = depth + 1, a + len(opener)
            else:
                depth, i = depth - 1, b + len(closer)
        out.append((tag, xml[m.end():i - len(closer)]))
        pos = i
        while pos < len(xml) and xml[pos].isspace():
            pos += 1
    return out


def _value(raw: str, schema: dict | None):
    """A tool argument from the template's XML: nested elements become dicts or lists (<item>),
    leaves are typed by the tool's JSON schema (a string stays a string)."""
    schema = schema or {}
    kind = schema.get("type")
    kids = _children(raw) if "<" in raw else None
    if kids:
        if all(tag == "item" for tag, _ in kids):
            return [_value(v, schema.get("items")) for _, v in kids]
        props = schema.get("properties") or {}
        return {tag: _value(v, props.get(tag)) for tag, v in kids}
    if kind == "string":
        return raw
    if kind in ("integer", "number", "boolean", "null", "array", "object") or not schema:
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def parse_minimax_tool_calls(text: str, tools=None) -> list[dict[str, Any]]:
    """`]<]minimax[>[<tool_call> ... <invoke name="f"><k>v</k></invoke> ... </tool_call>` to OpenAI calls."""
    schemas = {}
    for tool in tools or []:
        fn = tool.get("function", tool)
        schemas[fn.get("name")] = fn.get("parameters") or {}
    plain = text.replace(NS, "")
    start = plain.find("<tool_call>")
    if start < 0:
        return []
    calls = []
    for name, body in _INVOKE_RE.findall(plain[start:]):
        params = schemas.get(name, {}).get("properties") or {}
        args = {tag: _value(v, params.get(tag)) for tag, v in (_children(body) or [])}
        calls.append({"type": "function", "function": {"name": name, "arguments": args}})
    return calls


def _wanted(name: str) -> bool:
    return ".switch_mlp." not in name and ".mtp." not in name and not name.startswith("mtp.")


# HANDOFF 18.1: the decode-step overlap and one-layer-early routing prediction
DECODE_OVERLAP = os.environ.get("CACHALOT_MINIMAX_DECODE_OVERLAP", "1") != "0"
PREDICT_TOPK = int(os.environ.get("CACHALOT_MINIMAX_PREDICT_TOPK", "0"))
DECODE_BORROW = os.environ.get("CACHALOT_MINIMAX_DECODE_BORROW", "1") != "0"
DECODE_BORROW_KEEP = 16  # transient slots never borrowed (the store's PREDICT_SLOT_RESERVE)


class MiniMaxModel(GlmModel):
    # A 2,048-token chunk already reads nearly every routed expert (168 GiB), so a longer chunk reads the same
    # bytes for more tokens: 8,192 prefills at ~230 tok/s against ~74-85 at 2,048, same NLL (HANDOFF 18.1).
    PREFILL_CHUNK = int(os.environ.get("CACHALOT_MINIMAX_PREFILL_CHUNK", "8192"))

    def __init__(
        self,
        model_path,
        *,
        expert_budget_gib: float = 52.0,
        wired_limit_gib: float | None = None,
        load_workers: int = 8,
        heartbeat_seconds: float = 0.5,
        verbose: bool = True,
    ) -> None:
        t0 = time.perf_counter()
        self.model_path = Path(model_path)
        config = json.loads((self.model_path / "config.json").read_text())
        self.config = ModelArgs.from_dict(config)
        quant = config.get("quantization") or config.get("quantization_config") or {}

        self.expert_format, self.expert_index = build_minimax_expert_index(self.model_path)
        sizes = tensor_sizes(self.expert_format)
        expert_bytes = sum(sizes.values())

        # the same memory rules as GLM (HANDOFF 17.1): a capped MLX buffer cache, a wired set
        mx.set_cache_limit(int(float(os.environ.get("CACHALOT_GLM_MLX_CACHE_GIB", "2")) * 1024**3))
        if wired_limit_gib is None:
            wired_limit_gib = float(os.environ.get("CACHALOT_MLX_WIRED_LIMIT_GIB", "80"))
        if wired_limit_gib > 0:
            from cachalot.config import device_memory

            _, recommended = device_memory()
            mx.set_wired_limit(int(min(wired_limit_gib * 1024**3, recommended)))

        n_experts = self.config.num_local_experts
        self.store = ResidentExpertStore(
            int(expert_budget_gib * 1024**3),
            ExpertReader(),
            tensor_sizes=sizes,
            # one prefill layer's misses plus the next layer read early (glm.experts.PREFILL_SCAN)
            transient_slots=2 * n_experts + 16,
            load_workers=load_workers,
            verbose=verbose,
        )
        self.store.format = self.expert_format
        # decode holds the prefill transient slots as residents until the next prefill (HANDOFF 18.1)
        self.store.decode_borrow = max(0, self.store.transient_slots - DECODE_BORROW_KEEP) if DECODE_BORROW else 0

        self.model = Model(self.config)
        n_moe = 0
        for i, layer in enumerate(self.model.layers):
            if layer.is_sparse:
                moe: MiniMaxM3SparseMoeBlock = layer.block_sparse_moe
                moe.switch_mlp = StreamingSwitchGLU(i, self.store, self.expert_index, self.expert_format, moe.activation)
                n_moe += 1

        self._install_decode_hooks()

        weights = load_non_expert_weights(self.model_path, _wanted)

        def quantized(path, module):
            if not hasattr(module, "to_quantized") or f"{path}.scales" not in weights:
                return False
            spec = quant.get(path)
            return spec if isinstance(spec, dict) else True  # the routers are 8-bit in this conversion

        nn.quantize(
            self.model,
            group_size=int(quant.get("group_size", 64)),
            bits=int(quant.get("bits", 3)),
            mode=quant.get("mode", "affine"),
            class_predicate=quantized,
        )
        params = dict(nn.utils.tree_flatten(self.model.parameters()))
        missing = sorted(set(params) - set(weights))
        if missing:
            raise ValueError(f"{len(missing)} model parameters have no weight, e.g. {missing[:5]}")
        self.model.load_weights([(k, v) for k, v in weights.items() if k in params])
        mx.eval(self.model.parameters())
        self.model.eval()
        self.unused_weights = sorted(set(weights) - set(params))
        self.trunk_bytes = sum(v.nbytes for v in params.values())

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        eos = config.get("eos_token_id")
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()

        self.prefix = []
        self.disk = None
        self.max_seq_len = int(os.environ.get("CACHALOT_MINIMAX_MAX_SEQ_LEN", "131072"))
        self._lock = threading.RLock()
        self._busy = False
        self._idle_since = time.perf_counter()
        self._stop = threading.Event()
        if heartbeat_seconds > 0:
            threading.Thread(target=self._heartbeat, args=(heartbeat_seconds,), daemon=True, name="minimax-heartbeat").start()

        if verbose:
            print(
                f"MiniMax-M3: {len(self.model.layers)} layers ({n_moe} MoE), trunk {self.trunk_bytes / 1024**3:.1f} GiB, "
                f"{len(self.expert_index)} experts x {expert_bytes / 2**20:.2f} MiB, "
                f"{self.store.capacity} resident slots ({self.store.budget_bytes / 1024**3:.1f} GiB), "
                f"loaded in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    # -- decode ------------------------------------------------------------------------------------------
    def _install_decode_hooks(self) -> None:
        """Per MoE layer, what a decode token does between its routing and its routed experts (HANDOFF 18.1).

        DECODE_OVERLAP: the routing is evaluated on its own, then the shared expert is queued (async) so the
        GPU runs it while the misses are read, and the routed output is not evaluated at the end of the layer
        (the next layer's routing sync covers it): one GPU round trip per layer instead of two. Bit-identical.
        PREDICT_TOPK > 0: the next MoE layer's routing is predicted from this layer's residual (its own norm,
        gate and bias) and its misses start reading now, into transient slots (the store's prefetch path).
        """
        layers = self.model.layers
        store, index = self.store, self.expert_index

        def make(i, moe, nxt):
            def hook(x, residual, inds, weights):
                pred = None
                if nxt is not None and PREDICT_TOPK > 0 and residual is not None:
                    scores, _ = nxt.block_sparse_moe.route_scores(nxt.post_attention_layernorm(residual))
                    pred = mx.argsort(-scores, axis=-1)[..., :PREDICT_TOPK]
                mx.eval(inds, weights, *([pred] if pred is not None else []))
                shared = moe.shared_experts(x)
                mx.async_eval(shared)
                prefetch = None
                if pred is not None:
                    prefetch = [index[(i + 1, int(e))] for e in pred.reshape(-1).tolist()]
                return shared, prefetch
            return hook

        for i, layer in enumerate(layers):
            if not layer.is_sparse or not DECODE_OVERLAP:
                continue
            nxt = layers[i + 1] if i + 1 < len(layers) and layers[i + 1].is_sparse else None
            layer.block_sparse_moe.decode_hook = make(i, layer.block_sparse_moe, nxt)
            layer.block_sparse_moe.switch_mlp.decode_eval = False

    # -- model -------------------------------------------------------------------------------------------
    def new_cache(self):
        return [KVCache() for _ in self.model.layers]

    def _forward(self, tokens: list[int], cache) -> mx.array:
        # the lm_head on the last position only: a prefill chunk of 8,192 would otherwise build 8,192 x 200k
        # logits (3.3 GB) to keep one row (HANDOFF 18.1). One token (decode) is the same call as before.
        h = self.model.model(mx.array(tokens, dtype=mx.int32)[None], cache=cache)[:, -1:, :]
        if self.config.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(h)[:, -1, :]
        return self.model.lm_head(h)[:, -1, :]

    # -- chat format -------------------------------------------------------------------------------------
    @property
    def splitter_cls(self):
        return MiniMaxSplitter

    def render_chat(self, messages, *, tools=None, thinking=False, effort=None, add_generation_prompt=True) -> str:
        """M3's template takes thinking_mode enabled/disabled and writes the `<mm:think>` / `</mm:think>`
        prefix itself; it has no effort levels."""
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False,
            thinking_mode="enabled" if thinking else "disabled",
        )

    def parse_tool_calls(self, text: str, tools=None):
        return parse_minimax_tool_calls(text, tools)

    def encode_chat(self, messages, *, tools=None, reasoning_effort=None, add_generation_prompt=True) -> list[int]:
        text = self.render_chat(messages, tools=tools, thinking=reasoning_effort is not None,
                                add_generation_prompt=add_generation_prompt)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

