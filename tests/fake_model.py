"""A checkpoint-free stand-in for V41Model used by server tests."""

from __future__ import annotations

import types

import mlx.core as mx

from cachalot.model.prefix_cache import PrefixCache, SequenceSnapshot
from cachalot.model.text_decode_runtime import DecodeResult

VOCAB = 256
EOS = 0
THINK_END_TOKEN = 1  # decodes to "</think>"


class CharTokenizer:
    eos_token_id = EOS
    eos_token = "<eos>"

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), VOCAB - 1) for c in text]

    def decode(self, ids, skip_special_tokens=False) -> str:
        out = []
        for i in ids:
            if i == THINK_END_TOKEN:
                out.append("</think>")
            elif i == EOS:
                out.append("<eos>")
            else:
                out.append(chr(i))
        return "".join(out)


class ScriptedRuntime:
    """Emits a fixed reply regardless of the prompt."""

    def __init__(self, reply: str = "Hello, whale!", thinking: str | None = None, max_seq_len: int = 4096):
        self.tokenizer = CharTokenizer()
        self.max_seq_len = max_seq_len
        self.prefix_cache = PrefixCache()
        self.tokens: list[int] = []
        self.position = 0
        script = []
        if thinking is not None:
            script += self.tokenizer.encode(thinking) + [THINK_END_TOKEN]
        script += self.tokenizer.encode(reply) + [EOS]
        self.script = script
        self.step = 0
        self.prefills: list[int] = []
        self.image_prefills: list[int | None] = []
        self.image_spans: list = []
        self.model_path = "/fake"
        self.expert_store = types.SimpleNamespace(
            stats=lambda: types.SimpleNamespace(cache_hits=0, cache_misses=0, hit_rate=0.0, ssd_bytes_read=0),
            current_bytes=0,
            __len__=lambda self_: 0,
        )

    def _logits(self) -> mx.array:
        tok = self.script[min(self.step, len(self.script) - 1)]
        logits = mx.zeros((VOCAB,))
        return logits.at[tok].add(10.0)

    def reset(self):
        self.tokens = []
        self.position = 0
        self.step = 0
        self.image_spans = []

    def prefill_tokens(self, ids, image_rows=None, image_token_id=None, image_spans=(), next_token_ids=None):
        ids = list(ids)
        self.prefills.append(len(ids))
        self.lookaheads = list(getattr(self, "lookaheads", [])) + [
            None if next_token_ids is None else list(next_token_ids)
        ]
        self.image_prefills.append(None if image_rows is None else image_rows.shape[0])
        self.image_spans = list(getattr(self, "image_spans", [])) + list(image_spans)
        self.tokens.extend(ids)
        self.position += len(ids)
        self.step = 0
        return DecodeResult(logits=self._logits(), hidden=None, routes=(), position=self.position - 1)

    def decode_token(self, token_id):
        self.tokens.append(int(token_id))
        self.position += 1
        self.step += 1
        return DecodeResult(logits=self._logits(), hidden=None, routes=(), position=self.position - 1)

    def snapshot(self, logits=None):
        return SequenceSnapshot(
            tokens=tuple(self.tokens), position=self.position, logits=logits,
            windows={}, compressed_caches={}, compressor_kv={}, compressor_score={}, indexer_k={},
            engram_history=[], shared_compress_kv=None, shared_index_k_layer=None,
            shared_topk_idxs=None, shared_candidates=None,
            image_spans=tuple(self.image_spans),
        )

    def restore(self, snap):
        self.image_spans = list(snap.image_spans)
        self.tokens = list(snap.tokens)
        self.position = snap.position
        self.step = 0

    def close(self):
        pass


class FakeModel:
    def __init__(self, runtime: ScriptedRuntime):
        self.runtime = runtime
        self.model_path = runtime.model_path

    @property
    def tokenizer(self):
        return self.runtime.tokenizer

    @property
    def max_seq_len(self):
        return self.runtime.max_seq_len

    def stats(self):
        return {"resident_experts": 0}

    def close(self):
        pass


class FakeEncoding:
    """Minimal chat template: concatenates role-tagged messages."""

    @staticmethod
    def encode_messages(messages, thinking_mode="chat", reasoning_effort=None,
                        return_multi_modal_data=False, **_):
        parts = []
        images = []
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, list):
                images += [b for b in content if b.get("type") in ("image", "image_url")]
                content = "".join(b.get("text", "") for b in content)
            tools = m.get("tools")
            parts.append(f"<{m['role']}>{content}" + (f"<tools:{len(tools)}>" if tools else ""))
        parts.append("<assistant>" + ("<think>" if thinking_mode == "thinking" else ""))
        if return_multi_modal_data:
            return "".join(parts), {"images": images}
        return "".join(parts)

    @staticmethod
    def parse_message_from_completion_text(text, thinking_mode="chat"):
        reasoning = ""
        if thinking_mode == "thinking" and "</think>" in text:
            reasoning, text = text.split("</think>", 1)
        text = text.replace("<eos>", "")
        return {"role": "assistant", "content": text, "reasoning_content": reasoning, "tool_calls": []}
