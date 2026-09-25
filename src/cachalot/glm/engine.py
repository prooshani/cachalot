"""
The HTTP server's engine for GLM-5.3-Flash: same interface as `cachalot.server.engine.Engine`
(stream_chat / chat / stats), so `cachalot.server.app` serves either model.

GLM's chat template differs from DeepSeek's in three ways that matter here:

- it always opens the reply with `<think>`; a request with thinking off gets `</think>` appended
  to the prompt, so the model answers directly (GLM's own no-think form);
- tool calls come back as `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`;
  string arguments are raw text, others JSON (the template renders them that way), so values are
  typed by the tool's own JSON schema;
- the template reads an assistant message's tool-call arguments as a mapping, so the OpenAI
  JSON-string form is decoded before rendering.

Text only: image parts are passed to the template as text parts (the vision tower is not loaded).
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from cachalot.server.engine import ChatOutput, ChatRequest, Delta, _first_stop, _log_request, _with_ids, dump_reply

THINK_END = "</think>"
TOOL_START = "<tool_call>"
_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def _effort(value) -> str | None:
    """GLM knows 'low', 'high' and (default) 'max'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return "low" if value <= 33 else ("high" if value <= 80 else None)
    v = str(value).lower()
    if v in ("low", "minimal"):
        return "low"
    if v in ("medium", "high"):
        return "high"
    return None


def _schema_types(tools) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for tool in tools or []:
        fn = tool.get("function", tool)
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        out[fn.get("name", "")] = {k: (v or {}).get("type", "") for k, v in props.items()}
    return out


def parse_tool_calls(text: str, tools=None) -> list[dict[str, Any]]:
    types = _schema_types(tools)
    calls = []
    for block in _TOOL_RE.findall(text):
        name = block.split("<arg_key>", 1)[0].strip()
        args: dict[str, Any] = {}
        for key, raw in _ARG_RE.findall(block):
            key = key.strip()
            kind = types.get(name, {}).get(key, "")
            if kind == "string":
                args[key] = raw
                continue
            try:
                args[key] = json.loads(raw)
            except ValueError:
                args[key] = raw
        calls.append({"type": "function", "function": {"name": name, "arguments": args}})
    return calls


def _template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except ValueError:
                        fn["arguments"] = {"arguments": fn["arguments"]}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


class _GlmSplitter:
    """Reasoning until </think>, then content; a tool-call block is held back until the end."""

    THINK_END = THINK_END
    TOOL_START = TOOL_START

    def __init__(self, tokenizer, thinking: bool):
        self.tokenizer = tokenizer
        self.tokens: list[int] = []
        self.text = ""
        self.emitted_reasoning = 0
        self.emitted_content = 0
        self.in_tool_block = False
        self.think_end: int | None = None if thinking else -len(self.THINK_END)

    def push(self, token: int) -> Delta:
        self.tokens.append(token)
        text = self.tokenizer.decode(self.tokens, skip_special_tokens=False)
        if text.endswith("�"):
            return Delta()
        self.text = text
        return self._emit()

    def _emit(self) -> Delta:
        d = Delta()
        text = self.text
        if self.think_end is None:
            i = text.find(self.THINK_END)
            if i < 0:
                safe = max(len(text) - len(self.THINK_END), self.emitted_reasoning)
                d.reasoning = text[self.emitted_reasoning:safe]
                self.emitted_reasoning = safe
                return d
            d.reasoning = text[self.emitted_reasoning:i]
            self.emitted_reasoning = i
            self.think_end = i
            self.emitted_content = i + len(self.THINK_END)
        if self.in_tool_block:
            return d
        start = self.think_end + len(self.THINK_END)
        rel = text.find(self.TOOL_START, start)
        if rel >= 0:
            d.content = text[self.emitted_content:rel]
            self.emitted_content = rel
            self.in_tool_block = True
            return d
        safe = max(len(text) - len(self.TOOL_START), self.emitted_content)
        d.content = text[self.emitted_content:safe]
        self.emitted_content = safe
        return d

    def content_so_far(self) -> str:
        return "" if self.think_end is None else self.text[self.think_end + len(self.THINK_END):]

    def flush(self) -> Delta:
        d = Delta()
        if self.think_end is None:
            d.reasoning = self.text[self.emitted_reasoning:]
            self.emitted_reasoning = len(self.text)
        elif not self.in_tool_block:
            d.content = self.text[self.emitted_content:]
            self.emitted_content = len(self.text)
        return d


@dataclass
class GlmEngine:
    model: Any  # cachalot.glm.model.GlmModel
    model_id: str = "glm-5.3-flash"
    _lock: threading.Lock = field(default_factory=threading.Lock)
    requests_served: int = 0
    tokens_generated: int = 0
    started_at: float = field(default_factory=time.time)
    images_served: int = 0

    def __post_init__(self):
        self.tokenizer = self.model.tokenizer

    def _render(self, req: ChatRequest, messages, add_generation_prompt=True) -> list[int]:
        # the chat template, the thinking switch and the tool-call format belong to the model family
        text = self.model.render_chat(
            _template_messages(messages), tools=req.tools or None, thinking=req.thinking_mode == "thinking",
            effort=req.reasoning_effort, add_generation_prompt=add_generation_prompt,
        )
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_chat(self, req: ChatRequest) -> list[int]:
        return self._render(req, req.messages)

    def system_prefix_len(self, req: ChatRequest, tokens: list[int]) -> int:
        """Tokens of the rendered header + tools + leading system message, when a prefix of `tokens`."""
        if not req.messages or req.messages[0].get("role") != "system":
            return 0
        try:
            head = self._render(req, req.messages[:1], add_generation_prompt=False)
        except Exception:
            return 0
        n = len(head)
        return n if 0 < n < len(tokens) and list(tokens[:n]) == head else 0

    def _expert_counts(self):
        s = self.model.store.stats()
        return (s.cache_hits, s.cache_misses, s.reads, s.fast_reads, s.read_wall_seconds)

    def stream_chat(self, req: ChatRequest, cancel: threading.Event | None = None) -> Iterator[Delta]:
        with self._lock:
            if cancel is not None and cancel.is_set():
                return
            prompt = self.encode_chat(req)
            boundary = self.system_prefix_len(req, prompt)
            params = req.params
            room = self.model.max_seq_len - len(prompt)
            if req.max_tokens_defaulted and 0 < room < params.max_new_tokens:
                params = replace(params, max_new_tokens=room)
            splitter = self.model.splitter_cls(self.tokenizer, req.thinking_mode == "thinking")
            reused, prefill_s, finish, stop_hit, emitted = 0, 0.0, "length", False, 0
            decode_start = None
            for event in self.model.stream(
                prompt,
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                cancel=cancel,
                boundary=boundary,
            ):
                kind = event[0]
                if kind == "prefill":
                    reused, prefill_s = event[1], event[2]
                    decode_start = self._expert_counts()
                    yield Delta(prompt_tokens=len(prompt), reused_prefix_tokens=reused, prefill_seconds=prefill_s)
                elif kind == "token":
                    token = event[1]
                    if token in self.model.eos_ids:
                        continue
                    delta = splitter.push(token)
                    if req.stop and not stop_hit:
                        full = splitter.content_so_far()
                        cut = _first_stop(full, req.stop)
                        if cut is not None:
                            delta.content = full[:cut][emitted:]
                            stop_hit, finish = True, "stop"
                            if cancel is None:
                                cancel = threading.Event()
                            cancel.set()
                    emitted += len(delta.content)
                    if delta.content or delta.reasoning:
                        yield delta
                elif kind == "done":
                    if not stop_hit:
                        finish = "stop" if event[1] in ("stop", "cancel") else event[1]
                    tail = splitter.flush()
                    if stop_hit:
                        tail.content = ""
                    tool_calls = None
                    if splitter.in_tool_block and not stop_hit:
                        calls = self.model.parse_tool_calls(splitter.text, req.tools)
                        if calls:
                            tool_calls = _with_ids(calls)
                            finish = "tool_calls"
                        else:
                            tail.content += splitter.text[splitter.emitted_content:]
                    n_out = len(splitter.tokens) + (1 if event[1] == "stop" else 0)
                    self.requests_served += 1
                    self.tokens_generated += n_out
                    dump_reply(prompt, splitter.tokens, reused)
                    _log_request(len(prompt), reused, prefill_s, n_out, event[2], finish, 0, 0,
                                 self._expert_counts(), decode_start)
                    yield Delta(
                        content=tail.content,
                        reasoning=tail.reasoning,
                        tool_calls=tool_calls,
                        finish_reason=finish,
                        completion_tokens=n_out,
                        prompt_tokens=len(prompt),
                        reused_prefix_tokens=reused,
                        prefill_seconds=prefill_s,
                        decode_seconds=event[2],
                    )

    def chat(self, req: ChatRequest) -> ChatOutput:
        content, reasoning, tool_calls, last = [], [], None, None
        for d in self.stream_chat(req):
            content.append(d.content)
            reasoning.append(d.reasoning)
            if d.tool_calls:
                tool_calls = d.tool_calls
            last = d
        assert last is not None
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
        if req.thinking_mode == "thinking":
            message["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return ChatOutput(
            message=message,
            finish_reason=last.finish_reason or "length",
            prompt_tokens=last.prompt_tokens,
            completion_tokens=last.completion_tokens,
            reused_prefix_tokens=last.reused_prefix_tokens,
            prefill_seconds=last.prefill_seconds,
            decode_seconds=last.decode_seconds,
            raw_text="".join(content),
        )

    def stats(self) -> dict[str, Any]:
        s = self.model.store.stats()
        return {
            "model": self.model_id,
            "uptime_seconds": time.time() - self.started_at,
            "requests_served": self.requests_served,
            "tokens_generated": self.tokens_generated,
            "busy": self._lock.locked(),
            "expert_hit_rate": s.hit_rate,
            "resident_experts": len(self.model.store),
            "prefix_snapshots": len(self.model.prefix),
        }
