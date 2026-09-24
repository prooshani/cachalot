"""
Single-flight generation engine used by the HTTP server.

One model instance serves one request at a time (the runtime is stateful and
the machine is bandwidth-bound anyway). Requests queue on a lock; streaming
requests hand tokens to the asyncio side through a queue.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from cachalot.model.api import V41Model
from cachalot.model.generation import (
    SamplingParams,
    load_official_encoding,
    parse_completion,
    stream_tokens,
)

THINK_END = "</think>"
TOOL_CALLS_START = "\n\n<｜DSML｜ calls"


@dataclass
class ChatRequest:
    messages: list[dict[str, Any]]
    params: SamplingParams
    thinking_mode: str = "chat"
    reasoning_effort: str | int | None = None
    stop: tuple[str, ...] = ()
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
    # True when the client sent no max_tokens and params carries the
    # server's default; stream_chat then shortens it to fit max_seq_len
    # instead of refusing a long prompt (HANDOFF section 15.10).
    max_tokens_defaulted: bool = False


@dataclass
class Delta:
    """Incremental output for one streaming step."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    completion_tokens: int = 0
    prompt_tokens: int = 0
    reused_prefix_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0


@dataclass
class ChatOutput:
    message: dict[str, Any]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reused_prefix_tokens: int
    prefill_seconds: float
    decode_seconds: float
    raw_text: str


class _TextSplitter:
    """
    Turns the raw completion stream into reasoning / content deltas and
    holds back a tool-call block until it is complete.
    """

    def __init__(self, tokenizer, thinking_mode: str):
        self.tokenizer = tokenizer
        self.thinking = thinking_mode == "thinking"
        self.tokens: list[int] = []
        self.text = ""
        self.emitted_reasoning = 0
        self.emitted_content = 0
        self.in_tool_block = False
        self.think_end_idx: int | None = None if self.thinking else -len(THINK_END)

    def push(self, token: int) -> Delta:
        self.tokens.append(token)
        new_text = self.tokenizer.decode(self.tokens, skip_special_tokens=False)
        # Hold back trailing partial UTF-8 sequences.
        if new_text.endswith("�"):
            return Delta()
        self.text = new_text
        return self._emit()

    def _emit(self) -> Delta:
        delta = Delta()
        text = self.text

        if self.think_end_idx is None:
            idx = text.find(THINK_END)
            if idx < 0:
                # Still reasoning; keep a small tail in case </think> is split.
                safe = max(len(text) - len(THINK_END), self.emitted_reasoning)
                delta.reasoning = text[self.emitted_reasoning:safe]
                self.emitted_reasoning = safe
                return delta
            delta.reasoning = text[self.emitted_reasoning:idx]
            self.emitted_reasoning = idx
            self.think_end_idx = idx
            self.emitted_content = idx + len(THINK_END)

        content_start = self.think_end_idx + len(THINK_END)
        content = text[content_start:]
        if self.in_tool_block:
            return delta
        rel = content.find(TOOL_CALLS_START)
        if rel >= 0:
            end = content_start + rel
            delta.content = text[self.emitted_content:end]
            self.emitted_content = end
            self.in_tool_block = True
            return delta
        # Keep a tail so a split tool-call opener is not emitted as content.
        safe = max(len(text) - len(TOOL_CALLS_START), self.emitted_content)
        delta.content = text[self.emitted_content:safe]
        self.emitted_content = safe
        return delta

    def content_so_far(self) -> str:
        """Content region (after </think>) decoded so far, including held-back tail."""
        if self.think_end_idx is None:
            return ""
        return self.text[self.think_end_idx + len(THINK_END):]

    def flush(self) -> Delta:
        """Emit whatever is still held back (called at the end)."""
        delta = Delta()
        if self.think_end_idx is None:
            delta.reasoning = self.text[self.emitted_reasoning:]
            self.emitted_reasoning = len(self.text)
            return delta
        if not self.in_tool_block:
            delta.content = self.text[self.emitted_content:]
            self.emitted_content = len(self.text)
        return delta


@dataclass
class _OwnReply:
    """A reply this server generated: the prompt it answered (as prefilled),
    the reply's token ids, and the reply parsed into a message."""

    prompt: tuple[int, ...]
    reply: tuple[int, ...]
    message: Any


def _message_key(message: dict[str, Any] | None) -> Any:
    """What an assistant message says, independent of how it was serialized:
    tool arguments compare as JSON values, so key order and spacing drop out."""
    if not message:
        return None
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", tc)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        calls.append((fn.get("name"), json.dumps(args, sort_keys=True, ensure_ascii=False)))
    # Surrounding whitespace is not what a message says: Hermes sends an empty
    # thinking block back as reasoning_content " ".
    return (
        (message.get("content") or "").strip(),
        (message.get("reasoning_content") or "").strip(),
        tuple(calls),
    )


@dataclass
class Engine:
    model: V41Model
    model_id: str = "deepseek-v4.1-flash"
    _lock: threading.Lock = field(default_factory=threading.Lock)
    requests_served: int = 0
    tokens_generated: int = 0
    started_at: float = field(default_factory=time.time)
    encoding: Any = None
    # Image content (HANDOFF section 16 piece 4). The vision tower loads on
    # the first image, so a text-only server never pays for it.
    vision_enabled: bool = True
    _vision: Any = None
    images_served: int = 0
    # Recent replies, for splicing the model's own tokens back into a
    # client's re-rendered history (see _splice_own_replies).
    _own_replies: deque = field(default_factory=lambda: deque(maxlen=64))
    replies_spliced: int = 0

    def __post_init__(self):
        if self.encoding is None:
            self.encoding = load_official_encoding(self.model.model_path)
        self.tokenizer = self.model.tokenizer

    # ------------------------------------------------------------------
    def _vision_encoder(self):
        if not self.vision_enabled:
            return None
        if self._vision is None:
            from cachalot.model.vision_prompt import VisionEncoder

            index = getattr(self.model.runtime, "tensor_index", None)
            if index is None:
                return None
            self._vision = VisionEncoder(index)
        return self._vision

    def encode_chat_images(self, req: ChatRequest):
        """Token ids plus image spans: a vision_prompt.PromptImages."""
        from cachalot.model.vision_prompt import expand_prompt_images

        tokens, records = self._encode_chat(req)
        return expand_prompt_images(
            tokens, records, self._vision_encoder() if records else None
        )

    def encode_chat(self, req: ChatRequest) -> list[int]:
        return self._encode_chat(req)[0]

    def _chat_messages(self, req: ChatRequest) -> list[dict[str, Any]]:
        messages = [dict(m) for m in req.messages]
        if req.tools or req.response_format:
            if messages and messages[0].get("role") == "system":
                head = messages[0]
            else:
                head = {"role": "system", "content": ""}
                messages.insert(0, head)
            if req.tools:
                head["tools"] = req.tools
            if req.response_format:
                head["response_format"] = req.response_format
        return messages

    def _encode_chat(self, req: ChatRequest) -> tuple[list[int], list[dict[str, Any]]]:
        messages = self._chat_messages(req)
        # The official encoding accepts OpenAI content-part lists and turns
        # each image part into one placeholder token, returning the images
        # in prompt order.
        prompt, media = self.encoding.encode_messages(
            messages,
            thinking_mode=req.thinking_mode,
            reasoning_effort=req.reasoning_effort,
            return_multi_modal_data=True,
        )
        return list(self.tokenizer.encode(prompt)), list(media.get("images") or [])

    def system_prefix_len(self, req: ChatRequest, tokens: list[int]) -> int:
        """
        Tokens of `tokens` that are the rendered leading system message (with
        its tools and reasoning-effort header), or 0.

        An agent opens every conversation with the same long system prompt
        and tool schemas and diverges at the first user message. Snapshotting
        exactly here lets the next conversation reuse the whole system block
        instead of the last 4096-token chunk multiple inside it (HANDOFF
        section 15.4). The system block is rendered on its own and only used
        when its tokens are a prefix of the full prompt's, so a rendering or
        tokenization that depends on what follows it just disables the cut.
        """
        messages = self._chat_messages(req)
        if not messages or messages[0].get("role") != "system":
            return 0
        if not isinstance(messages[0].get("content") or "", str):
            return 0  # content parts (maybe images) would shift positions
        try:
            text = self.encoding.encode_messages(
                messages[:1],
                thinking_mode=req.thinking_mode,
                reasoning_effort=req.reasoning_effort,
            )
        except Exception:
            return 0
        head = list(self.tokenizer.encode(text))
        n = len(head)
        if 0 < n < len(tokens) and list(tokens[:n]) == head:
            return n
        return 0

    def _splice_own_replies(self, tokens: list[int], thinking_mode: str, not_before: int = 0) -> list[int]:
        """
        Put the model's own reply tokens back where a client re-rendered them.

        After a reply the prefix cache holds a snapshot of prompt + reply, and
        the next turn reuses it only if the client sends the reply back
        token for token. Agent harnesses re-serialize tool calls: Hermes sends
        a call's JSON arguments with its keys in a different order than the
        model wrote them, so the re-rendered reply diverges a few tokens in and
        the whole reply is prefilled again (HANDOFF section 15.5). When the
        client's copy of a reply parses to the same message as ours (same
        content, same tool names, same argument values), the prompt carries
        our tokens for it instead. The model then conditions on exactly what
        it wrote, and the snapshot is a prefix again. Anything that does not
        parse to the same message is left exactly as the client sent it.
        """
        eos = self.tokenizer.eos_token_id
        for rec in sorted(self._own_replies, key=lambda r: len(r.prompt)):
            p, r = len(rec.prompt), len(rec.reply)
            if p < not_before:
                continue
            if len(tokens) <= p or tuple(tokens[:p]) != rec.prompt:
                continue
            if tuple(tokens[p:p + r]) == rec.reply and len(tokens) > p + r and tokens[p + r] == eos:
                continue  # already verbatim
            try:
                e = tokens.index(eos, p)
            except ValueError:
                continue
            theirs = parse_completion(
                self.encoding,
                self.tokenizer.decode(tokens[p:e], skip_special_tokens=False),
                thinking_mode=thinking_mode,
                stopped_on_eos=True,
                eos_token=self.tokenizer.eos_token,
            )
            if theirs is None or _message_key(theirs) != rec.message:
                continue
            tokens = list(rec.prompt) + list(rec.reply) + tokens[e:]
            self.replies_spliced += 1
        return tokens

    def _remember_reply(self, prompt: list[int], reply: list[int], text: str, thinking_mode: str) -> None:
        parsed = parse_completion(
            self.encoding,
            text,
            thinking_mode=thinking_mode,
            stopped_on_eos=True,
            eos_token=self.tokenizer.eos_token,
        )
        key = _message_key(parsed)
        if key is not None:
            self._own_replies.append(_OwnReply(tuple(prompt), tuple(reply), key))

    def _expert_counts(self) -> tuple | None:
        """(hits, misses, reads, fast reads, read seconds) of the expert store
        so far, for the per-request log."""
        store = getattr(self.model.runtime, "expert_store", None)
        if store is None:
            return None
        try:
            s = store.stats()
            return (s.cache_hits, s.cache_misses, getattr(s, "reads", 0),
                    getattr(s, "fast_reads", 0), getattr(s, "read_wall_seconds", 0.0))
        except Exception:
            return None

    def stream_chat(self, req: ChatRequest, cancel: threading.Event | None = None) -> Iterator[Delta]:
        """Blocking generator; run it in a worker thread."""
        with self._lock:
            if cancel is not None and cancel.is_set():
                # the client left while this request was queued
                return
            images = self.encode_chat_images(req)
            prompt_tokens = images.tokens
            system_end = 0
            spliced_before = self.replies_spliced
            # A splice changes token counts after the reply it replaces, so it
            # must not move an image span: only replies after the last image.
            span_end = max((s.start + s.length for s in images.spans), default=0)
            prompt_tokens = self._splice_own_replies(prompt_tokens, req.thinking_mode, not_before=span_end)
            if images.spans:
                images = type(images)(tokens=prompt_tokens, spans=images.spans)
            if not images.spans:
                system_end = self.system_prefix_len(req, prompt_tokens)
            if images.spans:
                self.images_served += len(images.spans)
            else:
                images = None
            splitter = _TextSplitter(self.tokenizer, req.thinking_mode)
            n_prompt = len(prompt_tokens)
            params = req.params
            room = self.model.max_seq_len - n_prompt
            if req.max_tokens_defaulted and 0 < room < params.max_new_tokens:
                params = replace(params, max_new_tokens=room)
            reused = 0
            prefill_seconds = 0.0
            finish = "length"
            stop_hit = False
            content_emitted = 0
            decode_start = None

            for event in stream_tokens(
                self.model.runtime,
                prompt_tokens,
                params,
                cancel=cancel,
                images=images,
                boundaries=(system_end,) if system_end else (),
            ):
                if event.kind == "prefill":
                    decode_start = self._expert_counts()
                    reused = event.reused_prefix_tokens
                    prefill_seconds = event.prefill_seconds
                    yield Delta(prompt_tokens=n_prompt, reused_prefix_tokens=reused, prefill_seconds=prefill_seconds)
                elif event.kind == "token":
                    delta = splitter.push(event.token)
                    if req.stop and not stop_hit:
                        full = splitter.content_so_far()
                        cut = _first_stop(full, req.stop)
                        if cut is not None:
                            allowed = full[:cut]
                            delta.content = allowed[content_emitted:]
                            stop_hit = True
                            finish = "stop"
                            if cancel is None:
                                cancel = threading.Event()
                            cancel.set()
                    content_emitted += len(delta.content)
                    if delta.content or delta.reasoning:
                        yield delta
                elif event.kind == "done":
                    if not stop_hit:
                        finish = event.finish_reason or "length"
                    tail = splitter.flush()
                    if stop_hit:
                        tail.content = ""
                    tool_calls = None
                    if splitter.in_tool_block and not stop_hit:
                        parsed = parse_completion(
                            self.encoding,
                            splitter.text,
                            thinking_mode=req.thinking_mode,
                            stopped_on_eos=(finish == "stop"),
                            eos_token=self.tokenizer.eos_token,
                        )
                        if parsed and parsed.get("tool_calls"):
                            tool_calls = _with_ids(parsed["tool_calls"])
                            finish = "tool_calls"
                        else:
                            # Malformed block: surface it as plain text.
                            tail.content += splitter.text[splitter.emitted_content:]
                    self.requests_served += 1
                    self.tokens_generated += len(splitter.tokens)
                    dump_reply(prompt_tokens, splitter.tokens, reused)
                    if finish in ("stop", "tool_calls") and not stop_hit:
                        # ended on EOS, so a client's history renders it as
                        # reply + EOS: a candidate for the next turn's splice
                        self._remember_reply(prompt_tokens, splitter.tokens, splitter.text, req.thinking_mode)
                    _log_request(
                        n_prompt, reused, prefill_seconds, len(splitter.tokens),
                        event.decode_seconds, finish,
                        len(images.spans) if images is not None else 0,
                        self.replies_spliced - spliced_before,
                        self._expert_counts(), decode_start,
                    )
                    yield Delta(
                        content=tail.content,
                        reasoning=tail.reasoning,
                        tool_calls=tool_calls,
                        finish_reason=finish,
                        completion_tokens=len(splitter.tokens),
                        prompt_tokens=n_prompt,
                        reused_prefix_tokens=reused,
                        prefill_seconds=prefill_seconds,
                        decode_seconds=event.decode_seconds,
                    )

    def chat(self, req: ChatRequest) -> ChatOutput:
        content, reasoning = [], []
        tool_calls = None
        last: Delta | None = None
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
        s = self.model.stats()
        s.update(
            {
                "model": self.model_id,
                "uptime_seconds": time.time() - self.started_at,
                "requests_served": self.requests_served,
                "tokens_generated": self.tokens_generated,
                "busy": self._lock.locked(),
                "images_served": self.images_served,
                "replies_spliced": self.replies_spliced,
                "vision_loaded": bool(self._vision is not None and self._vision.loaded),
                "vision_rows_reused": self._vision.cache_hits if self._vision is not None else 0,
            }
        )
        return s


def _log_request(prompt, reused, prefill_s, completion, decode_s, finish, n_images, spliced=0,
                 experts_end=None, experts_start=None):
    """One stderr line per request: where the time went, for agent sessions.
    miss/tok is expert-cache misses per decoded token, which is what decode
    speed follows (HANDOFF section 15.5). read= is the mean wall time of one
    expert read during decode, and fast= the share of reads quick enough to
    have come from the page cache rather than the drive (section 15.7)."""
    tps = completion / decode_s if decode_s else 0.0
    misses = ""
    if experts_end and experts_start and completion:
        dh = experts_end[0] - experts_start[0]
        dm = experts_end[1] - experts_start[1]
        misses = f" miss/tok={dm / completion:.1f} hit={dh / max(1, dh + dm):.0%}"
        if len(experts_end) >= 5 and len(experts_start) >= 5:
            reads = experts_end[2] - experts_start[2]
            if reads:
                fast = experts_end[3] - experts_start[3]
                ms = 1000.0 * (experts_end[4] - experts_start[4]) / reads
                misses += f" read={ms:.2f}ms fast={fast / reads:.0%}"
    print(
        f"[request] prompt={prompt} reused={reused} prefilled={prompt - reused} "
        f"prefill={prefill_s:.2f}s completion={completion} decode={decode_s:.2f}s "
        f"({tps:.2f} tok/s) images={n_images} spliced={spliced}{misses} finish={finish}",
        file=sys.stderr,
        flush=True,
    )


def dump_request_body(body: dict) -> None:
    """Append a request body to $CACHALOT_SERVER_DUMP (JSON lines), if set."""
    path = os.environ.get("CACHALOT_SERVER_DUMP")
    if not path:
        return
    with open(path, "a") as fh:
        fh.write(json.dumps({"t": time.time(), "body": body}, ensure_ascii=False) + "\n")


def dump_reply(prompt_tokens: list[int], reply_tokens: list[int], reused: int) -> None:
    """With $CACHALOT_SERVER_DUMP set, also append each reply's token ids, so a
    prefix-cache miss on the next turn can be traced to the exact token where
    the client's re-rendered history left the model's own output."""
    path = os.environ.get("CACHALOT_SERVER_DUMP")
    if not path:
        return
    with open(path, "a") as fh:
        fh.write(json.dumps({
            "t": time.time(),
            "reply": {"prompt_tokens": list(prompt_tokens), "reply_tokens": list(reply_tokens), "reused": reused},
        }) + "\n")


def _first_stop(text: str, stops: tuple[str, ...]) -> int | None:
    best = None
    for s in stops:
        i = text.find(s)
        if i >= 0 and (best is None or i < best):
            best = i
    return best


def _with_ids(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, tc in enumerate(tool_calls):
        tc = dict(tc)
        tc.setdefault("id", f"call_{int(time.time() * 1000) % 10_000_000:07d}_{i}")
        tc.setdefault("type", "function")
        fn = dict(tc.get("function", {}))
        if not isinstance(fn.get("arguments"), str):
            import json

            fn["arguments"] = json.dumps(fn.get("arguments", {}), ensure_ascii=False)
        tc["function"] = fn
        out.append(tc)
    return out
