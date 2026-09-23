"""
FastAPI application exposing the OpenAI Chat Completions / Completions API.

Endpoints:
    GET  /health
    GET  /v1/models
    GET  /v1/stats
    POST /v1/chat/completions   (stream=true supported, SSE)
    POST /v1/completions        (raw prompt, no chat template)

Extensions (all optional, ignored by standard clients):
    "thinking": true            -> DeepSeek thinking mode (reasoning_content in response)
    "reasoning_effort": 1..100 | "low" | "high" | "max"
    "top_k": int
    "seed": int
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from cachalot.model.generation import SamplingParams
from cachalot.server.engine import ChatRequest, Delta, Engine, dump_request_body


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None
    thinking: bool | None = None
    reasoning_effort: str | int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    n: int = 1
    user: str | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int | None = 256
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    echo: bool = False
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    no_repeat_ngram_size: int | None = None


class ServerConfig(BaseModel):
    default_max_tokens: int = 1024
    default_temperature: float = 0.6
    # Repetition controls. A request may override them. The 62 % collapse rate
    # that once justified a nonzero default here was the transposed residual
    # mix (docs/HANDOFF.md section 9.9); through the fixed runtime the rate is
    # 0 of 12 with them off, on both banks, so the default now matches
    # `cachalot chat`'s and the official distribution: off.
    default_frequency_penalty: float = 0.0
    default_presence_penalty: float = 0.0
    default_no_repeat_ngram_size: int = 0
    default_penalty_window: int = 128
    default_thinking: bool = False
    default_reasoning_effort: str | int | None = None
    api_key: str | None = Field(default=None, description="If set, requests must carry it as a Bearer token.")


def _penalties(config: ServerConfig, body) -> dict:
    """Repetition controls for one request: the request's value, else the server default.

    0.0 from a request is honoured as "off", which `or` would silently turn back
    into the default -- so this tests for None rather than falsiness.
    """
    def pick(name: str, default):
        value = getattr(body, name, None)
        return default if value is None else value

    return {
        "frequency_penalty": pick("frequency_penalty", config.default_frequency_penalty),
        "presence_penalty": pick("presence_penalty", config.default_presence_penalty),
        "no_repeat_ngram_size": pick(
            "no_repeat_ngram_size", config.default_no_repeat_ngram_size
        ),
        "penalty_window": config.default_penalty_window,
    }


# OpenAI-style reasoning_effort values, mapped onto what the official V4.1
# encoding accepts (an int in [1, 100], or "low" / "high" / "max"). None means
# "no reasoning": thinking mode off. Hermes Agent sends "none" on its
# title-generation calls (HANDOFF section 15.1).
_EFFORT_ALIASES: dict[str, str | int | None] = {
    "none": None,
    "off": None,
    "minimal": None,
    "low": "low",
    "medium": 50,
    "high": "high",
    "xhigh": "max",
    "max": "max",
    "ultra": "max",
}


def _normalize_effort(effort):
    """(effort, disables_thinking) for a request's reasoning_effort field."""
    if effort is None:
        return None, False
    if isinstance(effort, bool):
        raise HTTPException(400, f"invalid reasoning_effort: {effort!r}")
    if isinstance(effort, int):
        if not 1 <= effort <= 100:
            raise HTTPException(400, "reasoning_effort must be an int within [1, 100]")
        return effort, False
    key = str(effort).strip().lower()
    if key.isdigit():
        return _normalize_effort(int(key))
    if key not in _EFFORT_ALIASES:
        raise HTTPException(
            400, f"invalid reasoning_effort {effort!r}; use 1-100, none, low, medium, high or max"
        )
    value = _EFFORT_ALIASES[key]
    return value, value is None


# Seconds of silence (queued, or in prefill) between SSE keep-alive comments.
KEEPALIVE_SECONDS = 15


def _stops(stop) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(stop)


def create_app(engine: Engine, config: ServerConfig | None = None) -> FastAPI:
    config = config or ServerConfig()
    app = FastAPI(title="Cachalot", version="0.2.0", docs_url="/docs")
    # Every generation runs on this one thread. MLX 0.32 ties an array that
    # is not evaluated yet to the thread that built it: evaluating it from
    # another thread raises "There is no Stream(gpu, N) in current thread".
    # With a thread per request, a request cancelled between prefill chunks
    # left such arrays in its last snapshot, and every retry that restored
    # it failed (HANDOFF section 15.6). The engine is single-flight anyway.
    generate_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cachalot-generate")
    app.state.generate_pool = generate_pool

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if config.api_key and request.url.path.startswith("/v1"):
            header = request.headers.get("authorization", "")
            if header != f"Bearer {config.api_key}":
                return JSONResponse({"error": {"message": "invalid api key", "type": "auth_error"}}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": engine.model_id, "busy": engine._lock.locked()}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "created": int(engine.started_at),
                    "owned_by": "cachalot",
                    "max_context_length": engine.model.max_seq_len,
                }
            ],
        }

    @app.get("/v1/stats")
    async def stats():
        return engine.stats()

    def build_chat_request(body: ChatCompletionRequest) -> ChatRequest:
        if body.n != 1:
            raise HTTPException(400, "n must be 1")
        if not body.messages:
            raise HTTPException(400, "messages must not be empty")
        max_tokens = body.max_completion_tokens or body.max_tokens or config.default_max_tokens
        thinking = config.default_thinking if body.thinking is None else body.thinking
        if body.reasoning_effort is not None:
            effort, no_reasoning = _normalize_effort(body.reasoning_effort)
            if no_reasoning and body.thinking is None:
                thinking = False  # "none" asks for no reasoning at all
        else:
            effort = config.default_reasoning_effort
        if effort is not None and not thinking:
            thinking = True  # asking for reasoning effort implies thinking mode
        params = SamplingParams(
            max_new_tokens=max_tokens,
            temperature=config.default_temperature if body.temperature is None else body.temperature,
            top_p=1.0 if body.top_p is None else body.top_p,
            top_k=body.top_k or 0,
            seed=body.seed,
            **_penalties(config, body),
        )
        return ChatRequest(
            messages=body.messages,
            params=params,
            thinking_mode="thinking" if thinking else "chat",
            reasoning_effort=effort,
            stop=_stops(body.stop),
            tools=body.tools,
            response_format=body.response_format,
        )

    def usage(d: Delta) -> dict[str, Any]:
        return {
            "prompt_tokens": d.prompt_tokens,
            "completion_tokens": d.completion_tokens,
            "total_tokens": d.prompt_tokens + d.completion_tokens,
            "cachalot": {
                "reused_prefix_tokens": d.reused_prefix_tokens,
                "prefill_seconds": round(d.prefill_seconds, 3),
                "decode_seconds": round(d.decode_seconds, 3),
                "decode_tok_per_s": round(d.completion_tokens / d.decode_seconds, 3) if d.decode_seconds else None,
            },
        }

    async def run_stream(request: Request, produce, sse_chunk, model_name: str, include_usage: bool,
                         keepalive_chunk: dict | None = None):
        """Run a blocking Delta generator in a thread and forward as SSE."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()
        sentinel = object()

        def worker():
            try:
                for delta in produce(cancel):
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
            except Exception as exc:  # pragma: no cover - surfaced to client
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        generate_pool.submit(worker)

        async def gen():
            idle_seconds = 0
            try:
                while True:
                    # Poll for a disconnect while waiting: a request queued
                    # behind the single-flight lock, or still in prefill,
                    # yields nothing for minutes, and a client that gave up
                    # (Hermes retries on a timeout) must not then cost a
                    # full prefill once the lock frees.
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except TimeoutError:
                        if await request.is_disconnected():
                            cancel.set()
                        idle_seconds += 1
                        if idle_seconds % KEEPALIVE_SECONDS == 0:
                            # an SSE comment keeps read timeouts and proxies
                            # from cutting a stream that is silent through a
                            # minutes-long prefill
                            yield ": keep-alive\n\n"
                            if keepalive_chunk is not None:
                                # but an OpenAI SDK never surfaces comments,
                                # and Hermes drops a local stream after 900 s
                                # without a parsed chunk: send an empty delta
                                yield f"data: {json.dumps(keepalive_chunk)}\n\n"
                        continue
                    idle_seconds = 0
                    if await request.is_disconnected():
                        cancel.set()
                    if item is sentinel:
                        break
                    if isinstance(item, Exception):
                        yield f"data: {json.dumps({'error': {'message': str(item)}})}\n\n"
                        break
                    for chunk in sse_chunk(item):
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                cancel.set()

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        dump_request_body(body.model_dump(exclude_none=True))
        req = build_chat_request(body)
        created = int(time.time())
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model_name = body.model or engine.model_id
        include_usage = bool(body.stream_options and body.stream_options.get("include_usage"))

        if body.stream:
            sent_role = {"done": False}

            def sse_chunk(d: Delta):
                delta: dict[str, Any] = {}
                if not sent_role["done"]:
                    delta["role"] = "assistant"
                    sent_role["done"] = True
                if d.reasoning:
                    delta["reasoning_content"] = d.reasoning
                if d.content:
                    delta["content"] = d.content
                if d.tool_calls:
                    delta["tool_calls"] = [
                        {"index": i, **tc} for i, tc in enumerate(d.tool_calls)
                    ]
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": d.finish_reason}],
                }
                if d.finish_reason and include_usage:
                    chunk["usage"] = usage(d)
                if not delta and not d.finish_reason:
                    return []
                return [chunk]

            keepalive = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            }
            return await run_stream(
                request, lambda cancel: engine.stream_chat(req, cancel), sse_chunk, model_name, include_usage,
                keepalive_chunk=keepalive,
            )

        try:
            out = await asyncio.get_running_loop().run_in_executor(generate_pool, engine.chat, req)
        except ValueError as exc:
            # a malformed request: bad image data, placeholder mismatch, ...
            raise HTTPException(400, str(exc)) from exc
        last = Delta(
            completion_tokens=out.completion_tokens,
            prompt_tokens=out.prompt_tokens,
            reused_prefix_tokens=out.reused_prefix_tokens,
            prefill_seconds=out.prefill_seconds,
            decode_seconds=out.decode_seconds,
        )
        return {
            "id": rid,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": out.message, "finish_reason": out.finish_reason, "logprobs": None}],
            "usage": usage(last),
        }

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        prompt = body.prompt if isinstance(body.prompt, str) else body.prompt[0]
        created = int(time.time())
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        model_name = body.model or engine.model_id
        params = SamplingParams(
            max_new_tokens=body.max_tokens or config.default_max_tokens,
            temperature=config.default_temperature if body.temperature is None else body.temperature,
            top_p=1.0 if body.top_p is None else body.top_p,
            top_k=body.top_k or 0,
            seed=body.seed,
            **_penalties(config, body),
        )
        stops = _stops(body.stop)

        def produce(cancel):
            from cachalot.model.generation import stream_tokens

            with engine._lock:
                ids = list(engine.tokenizer.encode(prompt))
                toks: list[int] = []
                emitted = 0
                text = ""
                for ev in stream_tokens(engine.model.runtime, ids, params, cancel=cancel):
                    if ev.kind == "token":
                        toks.append(ev.token)
                        text = engine.tokenizer.decode(toks, skip_special_tokens=False)
                        if text.endswith("�"):
                            continue
                        piece = text[emitted:]
                        cut = None
                        for s in stops:
                            i = text.find(s)
                            if i >= 0 and (cut is None or i < cut):
                                cut = i
                        if cut is not None:
                            piece = text[emitted:cut]
                            emitted = cut
                            cancel.set()
                            if piece:
                                yield Delta(content=piece)
                            continue
                        emitted = len(text)
                        if piece:
                            yield Delta(content=piece)
                    elif ev.kind == "done":
                        engine.requests_served += 1
                        engine.tokens_generated += len(toks)
                        yield Delta(
                            content=text[emitted:] if not stops else "",
                            finish_reason=ev.finish_reason if ev.finish_reason != "cancel" else "stop",
                            completion_tokens=len(toks),
                            prompt_tokens=len(ids),
                            reused_prefix_tokens=ev.reused_prefix_tokens,
                            prefill_seconds=ev.prefill_seconds,
                            decode_seconds=ev.decode_seconds,
                        )

        if body.stream:
            def sse_chunk(d: Delta):
                if not d.content and not d.finish_reason:
                    return []
                return [{
                    "id": rid,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "text": d.content, "finish_reason": d.finish_reason, "logprobs": None}],
                }]

            return await run_stream(request, produce, sse_chunk, model_name, False)

        def collect():
            parts, last = [], None
            for d in produce(threading.Event()):
                parts.append(d.content)
                last = d
            return "".join(parts), last

        text, last = await asyncio.get_running_loop().run_in_executor(generate_pool, collect)
        return {
            "id": rid,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "text": (prompt if body.echo else "") + text, "finish_reason": last.finish_reason, "logprobs": None}],
            "usage": usage(last),
        }

    return app
