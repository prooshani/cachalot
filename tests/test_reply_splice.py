"""
The server puts the model's own reply tokens back into a client's
re-rendered history when the client's copy says the same thing
(HANDOFF section 15.5). Uses the real tokenizer and the official encoding,
because the point is how those two render a re-serialized tool call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

MODEL_PATH = Path("/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash")
pytestmark = pytest.mark.skipif(
    not (MODEL_PATH / "tokenizer.json").exists(), reason="checkpoint not mounted"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    }
]


@pytest.fixture(scope="module")
def engine():
    from transformers import AutoTokenizer

    from cachalot.model.generation import load_official_encoding
    from cachalot.server.engine import Engine

    e = Engine.__new__(Engine)  # no model: only the tokenizer and encoding are used
    e.encoding = load_official_encoding(MODEL_PATH)
    e.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    from collections import deque

    e._own_replies = deque(maxlen=64)
    e.replies_spliced = 0
    return e


def _render(engine, messages):
    head = dict(messages[0], tools=TOOLS)
    text = engine.encoding.encode_messages([head] + messages[1:], thinking_mode="chat")
    return list(engine.tokenizer.encode(text))


def _model_reply(engine, prompt, args_json):
    """What the model generated: a write_file call with `path` first."""
    history = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "write it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "write_file", "arguments": args_json}}],
        },
    ]
    full = _render(engine, history)
    assert full[: len(prompt)] == prompt
    eos = engine.tokenizer.eos_token_id
    reply = full[len(prompt): full.index(eos, len(prompt))]
    text = engine.tokenizer.decode(reply, skip_special_tokens=False)
    return reply, text


def _next_turn(engine, args_json, tool_result="ok"):
    return _render(
        engine,
        [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "write it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": args_json}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": tool_result},
        ],
    )


def test_reordered_tool_arguments_are_spliced_back(engine):
    prompt = _render(engine, [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "write it"}])
    reply, text = _model_reply(engine, prompt, json.dumps({"path": "out.txt", "content": "narwhal\n"}))
    engine._remember_reply(prompt, reply, text, "chat")

    # Hermes sends the arguments back with the keys in the other order
    client = _next_turn(engine, json.dumps({"content": "narwhal\n", "path": "out.txt"}, separators=(",", ":")))
    assert client[: len(prompt) + len(reply)] != prompt + reply  # the miss this fixes

    spliced = engine._splice_own_replies(client, "chat")
    assert spliced[: len(prompt) + len(reply)] == prompt + reply
    eos = engine.tokenizer.eos_token_id
    # everything after the reply is the client's, untouched
    assert spliced[len(prompt) + len(reply):] == client[client.index(eos, len(prompt)):]


def test_a_different_argument_value_is_not_spliced(engine):
    prompt = _render(engine, [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "write it"}])
    reply, text = _model_reply(engine, prompt, json.dumps({"path": "out.txt", "content": "narwhal\n"}))
    engine._own_replies.clear()
    engine._remember_reply(prompt, reply, text, "chat")

    client = _next_turn(engine, json.dumps({"content": "orca\n", "path": "out.txt"}))
    assert engine._splice_own_replies(client, "chat") == client


def test_a_verbatim_history_is_left_alone(engine):
    prompt = _render(engine, [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "write it"}])
    args = json.dumps({"path": "out.txt", "content": "narwhal\n"})
    reply, text = _model_reply(engine, prompt, args)
    engine._own_replies.clear()
    engine._remember_reply(prompt, reply, text, "chat")

    client = _next_turn(engine, args)
    before = engine.replies_spliced
    assert engine._splice_own_replies(client, "chat") == client
    assert engine.replies_spliced == before


def test_an_unrelated_prompt_is_left_alone(engine):
    prompt = _render(engine, [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "write it"}])
    reply, text = _model_reply(engine, prompt, json.dumps({"path": "out.txt", "content": "narwhal\n"}))
    engine._own_replies.clear()
    engine._remember_reply(prompt, reply, text, "chat")

    other = _render(engine, [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "something else"}])
    assert engine._splice_own_replies(other, "chat") == other


def test_no_splice_would_move_an_image_span(engine):
    # a span that ends after the reply's prompt would shift if the reply's
    # token count changed, so such a reply is left as the client sent it
    prompt = _render(engine, [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "write it"}])
    reply, text = _model_reply(engine, prompt, json.dumps({"path": "out.txt", "content": "narwhal\n"}))
    engine._own_replies.clear()
    engine._remember_reply(prompt, reply, text, "chat")

    client = _next_turn(engine, json.dumps({"content": "narwhal\n", "path": "out.txt"}))
    assert engine._splice_own_replies(client, "chat", not_before=len(prompt) + 1) == client
    assert engine._splice_own_replies(client, "chat", not_before=len(prompt)) != client


def test_an_empty_thinking_block_sent_back_as_a_space_still_matches():
    # Hermes returns an empty reasoning block as reasoning_content " "; the
    # renderer then writes "<think> </think>" where the model wrote "<think></think>"
    from cachalot.server.engine import _message_key

    call = {"type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}
    mine = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [call]}
    theirs = {"role": "assistant", "content": "", "reasoning_content": " ", "tool_calls": [dict(call)]}
    assert _message_key(mine) == _message_key(theirs)
    theirs["reasoning_content"] = "I should read b.txt instead."
    assert _message_key(mine) != _message_key(theirs)
