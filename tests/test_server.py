import json

import pytest
from fastapi.testclient import TestClient

from cachalot.server.app import ServerConfig, create_app
from cachalot.server.engine import Engine
from fake_model import FakeEncoding, FakeModel, ScriptedRuntime


def make_client(reply="Hello, whale!", thinking=None, **cfg):
    rt = ScriptedRuntime(reply=reply, thinking=thinking)
    engine = Engine(FakeModel(rt), model_id="test-model", encoding=FakeEncoding())
    app = create_app(engine, ServerConfig(**cfg))
    return TestClient(app), rt


def test_models_and_health():
    client, _ = make_client()
    assert client.get("/health").json()["status"] == "ok"
    data = client.get("/v1/models").json()
    assert data["data"][0]["id"] == "test-model"


def test_chat_completion_non_stream():
    client, rt = make_client()
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello, whale!"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == len("Hello, whale!")
    assert body["usage"]["prompt_tokens"] == len("<user>hi<assistant>")


def test_chat_completion_stream_sse():
    client, _ = make_client()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True,
              "stream_options": {"include_usage": True}},
    ) as r:
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hello, whale!"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == len("Hello, whale!")


def test_thinking_mode_splits_reasoning():
    client, _ = make_client(reply="Answer.", thinking="deep thought", default_thinking=True)
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
    msg = r.json()["choices"][0]["message"]
    assert msg["reasoning_content"] == "deep thought"
    assert msg["content"] == "Answer."


def test_stop_sequence_truncates():
    client, _ = make_client(reply="one, two, three")
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "stop": [","]})
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "one"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_prefix_cache_reused_across_turns():
    client, rt = make_client(reply="ok")
    first = [{"role": "user", "content": "hello there"}]
    client.post("/v1/chat/completions", json={"messages": first})
    second = first + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
    r = client.post("/v1/chat/completions", json={"messages": second})
    usage = r.json()["usage"]["cachalot"]
    assert usage["reused_prefix_tokens"] > 0
    # second prefill only covered the suffix
    assert rt.prefills[-1] < usage["reused_prefix_tokens"] + rt.prefills[-1]
    assert rt.prefills[-1] < rt.prefills[0] + 10


def test_completions_endpoint():
    client, _ = make_client(reply="abc")
    r = client.post("/v1/completions", json={"prompt": "xyz", "max_tokens": 10})
    assert r.json()["choices"][0]["text"] == "abc"


def test_api_key_required():
    client, _ = make_client(api_key="secret")
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_tools_attach_to_system_message():
    client, rt = make_client(reply="ok")
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "tools": tools})
    assert r.status_code == 200
    # FakeEncoding renders <tools:1> inside the system message => prompt longer than without tools
    assert rt.prefills[-1] == len("<system><tools:1><user>x<assistant>")


@pytest.mark.parametrize("n", [0, 2])
def test_rejects_bad_n(n):
    client, _ = make_client()
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "n": n})
    assert r.status_code == 400
