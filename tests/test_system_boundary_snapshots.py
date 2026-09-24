"""HANDOFF section 15.4: a snapshot where the system prompt ends, kept on disk across restarts."""

import mlx.core as mx
import numpy as np
from fastapi.testclient import TestClient

from cachalot.model import generation, snapshot_store
from cachalot.model.generation import SamplingParams, prefill_chunks, stream_tokens
from cachalot.model.prefix_cache import PrefixCache, SequenceSnapshot
from cachalot.model.vision_prompt import ImageSpan
from cachalot.server.app import create_app
from cachalot.server.engine import Engine
from fake_model import FakeEncoding, FakeModel, ScriptedRuntime


def _span(start, length):
    return ImageSpan(start=start, length=length, digest=f"d{start}", rows=mx.zeros((length, 4)))


# ---------------------------------------------------------------- chunk cuts


def test_a_cut_ends_a_prefill_call_there():
    assert prefill_chunks(0, 10, chunk=4, cuts=(6,)) == [(0, 4), (4, 6), (6, 10)]
    assert prefill_chunks(0, 10, chunk=4, cuts=(4,)) == [(0, 4), (4, 8), (8, 10)]
    # a prompt shorter than one chunk is still cut
    assert prefill_chunks(0, 10, chunk=16, cuts=(7,)) == [(0, 7), (7, 10)]
    # chunking disabled: only the cut
    assert prefill_chunks(0, 10, chunk=0, cuts=(3,)) == [(0, 3), (3, 10)]


def test_cuts_outside_the_suffix_or_inside_an_image_are_ignored():
    assert prefill_chunks(5, 10, chunk=16, cuts=(0, 5, 10, 12)) == [(5, 10)]
    assert prefill_chunks(0, 12, [_span(2, 5)], chunk=16, cuts=(4,)) == [(0, 12)]


def test_stream_tokens_snapshots_at_the_boundary(monkeypatch):
    monkeypatch.setattr(generation, "PREFILL_CHUNK", 4)
    rt = ScriptedRuntime(reply="ok")
    persisted = []
    rt.prefix_cache.persist = persisted.append
    prompt = list(range(10, 20))
    list(stream_tokens(rt, prompt, SamplingParams(max_new_tokens=1, temperature=0.0), boundaries=(6,)))
    assert rt.prefills == [4, 2, 4]
    assert [len(s.tokens) for s in persisted] == [6]
    # a second conversation sharing the first 6 tokens reuses all 6
    other = prompt[:6] + [99, 98]
    events = list(stream_tokens(rt, other, SamplingParams(max_new_tokens=1, temperature=0.0)))
    assert events[0].reused_prefix_tokens == 6


def test_a_failing_persist_does_not_fail_the_request():
    cache = PrefixCache()

    def boom(_):
        raise OSError("disk full")

    cache.persist = boom
    snap = SequenceSnapshot((1, 2), 2, None, {}, {}, {}, {}, {}, [], None, None, None, None)
    cache.add(snap, boundary=True)
    assert len(cache) == 1


# ---------------------------------------------------------------- the server


class SystemPrefixEncoding(FakeEncoding):
    """Renders a lone system message without the assistant header, like the official encoding."""

    @staticmethod
    def encode_messages(messages, thinking_mode="chat", reasoning_effort=None, return_multi_modal_data=False, **kw):
        text = FakeEncoding.encode_messages(messages, thinking_mode, reasoning_effort, **kw)
        if messages and messages[-1]["role"] == "system":
            text = text[: text.rindex("<assistant>")]
        return (text, {"images": []}) if return_multi_modal_data else text


def _body(user):
    return {
        "model": "m",
        "max_tokens": 2,
        "messages": [{"role": "system", "content": "S" * 30}, {"role": "user", "content": user}],
    }


def test_a_new_conversation_reuses_the_whole_system_prompt():
    rt = ScriptedRuntime(reply="ok")
    engine = Engine(FakeModel(rt), model_id="m", encoding=SystemPrefixEncoding())
    client = TestClient(create_app(engine))
    client.post("/v1/chat/completions", json=_body("first question"))
    r = client.post("/v1/chat/completions", json=_body("an unrelated second one"))
    assert r.json()["usage"]["cachalot"]["reused_prefix_tokens"] == len("<system>" + "S" * 30)


def test_no_boundary_when_the_system_block_is_not_a_token_prefix():
    # FakeEncoding appends "<assistant>" even after a lone system message
    from types import SimpleNamespace

    engine = Engine(FakeModel(ScriptedRuntime()), model_id="m", encoding=FakeEncoding())
    body = _body("q")
    req = SimpleNamespace(messages=body["messages"], tools=None, response_format=None,
                          thinking_mode="chat", reasoning_effort=None)
    tokens = engine._encode_chat(req)[0]
    assert engine.system_prefix_len(req, tokens) == 0


# ---------------------------------------------------------------- disk


def _random_snapshot(n_tokens=9, logits=True):
    rng = np.random.default_rng(0)

    def arr(*shape, dtype=mx.bfloat16):
        return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(dtype)

    return SequenceSnapshot(
        tokens=tuple(range(100, 100 + n_tokens)),
        position=n_tokens,
        logits=arr(1, 32, dtype=mx.float32) if logits else None,
        windows={0: arr(8, 4), 1: arr(8, 4)},
        compressed_caches={2: arr(5, 4), 20: arr(10, 4)},
        compressor_kv={2: arr(1, 4, 4)},
        compressor_score={2: arr(1, 4, 4)},
        indexer_k={2: arr(5, 4)},
        engram_history=[1, 2**40, 7],
        shared_compress_kv=None,
        shared_index_k_layer=2,
        shared_topk_idxs=mx.array([[0, 3, 1]], dtype=mx.int32),
        shared_candidates=None,
        image_spans=((1, 3, "abc"),),
        shared_compress_kv_layer=20,
    )


def _same(a: SequenceSnapshot, b: SequenceSnapshot):
    assert a.tokens == b.tokens and a.position == b.position
    assert a.engram_history == b.engram_history and a.image_spans == b.image_spans
    assert a.shared_index_k_layer == b.shared_index_k_layer
    assert a.shared_compress_kv_layer == b.shared_compress_kv_layer
    for name in ("windows", "compressed_caches", "compressor_kv", "compressor_score", "indexer_k"):
        ga, gb = getattr(a, name), getattr(b, name)
        assert ga.keys() == gb.keys()
        for k in ga:
            assert ga[k].dtype == gb[k].dtype and mx.array_equal(ga[k], gb[k]).item()
    for name in ("logits", "shared_compress_kv", "shared_topk_idxs", "shared_candidates"):
        va, vb = getattr(a, name), getattr(b, name)
        assert (va is None) == (vb is None)
        if va is not None:
            assert mx.array_equal(va, vb).item()


def test_disk_round_trip_is_exact(tmp_path):
    snap = _random_snapshot()
    snapshot_store.save(snap, tmp_path, "id-1")
    (loaded,) = snapshot_store.load_all(tmp_path, "id-1")
    _same(snap, loaded)


def test_another_identity_is_not_loaded_but_kept(tmp_path):
    snapshot_store.save(_random_snapshot(), tmp_path, "bank-a")
    assert snapshot_store.load_all(tmp_path, "bank-b") == []
    assert len(snapshot_store.load_all(tmp_path, "bank-a")) == 1


def test_only_the_newest_files_are_kept(tmp_path):
    for n in range(3, 9):
        snapshot_store.save(_random_snapshot(n, logits=False), tmp_path, "id", keep=4)
    loaded = snapshot_store.load_all(tmp_path, "id")
    assert sorted(len(s.tokens) for s in loaded) == [5, 6, 7, 8]


def test_identity_changes_with_the_bank_and_the_version(tmp_path):
    model, bank = tmp_path / "model", tmp_path / "bank"
    model.mkdir()
    bank.mkdir()
    (model / "config.json").write_text("{}")
    (bank / "config.json").write_text("{}")
    (bank / "model-1.safetensors").write_bytes(b"x")
    base = snapshot_store.runtime_identity(model, bank, 65536, "1.0")
    assert base == snapshot_store.runtime_identity(model, bank, 65536, "1.0")
    assert base != snapshot_store.runtime_identity(model, bank, 32768, "1.0")
    assert base != snapshot_store.runtime_identity(model, bank, 65536, "1.1")
    (bank / "model-1.safetensors").write_bytes(b"xy")
    assert base != snapshot_store.runtime_identity(model, bank, 65536, "1.0")


def _snap(tokens):
    return SequenceSnapshot(tuple(tokens), len(tokens), None, {}, {}, {}, {}, {}, [], None, None, None, None)


def test_a_long_session_does_not_evict_the_system_block():
    # HANDOFF section 15.5: two snapshots per request pushed the system block
    # out of the LRU, and the next new session paid the cold prefill again.
    cache = PrefixCache(max_entries=4)
    system = _snap(range(10))
    cache.add(system, boundary=True)
    for turn in range(20):
        cache.add(_snap(list(range(10)) + [1000 + turn] * (turn + 1)))
    assert len(cache) == 4
    assert cache.find(tuple(range(10)) + (7, 7, 7)) is system


def test_pinned_snapshots_are_capped_and_the_oldest_unpins_first():
    cache = PrefixCache(max_entries=4, max_pinned=2)
    blocks = [_snap([b] * 5) for b in range(3)]
    for s in blocks:
        cache.add(s, boundary=True)
    for turn in range(10):
        cache.add(_snap([9] * (turn + 1)))
    kept = {s.tokens for s in cache._entries}
    assert blocks[0].tokens not in kept  # unpinned, then evicted like any entry
    assert blocks[1].tokens in kept and blocks[2].tokens in kept


def test_chunk_snapshots_inside_the_system_block_are_pinned(monkeypatch):
    # HANDOFF section 15.7: Hermes compression inserts a tool mid-list, so the
    # system block changes part-way through; the chunk snapshot before the
    # change must survive a long session to be reused.
    monkeypatch.setattr(generation, "PREFILL_CHUNK", 4)
    rt = ScriptedRuntime(reply="ok")
    rt.prefix_cache = PrefixCache(max_entries=6)
    persisted = []
    rt.prefix_cache.persist = persisted.append
    system = list(range(100, 110))  # boundary at 10: chunks at 4 and 8 lie inside it
    prompt = system + [7, 7]
    list(stream_tokens(rt, prompt, SamplingParams(max_new_tokens=1, temperature=0.0), boundaries=(10,)))
    assert [len(s.tokens) for s in persisted] == [10]  # only the block itself goes to disk
    for turn in range(20):  # a long session's per-turn snapshots
        rt.prefix_cache.add(_snap(prompt + [1000 + turn] * (turn + 1)))
    changed = system[:9] + [555] + [7, 7]  # the block diverges at token 9
    events = list(stream_tokens(rt, changed, SamplingParams(max_new_tokens=1, temperature=0.0)))
    assert events[0].reused_prefix_tokens == 8


def test_chunks_after_the_system_block_are_not_pinned(monkeypatch):
    monkeypatch.setattr(generation, "PREFILL_CHUNK", 4)
    rt = ScriptedRuntime(reply="ok")
    list(stream_tokens(rt, list(range(20)), SamplingParams(max_new_tokens=1, temperature=0.0), boundaries=(6,)))
    pinned = sorted(len(t) for t in rt.prefix_cache._pinned)
    assert pinned == [4, 6]  # the chunk inside the block and the block; not 10, 14, 18
