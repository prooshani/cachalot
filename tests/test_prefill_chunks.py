"""HANDOFF section 15.2: long prompts are prefilled in chunks."""

import threading

import mlx.core as mx

from cachalot.model import generation
from cachalot.model.generation import SamplingParams, prefill_chunks, stream_tokens
from cachalot.model.vision_prompt import IMAGE_TOKEN_ID, ImageSpan, PromptImages
from fake_model import ScriptedRuntime


def _span(start, length):
    return ImageSpan(start=start, length=length, digest=f"d{start}", rows=mx.zeros((length, 4)))


def test_chunks_cover_the_suffix_in_order():
    assert prefill_chunks(0, 10, chunk=4) == [(0, 4), (4, 8), (8, 10)]
    assert prefill_chunks(3, 10, chunk=4) == [(3, 7), (7, 10)]
    assert prefill_chunks(0, 4, chunk=4) == [(0, 4)]
    assert prefill_chunks(0, 10, chunk=0) == [(0, 10)]
    assert prefill_chunks(5, 5, chunk=4) == []


def test_a_chunk_never_splits_an_image_span():
    spans = [_span(2, 5)]  # positions 2..6
    chunks = prefill_chunks(0, 12, spans, chunk=4)
    assert chunks == [(0, 7), (7, 11), (11, 12)]
    # a span longer than a chunk goes whole into one call
    assert prefill_chunks(0, 20, [_span(4, 10)], chunk=4) == [(0, 4), (4, 14), (14, 18), (18, 20)]


def test_stream_tokens_prefills_in_chunks(monkeypatch):
    monkeypatch.setattr(generation, "PREFILL_CHUNK", 4)
    rt = ScriptedRuntime(reply="ok")
    events = list(stream_tokens(rt, list(range(10, 20)), SamplingParams(max_new_tokens=2, temperature=0.0)))
    assert rt.prefills == [4, 4, 2]
    assert rt.tokens[:10] == list(range(10, 20))
    assert events[-1].kind == "done"


def test_image_rows_follow_their_chunk(monkeypatch):
    monkeypatch.setattr(generation, "PREFILL_CHUNK", 4)
    rt = ScriptedRuntime(reply="ok")
    tokens = [1, 2] + [IMAGE_TOKEN_ID] * 3 + [3, 4, 5, 6, 7]
    images = PromptImages(tokens=tokens, spans=[_span(2, 3)])
    list(stream_tokens(rt, tokens, SamplingParams(max_new_tokens=1, temperature=0.0), images=images))
    assert rt.prefills == [5, 4, 1]
    assert rt.image_prefills == [3, None, None]


def test_cancel_between_chunks_stops_and_keeps_only_completed_chunks(monkeypatch):
    monkeypatch.setattr(generation, "PREFILL_CHUNK", 4)
    rt = ScriptedRuntime(reply="ok")
    cancel = threading.Event()
    original = rt.prefill_tokens

    def prefill_then_cancel(ids, **kw):
        cancel.set()
        return original(ids, **kw)

    rt.prefill_tokens = prefill_then_cancel
    events = list(stream_tokens(rt, list(range(10, 22)), SamplingParams(max_new_tokens=2, temperature=0.0),
                                cancel=cancel))
    assert rt.prefills == [4]
    assert events[-1].finish_reason == "cancel"
    # only the completed chunk's boundary is cached, so a retry resumes there
    assert [len(s.tokens) for s in rt.prefix_cache._entries] == [4]


def test_chunk_boundaries_leave_snapshots_a_new_session_can_reuse(monkeypatch):
    monkeypatch.setattr(generation, "PREFILL_CHUNK", 4)
    rt = ScriptedRuntime(reply="ok")
    system = list(range(10, 20))  # a long shared prefix, like an agent's system prompt
    list(stream_tokens(rt, system + [50, 51], SamplingParams(max_new_tokens=1, temperature=0.0)))
    # a new conversation: same prefix, different user turn
    events = list(stream_tokens(rt, system + [60, 61, 62], SamplingParams(max_new_tokens=1, temperature=0.0)))
    assert events[0].reused_prefix_tokens == 8  # the last chunk boundary inside the shared prefix


def test_prefix_cache_evicts_least_recently_used():
    from cachalot.model.prefix_cache import PrefixCache
    from test_vision_prompt import _snap

    cache = PrefixCache(max_entries=2)
    cache.add(_snap((1,)))
    cache.add(_snap((2,)))
    assert cache.find((1, 9)) is not None  # touch (1,)
    cache.add(_snap((3,)))  # evicts (2,), not the recently used (1,)
    assert cache.find((1, 9)) is not None
    assert cache.find((2, 9)) is None
