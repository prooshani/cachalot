"""MiniMax-M3 pieces that need no checkpoint: tool-call parsing, the causal chunk mask, the stacked-expert index."""

import json

import mlx.core as mx
import numpy as np

from cachalot.minimax.experts import build_minimax_expert_index
from cachalot.minimax.model import NS, MiniMaxSplitter, parse_minimax_tool_calls

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}, "zip": {"type": "string"},
    "opts": {"type": "object", "properties": {"units": {"type": "string"}, "hourly": {"type": "boolean"}}},
    "tags": {"type": "array", "items": {"type": "string"}}}}}}]


def _call(inner):
    return f"{NS}<tool_call>\n{NS}<invoke name=\"get_weather\">{inner}{NS}</invoke>\n{NS}</tool_call>"


def test_tool_call_types_follow_the_schema():
    text = _call(f"{NS}<city>Paris{NS}</city>{NS}<days>3{NS}</days>{NS}<zip>01234{NS}</zip>")
    (call,) = parse_minimax_tool_calls("I will check." + text, TOOLS)
    assert call["function"] == {"name": "get_weather", "arguments": {"city": "Paris", "days": 3, "zip": "01234"}}


def test_tool_call_nested_objects_and_lists():
    text = _call(f"{NS}<opts>{NS}<units>metric{NS}</units>{NS}<hourly>true{NS}</hourly>{NS}</opts>"
                 f"{NS}<tags>{NS}<item>a{NS}</item>{NS}<item>7{NS}</item>{NS}</tags>")
    (call,) = parse_minimax_tool_calls(text, TOOLS)
    assert call["function"]["arguments"] == {"opts": {"units": "metric", "hourly": True}, "tags": ["a", "7"]}


def test_two_invokes_and_no_block():
    text = (f"{NS}<tool_call>\n{NS}<invoke name=\"a\">{NS}<x>1{NS}</x>{NS}</invoke>\n"
            f"{NS}<invoke name=\"b\">{NS}</invoke>\n{NS}</tool_call>")
    calls = parse_minimax_tool_calls(text)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert calls[0]["function"]["arguments"] == {"x": 1}
    assert parse_minimax_tool_calls("plain answer") == []


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(ids)


def test_splitter_separates_reasoning_content_and_tool_block():
    s = MiniMaxSplitter(_Tok(), thinking=True)
    out = [s.push(c) for c in ["think", "ing", "</mm:think>", "Answer ", "now", NS + "<tool_call>", "x"]]
    out.append(s.flush())
    assert "".join(d.reasoning for d in out) == "thinking"
    assert "".join(d.content for d in out) == "Answer now"
    assert s.in_tool_block


def test_causal_mask_is_aligned_to_the_end_of_the_keys():
    mx.random.seed(1)
    q = mx.random.normal((1, 2, 3, 8))
    k = mx.random.normal((1, 2, 7, 8))
    v = mx.random.normal((1, 2, 7, 8))
    got = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask="causal")
    allowed = np.tril(np.ones((3, 7), dtype=bool), k=4)  # query i sees keys 0..4+i
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mx.array(allowed))
    assert np.allclose(np.array(got), np.array(ref), atol=1e-5)


def test_stacked_expert_index(tmp_path):
    n, rows, cols = 4, 3, 2
    tensors, offset = {}, 0
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for field, dtype, size in (("weight", "U32", 4), ("scales", "BF16", 2), ("biases", "BF16", 2)):
            nbytes = n * rows * cols * size
            tensors[f"model.layers.3.block_sparse_moe.switch_mlp.{proj}.{field}"] = {
                "dtype": dtype, "shape": [n, rows, cols], "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
    header = json.dumps(tensors).encode()
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + bytes(offset))
    (tmp_path / "config.json").write_text(json.dumps(
        {"num_local_experts": n, "quantization": {"bits": 3, "group_size": 64}}))
    fmt, index = build_minimax_expert_index(tmp_path)
    assert len(index) == n and fmt.bits == 3
    e2 = {t.name.rsplit(".", 2)[-2] + "." + t.name.rsplit(".", 1)[-1]: t for t in index[(3, 2)].tensors}
    w1 = e2["w1.weight"]
    base = tensors["model.layers.3.block_sparse_moe.switch_mlp.gate_proj.weight"]["data_offsets"][0]
    assert w1.end - w1.start == rows * cols * 4
    assert w1.start - (8 + len(header)) == base + 2 * rows * cols * 4


def test_decode_hook_is_bit_identical():
    """The decode overlap (routing evaluated first, shared expert queued, prefetch passed) changes no output."""
    from cachalot.minimax.language import MiniMaxM3SparseMoeBlock, ModelArgs

    args = ModelArgs(model_type="minimax_m3", hidden_size=32, intermediate_size=16, dense_intermediate_size=16,
                     shared_intermediate_size=16, num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=2,
                     num_local_experts=8, num_experts_per_tok=2, rms_norm_eps=1e-6, rope_theta=1e4, rotary_dim=8,
                     vocab_size=10)
    mx.random.seed(0)
    moe = MiniMaxM3SparseMoeBlock(args)
    moe.e_score_correction_bias = mx.random.normal((8,)) * 0.1
    table = mx.random.normal((8, 32))
    seen = []

    def switch(x, inds, prefetch=None):
        seen.append(prefetch)
        return x[..., None, :] * table[inds]  # [..., k, D], a stand-in routed expert per index

    moe.switch_mlp = switch
    x = mx.random.normal((1, 1, 32))
    plain = moe(x)
    moe.decode_hook = lambda x, r, inds, w: (moe.shared_experts(x), ["p"])
    hooked = moe(x, residual=x)
    assert mx.array_equal(plain, hooked).item()
    assert seen == [None, ["p"]]


def test_stacked_quantized_linears_are_bit_identical():
    """HANDOFF 18.2: q/k/v (and gate/up) as one quantized matmul give the same rows as separate ones."""
    import mlx.nn as nn

    from cachalot.minimax.language import _stack

    mx.random.seed(3)
    parts = []
    for rows in (256, 64, 64):
        lin = nn.Linear(128, rows, bias=False)
        parts.append(nn.QuantizedLinear.from_linear(lin, group_size=64, bits=3))
    fused = _stack(parts)
    for n in (1, 7):
        x = mx.random.normal((1, n, 128)).astype(mx.bfloat16)
        want = mx.concatenate([p(x) for p in parts], axis=-1)
        assert mx.array_equal(fused(x), want).item()


def test_fast_gemma_norm_matches_the_reference_at_decode_shapes(monkeypatch):
    from cachalot.minimax import language

    mx.random.seed(4)
    norm = language.GemmaRMSNorm(128)
    norm.weight = (mx.random.normal((128,)) * 0.1).astype(mx.bfloat16)
    x = (mx.random.normal((1, 1, 64, 128)) * 3).astype(mx.bfloat16)
    monkeypatch.setattr(language, "FAST_NORM", False)
    ref = norm(x)
    monkeypatch.setattr(language, "FAST_NORM", True)
    assert mx.array_equal(norm(x), ref).item()


def test_warm_set_round_trip(tmp_path):
    """The resident set is saved after a request and read back at startup (HANDOFF 18.2)."""
    from cachalot.cache.resident_store import ResidentExpertStore
    from cachalot.glm.model import GlmModel
    from fakes import EXPERT_BYTES, FAKE_TENSOR_SIZES, FakeReader, make_index

    idx = make_index(2, 4)

    def model():
        m = GlmModel.__new__(GlmModel)
        m.model_path, m.expert_index, m.expert_format = tmp_path, idx, "fmt"
        m.store = ResidentExpertStore(budget_bytes=3 * EXPERT_BYTES, reader=FakeReader(),
                                      tensor_sizes=FAKE_TENSOR_SIZES, transient_slots=2)
        return m

    a = model()
    for key in [(0, 1), (1, 2), (0, 3)]:
        a.store.get(idx[key])
    a._warm_path = tmp_path / "resident-set.json"
    a._save_warm_set()
    b = model()
    b.expert_format = "fmt"
    status = b.start_warm_set(tmp_path / "resident-set.json")
    b._wait_warm_set()
    assert "3 experts" in status and b.store.resident_keys() == [(0, 1), (1, 2), (0, 3)]
    c = model()
    c.expert_format = "other"
    assert "another model" in c.start_warm_set(tmp_path / "resident-set.json")


def test_snapshot_at_shares_buffers_and_a_restore_never_writes_into_them():
    """HANDOFF 18.6: the prompt snapshot is the reply snapshot's buffers trimmed; writing into a restored
    copy must leave both snapshots' data as it was."""
    import mlx.core as mx

    from cachalot.glm.model import GlmModel
    from cachalot.third_party.mlx_vlm.models.cache import KVCache

    cache = [KVCache(), KVCache()]
    k = mx.random.normal((1, 2, 10, 4))
    for c in cache:
        c.update_and_fetch(k, k + 1)
    reply = GlmModel.snapshot_at(tuple(range(10)), cache)
    prompt = GlmModel.snapshot_at(tuple(range(6)), cache, mx.zeros((1, 5)))
    assert [c.offset for c in prompt.cache] == [6, 6] and [c.offset for c in reply.cache] == [10, 10]
    before = [mx.array(c.keys[..., :10, :]) for c in reply.cache]
    live = GlmModel.restore(prompt)
    new = mx.ones((1, 2, 1, 4)) * 7
    for c in live:
        keys, _ = c.update_and_fetch(new, new)
        assert keys.shape[2] == 7 and mx.array_equal(keys[..., 6:, :], new).item()
        assert mx.array_equal(keys[..., :6, :], k[..., :6, :]).item()
    for c, b in zip(reply.cache, before, strict=True):
        assert mx.array_equal(c.keys[..., :10, :], b).item()


def test_consume_takes_the_whole_group():
    from cachalot.glm.model import GlmModel, Snapshot

    class Holder:
        prefix: list

    h = Holder()
    g = object()
    a = Snapshot((1, 2), [], None, consumable=True, group=g)
    b = Snapshot((1, 2, 3), [], None, consumable=True, group=g)
    block = Snapshot((1,), [], None)
    h.prefix = [block, a, b]
    GlmModel._consume(h, b)
    assert h.prefix == [block]
