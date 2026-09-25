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
