"""HANDOFF section 17: the GLM-5.3-Flash backend's pieces that run without the checkpoint."""

import json
import struct

import mlx.core as mx
import numpy as np

from cachalot.glm.engine import _effort, _GlmSplitter, _template_messages, parse_tool_calls
from cachalot.glm.experts import build_glm_expert_index, tensor_sizes
from cachalot.glm.model import GlmModel, _NoProjectedCache
from cachalot.third_party.mlx_vlm.models.cache import ArraysCache, CacheList, KVCache

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}, "zip": {"type": "string"}, "opts": {"type": "object"}}}}}]


def test_tool_calls_are_typed_by_the_schema():
    text = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            "<arg_key>days</arg_key><arg_value>3</arg_value><arg_key>zip</arg_key><arg_value>75001</arg_value>"
            "<arg_key>opts</arg_key><arg_value>{\"a\": [1, 2]}</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call>")
    calls = parse_tool_calls(text, TOOLS)
    assert [c["function"]["name"] for c in calls] == ["get_weather", "get_weather"]
    # a string parameter keeps "75001" a string; integer and object parameters are decoded
    assert calls[0]["function"]["arguments"] == {"city": "Paris", "days": 3, "zip": "75001", "opts": {"a": [1, 2]}}
    assert calls[1]["function"]["arguments"] == {"city": "Rome"}


def test_openai_tool_call_arguments_become_a_mapping_for_the_template():
    msgs = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1}"}}]}]
    out = _template_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "{\"x\": 1}"  # caller's copy untouched


def test_reasoning_effort_maps_onto_glms_three_levels():
    assert _effort("low") == "low" and _effort("medium") == "high" and _effort("high") == "high"
    assert _effort("max") is None and _effort(None) is None and _effort(90) is None and _effort(10) == "low"


class _CharTokenizer:
    """One character per token."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_splitter_separates_reasoning_content_and_a_held_back_tool_block():
    s = _GlmSplitter(_CharTokenizer(), thinking=True)
    text = "plan it</think>Sure.<tool_call>f<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
    reasoning = content = ""
    for ch in text:
        d = s.push(ord(ch))
        reasoning += d.reasoning
        content += d.content
    tail = s.flush()
    reasoning += tail.reasoning
    content += tail.content
    assert reasoning == "plan it"
    assert content == "Sure."
    assert s.in_tool_block and parse_tool_calls(s.text)[0]["function"]["arguments"] == {"x": 1}


def _write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        raw = arr.tobytes()
        dtype = {np.dtype("uint32"): "U32", np.dtype("uint16"): "BF16"}[arr.dtype]
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_expert_index_maps_glm_projections_onto_slot_names(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 64}}))
    tensors = {}
    for e in range(2):
        for proj, (rows, cols) in {"gate_proj": (8, 4), "up_proj": (8, 4), "down_proj": (16, 2)}.items():
            base = f"model.language_model.layers.3.mlp.experts.{e}.{proj}"
            tensors[f"{base}.weight"] = np.full((rows, cols), e, dtype=np.uint32)
            tensors[f"{base}.scales"] = np.ones((rows, 1), dtype=np.uint16)
            tensors[f"{base}.biases"] = np.zeros((rows, 1), dtype=np.uint16)
    _write_safetensors(tmp_path / "model-00001-of-00001.safetensors", tensors)
    fmt, index = build_glm_expert_index(tmp_path)
    assert sorted(index) == [(3, 0), (3, 1)]
    assert fmt.shapes["w1.weight"] == (8, 4) and fmt.shapes["w2.weight"] == (16, 2)  # gate -> w1, down -> w2
    assert fmt.dtypes["w3.scales"] == "bfloat16" and (fmt.bits, fmt.group_size) == (4, 64)
    assert sum(tensor_sizes(fmt).values()) == sum(t.size for t in index[(3, 0)].tensors)


def test_a_snapshot_is_not_changed_by_later_in_place_cache_updates():
    kv, arrays = KVCache(), ArraysCache(size=2)
    kv.update_and_fetch(mx.ones((1, 1, 3, 4)), mx.ones((1, 1, 3, 4)))
    arrays[0] = mx.zeros((1, 3))
    cache = [CacheList(kv, _NoProjectedCache()), arrays]
    snap = GlmModel.snapshot((1, 2, 3), cache)
    # the live cache moves on: an in-place write and a rebind
    kv.keys[..., 0:1, :] = 7.0
    arrays[0] = mx.full((1, 3), 5.0)
    restored = GlmModel.restore(snap)
    assert mx.all(restored[0][0].keys[..., :3, :] == 1).item()
    assert mx.all(restored[1][0] == 0).item()
    assert restored[0][1].size() == -1  # the projected-cache stand-in survives the round trip
