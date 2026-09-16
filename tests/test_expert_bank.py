"""Affine (oMLX-style stacked) expert bank: index, slot reads and expert math."""
from __future__ import annotations

import json
import struct

import mlx.core as mx
import numpy as np
import pytest

from cachalot.cache.resident_store import tensor_sizes_from_entry
from cachalot.cache.slots import ExpertSlotPool
from cachalot.model.expert_affine import affine_expert_forward, affine_expert_forward_batched
from cachalot.storage.index import FP4_FORMAT, detect_expert_bank
from cachalot.storage.reader import ExpertReader

HIDDEN, INTER, N_EXPERTS, N_LAYERS, BITS, GROUP = 128, 64, 3, 2, 3, 64
_ST_DTYPE = {mx.uint32: "U32", mx.float16: "F16"}


def _write_safetensors(path, tensors: dict[str, mx.array]) -> None:
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        raw = np.array(arr).tobytes() if arr.dtype != mx.float16 else np.array(arr.astype(mx.float32)).astype(np.float16).tobytes()
        header[name] = {"dtype": _ST_DTYPE[arr.dtype], "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    h = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(h)))
        fh.write(h)
        for b in blobs:
            fh.write(b)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    """A tiny oMLX-layout checkpoint: stacked 3-bit affine experts for two layers, plus the dense weights they encode."""
    root = tmp_path_factory.mktemp("oq3e")
    mx.random.seed(7)
    dense = {}   # (layer, expert) -> {"w1": W [INTER, HIDDEN], "w2": [HIDDEN, INTER], "w3": ...} as dequantized fp32
    modules = {}
    tensors = {}
    for layer in range(N_LAYERS):
        for proj, shape in (("w1", (INTER, HIDDEN)), ("w2", (HIDDEN, INTER)), ("w3", (INTER, HIDDEN))):
            qs, ss, bs = [], [], []
            for e in range(N_EXPERTS):
                w = mx.random.normal(shape).astype(mx.bfloat16)
                q, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
                s, b = s.astype(mx.float16), b.astype(mx.float16)
                dense.setdefault((layer, e), {})[proj] = mx.dequantize(q, s, b, group_size=GROUP, bits=BITS).astype(mx.float32)
                qs.append(q)
                ss.append(s)
                bs.append(b)
            name = f"language_model.layers.{layer}.ffn.experts.{proj}"
            modules[name] = {"bits": BITS, "group_size": GROUP, "mode": "affine", "quantize_input": True}
            tensors[name + ".weight"] = mx.stack(qs)
            tensors[name + ".scales"] = mx.stack(ss)
            tensors[name + ".biases"] = mx.stack(bs)
    mx.eval(*tensors.values(), *(w for d in dense.values() for w in d.values()))
    _write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    (root / "config.json").write_text(json.dumps({"omlx_deepseek_v41": {"version": 1, "quantized_modules": modules}}))
    return root, dense


def test_detect_stacked_bank_builds_per_expert_ranges(bank):
    root, _ = bank
    fmt, index = detect_expert_bank(root)
    assert fmt.kind == "affine" and fmt.bits == BITS and fmt.group_size == GROUP
    assert len(fmt.tensor_names) == 9 and set(fmt.tensor_names) == set(tensor_sizes_from_entry(next(iter(index.values()))))
    assert len(index) == N_LAYERS * N_EXPERTS
    assert fmt.shapes["w1.weight"] == (INTER, HIDDEN * BITS // 32) and fmt.dtypes["w1.weight"] == "uint32"
    assert fmt.shapes["w2.scales"] == (HIDDEN, INTER // GROUP) and fmt.dtypes["w2.scales"] == "float16"
    sizes = tensor_sizes_from_entry(index[(1, 2)])
    assert sizes["w1.weight"] == INTER * (HIDDEN * BITS // 32) * 4
    assert sizes["w1.scales"] == INTER * (HIDDEN // GROUP) * 2


def test_detect_falls_back_to_fp4_without_omlx_config(tmp_path):
    fmt, index = detect_expert_bank(tmp_path)
    assert fmt is FP4_FORMAT and index == {}


def test_slot_pool_accepts_nine_tensor_layout(bank):
    root, _ = bank
    _, index = detect_expert_bank(root)
    sizes = tensor_sizes_from_entry(index[(0, 0)])
    pool = ExpertSlotPool(sizes, 2)
    slot = pool.try_acquire()
    assert set(slot.arrays) == set(sizes) and slot.size == sum(sizes.values())


def test_read_expert_matches_dense_math(bank):
    root, dense = bank
    fmt, index = detect_expert_bank(root)
    reader = ExpertReader(bypass_page_cache=False)
    try:
        for key in ((0, 0), (1, 2)):
            sizes = tensor_sizes_from_entry(index[key])
            views = {n: bytearray(sz) for n, sz in sizes.items()}
            assert reader.read_expert_into(index[key], views) == sum(sizes.values())
            arrays = {n: mx.array(np.frombuffer(v, dtype=np.uint8)) for n, v in views.items()}
            x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
            got = affine_expert_forward(x, arrays, fmt, 0.7, swiglu_limit=10.0)
            d = dense[key]
            xb = x.astype(mx.float32)
            gate = mx.minimum((d["w1"] @ xb).astype(mx.bfloat16).astype(mx.float32), 10.0)
            up = mx.clip((d["w3"] @ xb).astype(mx.bfloat16).astype(mx.float32), -10.0, 10.0)
            hidden = (mx.sigmoid(gate) * gate * up * 0.7).astype(mx.bfloat16).astype(mx.float32)
            want = (d["w2"] @ hidden).astype(mx.bfloat16).astype(mx.float32)
            scale = float(mx.max(mx.abs(want)))
            assert float(mx.max(mx.abs(got - want))) < 0.02 * scale, key
            xm = mx.random.normal((4, HIDDEN)).astype(mx.bfloat16)
            batched = affine_expert_forward_batched(xm, arrays, fmt, mx.array([0.7, 0.1, 1.0, 0.3]), swiglu_limit=10.0)
            single = affine_expert_forward(xm[1], arrays, fmt, 0.1, swiglu_limit=10.0)
            assert float(mx.max(mx.abs(batched[1] - single))) < 0.02 * max(1e-6, float(mx.max(mx.abs(single))))
    finally:
        reader.close()
