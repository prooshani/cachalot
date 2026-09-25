"""
GLM-5.3-Flash routed experts streamed from SSD through Cachalot's expert store.

MLX checkpoints of GLM-5.3-Flash (Vontra/GLM-5.3-Flash-MLX-4bit-MTP and the like) keep
every routed expert as its own nine tensors,

    model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.{weight,scales,biases}

affine-quantized (4-bit, group 64 in the 4-bit build). `build_glm_expert_index` maps each
expert to byte ranges in the safetensors shards, so `ResidentExpertStore` reads it straight
into a wired slot, exactly as it does for DeepSeek's banks. Slot names follow the DeepSeek
convention: w1 = gate_proj, w3 = up_proj, w2 = down_proj.

`StreamingSwitchGLU` replaces mlx-vlm's `SwitchGLU` in each MoE layer: same call signature
as mlx-vlm's `OffloadedSwitchGLU` (`(x, indices) -> [..., K, D]`, the caller weights and sums),
computing only the routed experts with `mx.quantized_matmul` on the resident slots.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cachalot.storage.index import (
    _ITEM_BYTES,
    _MX_DTYPE,
    ExpertEntry,
    ExpertFormat,
    TensorRange,
    read_safetensors_header,
)

_GLM_EXPERT_RE = re.compile(
    r"^(?:model\.language_model|language_model\.model)\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
_PROJ_TO_SLOT = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}


def build_glm_expert_index(model_path) -> tuple[ExpertFormat, dict[tuple[int, int], ExpertEntry]]:
    """(format, {(layer, expert): entry}) for a per-expert MLX GLM checkpoint."""
    root = Path(model_path)
    config = json.loads((root / "config.json").read_text())
    quant = config.get("quantization") or config.get("quantization_config") or {}
    entries: dict[tuple[int, int], list[TensorRange]] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for shard in sorted(root.glob("model-*.safetensors")):
        header, data_start = read_safetensors_header(shard)
        for name, meta in header.items():
            m = _GLM_EXPERT_RE.match(name)
            if not m:
                continue
            layer, expert, proj, field = int(m[1]), int(m[2]), m[3], m[4]
            short = f"{_PROJ_TO_SLOT[proj]}.{field}"
            shape = tuple(int(v) for v in meta["shape"])
            if shapes.setdefault(short, shape) != shape:
                raise ValueError(f"{name}: shape {shape} differs from {shapes[short]}")
            dtypes[short] = _MX_DTYPE[meta["dtype"]]
            a, b = meta["data_offsets"]
            entries.setdefault((layer, expert), []).append(
                TensorRange(
                    name=f"layers.{layer}.experts.{expert}.{short}",
                    shard=shard,
                    start=data_start + a,
                    end=data_start + b,
                )
            )
    if not entries:
        raise ValueError(f"no per-expert GLM routed-expert tensors under {root}")
    counts = {len(v) for v in entries.values()}
    if counts != {9}:
        raise ValueError(f"expected 9 tensors per expert, found {sorted(counts)}")
    # every projection must share one quantization (per-module overrides would say otherwise)
    for (layer, expert) in list(entries)[:1]:
        for proj in _PROJ_TO_SLOT:
            spec = quant.get(f"language_model.model.layers.{layer}.mlp.experts.{expert}.{proj}")
            if spec is not None and (spec.get("bits"), spec.get("group_size")) != (quant.get("bits"), quant.get("group_size")):
                raise ValueError(f"mixed routed-expert quantization is not supported: {proj} {spec}")
    fmt = ExpertFormat(
        kind="affine",
        bits=int(quant.get("bits", 4)),
        group_size=int(quant.get("group_size", 64)),
        tensor_names=tuple(sorted(shapes)),
        shapes=shapes,
        dtypes=dtypes,
    )
    index = {
        key: ExpertEntry(layer=key[0], expert=key[1], tensors=tuple(sorted(t, key=lambda x: (str(x.shard), x.start))))
        for key, t in entries.items()
    }
    return fmt, index


def tensor_sizes(fmt: ExpertFormat) -> dict[str, int]:
    """Slot bytes per short tensor name."""
    item = {"uint32": 4, "float32": 4, "float16": 2, "bfloat16": 2, "uint8": 1}
    return {name: int(np.prod(fmt.shapes[name])) * item[fmt.dtypes[name]] for name in fmt.tensor_names}


def _typed(slot, fmt: ExpertFormat, proj: str):
    """(weight, scales, biases) views of a slot, built once per slot (they alias its memory)."""
    views = slot.typed.get(proj)
    if views is None:
        views = tuple(
            slot.arrays[f"{proj}.{field}"].view(getattr(mx, fmt.dtypes[f"{proj}.{field}"])).reshape(
                fmt.shapes[f"{proj}.{field}"]
            )
            for field in ("weight", "scales", "biases")
        )
        slot.typed[proj] = views
    return views


class StreamingSwitchGLU(nn.Module):
    """Routed experts of one GLM MoE layer, read on demand into the shared expert store."""

    def __init__(self, layer: int, store, index, fmt: ExpertFormat, activation):
        super().__init__()
        # plain attributes, not parameters: nothing here is saved, loaded or quantized
        self._layer = layer
        self._store = store
        self._index = index
        self._fmt = fmt
        self._activation = activation

    def _qmm(self, x, slot, proj):
        w, s, b = _typed(slot, self._fmt, proj)
        return mx.quantized_matmul(
            x, w, s, b, transpose=True, group_size=self._fmt.group_size, bits=self._fmt.bits
        )

    def __call__(self, x, indices):
        shape = x.shape
        dim = shape[-1]
        k = indices.shape[-1]
        flat_x = x.reshape(-1, dim)
        routes = np.array(indices.reshape(-1).astype(mx.int32))  # syncs on the router
        order = np.argsort(routes, kind="stable")
        experts, starts = np.unique(routes[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        residents = self._store.get_many([self._index[(self._layer, int(e))] for e in experts])
        outputs = []
        for resident, a, b in zip(residents, starts, ends):
            rows = mx.array((order[a:b] // k).astype(np.int32))
            xe = flat_x[rows]
            gate = self._qmm(xe, resident.slot, "w1")
            up = self._qmm(xe, resident.slot, "w3")
            outputs.append(self._qmm(self._activation(up, gate), resident.slot, "w2"))
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        y = mx.concatenate(outputs, axis=0)[mx.array(inverse.astype(np.int32))]
        y = y.reshape(*shape[:-1], k, dim)
        # the slots may be refilled by the next layer's misses: finish reading them first
        mx.eval(y)
        return y
