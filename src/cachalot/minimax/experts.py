"""
MiniMax-M3 routed experts as byte ranges of the checkpoint's stacked expert tensors.

The MLX conversion stores each MoE layer's experts stacked, nine tensors per layer,

    model.layers.L.block_sparse_moe.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}   [128, ...]

row-major with the expert first, so expert E of a tensor is one contiguous slice at
`start + E * (size / 128)`. `build_minimax_expert_index` turns that into the same
`(format, {(layer, expert): ExpertEntry})` that `cachalot.glm.experts` builds for GLM's per-expert
tensors, so `ResidentExpertStore`, `ExpertReader` and `StreamingSwitchGLU` are reused unchanged
(w1 = gate_proj, w3 = up_proj, w2 = down_proj). At 3-bit an expert is 24.8 MB.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from cachalot.storage.index import _MX_DTYPE, ExpertEntry, ExpertFormat, TensorRange, read_safetensors_header

_STACKED_RE = re.compile(
    r"^model\.layers\.(\d+)\.block_sparse_moe\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
_PROJ_TO_SLOT = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}


def build_minimax_expert_index(model_path) -> tuple[ExpertFormat, dict[tuple[int, int], ExpertEntry]]:
    root = Path(model_path)
    config = json.loads((root / "config.json").read_text())
    quant = config.get("quantization") or config.get("quantization_config") or {}
    n_experts = int(config["num_local_experts"])
    ranges: dict[tuple[int, int], list[TensorRange]] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for shard in sorted(root.glob("model-*.safetensors")):
        header, data_start = read_safetensors_header(shard)
        for name, meta in header.items():
            m = _STACKED_RE.match(name)
            if not m:
                continue
            layer, proj, field = int(m[1]), m[2], m[3]
            spec = quant.get(f"model.layers.{layer}.block_sparse_moe.switch_mlp.{proj}")
            if isinstance(spec, dict) and (spec.get("bits"), spec.get("group_size")) != (quant.get("bits"), quant.get("group_size")):
                raise ValueError(f"mixed routed-expert quantization is not supported: layer {layer} {proj} {spec}")
            shape = tuple(int(v) for v in meta["shape"])
            if shape[0] != n_experts:
                raise ValueError(f"{name}: leading dimension {shape[0]} is not {n_experts} experts")
            short = f"{_PROJ_TO_SLOT[proj]}.{field}"
            per = shape[1:]
            if shapes.setdefault(short, per) != per:
                raise ValueError(f"{name}: expert shape {per} differs from {shapes[short]}")
            dtypes[short] = _MX_DTYPE[meta["dtype"]]
            a, b = meta["data_offsets"]
            if (b - a) % n_experts:
                raise ValueError(f"{name}: {b - a} bytes do not split into {n_experts} experts")
            step = (b - a) // n_experts
            for e in range(n_experts):
                start = data_start + a + e * step
                ranges.setdefault((layer, e), []).append(
                    TensorRange(name=f"layers.{layer}.experts.{e}.{short}", shard=shard, start=start, end=start + step)
                )
    if not ranges:
        raise ValueError(f"no stacked MiniMax routed-expert tensors under {root}")
    counts = {len(v) for v in ranges.values()}
    if counts != {9}:
        raise ValueError(f"expected 9 tensors per expert, found {sorted(counts)}")
    fmt = ExpertFormat(
        kind="affine",
        bits=int(quant.get("bits", 3)),
        group_size=int(quant.get("group_size", 64)),
        tensor_names=tuple(sorted(shapes)),
        shapes=shapes,
        dtypes=dtypes,
    )
    index = {
        key: ExpertEntry(layer=key[0], expert=key[1], tensors=tuple(sorted(t, key=lambda x: (str(x.shard), x.start))))
        for key, t in ranges.items()
    }
    return fmt, index
