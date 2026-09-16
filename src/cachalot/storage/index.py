from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
# oMLX-converted V4.1 checkpoints stack every layer's routed experts into one
# tensor per projection field: [n_experts, rows, cols].
_STACKED_RE = re.compile(
    r"^language_model\.layers\.(\d+)\.ffn\.experts\.(w[123])\.(weight|scales|biases)$"
)
_ITEM_BYTES = {"U32": 4, "F32": 4, "F16": 2, "BF16": 2, "U8": 1}
_MX_DTYPE = {"U32": "uint32", "F32": "float32", "F16": "float16", "BF16": "bfloat16", "U8": "uint8"}


@dataclass(frozen=True)
class TensorRange:
    name: str
    shard: Path
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ExpertEntry:
    layer: int
    expert: int
    tensors: tuple[TensorRange, ...]


@dataclass(frozen=True)
class ExpertFormat:
    """
    How a routed expert's slot bytes are laid out and multiplied.

    kind "fp4": the shipped checkpoint (E2M1 nibbles + UE8M0 scales, six
    tensors, handled by the fused FP4 kernels). kind "affine": MLX affine
    quantization (packed uint32 weights + scales + biases per projection),
    multiplied with mx.quantized_matmul; shapes/dtypes describe the slot
    views per short tensor name ("w1.weight", "w1.scales", ...).
    """

    kind: str
    bits: int
    group_size: int
    tensor_names: tuple[str, ...]
    shapes: dict[str, tuple[int, int]]
    dtypes: dict[str, str]


FP4_FORMAT = ExpertFormat(
    kind="fp4",
    bits=4,
    group_size=32,
    tensor_names=("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale"),
    shapes={},
    dtypes={},
)


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))

    data_start = 8 + header_len
    return header, data_start


def build_expert_index(model_path: str | Path) -> dict[tuple[int, int], ExpertEntry]:
    root = Path(model_path)
    entries: dict[tuple[int, int], list[TensorRange]] = {}

    for shard in sorted(root.glob("model-*.safetensors")):
        try:
            header, data_start = read_safetensors_header(shard)
        except (OSError, json.JSONDecodeError, struct.error):
            # Allows indexing while other shards are still downloading.
            continue

        for name, meta in header.items():
            match = _EXPERT_RE.match(name)
            if not match:
                continue

            layer = int(match.group(1))
            expert = int(match.group(2))
            relative_start, relative_end = meta["data_offsets"]
            start = data_start + relative_start
            end = data_start + relative_end

            entries.setdefault((layer, expert), []).append(
                TensorRange(
                    name=name,
                    shard=shard,
                    start=start,
                    end=end,
                )
            )

    return {
        key: ExpertEntry(
            layer=key[0],
            expert=key[1],
            tensors=tuple(sorted(tensors, key=lambda x: x.start)),
        )
        for key, tensors in entries.items()
    }


def build_stacked_expert_index(
    model_path: str | Path,
    quantized_modules: dict,
) -> tuple[ExpertFormat, dict[tuple[int, int], ExpertEntry]]:
    """Index an oMLX-converted checkpoint whose experts are stacked per layer."""
    root = Path(model_path)
    entries: dict[tuple[int, int], list[TensorRange]] = {}
    shapes: dict[str, tuple[int, int]] = {}
    dtypes: dict[str, str] = {}
    bits: int | None = None
    group: int | None = None
    for shard in sorted(root.glob("model-*.safetensors")):
        try:
            header, data_start = read_safetensors_header(shard)
        except (OSError, json.JSONDecodeError, struct.error):
            continue
        for name, meta in header.items():
            match = _STACKED_RE.match(name)
            if not match:
                continue
            layer, proj, field = int(match.group(1)), match.group(2), match.group(3)
            spec = quantized_modules.get(f"language_model.layers.{layer}.ffn.experts.{proj}")
            if spec is None or spec.get("mode") != "affine":
                raise ValueError(f"{name}: routed experts must be affine-quantized, got {spec}")
            if bits is None:
                bits, group = int(spec["bits"]), int(spec.get("group_size", 64))
            elif (int(spec["bits"]), int(spec.get("group_size", 64))) != (bits, group):
                raise ValueError(f"{name}: mixed expert quantization ({spec} vs {bits}-bit/{group})")
            n_experts, rows, cols = (int(v) for v in meta["shape"])
            short = f"{proj}.{field}"
            shape = (rows, cols)
            if shapes.setdefault(short, shape) != shape:
                raise ValueError(f"{name}: shape {shape} differs from {shapes[short]}")
            dtypes[short] = _MX_DTYPE[meta["dtype"]]
            row_bytes = rows * cols * _ITEM_BYTES[meta["dtype"]]
            base = data_start + meta["data_offsets"][0]
            for expert in range(n_experts):
                entries.setdefault((layer, expert), []).append(
                    TensorRange(
                        name=f"layers.{layer}.ffn.experts.{expert}.{short}",
                        shard=shard,
                        start=base + expert * row_bytes,
                        end=base + (expert + 1) * row_bytes,
                    )
                )
    if bits is None:
        raise ValueError(f"no stacked routed-expert tensors under {root}")
    fmt = ExpertFormat(
        kind="affine",
        bits=bits,
        group_size=group,
        tensor_names=tuple(sorted(shapes)),
        shapes=shapes,
        dtypes=dtypes,
    )
    index = {
        key: ExpertEntry(layer=key[0], expert=key[1], tensors=tuple(sorted(tensors, key=lambda x: x.start)))
        for key, tensors in entries.items()
    }
    return fmt, index


def detect_expert_bank(model_path: str | Path) -> tuple[ExpertFormat, dict[tuple[int, int], ExpertEntry]]:
    """(format, index) of the routed experts under `model_path`: oMLX-converted stacked affine or shipped FP4."""
    root = Path(model_path)
    config = root / "config.json"
    if config.exists():
        try:
            spec = json.loads(config.read_text()).get("omlx_deepseek_v41")
        except (OSError, json.JSONDecodeError):
            spec = None
        if spec:
            return build_stacked_expert_index(root, spec.get("quantized_modules", {}))
    return FP4_FORMAT, build_expert_index(root)


@dataclass(frozen=True)
class ReadRange:
    shard: Path
    start: int
    end: int
    tensors: tuple[TensorRange, ...]

    @property
    def size(self) -> int:
        return self.end - self.start


def merge_contiguous_ranges(entry: ExpertEntry) -> tuple[ReadRange, ...]:
    if not entry.tensors:
        return ()

    groups: list[list[TensorRange]] = []

    for tensor in entry.tensors:
        if (
            groups
            and groups[-1][-1].shard == tensor.shard
            and groups[-1][-1].end == tensor.start
        ):
            groups[-1].append(tensor)
        else:
            groups.append([tensor])

    return tuple(
        ReadRange(
            shard=group[0].shard,
            start=group[0].start,
            end=group[-1].end,
            tensors=tuple(group),
        )
        for group in groups
    )
