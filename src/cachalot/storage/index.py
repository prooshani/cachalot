from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path


_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)


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
