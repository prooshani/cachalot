from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cachalot.storage.index import read_safetensors_header


@dataclass(frozen=True)
class CheckpointTensor:
    name: str
    shard: Path

    dtype: str
    shape: tuple[int, ...]

    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


def build_tensor_index(
    model_path: str | Path,
) -> dict[str, CheckpointTensor]:
    root = Path(model_path)

    result: dict[str, CheckpointTensor] = {}

    for shard in sorted(
        root.glob("model-*.safetensors")
    ):
        header, data_start = (
            read_safetensors_header(shard)
        )

        for name, meta in header.items():
            if name == "__metadata__":
                continue

            relative_start, relative_end = (
                meta["data_offsets"]
            )

            if name in result:
                raise ValueError(
                    f"Duplicate tensor {name!r}"
                )

            result[name] = CheckpointTensor(
                name=name,
                shard=shard,
                dtype=meta["dtype"],
                shape=tuple(
                    int(x)
                    for x in meta["shape"]
                ),
                start=(
                    data_start
                    + int(relative_start)
                ),
                end=(
                    data_start
                    + int(relative_end)
                ),
            )

    return result
