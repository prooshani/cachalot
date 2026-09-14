from __future__ import annotations

from dataclasses import dataclass

from cachalot.storage.tensor_index import CheckpointTensor
from cachalot.storage.tensor_loader import (
    ResidentTensor,
    load_resident_tensor,
)


@dataclass(frozen=True)
class ResidentLayer:
    layer_id: int
    tensors: dict[str, ResidentTensor]

    @property
    def size(self) -> int:
        return sum(
            tensor.size
            for tensor in self.tensors.values()
        )


def _should_load_layer_tensor(
    name: str,
) -> bool:
    # Routed experts have their own bounded resident cache.
    if ".ffn.experts." in name:
        return False

    # Giant Engram embedding tables are mmap/sparse.
    if ".engram.embed." in name:
        return False

    return True


def load_resident_layer(
    index: dict[str, CheckpointTensor],
    layer_id: int,
) -> ResidentLayer:
    prefix = f"layers.{layer_id}."

    tensors: dict[str, ResidentTensor] = {}

    for name in sorted(index):
        if not name.startswith(prefix):
            continue

        if not _should_load_layer_tensor(name):
            continue

        tensors[name] = load_resident_tensor(
            index[name]
        )

    if not tensors:
        raise KeyError(
            f"No resident tensors found for layer {layer_id}"
        )

    return ResidentLayer(
        layer_id=layer_id,
        tensors=tensors,
    )
