from __future__ import annotations

from dataclasses import dataclass

from cachalot.storage.tensor_index import CheckpointTensor
from cachalot.storage.tensor_loader import (
    ResidentTensor,
    load_resident_tensor,
)


@dataclass(frozen=True)
class ResidentTrunk:
    tensors: dict[str, ResidentTensor]

    @property
    def size(self) -> int:
        return sum(
            tensor.size
            for tensor in self.tensors.values()
        )

    def get(self, name: str) -> ResidentTensor:
        return self.tensors[name]


def should_load_text_trunk_tensor(
    name: str,
) -> bool:
    # Routed experts use ResidentExpertStore.
    if ".ffn.experts." in name:
        return False

    # Giant Engram tables use sparse mmap access.
    if ".engram.embed." in name:
        return False

    lower = name.lower()

    # Vision + MTP paths are deliberately excluded from the
    # first text-only runtime.
    if (
        name.startswith("mtp.")
        or ".mtp." in name
        or name.startswith("vision.")
        or ".vision." in name
        or name.startswith("aligner.")
        or ".aligner." in name
        or "image_start" in lower
        or "image_end" in lower
        or "image_newline" in lower
    ):
        return False

    return True


def load_resident_trunk(
    index: dict[str, CheckpointTensor],
) -> ResidentTrunk:
    tensors: dict[str, ResidentTensor] = {}

    for name in sorted(index):
        if not should_load_text_trunk_tensor(name):
            continue

        tensors[name] = load_resident_tensor(
            index[name]
        )

    return ResidentTrunk(
        tensors=tensors,
    )
