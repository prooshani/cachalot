from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from cachalot.storage.index import ExpertEntry
from cachalot.storage.store import ExpertPayload
from cachalot.storage.tensors import extract_expert_tensors


@dataclass(frozen=True)
class ResidentExpert:
    layer: int
    expert: int

    w1_weight: mx.array
    w1_scale: mx.array

    w2_weight: mx.array
    w2_scale: mx.array

    w3_weight: mx.array
    w3_scale: mx.array

    @property
    def size(self) -> int:
        return sum(
            tensor.nbytes
            for tensor in (
                self.w1_weight,
                self.w1_scale,
                self.w2_weight,
                self.w2_scale,
                self.w3_weight,
                self.w3_scale,
            )
        )

    def as_model_dict(self) -> dict[str, mx.array]:
        return {
            "w1.weight": self.w1_weight,
            "w1.scale": self.w1_scale,
            "w2.weight": self.w2_weight,
            "w2.scale": self.w2_scale,
            "w3.weight": self.w3_weight,
            "w3.scale": self.w3_scale,
        }


def promote_expert(
    entry: ExpertEntry,
    payload: ExpertPayload,
) -> ResidentExpert:
    tensors = extract_expert_tensors(entry, payload)

    def make(name: str) -> mx.array:
        view = np.frombuffer(
            tensors[name].data,
            dtype=np.uint8,
        )
        return mx.array(view)

    resident = ResidentExpert(
        layer=entry.layer,
        expert=entry.expert,

        w1_weight=make("w1.weight"),
        w1_scale=make("w1.scale"),

        w2_weight=make("w2.weight"),
        w2_scale=make("w2.scale"),

        w3_weight=make("w3.weight"),
        w3_scale=make("w3.scale"),
    )

    # Make sure promotion/copy is complete before the raw payload
    # is allowed to disappear.
    mx.eval(
        resident.w1_weight,
        resident.w1_scale,
        resident.w2_weight,
        resident.w2_scale,
        resident.w3_weight,
        resident.w3_scale,
    )

    return resident
