from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from cachalot.cache.slots import ExpertSlot


@dataclass(eq=False)
class ResidentExpert:
    """A routed expert living in a pool slot. Arrays alias the slot's memory."""

    layer: int
    expert: int
    slot: ExpertSlot
    transient: bool = False

    @property
    def w1_weight(self) -> mx.array:
        return self.slot.arrays["w1.weight"]

    @property
    def w1_scale(self) -> mx.array:
        return self.slot.arrays["w1.scale"]

    @property
    def w2_weight(self) -> mx.array:
        return self.slot.arrays["w2.weight"]

    @property
    def w2_scale(self) -> mx.array:
        return self.slot.arrays["w2.scale"]

    @property
    def w3_weight(self) -> mx.array:
        return self.slot.arrays["w3.weight"]

    @property
    def w3_scale(self) -> mx.array:
        return self.slot.arrays["w3.scale"]

    @property
    def size(self) -> int:
        return self.slot.size

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer, self.expert)

    def as_model_dict(self) -> dict[str, mx.array]:
        return self.slot.arrays
