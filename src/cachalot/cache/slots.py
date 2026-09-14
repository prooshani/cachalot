"""
Pre-allocated, wired expert slots.

Every routed expert of DeepSeek V4.1 Flash has the same six tensors with the
same byte sizes, so the resident cache can be a fixed pool of slots allocated
once at start-up. Experts are read from SSD straight into a slot's unified
memory through a writable NumPy view of the MLX buffer:

    SSD ── preadv ──▶ slot.views[name] (== slot.arrays[name] memory) ──▶ GPU

This removes per-expert mx.array allocation, the memcpy, allocator churn and,
with mx.set_wired_limit, the per-allocation Metal residency-set updates that
made concurrent promotion 25x slower.

Safety rule: a slot may only be released/overwritten after every MLX
operation that read it has been evaluated. The store enforces this by
evicting only after the per-layer synchronisation point (route eval) and by
holding bypass ("transient") slots until the caller reports evaluation.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

TENSOR_NAMES = (
    "w1.weight",
    "w1.scale",
    "w2.weight",
    "w2.scale",
    "w3.weight",
    "w3.scale",
)


@dataclass(eq=False)
class ExpertSlot:
    index: int
    arrays: dict[str, mx.array]
    views: dict[str, np.ndarray]

    @property
    def size(self) -> int:
        return sum(a.nbytes for a in self.arrays.values())


def _writable_view(array: mx.array) -> np.ndarray:
    view = np.array(array, copy=False)
    view.setflags(write=True)
    return view


class ExpertSlotPool:
    def __init__(
        self,
        tensor_sizes: dict[str, int],
        n_slots: int,
        *,
        chunk: int = 64,
        verbose: bool = False,
    ) -> None:
        if n_slots <= 0:
            raise ValueError(f"n_slots must be positive, got {n_slots}")

        missing = set(TENSOR_NAMES) - set(tensor_sizes)
        if missing:
            raise ValueError(f"tensor_sizes missing {sorted(missing)}")

        self.tensor_sizes = dict(tensor_sizes)
        self.slot_bytes = sum(tensor_sizes[n] for n in TENSOR_NAMES)
        self._slots: list[ExpertSlot] = []
        self._free: deque[int] = deque()
        self._cond = threading.Condition()

        for start in range(0, n_slots, chunk):
            batch = []
            for index in range(start, min(start + chunk, n_slots)):
                arrays = {
                    name: mx.zeros((tensor_sizes[name],), dtype=mx.uint8)
                    for name in TENSOR_NAMES
                }
                batch.append((index, arrays))
            mx.eval(*(a for _, arrays in batch for a in arrays.values()))
            for index, arrays in batch:
                views = {name: _writable_view(a) for name, a in arrays.items()}
                self._slots.append(ExpertSlot(index, arrays, views))
                self._free.append(index)
            if verbose:
                print(f"  slots {len(self._slots)}/{n_slots}", flush=True)

    @property
    def capacity(self) -> int:
        return len(self._slots)

    @property
    def free_count(self) -> int:
        with self._cond:
            return len(self._free)

    def try_acquire(self) -> ExpertSlot | None:
        with self._cond:
            if not self._free:
                return None
            return self._slots[self._free.popleft()]

    def acquire(self, timeout: float | None = None) -> ExpertSlot:
        """Block until a slot is free."""
        with self._cond:
            if not self._cond.wait_for(lambda: bool(self._free), timeout=timeout):
                raise TimeoutError("no free expert slot")
            return self._slots[self._free.popleft()]

    def release(self, slot: ExpertSlot) -> None:
        with self._cond:
            self._free.append(slot.index)
            self._cond.notify()

    def release_many(self, slots) -> None:
        with self._cond:
            for slot in slots:
                self._free.append(slot.index)
            self._cond.notify_all()
