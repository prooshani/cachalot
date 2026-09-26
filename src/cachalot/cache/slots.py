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
from dataclasses import dataclass, field

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
    typed: dict[str, tuple[mx.array, mx.array, mx.array]] = field(default_factory=dict)
    """
    Per-projection (weight, scales, biases) views of an affine bank's slot
    bytes, built once and reused for every expert that later occupies this
    slot. `.view(dtype).reshape(shape)` on a contiguous buffer is zero-copy in
    MLX, so these arrays alias the slot's memory exactly as `views` does and
    keep reflecting it after a refill -- there is nothing to invalidate. Only
    the shapes and dtypes matter, and one bank has one format for every expert.
    Without this, moe_layer_forward rebuilt nine views per expert per layer:
    4,320 MLX op constructions per decoded token on the critical path.
    """

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

        missing = {"w1.weight", "w2.weight", "w3.weight"} - set(tensor_sizes)
        if missing:
            raise ValueError(f"tensor_sizes missing {sorted(missing)}")

        # Slot layout follows the expert bank's tensor set (FP4: weight+scale,
        # affine: weight+scales+biases per projection), see storage.index.
        self.tensor_sizes = dict(tensor_sizes)
        self.tensor_names = tuple(self.tensor_sizes)
        self.slot_bytes = sum(self.tensor_sizes.values())
        self._slots: list[ExpertSlot] = []
        self._free: deque[int] = deque()
        self._cond = threading.Condition()
        self._parked: list[int] = []

        for start in range(0, n_slots, chunk):
            batch = []
            for index in range(start, min(start + chunk, n_slots)):
                arrays = {
                    name: mx.zeros((self.tensor_sizes[name],), dtype=mx.uint8)
                    for name in self.tensor_names
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
    def parked(self) -> int:
        """Slots whose memory was given back (park); they are never handed out until unparked."""
        with self._cond:
            return len(self._parked)

    def park(self, n: int) -> int:
        """Free the memory of up to `n` free slots (HANDOFF 18.6: a long context's KV cache needs the room).

        Only free slots are parked, so nothing still reads them; their arrays, views and typed views are
        dropped and MLX can return the buffers. Returns how many were parked."""
        done = 0
        with self._cond:
            while done < n and self._free:
                slot = self._slots[self._free.pop()]
                slot.arrays, slot.views, slot.typed = {}, {}, {}
                self._parked.append(slot.index)
                done += 1
        return done

    def unpark(self, n: int) -> int:
        """Allocate memory for up to `n` parked slots again and make them free. Returns how many."""
        with self._cond:
            indices = [self._parked.pop() for _ in range(min(n, len(self._parked)))]
        if not indices:
            return 0
        fresh = []
        for index in indices:
            arrays = {name: mx.zeros((self.tensor_sizes[name],), dtype=mx.uint8) for name in self.tensor_names}
            fresh.append((index, arrays))
        mx.eval(*(a for _, arrays in fresh for a in arrays.values()))
        with self._cond:
            for index, arrays in fresh:
                slot = self._slots[index]
                slot.arrays = arrays
                slot.views = {name: _writable_view(a) for name, a in arrays.items()}
                slot.typed = {}
                self._free.append(index)
            self._cond.notify_all()
        return len(indices)

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
