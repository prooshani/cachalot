"""
Resident routed-expert cache backed by a fixed pool of wired slots.

Two access paths, mirroring the runtime:

decode  get() / get_many()
    Global LRU. Misses are read straight into a free slot (evicting the
    LRU resident first) and always admitted.

prefill prepare_prefill_layer() + get_prefill()
    Deterministic per-layer quotas decided before asynchronous prefetch
    starts, so SSD completion order cannot change cache membership.
    Misses the plan does not admit are loaded into *transient* slots and
    returned to the pool once the caller reports that the MLX operations
    that consumed them have been evaluated (release_transients()).

Invariant for slot reuse: a slot is released only after the last MLX op
reading it has been evaluated. Decode evicts right after the per-layer
route eval; prefill evicts in prepare_prefill_layer(), which also runs
after the route eval of its layer.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from time import perf_counter

from cachalot.cache.resident import ResidentExpert
from cachalot.cache.slots import TENSOR_NAMES, ExpertSlot, ExpertSlotPool
from cachalot.storage.index import ExpertEntry
from cachalot.storage.reader import ExpertReader

Key = tuple[int, int]


@dataclass(frozen=True)
class ResidentStoreStats:
    cache_hits: int
    cache_misses: int
    ssd_bytes_read: int
    ssd_read_seconds: float
    promotion_seconds: float

    @property
    def requests(self) -> int:
        return self.cache_hits + self.cache_misses

    @property
    def hit_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.cache_hits / self.requests

    @property
    def ssd_throughput_mib_s(self) -> float:
        if self.ssd_read_seconds == 0:
            return 0.0
        return (self.ssd_bytes_read / 1024**2) / self.ssd_read_seconds


def tensor_sizes_from_entry(entry: ExpertEntry) -> dict[str, int]:
    sizes = {}
    for tensor in entry.tensors:
        short = ".".join(tensor.name.rsplit(".", 2)[-2:])
        sizes[short] = tensor.size
    missing = set(TENSOR_NAMES) - set(sizes)
    if missing:
        raise ValueError(f"expert entry lacks tensors {sorted(missing)}")
    return sizes


class ResidentExpertStore:
    def __init__(
        self,
        budget_bytes: int,
        reader: ExpertReader | None = None,
        *,
        tensor_sizes: dict[str, int] | None = None,
        slot_pool: ExpertSlotPool | None = None,
        transient_slots: int = 128,
        load_workers: int = 8,
        verbose: bool = False,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")

        self.reader = reader or ExpertReader()

        if slot_pool is None:
            if tensor_sizes is None:
                raise ValueError("tensor_sizes or slot_pool is required")
            expert_bytes = sum(tensor_sizes[n] for n in TENSOR_NAMES)
            capacity = budget_bytes // expert_bytes
            if capacity <= 0:
                raise ValueError(
                    f"budget {budget_bytes} smaller than one expert ({expert_bytes})"
                )
            if verbose:
                print(
                    f"Allocating {capacity + transient_slots} expert slots "
                    f"({(capacity + transient_slots) * expert_bytes / 1024**3:.1f} GiB)...",
                    flush=True,
                )
            slot_pool = ExpertSlotPool(
                tensor_sizes,
                capacity + max(1, transient_slots),
                verbose=verbose,
            )
        else:
            expert_bytes = slot_pool.slot_bytes
            capacity = min(budget_bytes // expert_bytes, slot_pool.capacity - 1)

        self.pool = slot_pool
        self.expert_bytes = expert_bytes
        self.capacity = int(capacity)
        self.budget_bytes = self.capacity * expert_bytes
        self.transient_slots = slot_pool.capacity - self.capacity

        self._items: OrderedDict[Key, ResidentExpert] = OrderedDict()
        self._lock = RLock()
        self._key_locks: dict[Key, RLock] = {}

        # Deterministic prefill plan (see prepare_prefill_layer)
        self._prefill_layer_keys: dict[int, list[Key]] = {}
        self._prefill_admit: set[Key] = set()

        # Bypass loads awaiting release after the consumer's eval.
        self._transients: dict[Key, ResidentExpert] = {}
        # how often each expert was requested (decode or prefill); orders
        # speculative next-layer loads in prefill
        self.use_counts: dict[Key, int] = defaultdict(int)
        self._transient_count = 0  # includes loads in flight
        self._transient_cond = threading.Condition(self._lock)

        # Resident slots handed out but not yet admitted (loads in flight).
        self._reserved = 0

        self._load_pool = ThreadPoolExecutor(
            max_workers=max(1, int(load_workers)),
            thread_name_prefix="expert-load",
        )

        self.cache_hits = 0
        self.cache_misses = 0
        self.skipped_experts = 0
        self.decode_miss_budget: int | None = None  # opt-in approximation
        self.ssd_bytes_read = 0
        self.ssd_read_seconds = 0.0
        self.promotion_seconds = 0.0

    # ------------------------------------------------------------------
    # basics
    # ------------------------------------------------------------------
    @property
    def current_bytes(self) -> int:
        with self._lock:
            return len(self._items) * self.expert_bytes

    def _key_lock(self, key: Key) -> RLock:
        with self._lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = RLock()
                self._key_locks[key] = lock
            return lock

    def _evict_lru_locked(self, avoid_layer: int | None = None) -> None:
        """Evict one resident (LRU, optionally skipping a layer) and free its slot."""
        for key in self._items:
            if avoid_layer is None or key[0] != avoid_layer:
                victim = self._items.pop(key)
                self._release_slot_locked(victim.slot)
                return
        _, victim = self._items.popitem(last=False)
        self._release_slot_locked(victim.slot)

    def _admit_reserved_locked(self, resident: ResidentExpert) -> None:
        """Admit an expert whose slot was reserved by _acquire_resident_slot_locked."""
        self._reserved = max(0, self._reserved - 1)
        self._items[resident.key] = resident

    def _read_into(self, entry: ExpertEntry, slot: ExpertSlot) -> tuple[int, float]:
        t0 = perf_counter()
        nbytes = self.reader.read_expert_into(entry, slot.views)
        return nbytes, perf_counter() - t0

    def _record_miss(self, nbytes: int, read_seconds: float) -> None:
        self.cache_misses += 1
        self.ssd_bytes_read += nbytes
        self.ssd_read_seconds += read_seconds

    def _acquire_resident_slot_locked(self) -> ExpertSlot:
        """
        Reserve a free slot for a future resident, evicting LRU residents so
        that residents + reservations never exceed capacity.
        """
        while len(self._items) + self._reserved >= self.capacity and self._items:
            self._evict_lru_locked()
        while True:
            slot = self.pool.try_acquire()
            if slot is not None:
                self._reserved += 1
                return slot
            if not self._items:
                # Only transients hold the pool; wait for a release.
                self._transient_cond.wait(timeout=5.0)
                continue
            self._evict_lru_locked()

    def _release_slot_locked(self, slot: ExpertSlot) -> None:
        self.pool.release(slot)
        self._transient_cond.notify_all()

    # ------------------------------------------------------------------
    # decode path
    # ------------------------------------------------------------------
    def get(self, entry: ExpertEntry) -> ResidentExpert:
        key = (entry.layer, entry.expert)

        with self._lock:
            self.use_counts[key] += 1
            cached = self._items.get(key)
            if cached is not None:
                self._items.move_to_end(key)
                self.cache_hits += 1
                return cached

        with self._key_lock(key):
            with self._lock:
                cached = self._items.get(key)
                if cached is not None:
                    self._items.move_to_end(key)
                    self.cache_hits += 1
                    return cached
                slot = self._acquire_resident_slot_locked()

            nbytes, read_seconds = self._read_into(entry, slot)
            resident = ResidentExpert(entry.layer, entry.expert, slot)

            with self._lock:
                self._admit_reserved_locked(resident)
                self._record_miss(nbytes, read_seconds)

            return resident

    def get_many(
        self,
        entries: list[ExpertEntry],
        *,
        max_misses: int | None = None,
        priorities: list[float] | None = None,
    ) -> list[ResidentExpert | None]:
        """
        Acquire several experts for one decode step. Misses are read
        concurrently on the load pool and admitted in caller order so the
        LRU order does not depend on completion order.

        max_misses (opt-in approximation): load at most this many misses,
        highest `priorities` first; the rest are returned as None and
        counted in `skipped_experts`. None (default) loads every expert.
        """
        results: list[ResidentExpert | None] = [None] * len(entries)
        pending: list[tuple[int, ExpertEntry, ExpertSlot | None]] = []

        with self._lock:
            miss_idx = []
            for i, entry in enumerate(entries):
                self.use_counts[(entry.layer, entry.expert)] += 1
                key = (entry.layer, entry.expert)
                cached = self._items.get(key)
                if cached is not None:
                    self._items.move_to_end(key)
                    self.cache_hits += 1
                    results[i] = cached
                else:
                    miss_idx.append(i)

            if max_misses is not None and len(miss_idx) > max_misses:
                order = sorted(
                    miss_idx,
                    key=lambda i: -(priorities[i] if priorities is not None else 0.0),
                )
                keep = set(order[:max_misses])
                self.skipped_experts += len(miss_idx) - len(keep)
                miss_idx = [i for i in miss_idx if i in keep]

            unique: dict[Key, ExpertSlot] = {}
            for i in miss_idx:
                entry = entries[i]
                key = (entry.layer, entry.expert)
                if key not in unique:
                    unique[key] = self._acquire_resident_slot_locked()
                pending.append((i, entry, unique[key]))

        if not pending:
            return results

        futures = {}
        for _i, entry, slot in pending:
            key = (entry.layer, entry.expert)
            if key not in futures:
                futures[key] = self._load_pool.submit(self._read_into, entry, slot)

        loaded = {key: fut.result() for key, fut in futures.items()}

        with self._lock:
            for i, entry, slot in pending:
                key = (entry.layer, entry.expert)
                existing = self._items.get(key)
                if existing is not None:
                    results[i] = existing
                    continue
                resident = ResidentExpert(entry.layer, entry.expert, slot)
                self._admit_reserved_locked(resident)
                nbytes, read_seconds = loaded[key]
                self._record_miss(nbytes, read_seconds)
                results[i] = resident

        return results

    # ------------------------------------------------------------------
    # prefill path
    # ------------------------------------------------------------------
    def prepare_prefill_layer(
        self,
        layer_id: int,
        entries: list[ExpertEntry],
        *,
        num_layers: int = 40,
    ) -> None:
        """
        Decide this layer's resident set before asynchronous prefetch.

        1. total capacity is divided evenly across layers;
        2. residents this layer needs are kept (planned ones first);
        3. stale residents of this layer are evicted;
        4. misses are reserved in logical order until the quota is full;
        5. capacity is reclaimed from other layers above their quota,
           preferring residents added by decode.

        Must run after the layer's route eval, so evicted slots are idle.
        """
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

        ordered: list[Key] = []
        seen: set[Key] = set()
        for entry in entries:
            if entry.layer != layer_id:
                raise ValueError(
                    f"prepare_prefill_layer received entry layer={entry.layer} for layer_id={layer_id}"
                )
            key = (entry.layer, entry.expert)
            if key not in seen:
                seen.add(key)
                ordered.append(key)

        base_slots, remainder = divmod(self.capacity, num_layers)

        def quota(layer: int) -> int:
            return base_slots + (1 if layer < remainder else 0)

        layer_slots = quota(layer_id)
        needed = set(ordered)

        with self._lock:
            # Release transients consumed in the previous layer (the route
            # eval that precedes this call evaluated everything using them)
            # and speculative loads of this layer that turned out unneeded.
            # Speculative transients this layer does need are kept and, while
            # the quota has room, promoted to residents below.
            self._release_transients_locked(
                lambda key: key[0] != layer_id or key not in needed
            )

            if not ordered:
                self._prefill_layer_keys[layer_id] = []
                self._prefill_admit = set()
                return

            previous = self._prefill_layer_keys.get(layer_id, [])
            retained: list[Key] = []
            retained_set: set[Key] = set()
            for key in previous:
                if key in needed and key in self._items and len(retained) < layer_slots:
                    retained.append(key)
                    retained_set.add(key)
            for key in ordered:
                if len(retained) >= layer_slots:
                    break
                if key in retained_set or key not in self._items:
                    continue
                retained.append(key)
                retained_set.add(key)
            for key in ordered:
                if len(retained) >= layer_slots or len(self._items) + self._reserved >= self.capacity:
                    break
                resident = self._transients.get(key)
                if resident is None or key in retained_set:
                    continue
                # promote a speculative transient into this layer's quota
                del self._transients[key]
                self._transient_count -= 1
                resident.transient = False
                self._items[key] = resident
                retained.append(key)
                retained_set.add(key)
            self._transient_cond.notify_all()

            for key in [k for k in self._items if k[0] == layer_id and k not in retained_set]:
                victim = self._items.pop(key)
                self.pool.release(victim.slot)

            desired = list(retained)
            desired_set = set(desired)
            admit: set[Key] = set()
            for key in ordered:
                if len(desired) >= layer_slots:
                    break
                if key in desired_set:
                    continue
                desired.append(key)
                desired_set.add(key)
                if key not in self._items:
                    admit.add(key)

            to_reclaim = max(0, len(self._items) + len(admit) - self.capacity)
            if to_reclaim > 0:
                for other in range(num_layers):
                    if to_reclaim <= 0:
                        break
                    if other == layer_id:
                        continue
                    resident_other = [k for k in self._items if k[0] == other]
                    overflow = max(0, len(resident_other) - quota(other))
                    if overflow == 0:
                        continue
                    planned = set(self._prefill_layer_keys.get(other, []))
                    victims = (
                        sorted(k for k in resident_other if k not in planned)
                        + sorted(k for k in resident_other if k in planned)
                    )[:overflow]
                    for victim_key in victims:
                        if to_reclaim <= 0:
                            break
                        victim = self._items.pop(victim_key, None)
                        if victim is not None:
                            self.pool.release(victim.slot)
                            to_reclaim -= 1

            if to_reclaim > 0:
                raise RuntimeError(
                    "Unable to reclaim enough resident capacity for prepared "
                    f"prefill layer {layer_id}: short by {to_reclaim} slots"
                )

            self._prefill_layer_keys[layer_id] = desired
            self._prefill_admit = admit

    def is_resident(self, key: Key) -> bool:
        with self._lock:
            return key in self._items or key in self._transients

    def speculative_candidates(self, entries: list[ExpertEntry], limit: int) -> list[ExpertEntry]:
        """
        Up to `limit` experts of one layer that are neither resident nor
        transient, most requested first (ties by expert id). Used to fill the
        SSD-idle gap before a prefill layer's routing is known.
        """
        if limit <= 0:
            return []
        with self._lock:
            missing = [
                e for e in entries
                if (e.layer, e.expert) not in self._items and (e.layer, e.expert) not in self._transients
            ]
            missing.sort(key=lambda e: (-self.use_counts.get((e.layer, e.expert), 0), e.expert))
        return missing[:limit]

    def get_prefill(self, entry: ExpertEntry) -> ResidentExpert:
        """
        Acquire an expert for layer-major prefill.

        resident hit      -> returned without touching decode LRU order
        planned miss      -> read into a resident slot and admitted
        unplanned miss    -> read into a transient slot (not admitted);
                             freed by release_transients() after eval
        """
        key = (entry.layer, entry.expert)

        with self._lock:
            self.use_counts[key] += 1
            cached = self._items.get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached
            transient = self._transients.get(key)
            if transient is not None:
                self.cache_hits += 1
                return transient

        with self._key_lock(key):
            with self._lock:
                cached = self._items.get(key) or self._transients.get(key)
                if cached is not None:
                    self.cache_hits += 1
                    return cached
                admit = key in self._prefill_admit
                slot = self._acquire_resident_slot_locked() if admit else None

            if slot is None:
                # Transient: block until a bypass slot is free. The consumer
                # releases them after evaluating its outputs.
                slot = self._acquire_transient_slot()

            nbytes, read_seconds = self._read_into(entry, slot)
            resident = ResidentExpert(entry.layer, entry.expert, slot, transient=not admit)

            with self._lock:
                if admit:
                    self._admit_reserved_locked(resident)
                else:
                    self._transients[key] = resident
                self._record_miss(nbytes, read_seconds)

            return resident

    def _acquire_transient_slot(self) -> ExpertSlot:
        with self._transient_cond:
            while True:
                if self._transient_count < self.transient_slots:
                    slot = self.pool.try_acquire()
                    if slot is not None:
                        self._transient_count += 1
                        return slot
                self._transient_cond.wait(timeout=5.0)

    def transient_free(self) -> int:
        """Bypass slots still available for prefill loads."""
        with self._lock:
            return self.transient_slots - self._transient_count

    def release_transients(self, keys) -> None:
        """Return transient slots whose consumers have been evaluated."""
        with self._lock:
            for key in keys:
                resident = self._transients.pop(key, None)
                if resident is not None:
                    self.pool.release(resident.slot)
                    self._transient_count -= 1
            self._transient_cond.notify_all()

    def _release_transients_locked(self, predicate) -> None:
        victims = [k for k in self._transients if predicate(k)]
        for key in victims:
            resident = self._transients.pop(key)
            self.pool.release(resident.slot)
            self._transient_count -= 1
        if victims:
            self._transient_cond.notify_all()

    def _release_all_transients_locked(self) -> None:
        if self._transients:
            for r in self._transients.values():
                self.pool.release(r.slot)
            self._transient_count -= len(self._transients)
            self._transients.clear()
            self._transient_cond.notify_all()

    def release_all_transients(self) -> None:
        with self._lock:
            self._release_all_transients_locked()

    # ------------------------------------------------------------------
    def stats(self) -> ResidentStoreStats:
        with self._lock:
            return ResidentStoreStats(
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                ssd_bytes_read=self.ssd_bytes_read,
                ssd_read_seconds=self.ssd_read_seconds,
                promotion_seconds=self.promotion_seconds,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def utilization(self) -> float:
        with self._lock:
            return len(self._items) / self.capacity if self.capacity else 0.0

    def close(self) -> None:
        self._load_pool.shutdown(wait=True, cancel_futures=True)
        self.reader.close()

    def __enter__(self) -> ResidentExpertStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
