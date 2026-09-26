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

import os
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from time import perf_counter

from cachalot.cache.resident import ResidentExpert
from cachalot.cache.slots import ExpertSlot, ExpertSlotPool
from cachalot.storage.index import ExpertEntry
from cachalot.storage.reader import ExpertReader

Key = tuple[int, int]


EVICT_POLICY = os.environ.get("CACHALOT_EVICT", "lru")
EVICT_SAMPLE = int(os.environ.get("CACHALOT_EVICT_SAMPLE", "64"))
# Segmented LRU (CACHALOT_EVICT=slru): share of the resident set an expert
# requested twice may occupy. An offline replay of the routing trace through
# benchmarks/simulate_policies.py puts slru 1.5 points of decode hit rate above
# LRU on the 3-bit bank at a 44 GiB budget, against 0.7 for frequency+decay.
SLRU_PROTECTED_FRACTION = float(os.environ.get("CACHALOT_SLRU_PROTECTED", "0.8"))
PREDICT_SLOT_RESERVE = 16   # transient slots kept free for prefill bypass loads
# a prefill gives back all of decode's borrow at once (the 0.20.0 behaviour), not just what it reads (HANDOFF 18.2)
PREFILL_SHRINK_ALL = os.environ.get("CACHALOT_PREFILL_SHRINK_ALL", "0") != "0"
# A read of one expert faster than this came from the page cache, not the drive:
# 9.49 MiB copies from RAM in ~0.5 ms and takes 1.9 ms or more from the SSD
# (HANDOFF section 15.7). Only used to label reads in the statistics.
FAST_READ_SECONDS = float(os.environ.get("CACHALOT_FAST_READ_MS", "1.0")) / 1000.0


@dataclass(frozen=True)

class ResidentStoreStats:
    cache_hits: int
    cache_misses: int
    ssd_bytes_read: int
    ssd_read_seconds: float
    promotion_seconds: float
    # every expert read, demand and predicted: how many, how many were fast
    # enough to have come from the page cache, and their summed wall time
    reads: int = 0
    fast_reads: int = 0
    read_wall_seconds: float = 0.0

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
    missing = {"w1.weight", "w2.weight", "w3.weight"} - set(sizes)
    if missing:
        raise ValueError(f"expert entry lacks tensors {sorted(missing)}")
    return dict(sorted(sizes.items()))


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
            expert_bytes = sum(tensor_sizes.values())
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
        # Transient slots decode may hold as residents while no prefill needs them (GLM/MiniMax scan
        # prefill, HANDOFF 18.1). get_many_prefill evicts back down to `capacity` before it takes any.
        self.decode_borrow = 0
        self.budget_bytes = self.capacity * expert_bytes
        self.transient_slots = slot_pool.capacity - self.capacity

        self._items: OrderedDict[Key, ResidentExpert] = OrderedDict()
        # Segmented LRU only: keys requested more than once, in recency order.
        # Residents are held in _items either way; this records the segment.
        self._protected: OrderedDict[Key, None] = OrderedDict()
        self._lock = RLock()
        self._key_locks: dict[Key, RLock] = {}

        # Deterministic prefill plan (see prepare_prefill_layer)
        self._prefill_layer_keys: dict[int, list[Key]] = {}
        self._prefill_admit: set[Key] = set()

        # Bypass loads awaiting release after the consumer's eval.
        self._transients: dict[Key, ResidentExpert] = {}
        # Scan-resistant prefill loads in flight: key -> (future, slot, entry, admit)
        self._scan_inflight: dict[Key, tuple[Future, ExpertSlot, ExpertEntry, bool]] = {}
        # how often each expert was requested (decode or prefill); orders
        # speculative next-layer loads in prefill
        self.use_counts: dict[Key, int] = defaultdict(int)
        self._transient_count = 0  # includes loads in flight
        self._transient_cond = threading.Condition(self._lock)

        # Resident slots handed out but not yet admitted (loads in flight).
        self._reserved = 0
        # Predictive decode prefetch: loads issued a layer early, keyed by
        # expert; each holds a reserved slot until admitted.
        self._inflight: dict[Key, tuple[Future, ExpertSlot, ExpertEntry]] = {}
        # The deadline each in-flight prediction was issued for: the pass it
        # belongs to and the layer that is expected to consume it. Without this
        # the sweep cannot tell a stale prediction from a correct one for a
        # later layer, and releases whichever happened to finish first --
        # see _sweep_inflight_locked.
        self._inflight_deadline: dict[Key, tuple[int, int]] = {}
        # Decode walks layers in ascending order, once per token. A requested
        # layer that does not advance means a new token or a new sequence, so
        # every prediction from the previous walk has expired.
        self._decode_pass = 0
        self._decode_layer = -1
        self.predicted_expired = 0
        self.predicted_loads = 0
        self.predicted_used = 0
        self.predicted_wasted_bytes = 0
        self._predict_pool = ThreadPoolExecutor(
            max_workers=max(1, int(os.environ.get("CACHALOT_PREDICT_WORKERS", "2"))),
            thread_name_prefix="expert-predict",
        )
        # gate weights per layer for one-layer-early routing prediction (set by the runtime)
        self.decode_gates: dict[int, tuple] = {}
        # Expert bank layout (storage.index.ExpertFormat); None means the
        # shipped FP4 layout. Compute paths branch on it.
        self.format = None

        self._load_pool = ThreadPoolExecutor(
            max_workers=max(1, int(load_workers)),
            thread_name_prefix="expert-load",
        )

        # Startup hotlist preload. Counted separately from the decode path so
        # a preloaded session's hit rate still means what it meant before.
        self.preloaded_experts = 0
        self.preload_bytes = 0
        self.preload_seconds = 0.0

        self.cache_hits = 0
        self.cache_misses = 0
        self.skipped_experts = 0
        self.decode_miss_budget: int | None = None  # opt-in approximation
        self.ssd_bytes_read = 0
        self.ssd_read_seconds = 0.0
        self.promotion_seconds = 0.0
        self._read_tally_lock = threading.Lock()
        self.reads = 0
        self.fast_reads = 0
        self.read_wall_seconds = 0.0

    def prefetch_decode(self, entries: list[ExpertEntry]) -> int:
        """
        Start loading experts predicted for an upcoming decode layer into
        free transient slots (never evicting a resident). A predicted expert
        becomes a resident only when a layer actually requests it; finished
        loads nobody asked for return their slot at the next sweep.
        """
        submitted = 0
        with self._lock:
            for entry in entries:
                key = (entry.layer, entry.expert)
                if key in self._items or key in self._inflight:
                    continue
                if self._transient_count >= self.transient_slots - PREDICT_SLOT_RESERVE:
                    break
                slot = self.pool.try_acquire()
                if slot is None:
                    break
                self._transient_count += 1
                future = self._predict_pool.submit(self._read_into, entry, slot)
                self._inflight[key] = (future, slot, entry)
                self._inflight_deadline[key] = (self._decode_pass, entry.layer)
                self.predicted_loads += 1
                submitted += 1
        return submitted

    def preload(
        self,
        entries: list[ExpertEntry],
        *,
        max_experts: int | None = None,
        reserve_fraction: float = 0.25,
    ) -> int:
        """
        Admit a recorded hot set as residents before the first prompt arrives.

        A session's first turn pays full miss cost while later turns run at
        87 % because they reuse what the first one dragged in. Routing is
        concentrated enough that a hot set ranked on other sessions covers
        about 30 % of an unseen prompt's requests at 5.6 % of the bank
        (benchmarks/hotlist_coverage.py), so reading it once at startup is
        worth 1.3 s of the 16.4 s a session already spends becoming ready.

        This never evicts anything, because at startup there is nothing to
        evict, and it deliberately leaves `reserve_fraction` of the capacity
        empty: the prefill quota planner needs room to admit what the actual
        prompt wants, and a hot set that filled the cache would be working
        against the turn it is meant to help. LRU takes care of the rest --
        whatever the session does not use is the first thing evicted.

        Returns the number of experts admitted. Hit and miss counters are not
        touched; see `preloaded_experts`, `preload_bytes` and `preload_seconds`.
        """
        if not entries:
            return 0

        t0 = perf_counter()
        room = int(self.capacity * (1.0 - reserve_fraction))
        if max_experts is not None:
            room = min(room, max_experts)

        admitted = 0
        batch = max(1, self._load_pool._max_workers)

        for start in range(0, len(entries), batch):
            chunk: list[tuple[ExpertEntry, ExpertSlot]] = []
            with self._lock:
                if len(self._items) + self._reserved >= room:
                    break
                for entry in entries[start : start + batch]:
                    key = (entry.layer, entry.expert)
                    if key in self._items or key in self._inflight:
                        continue
                    if len(self._items) + self._reserved >= room:
                        break
                    slot = self.pool.try_acquire()
                    if slot is None:
                        break
                    self._reserved += 1
                    chunk.append((entry, slot))

            if not chunk:
                if len(self._items) + self._reserved >= room:
                    break
                continue

            futures = [
                (entry, slot, self._load_pool.submit(self._read_into, entry, slot))
                for entry, slot in chunk
            ]
            for entry, slot, future in futures:
                nbytes, _read_seconds = future.result()
                with self._lock:
                    key = (entry.layer, entry.expert)
                    if key in self._items:
                        self._reserved = max(0, self._reserved - 1)
                        self._release_slot_locked(slot)
                        continue
                    self._admit_reserved_locked(
                        ResidentExpert(entry.layer, entry.expert, slot)
                    )
                    self.preload_bytes += nbytes
                    admitted += 1

        self.preloaded_experts += admitted
        self.preload_seconds += perf_counter() - t0
        return admitted

    def _advance_decode_pass_locked(self, layer: int) -> None:
        """Note which layer is being requested, and detect a new walk.

        Decode visits layers in ascending order, once per token. A requested
        layer that does not advance therefore means a new token or a new
        sequence, and every prediction issued during the previous walk has
        expired whatever layer it was aimed at.
        """
        if layer <= self._decode_layer:
            self._decode_pass += 1
        self._decode_layer = layer

    def _sweep_inflight_locked(self, keep: set[Key]) -> None:
        """Release finished predicted loads whose deadline has passed.

        This used to release every finished load the current layer had not
        asked for, which discarded a correct prediction for a later layer as
        soon as an earlier layer ran -- so with CACHALOT_PREDICT_AHEAD=2, an
        L+2 read that completed before the L+1 acquisition was thrown away
        before L+2 could consume it, and a prediction was punished for
        finishing early. A load still in flight survived only because the
        sweep took `f.done()` entries.

        A prediction is expired when its walk is over (a new token or sequence
        began) or when the layer it was aimed at has already been requested --
        at which point it was either consumed, in which case it is no longer
        in flight, or mispredicted, in which case it is genuinely waste. A
        prediction for a layer this walk has not reached yet is kept, whether
        or not its read has finished.

        Slot pressure is bounded at the other end: prefetch_decode refuses to
        start a load once the transient slots are down to PREDICT_SLOT_RESERVE,
        so a longer lifetime costs prefetch depth rather than the demand path.
        """
        expired = []
        for key, (future, _slot, _entry) in self._inflight.items():
            if key in keep or not future.done():
                continue
            pass_id, target_layer = self._inflight_deadline.get(
                key, (self._decode_pass, self._decode_layer)
            )
            if pass_id == self._decode_pass and target_layer > self._decode_layer:
                continue
            expired.append(key)

        for key in expired:
            future, slot, _entry = self._inflight.pop(key)
            self._inflight_deadline.pop(key, None)
            nbytes, read_seconds = future.result()
            self.pool.release(slot)
            self._transient_count -= 1
            self.predicted_expired += 1
            self.predicted_wasted_bytes += nbytes
            self.ssd_bytes_read += nbytes
            self.ssd_read_seconds += read_seconds
        self._transient_cond.notify_all()

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

    def _drop_locked(self, key: Key) -> None:
        """Remove one resident, forget its segment and return its slot."""
        victim = self._items.pop(key)
        self._protected.pop(key, None)
        self._release_slot_locked(victim.slot)

    def _protect_locked(self, key: Key) -> None:
        """
        Segmented LRU: a second request promotes an expert from probation to
        the protected segment, where eviction only reaches it once probation
        is empty. Overflow demotes the coldest protected expert back to the
        LRU end of probation rather than evicting it.
        """
        if EVICT_POLICY != "slru":
            return
        if key in self._protected:
            self._protected.move_to_end(key)
            return
        self._protected[key] = None
        cap = max(1, int(self.capacity * SLRU_PROTECTED_FRACTION))
        while len(self._protected) > cap:
            demoted, _ = self._protected.popitem(last=False)
            if demoted in self._items:
                self._items.move_to_end(demoted, last=False)

    def _evict_slru_locked(self, avoid_layer: int | None = None) -> None:
        """Evict from probation first, then from the coldest protected expert."""
        for key in self._items:
            if avoid_layer is not None and key[0] == avoid_layer:
                continue
            if key in self._protected:
                continue
            self._drop_locked(key)
            return
        for key in list(self._protected):
            if avoid_layer is not None and key[0] == avoid_layer:
                continue
            if key in self._items:
                self._drop_locked(key)
                return
            self._protected.pop(key, None)
        key, victim = self._items.popitem(last=False)
        self._protected.pop(key, None)
        self._release_slot_locked(victim.slot)

    def _evict_lru_locked(self, avoid_layer: int | None = None) -> None:
        """
        Evict one resident and free its slot. Policy "lru" takes the least
        recently used; "lfu" (CACHALOT_EVICT=lfu) takes the least requested
        among the EVICT_SAMPLE least recently used, ties to the older one;
        "slru" (CACHALOT_EVICT=slru) keeps twice-requested experts in a
        protected segment and evicts from probation first.
        """
        if EVICT_POLICY == "slru":
            self._evict_slru_locked(avoid_layer)
            return
        candidates = []
        for key in self._items:
            if avoid_layer is None or key[0] != avoid_layer:
                if EVICT_POLICY != "lfu":
                    self._drop_locked(key)
                    return
                candidates.append(key)
                if len(candidates) >= EVICT_SAMPLE:
                    break
        if candidates:
            key = min(candidates, key=lambda k: self.use_counts.get(k, 0))
            self._drop_locked(key)
            return
        key, victim = self._items.popitem(last=False)
        self._protected.pop(key, None)
        self._release_slot_locked(victim.slot)

    def _admit_reserved_locked(self, resident: ResidentExpert) -> None:
        """Admit an expert whose slot was reserved by _acquire_resident_slot_locked."""
        self._reserved = max(0, self._reserved - 1)
        self._items[resident.key] = resident

    def _read_into(self, entry: ExpertEntry, slot: ExpertSlot) -> tuple[int, float]:
        t0 = perf_counter()
        nbytes = self.reader.read_expert_into(entry, slot.views)
        seconds = perf_counter() - t0
        with self._read_tally_lock:
            self.reads += 1
            self.fast_reads += seconds < FAST_READ_SECONDS
            self.read_wall_seconds += seconds
        return nbytes, seconds

    def _record_miss(self, nbytes: int, read_seconds: float) -> None:
        self.cache_misses += 1
        self.ssd_bytes_read += nbytes
        self.ssd_read_seconds += read_seconds

    def set_capacity(self, target: int) -> int:
        """Move the resident capacity towards `target` slots by parking free slots (their memory is given back)
        or unparking them, never above the capacity the store was built with (HANDOFF 18.6).

        Only between requests or between decode tokens: no prefill in flight, every read evaluated. Residents
        above the new capacity (plus the decode borrow) are evicted LRU first. Returns the new capacity."""
        with self._lock:
            if not hasattr(self, "_full_capacity"):
                self._full_capacity = self.capacity
            target = max(1, min(int(target), self._full_capacity))
            if target < self.capacity:
                want = self.capacity - target
                keep = target + self.decode_borrow
                while len(self._items) + self._reserved > keep and self._items:
                    self._evict_lru_locked()
                # free slots beyond what capacity + borrow + transients can use are the ones to give back
                while self.pool.free_count < want and self._items:
                    self._evict_lru_locked()
                self.capacity -= self.pool.park(want)
            elif target > self.capacity:
                self.capacity += self.pool.unpark(target - self.capacity)
            self.budget_bytes = self.capacity * self.expert_bytes
            return self.capacity

    def _decode_capacity(self) -> int:
        return self.capacity + self.decode_borrow

    def _acquire_resident_slot_locked(self) -> ExpertSlot:
        """
        Reserve a free slot for a future resident, evicting LRU residents so
        that residents + reservations never exceed capacity (plus what decode
        may borrow from the transient slots).
        """
        while len(self._items) + self._reserved >= self._decode_capacity() and self._items:
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
                self._protect_locked(key)
                self.cache_hits += 1
                return cached

        with self._key_lock(key):
            with self._lock:
                cached = self._items.get(key)
                if cached is not None:
                    self._items.move_to_end(key)
                    self._protect_locked(key)
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
        prefetch: list[ExpertEntry] | None = None,
        on_hits=None,
    ) -> list[ResidentExpert | None]:
        """
        Acquire several experts for one decode step. Misses are read
        concurrently on the load pool and admitted in caller order so the
        LRU order does not depend on completion order.

        prefetch: experts predicted for a later layer; their loads are
        submitted right after this layer's real misses (prefetch_decode) so
        they stream while the GPU works. Misses already in flight from an
        earlier prediction are awaited instead of re-read.

        max_misses (opt-in approximation): load at most this many misses,
        highest `priorities` first; the rest are returned as None and
        counted in `skipped_experts`. None (default) loads every expert.

        on_hits: called once, on the calling thread, with the results so far
        (hits filled, misses None) after the misses' reads are submitted and
        before they are awaited, so the caller can queue GPU work on the hits
        while the reads run. Not called when every expert is a hit.
        """
        results: list[ResidentExpert | None] = [None] * len(entries)
        pending: list[tuple[int, ExpertEntry, ExpertSlot | None]] = []
        awaited: list[tuple[int, Key]] = []          # misses whose load is already in flight

        with self._lock:
            requested = {(e.layer, e.expert) for e in entries}
            if entries:
                # Decode asks for one layer's experts at a time, so the first
                # entry names the layer; prefill does not come through here.
                self._advance_decode_pass_locked(entries[0].layer)
            self._sweep_inflight_locked(keep=requested)
            miss_idx = []
            for i, entry in enumerate(entries):
                self.use_counts[(entry.layer, entry.expert)] += 1
                key = (entry.layer, entry.expert)
                cached = self._items.get(key)
                if cached is not None:
                    self._items.move_to_end(key)
                    self._protect_locked(key)
                    self.cache_hits += 1
                    results[i] = cached
                elif key in self._inflight:
                    awaited.append((i, key))
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

        futures = {}
        for _i, entry, slot in pending:
            key = (entry.layer, entry.expert)
            if key not in futures:
                futures[key] = self._load_pool.submit(self._read_into, entry, slot)

        if prefetch:
            self.prefetch_decode(prefetch)

        if on_hits is not None and (pending or awaited):
            on_hits(list(results))

        # predicted loads this layer needs: wait for them and admit
        for i, key in awaited:
            with self._lock:
                item = self._inflight.get(key)
                existing = self._items.get(key)
            if existing is not None:
                results[i] = existing
                continue
            if item is None:
                # admitted by a concurrent sweep between the two locked sections
                with self._lock:
                    results[i] = self._items[key]
                continue
            future, slot, entry = item
            nbytes, read_seconds = future.result()
            with self._lock:
                if key in self._inflight:
                    del self._inflight[key]
                    self._inflight_deadline.pop(key, None)
                    # the data already sits in a pool slot; make room among
                    # the residents and register it (no copy)
                    while len(self._items) + self._reserved >= self._decode_capacity() and self._items:
                        self._evict_lru_locked()
                    self._transient_count -= 1
                    self._transient_cond.notify_all()
                    self._items[key] = ResidentExpert(entry.layer, entry.expert, slot)
                    self._record_miss(nbytes, read_seconds)
                    self.predicted_used += 1
                results[i] = self._items[key]
                self._items.move_to_end(key)

        if not pending:
            return results

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

            # Quota room that this prompt's own misses will not fill stays with
            # the layer's current residents, most recently used first. A long
            # prompt needs more than the quota and this keeps nothing; a short
            # prompt (a chat follow-up, a typing-time prefill of a few tokens)
            # displaces only as many residents as it has misses, LRU-style,
            # instead of evicting every resident it does not route to. Measured
            # 2026-09-16: without this, tiny prefills shrank the resident set
            # from 1599 to 453 experts and cost 4 points of decode hit rate.
            remaining_misses = sum(1 for k in ordered if k not in retained_set)
            old_room = layer_slots - len(retained) - remaining_misses
            if old_room > 0:
                for key in reversed(self._items):
                    if old_room <= 0:
                        break
                    if key[0] != layer_id or key in retained_set:
                        continue
                    retained.append(key)
                    retained_set.add(key)
                    old_room -= 1

            for key in [k for k in self._items if k[0] == layer_id and k not in retained_set]:
                victim = self._items.pop(key)
                self._protected.pop(key, None)
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
                            self._protected.pop(victim_key, None)
                            self.pool.release(victim.slot)
                            to_reclaim -= 1

            if to_reclaim > 0:
                raise RuntimeError(
                    "Unable to reclaim enough resident capacity for prepared "
                    f"prefill layer {layer_id}: short by {to_reclaim} slots"
                )

            self._prefill_layer_keys[layer_id] = desired
            self._prefill_admit = admit

    # ------------------------------------------------------------------
    # scan-resistant prefill (GLM): never evicts, reads the next layer early
    # ------------------------------------------------------------------
    def _scan_slot_locked(self, block: bool, layer: int | None = None) -> tuple[ExpertSlot, bool] | None:
        """(slot, admit): free resident capacity first, then a transient slot."""
        while True:
            if len(self._items) + self._reserved < self.capacity:
                slot = self.pool.try_acquire()
                if slot is not None:
                    self._reserved += 1
                    return slot, True
            if self._transient_count < self.transient_slots:
                slot = self.pool.try_acquire()
                if slot is not None:
                    self._transient_count += 1
                    return slot, False
            if not block:
                return None
            if len(self._items) + self._reserved > self.capacity and self._items:
                # every free slot is held by residents decode borrowed: give one back rather than wait for a
                # release that only this call could make (HANDOFF 18.6)
                self._evict_lru_locked(avoid_layer=layer)
                continue
            self._transient_cond.wait(timeout=5.0)

    def _scan_finish_locked(self, key: Key, nbytes: int, read_seconds: float, keep: bool) -> None:
        """Register a finished scan load: admitted, kept as a transient, or (unused) freed."""
        _future, slot, entry, admit = self._scan_inflight.pop(key)
        self._record_miss(nbytes, read_seconds)
        resident = ResidentExpert(entry.layer, entry.expert, slot, transient=not admit)
        if admit:
            self._admit_reserved_locked(resident)
        elif keep:
            self._transients[key] = resident
        else:
            self._transient_count -= 1
            self._release_slot_locked(slot)

    def get_many_prefill(
        self, entries: list[ExpertEntry], *, speculate: list[ExpertEntry] | None = None
    ) -> list[ResidentExpert]:
        """
        One prefill layer's experts without evicting anything.

        A long prefill chunk routes to nearly every expert of every layer, far
        more than the cache holds, so the LRU path turns each chunk into a scan
        that evicts every resident before it is reused (0 hits, HANDOFF 17.1).
        Here residents are returned without touching LRU order, misses fill free
        capacity (admitted) and then transient slots, which the caller frees
        with release_prefill_layer() once its outputs are evaluated.

        speculate: the next layer's likely experts; their reads are queued
        behind this layer's misses (same FIFO pool) so the drive keeps working
        while this layer computes. Stops at the transient budget, never blocks.
        """
        results: list[ResidentExpert | None] = [None] * len(entries)
        waits: list[tuple[int, Key]] = []
        with self._lock:
            # give back what decode borrowed: the transient slots are this path's
            if PREFILL_SHRINK_ALL:
                while len(self._items) + self._reserved > self.capacity and self._items:
                    self._evict_lru_locked()
            else:
                # only as many as this call's reads need (HANDOFF 18.2): a short follow-up prefill touches a few
                # dozen experts per layer, and evicting all of the borrow cost the next reply its hits
                need = sum(1 for e in entries if (e.layer, e.expert) not in self._items
                           and (e.layer, e.expert) not in self._transients
                           and (e.layer, e.expert) not in self._scan_inflight)
                need += len(speculate or ())
                # never this layer's residents: they were counted as hits in `need`
                layer = entries[0].layer if entries else None
                while (len(self._items) + self._reserved > self.capacity and self._items
                       and self.pool.free_count < need):
                    self._evict_lru_locked(avoid_layer=layer)
            for i, entry in enumerate(entries):
                key = (entry.layer, entry.expert)
                self.use_counts[key] += 1
                cached = self._items.get(key) or self._transients.get(key)
                if cached is not None:
                    self.cache_hits += 1
                    results[i] = cached
                    continue
                if key not in self._scan_inflight:
                    slot, admit = self._scan_slot_locked(block=True, layer=entry.layer)
                    future = self._load_pool.submit(self._read_into, entry, slot)
                    self._scan_inflight[key] = (future, slot, entry, admit)
                waits.append((i, key))
            for entry in speculate or ():
                key = (entry.layer, entry.expert)
                if key in self._items or key in self._transients or key in self._scan_inflight:
                    continue
                got = self._scan_slot_locked(block=False)
                if got is None:
                    break
                slot, admit = got
                future = self._load_pool.submit(self._read_into, entry, slot)
                self._scan_inflight[key] = (future, slot, entry, admit)
        for i, key in waits:
            with self._lock:
                item = self._scan_inflight.get(key)
            if item is not None:
                nbytes, read_seconds = item[0].result()
                with self._lock:
                    if key in self._scan_inflight:
                        self._scan_finish_locked(key, nbytes, read_seconds, keep=True)
            with self._lock:
                results[i] = self._items.get(key) or self._transients[key]
        return results

    def release_prefill_layer(self, layer: int) -> None:
        """After a prefill layer's outputs are evaluated: free its transients and
        any speculative load of it that nobody asked for (admitted ones stay)."""
        with self._lock:
            pending = [(k, v[0]) for k, v in self._scan_inflight.items() if k[0] == layer]
        for key, future in pending:
            nbytes, read_seconds = future.result()
            with self._lock:
                if key in self._scan_inflight:
                    self._scan_finish_locked(key, nbytes, read_seconds, keep=False)
        with self._lock:
            self._release_transients_locked(lambda k: k[0] == layer)

    def release_prefill(self) -> None:
        """End of a prefill (or an aborted one): nothing scan-loaded stays transient."""
        with self._lock:
            layers = {k[0] for k in self._scan_inflight} | {k[0] for k in self._transients}
        for layer in sorted(layers):
            self.release_prefill_layer(layer)

    def resident_keys(self) -> list[Key]:
        """Resident experts, least recently used first (transients excluded)."""
        with self._lock:
            return list(self._items)

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

    def expire_predictions(self) -> int:
        """End the current speculative walk, so no prediction outlives it.

        The sweep in `get_many` detects a new walk from the requested layer not
        advancing, which covers every token boundary during decode. A sequence
        reset does not necessarily produce one -- the next request could be for
        a higher layer than the last -- so the runtime says so explicitly. A
        prediction still in flight keeps its slot until it finishes and is then
        released by the next sweep; only finished ones can be returned here.
        """
        with self._lock:
            self._decode_pass += 1
            self._decode_layer = -1
            before = self.predicted_expired
            self._sweep_inflight_locked(keep=set())
            return self.predicted_expired - before

    # ------------------------------------------------------------------
    def stats(self) -> ResidentStoreStats:
        with self._lock:
            return ResidentStoreStats(
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                ssd_bytes_read=self.ssd_bytes_read,
                ssd_read_seconds=self.ssd_read_seconds,
                promotion_seconds=self.promotion_seconds,
                reads=self.reads,
                fast_reads=self.fast_reads,
                read_wall_seconds=self.read_wall_seconds,
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
        self._predict_pool.shutdown(wait=True, cancel_futures=True)
        self.reader.close()

    def __enter__(self) -> ResidentExpertStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
