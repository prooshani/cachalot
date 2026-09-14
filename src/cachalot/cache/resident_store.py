from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from time import perf_counter

from cachalot.cache.resident import ResidentExpert, promote_expert
from cachalot.storage.index import ExpertEntry
from cachalot.storage.reader import ExpertReader
from cachalot.storage.store import ExpertPayload


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
        return (
            self.ssd_bytes_read / 1024**2
        ) / self.ssd_read_seconds


class ResidentExpertStore:
    def __init__(
        self,
        budget_bytes: int,
        reader: ExpertReader | None = None,
    ) -> None:
        self.budget_bytes = budget_bytes
        self.current_bytes = 0

        self.reader = reader or ExpertReader()

        self._items: OrderedDict[
            tuple[int, int],
            ResidentExpert,
        ] = OrderedDict()

        self._lock = RLock()

        self._key_locks: dict[
            tuple[int, int],
            RLock,
        ] = {}

        # --------------------------------------------------------
        # Deterministic layer-major prefill cache policy.
        #
        # _prefill_layer_keys:
        #     Desired resident keys for each layer, kept in logical
        #     expert-work order rather than async completion order.
        #
        # _prefill_admit:
        #     Misses selected by prepare_prefill_layer() for cache
        #     admission during the current layer.
        # --------------------------------------------------------
        self._prefill_layer_keys: dict[
            int,
            list[tuple[int, int]],
        ] = {}

        self._prefill_admit: set[
            tuple[int, int]
        ] = set()

        self.cache_hits = 0
        self.cache_misses = 0

        self.ssd_bytes_read = 0
        self.ssd_read_seconds = 0.0
        self.promotion_seconds = 0.0

    def _key_lock(
        self,
        key: tuple[int, int],
    ) -> RLock:
        with self._lock:
            lock = self._key_locks.get(key)

            if lock is None:
                lock = RLock()
                self._key_locks[key] = lock

            return lock

    def get(
        self,
        entry: ExpertEntry,
    ) -> ResidentExpert:
        key = (entry.layer, entry.expert)

        # Fast resident-cache path.
        with self._lock:
            cached = self._items.get(key)

            if cached is not None:
                self._items.move_to_end(key)
                self.cache_hits += 1
                return cached

        # Serialize loading/promotion of the same expert while
        # still allowing different experts to load concurrently.
        key_lock = self._key_lock(key)

        with key_lock:
            # Re-check after acquiring the per-expert lock.
            with self._lock:
                cached = self._items.get(key)

                if cached is not None:
                    self._items.move_to_end(key)
                    self.cache_hits += 1
                    return cached

            t0 = perf_counter()
            chunks = self.reader.read_expert(entry)
            read_seconds = perf_counter() - t0

            payload = ExpertPayload(chunks=chunks)

            t1 = perf_counter()
            resident = promote_expert(entry, payload)
            promotion_seconds = perf_counter() - t1

            if resident.size > self.budget_bytes:
                raise ValueError(
                    f"Expert size {resident.size} exceeds "
                    f"resident cache budget {self.budget_bytes}"
                )

            with self._lock:
                while (
                    self._items
                    and self.current_bytes + resident.size
                    > self.budget_bytes
                ):
                    _, evicted = self._items.popitem(last=False)
                    self.current_bytes -= evicted.size

                self._items[key] = resident
                self.current_bytes += resident.size

                self.cache_misses += 1
                self.ssd_bytes_read += payload.size
                self.ssd_read_seconds += read_seconds
                self.promotion_seconds += promotion_seconds

            return resident

    def prepare_prefill_layer(
        self,
        layer_id: int,
        entries: list[ExpertEntry],
        *,
        num_layers: int = 40,
    ) -> None:
        """
        Reconcile one layer's resident working set before async
        layer-major prefetch begins.

        The decision is made synchronously from logical expert-work
        order, so async SSD completion order cannot affect which
        experts become resident.

        Policy:

        1. Divide total expert capacity across transformer layers.
        2. Preserve already-resident experts needed by this layer.
        3. Evict stale residents from this layer.
        4. Reserve current misses, in logical order, until this
           layer reaches its quota.
        5. get_prefill() later admits only those reserved misses.

        Normal decode continues to use get() and its global LRU.
        """
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be > 0, got {num_layers}"
            )

        if not entries:
            with self._lock:
                self._prefill_layer_keys[
                    layer_id
                ] = []
                self._prefill_admit.clear()
            return

        ordered_keys: list[
            tuple[int, int]
        ] = []

        seen: set[
            tuple[int, int]
        ] = set()

        expert_size: int | None = None

        for entry in entries:
            if entry.layer != layer_id:
                raise ValueError(
                    "prepare_prefill_layer received "
                    f"entry layer={entry.layer} for "
                    f"layer_id={layer_id}"
                )

            key = (
                entry.layer,
                entry.expert,
            )

            if key in seen:
                continue

            seen.add(key)
            ordered_keys.append(key)

            entry_size = sum(
                tensor.size
                for tensor in entry.tensors
            )

            if expert_size is None:
                expert_size = entry_size
            elif entry_size != expert_size:
                raise ValueError(
                    "Layer-prefill partitioning currently "
                    "requires equal routed-expert sizes: "
                    f"expected {expert_size}, got {entry_size} "
                    f"for key={key}"
                )

        if expert_size is None:
            return

        if expert_size <= 0:
            raise ValueError(
                f"Invalid expert size: {expert_size}"
            )

        total_slots = (
            self.budget_bytes
            // expert_size
        )

        base_slots = (
            total_slots
            // num_layers
        )

        remainder = (
            total_slots
            % num_layers
        )

        layer_slots = (
            base_slots
            + (
                1
                if layer_id < remainder
                else 0
            )
        )

        needed_set = set(
            ordered_keys
        )

        with self._lock:
            # ----------------------------------------------------
            # Start from the deterministic desired set produced by
            # the previous prefill, but retain only entries that
            # are still physically resident and needed now.
            # ----------------------------------------------------
            previous = (
                self._prefill_layer_keys.get(
                    layer_id,
                    []
                )
            )

            retained: list[
                tuple[int, int]
            ] = []

            retained_set: set[
                tuple[int, int]
            ] = set()

            for key in previous:
                if (
                    key in needed_set
                    and key in self._items
                    and len(retained) < layer_slots
                ):
                    retained.append(key)
                    retained_set.add(key)

            # ----------------------------------------------------
            # Decode may have admitted useful experts that are not
            # represented in the previous prefill metadata.
            #
            # Add those in CURRENT LOGICAL order, not OrderedDict
            # order, so the decision remains deterministic.
            # ----------------------------------------------------
            for key in ordered_keys:
                if len(retained) >= layer_slots:
                    break

                if key in retained_set:
                    continue

                if key in self._items:
                    retained.append(key)
                    retained_set.add(key)

            # ----------------------------------------------------
            # Evict stale residents belonging to THIS layer.
            #
            # Do not disturb other layer partitions here.
            # ----------------------------------------------------
            stale = [
                key
                for key in self._items.keys()
                if (
                    key[0] == layer_id
                    and key not in retained_set
                )
            ]

            for key in stale:
                evicted = self._items.pop(
                    key,
                    None,
                )

                if evicted is not None:
                    self.current_bytes -= (
                        evicted.size
                    )

            # ----------------------------------------------------
            # Reserve misses in logical order until this layer's
            # deterministic quota is filled.
            # ----------------------------------------------------
            desired = list(
                retained
            )

            desired_set = set(
                desired
            )

            admit: set[
                tuple[int, int]
            ] = set()

            for key in ordered_keys:
                if len(desired) >= layer_slots:
                    break

                if key in desired_set:
                    continue

                desired.append(key)
                desired_set.add(key)

                if key not in self._items:
                    admit.add(key)

            # ----------------------------------------------------
            # Normal autoregressive decode uses the global LRU and
            # may redistribute a full cache unevenly across layers.
            #
            # Before reserving this layer's missing quota, reclaim
            # any required capacity from OTHER layers that are
            # currently above their own deterministic quotas.
            #
            # Prefer entries that are not part of the other layer's
            # last planned prefill set. Those are normally experts
            # introduced by decode after the previous prefill.
            #
            # Victim selection is deterministic; it does not depend
            # on async prefill completion order.
            # ----------------------------------------------------
            required_bytes = (
                len(admit)
                * expert_size
            )

            bytes_to_reclaim = max(
                0,
                (
                    self.current_bytes
                    + required_bytes
                    - self.budget_bytes
                ),
            )

            if bytes_to_reclaim > 0:
                for other_layer in range(
                    num_layers
                ):
                    if (
                        bytes_to_reclaim <= 0
                    ):
                        break

                    if (
                        other_layer
                        == layer_id
                    ):
                        continue

                    other_slots = (
                        base_slots
                        + (
                            1
                            if (
                                other_layer
                                < remainder
                            )
                            else 0
                        )
                    )

                    resident_other = [
                        key
                        for key
                        in self._items.keys()
                        if (
                            key[0]
                            == other_layer
                        )
                    ]

                    overflow_count = max(
                        0,
                        (
                            len(
                                resident_other
                            )
                            - other_slots
                        ),
                    )

                    if (
                        overflow_count
                        == 0
                    ):
                        continue

                    planned_other = set(
                        self._prefill_layer_keys.get(
                            other_layer,
                            [],
                        )
                    )

                    # Decode-added/non-planned residents first.
                    preferred = sorted(
                        key
                        for key
                        in resident_other
                        if (
                            key
                            not in planned_other
                        )
                    )

                    # Defensive fallback. Normally overflow created
                    # by decode should be represented entirely by
                    # non-planned entries.
                    fallback = sorted(
                        key
                        for key
                        in resident_other
                        if (
                            key
                            in planned_other
                        )
                    )

                    victims = (
                        preferred
                        + fallback
                    )[
                        :overflow_count
                    ]

                    for victim_key in victims:
                        if (
                            bytes_to_reclaim
                            <= 0
                        ):
                            break

                        evicted = (
                            self._items.pop(
                                victim_key,
                                None,
                            )
                        )

                        if (
                            evicted
                            is None
                        ):
                            continue

                        self.current_bytes -= (
                            evicted.size
                        )

                        bytes_to_reclaim -= (
                            evicted.size
                        )

            if bytes_to_reclaim > 0:
                raise RuntimeError(
                    "Unable to reclaim enough resident "
                    "capacity for prepared prefill layer: "
                    f"layer={layer_id}, "
                    f"remaining_bytes={bytes_to_reclaim}, "
                    f"current={self.current_bytes}, "
                    f"required={required_bytes}, "
                    f"budget={self.budget_bytes}"
                )

            self._prefill_layer_keys[
                layer_id
            ] = desired

            self._prefill_admit = admit

    def get_prefill(
        self,
        entry: ExpertEntry,
    ) -> ResidentExpert:
        """
        Acquire a routed expert for layer-major prefill.

        Cache membership for the current layer is decided beforehand
        by prepare_prefill_layer(). This method only executes that
        deterministic decision:

        - resident hit:
            reuse the expert without perturbing global decode-LRU
            order

        - reserved miss:
            load, promote, and admit the expert

        - non-reserved miss:
            load and promote the expert for immediate computation,
            but bypass resident-cache admission

        Because admission reservations are selected synchronously
        before asynchronous prefetch starts, worker completion order
        cannot change the final resident set.

        Normal token-by-token decode continues to use get() and its
        existing global-LRU semantics.
        """
        key = (
            entry.layer,
            entry.expert,
        )

        # --------------------------------------------------------
        # Fast resident path.
        #
        # Deliberately do NOT move_to_end(). Prefill should not
        # perturb the normal decode LRU order merely by scanning
        # through the cache.
        # --------------------------------------------------------
        with self._lock:
            cached = self._items.get(
                key
            )

            if cached is not None:
                self.cache_hits += 1
                return cached

        # Keep duplicate loading/promotion suppression identical to
        # the normal get() path.
        key_lock = self._key_lock(
            key
        )

        with key_lock:
            # Another worker may have admitted this expert while we
            # were waiting for its per-key lock.
            with self._lock:
                cached = self._items.get(
                    key
                )

                if cached is not None:
                    self.cache_hits += 1
                    return cached

            t0 = perf_counter()

            chunks = self.reader.read_expert(
                entry
            )

            read_seconds = (
                perf_counter()
                - t0
            )

            payload = ExpertPayload(
                chunks=chunks
            )

            t1 = perf_counter()

            resident = promote_expert(
                entry,
                payload,
            )

            promotion_seconds = (
                perf_counter()
                - t1
            )

            if resident.size > self.budget_bytes:
                raise ValueError(
                    f"Expert size {resident.size} exceeds "
                    f"resident cache budget {self.budget_bytes}"
                )

            with self._lock:
                # ------------------------------------------------
                # Deterministic adaptive prefill admission.
                #
                # prepare_prefill_layer() selected this set before
                # any async reads started. Worker completion order
                # therefore cannot alter cache membership.
                # ------------------------------------------------
                should_admit = (
                    key
                    in self._prefill_admit
                )

                if should_admit:
                    if (
                        self.current_bytes
                        + resident.size
                        > self.budget_bytes
                    ):
                        raise RuntimeError(
                            "Prepared prefill admission exceeds "
                            "resident cache budget: "
                            f"key={key}, "
                            f"current={self.current_bytes}, "
                            f"resident={resident.size}, "
                            f"budget={self.budget_bytes}"
                        )

                    self._items[
                        key
                    ] = resident

                    self.current_bytes += (
                        resident.size
                    )

                # A bypassed resident is still a real cache miss and
                # still incurred SSD + promotion work.
                self.cache_misses += 1

                self.ssd_bytes_read += (
                    payload.size
                )

                self.ssd_read_seconds += (
                    read_seconds
                )

                self.promotion_seconds += (
                    promotion_seconds
                )

            return resident

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
            if self.budget_bytes == 0:
                return 0.0
            return self.current_bytes / self.budget_bytes

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> ResidentExpertStore:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
