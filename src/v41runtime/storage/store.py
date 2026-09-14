from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import perf_counter

from v41runtime.cache.lru import ByteLRUCache
from v41runtime.storage.index import ExpertEntry
from v41runtime.storage.reader import ExpertReader, ReadChunk


@dataclass(frozen=True)
class ExpertPayload:
    chunks: tuple[ReadChunk, ...]

    @property
    def size(self) -> int:
        return sum(len(chunk.data) for chunk in self.chunks)


@dataclass(frozen=True)
class ExpertStoreStats:
    cache_hits: int
    cache_misses: int
    ssd_bytes_read: int
    cache_bytes_served: int
    ssd_read_seconds: float

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


class ExpertStore:
    def __init__(
        self,
        cache_budget_bytes: int,
        reader: ExpertReader | None = None,
    ) -> None:
        self.cache = ByteLRUCache(cache_budget_bytes)
        self.reader = reader or ExpertReader()

        self.cache_hits = 0
        self.cache_misses = 0

        self.ssd_bytes_read = 0
        self.cache_bytes_served = 0
        self.ssd_read_seconds = 0.0

        self._lock = RLock()
        self._key_locks: dict[tuple[int, int], RLock] = {}

    def _key_lock(self, key: tuple[int, int]) -> RLock:
        with self._lock:
            lock = self._key_locks.get(key)

            if lock is None:
                lock = RLock()
                self._key_locks[key] = lock

            return lock

    def get(self, entry: ExpertEntry) -> ExpertPayload:
        key = (entry.layer, entry.expert)

        cached = self.cache.get(key)

        if cached is not None:
            with self._lock:
                self.cache_hits += 1
                self.cache_bytes_served += cached.size

            return cached

        key_lock = self._key_lock(key)

        with key_lock:
            cached = self.cache.get(key)

            if cached is not None:
                with self._lock:
                    self.cache_hits += 1
                    self.cache_bytes_served += cached.size

                return cached

            t0 = perf_counter()

            payload = ExpertPayload(
                chunks=self.reader.read_expert(entry)
            )

            read_seconds = perf_counter() - t0

            with self._lock:
                self.cache_misses += 1
                self.ssd_bytes_read += payload.size
                self.ssd_read_seconds += read_seconds

            self.cache.put(
                key,
                payload,
                payload.size,
            )

            return payload

    def stats(self) -> ExpertStoreStats:
        with self._lock:
            return ExpertStoreStats(
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                ssd_bytes_read=self.ssd_bytes_read,
                cache_bytes_served=self.cache_bytes_served,
                ssd_read_seconds=self.ssd_read_seconds,
            )

    @property
    def hit_rate(self) -> float:
        return self.stats().hit_rate

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> ExpertStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
