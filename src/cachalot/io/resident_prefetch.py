from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock

from cachalot.cache.resident import ResidentExpert
from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.storage.index import ExpertEntry


class ResidentExpertPrefetcher:
    def __init__(
        self,
        store: ResidentExpertStore,
        workers: int = 2,
    ) -> None:
        if workers <= 0:
            raise ValueError(
                f"workers must be > 0, got {workers}"
            )

        self.store = store
        self.workers = int(workers)

        self._pool = ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="resident-prefetch",
        )

        self._pending: dict[
            tuple[int, int],
            Future[ResidentExpert],
        ] = {}

        self._lock = RLock()

    def prefetch(
        self,
        entry: ExpertEntry,
    ) -> None:
        key = (entry.layer, entry.expert)

        # Avoid touching the store's hit/miss statistics merely
        # to check residency.
        with self.store._lock:
            if key in self.store._items:
                return

        with self._lock:
            future = self._pending.get(key)

            if future is not None:
                return

            self._pending[key] = self._pool.submit(
                self.store.get_prefill,
                entry,
            )

    def get(
        self,
        entry: ExpertEntry,
    ) -> ResidentExpert:
        key = (entry.layer, entry.expert)

        with self._lock:
            future = self._pending.pop(
                key,
                None,
            )

        if future is not None:
            return future.result()

        return self.store.get_prefill(entry)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def close(self) -> None:
        with self._lock:
            futures = list(self._pending.values())
            self._pending.clear()

        for future in futures:
            future.cancel()

        self._pool.shutdown(
            wait=True,
            cancel_futures=True,
        )

    def __enter__(
        self,
    ) -> "ResidentExpertPrefetcher":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
