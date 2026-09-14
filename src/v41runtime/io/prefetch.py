from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from v41runtime.storage.index import ExpertEntry
from v41runtime.storage.store import ExpertPayload, ExpertStore


class ExpertPrefetcher:
    def __init__(
        self,
        store: ExpertStore,
        workers: int = 2,
    ) -> None:
        self.store = store
        self._pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="expert-prefetch",
        )
        self._pending: dict[
            tuple[int, int],
            Future[ExpertPayload],
        ] = {}

    def prefetch(self, entry: ExpertEntry) -> None:
        key = (entry.layer, entry.expert)

        if self.store.cache.get(key) is not None:
            return

        if key in self._pending:
            return

        self._pending[key] = self._pool.submit(
            self.store.get,
            entry,
        )

    def get(self, entry: ExpertEntry) -> ExpertPayload:
        key = (entry.layer, entry.expert)

        future = self._pending.pop(key, None)

        if future is not None:
            return future.result()

        return self.store.get(entry)

    def close(self) -> None:
        for future in self._pending.values():
            future.cancel()

        self._pending.clear()
        self._pool.shutdown(wait=True)

    def __enter__(self) -> ExpertPrefetcher:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
