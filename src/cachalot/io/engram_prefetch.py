from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock

import mlx.core as mx
import numpy as np

from cachalot.model.engram_rows import load_engram_rows
from cachalot.storage.engram_reader import (
    EngramRowReader,
    EngramTableLayout,
)


@dataclass(frozen=True)
class PrefetchedEngramRows:
    requested_ids: np.ndarray
    unique_ids: np.ndarray
    inverse: np.ndarray
    unique_values: mx.array

    def materialize_requested(self) -> mx.array:
        if self.requested_ids.size == 0:
            return mx.zeros(
                (0, self.unique_values.shape[-1]),
                dtype=self.unique_values.dtype,
            )

        return self.unique_values[
            mx.array(
                self.inverse,
                dtype=mx.int32,
            )
        ]


def _load_deduplicated(
    reader: EngramRowReader,
    layout: EngramTableLayout,
    row_ids: np.ndarray,
) -> PrefetchedEngramRows:
    requested = np.asarray(
        row_ids,
        dtype=np.int64,
    ).reshape(-1)

    if requested.size == 0:
        unique = requested
        inverse = np.empty(
            (0,),
            dtype=np.int64,
        )
    else:
        unique, inverse = np.unique(
            requested,
            return_inverse=True,
        )

    values = load_engram_rows(
        reader,
        layout,
        unique,
    )

    return PrefetchedEngramRows(
        requested_ids=requested,
        unique_ids=unique,
        inverse=inverse,
        unique_values=values,
    )


class EngramPrefetcher:
    def __init__(
        self,
        reader: EngramRowReader,
        workers: int = 2,
    ) -> None:
        self.reader = reader

        self._pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="engram-prefetch",
        )

        self._pending: dict[
            tuple[int, tuple[int, ...]],
            Future[PrefetchedEngramRows],
        ] = {}

        self._lock = RLock()

    @staticmethod
    def _key(
        layout: EngramTableLayout,
        row_ids: np.ndarray,
    ) -> tuple[int, tuple[int, ...]]:
        ids = np.asarray(
            row_ids,
            dtype=np.int64,
        ).reshape(-1)

        return (
            layout.layer,
            tuple(int(x) for x in ids),
        )

    def prefetch(
        self,
        layout: EngramTableLayout,
        row_ids: np.ndarray,
    ) -> None:
        key = self._key(
            layout,
            row_ids,
        )

        with self._lock:
            if key in self._pending:
                return

            self._pending[key] = self._pool.submit(
                _load_deduplicated,
                self.reader,
                layout,
                np.asarray(
                    row_ids,
                    dtype=np.int64,
                ).copy(),
            )

    def get(
        self,
        layout: EngramTableLayout,
        row_ids: np.ndarray,
    ) -> PrefetchedEngramRows:
        key = self._key(
            layout,
            row_ids,
        )

        with self._lock:
            future = self._pending.pop(
                key,
                None,
            )

        if future is not None:
            return future.result()

        return _load_deduplicated(
            self.reader,
            layout,
            row_ids,
        )

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def close(self) -> None:
        with self._lock:
            futures = list(
                self._pending.values()
            )
            self._pending.clear()

        for future in futures:
            future.cancel()

        self._pool.shutdown(
            wait=True,
            cancel_futures=True,
        )

    def __enter__(
        self,
    ) -> "EngramPrefetcher":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
