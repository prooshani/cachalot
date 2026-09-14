from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock


@dataclass
class CacheEntry:
    value: object
    size: int


class ByteLRUCache:
    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self.current_bytes = 0
        self._items: OrderedDict[object, CacheEntry] = OrderedDict()
        self._lock = RLock()

    def get(self, key: object):
        with self._lock:
            entry = self._items.get(key)

            if entry is None:
                return None

            self._items.move_to_end(key)
            return entry.value

    def put(self, key: object, value: object, size: int) -> None:
        if size > self.budget_bytes:
            raise ValueError(
                f"Entry size {size} exceeds cache budget {self.budget_bytes}"
            )

        with self._lock:
            existing = self._items.pop(key, None)

            if existing is not None:
                self.current_bytes -= existing.size

            while self._items and self.current_bytes + size > self.budget_bytes:
                _, evicted = self._items.popitem(last=False)
                self.current_bytes -= evicted.size

            self._items[key] = CacheEntry(
                value=value,
                size=size,
            )

            self.current_bytes += size

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def utilization(self) -> float:
        with self._lock:
            if self.budget_bytes == 0:
                return 0.0

            return self.current_bytes / self.budget_bytes
