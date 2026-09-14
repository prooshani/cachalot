"""Shared helpers for Cachalot benchmark scripts."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx

MODEL_PATH = "/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class StoreSnapshot:
    cache_hits: int
    cache_misses: int
    ssd_bytes_read: int
    ssd_read_seconds: float
    promotion_seconds: float

    @classmethod
    def take(cls, runtime) -> StoreSnapshot:
        s = runtime.expert_store.stats()
        return cls(
            s.cache_hits,
            s.cache_misses,
            s.ssd_bytes_read,
            s.ssd_read_seconds,
            s.promotion_seconds,
        )

    def delta(self, later: StoreSnapshot) -> StoreSnapshot:
        return StoreSnapshot(
            later.cache_hits - self.cache_hits,
            later.cache_misses - self.cache_misses,
            later.ssd_bytes_read - self.ssd_bytes_read,
            later.ssd_read_seconds - self.ssd_read_seconds,
            later.promotion_seconds - self.promotion_seconds,
        )


def mlx_memory() -> dict[str, float]:
    gib = 1024**3
    return {
        "active_gib": mx.get_active_memory() / gib,
        "cache_gib": mx.get_cache_memory() / gib,
        "peak_gib": mx.get_peak_memory() / gib,
    }


def resident_layer_counts(runtime, n_layers: int = 40) -> list[int]:
    counts = [0] * n_layers
    with runtime.expert_store._lock:
        for layer, _expert in runtime.expert_store._items.keys():
            counts[layer] += 1
    return counts


def phase_report(name: str, seconds: float, tokens: int, delta: StoreSnapshot, runtime) -> dict:
    requests = delta.cache_hits + delta.cache_misses
    counts = resident_layer_counts(runtime)
    report = {
        "phase": name,
        "seconds": seconds,
        "tokens": tokens,
        "tok_per_s": tokens / seconds if seconds else 0.0,
        "sec_per_token": seconds / tokens if tokens else 0.0,
        "requests": requests,
        "hits": delta.cache_hits,
        "misses": delta.cache_misses,
        "hit_rate": delta.cache_hits / requests if requests else 0.0,
        "ssd_gib": delta.ssd_bytes_read / 1024**3,
        "ssd_worker_seconds": delta.ssd_read_seconds,
        "promotion_worker_seconds": delta.promotion_seconds,
        "resident_experts": len(runtime.expert_store),
        "resident_gib": runtime.expert_store.current_bytes / 1024**3,
        "layer_min": min(counts),
        "layer_max": max(counts),
        **mlx_memory(),
    }
    print(json.dumps(report, indent=None), flush=True)
    return report


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)))
