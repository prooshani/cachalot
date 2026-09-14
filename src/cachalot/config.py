"""
Runtime configuration.

Defaults target a 96 GB Apple Silicon Mac. Every field can be overridden with
an environment variable named CACHALOT_<FIELD> (upper-case), e.g.

    CACHALOT_EXPERT_CACHE_BUDGET_GIB=48 CACHALOT_MODEL_PATH=/Volumes/NVMe/DeepSeek-V4.1-Flash cachalot serve
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace

GiB = 1024**3


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: str = "/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash"

    # MLX working-set limit (mx.set_memory_limit).
    mlx_memory_limit_bytes: int = 64 * GiB

    # Cap on MLX's free-buffer cache. Larger values recreate allocator stalls
    # under concurrent expert materialization; do not raise without evidence.
    mlx_cache_limit_bytes: int = 2 * GiB

    # Routed experts kept resident in MLX memory. 40 GiB fits a 96 GB Mac
    # together with the ~12 GiB trunk while leaving room for macOS and an
    # agent harness.
    expert_cache_budget_bytes: int = 40 * GiB

    # Prefetch / load threads. The SSD is the bottleneck, not thread count.
    io_workers: int = 8

    # Sequence capacity for KV / compressed caches (must be even).
    max_seq_len: int = 32768

    # Server defaults
    host: str = "127.0.0.1"
    port: int = 8000
    model_id: str = "deepseek-v4.1-flash"

    @property
    def expert_cache_budget_gib(self) -> float:
        return self.expert_cache_budget_bytes / GiB

    def with_env_overrides(self, environ: dict[str, str] | None = None) -> RuntimeConfig:
        """Apply CACHALOT_* environment variables. *_GIB variants set byte fields."""
        env = os.environ if environ is None else environ
        updates: dict[str, object] = {}
        for f in fields(self):
            key = f"CACHALOT_{f.name.upper()}"
            if f.name.endswith("_bytes"):
                gib_key = key[: -len("_BYTES")] + "_GIB"
                if gib_key in env:
                    updates[f.name] = int(float(env[gib_key]) * GiB)
                    continue
            if key in env:
                raw = env[key]
                updates[f.name] = type(getattr(self, f.name))(raw) if f.type != "str" else raw
        return replace(self, **updates) if updates else self


DEFAULT_CONFIG = RuntimeConfig()


def load_config() -> RuntimeConfig:
    return DEFAULT_CONFIG.with_env_overrides()
