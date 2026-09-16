"""
Runtime configuration.

Defaults target a 96 GB Apple Silicon Mac. Every field can be overridden with
an environment variable named CACHALOT_<FIELD> (upper-case), e.g.

    CACHALOT_EXPERT_CACHE_BUDGET_GIB=48 CACHALOT_MODEL_PATH=/Volumes/NVMe/DeepSeek-V4.1-Flash cachalot serve
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

GiB = 1024**3


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: str = ""  # empty = discover (see default_model_path)

    # MLX working-set limit (mx.set_memory_limit).
    mlx_memory_limit_bytes: int = 64 * GiB

    # Bytes of MLX memory kept wired (Metal residency set). Without this,
    # macOS compresses cold expert buffers under memory pressure and every
    # GPU access to them pays a page decompression (measured 3-10x slower
    # decode). 0 means auto: trunk + expert budget + MLX cache, capped at
    # the Metal recommended working set.
    mlx_wired_limit_bytes: int = 0

    # Cap on MLX's free-buffer cache. Larger values recreate allocator stalls
    # under concurrent expert materialization; do not raise without evidence.
    mlx_cache_limit_bytes: int = 2 * GiB

    # Routed experts kept resident in MLX memory. 0 means auto:
    #   total unified memory - trunk - MLX cache - system reserve,
    # clamped to [8 GiB, all experts] and to the Metal recommended working
    # set. On a 96 GB Mac this yields ~50 GiB; on 64 GB ~18 GiB; on 256 GB
    # the full 270 GiB expert set becomes resident.
    expert_cache_budget_bytes: int = 0

    # Memory left to macOS, page cache and other apps when auto-sizing.
    system_reserve_bytes: int = 32 * GiB

    # Prefetch / load threads. The SSD is the bottleneck, not thread count.
    io_workers: int = 8

    # Sequence capacity for KV / compressed caches (must be even).
    max_seq_len: int = 32768

    # Prefix-cache snapshots kept (2 per active conversation).
    prefix_cache_entries: int = 16
    # Idle heartbeat period in seconds (0 disables). When the runtime has run
    # no forward pass for this long, a background thread evaluates a trivial
    # MLX op every period. Measured 2026-09-16: within ~6 s of an idle Metal
    # queue macOS un-wires the whole working set (45 GiB wired -> 6 GiB) and
    # compresses it; the next chat turn then paid ~7 s of decompression
    # (13-token prefill 10.5 s instead of 3 s). A 0.5 s heartbeat keeps the
    # residency set wired across idle gaps.
    idle_heartbeat_seconds: float = 0.5

    # Server defaults
    host: str = "127.0.0.1"
    port: int = 8000
    model_id: str = "deepseek-v4.1-flash"

    @property
    def resolved_model_path(self) -> str:
        return self.model_path or default_model_path()

    @property
    def expert_cache_budget_gib(self) -> float:
        return resolve_expert_budget(self) / GiB

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


MODEL_DIR_NAME = "DeepSeek-V4.1-Flash"
MODEL_SEARCH_ROOTS = (
    Path.home(),
    Path.home() / "models",
    Path("/Volumes"),
)


def default_model_path() -> str:
    """CACHALOT_MODEL_PATH, else the first DeepSeek-V4.1-Flash directory found in common roots."""
    env = os.environ.get("CACHALOT_MODEL_PATH")
    if env:
        return env
    for root in MODEL_SEARCH_ROOTS:
        direct = root / MODEL_DIR_NAME
        if (direct / "config.json").exists():
            return str(direct)
        if root.is_dir() and root.name == "Volumes":
            for volume in sorted(root.glob("*")):
                for hit in sorted(volume.glob(f"*/{MODEL_DIR_NAME}/config.json"))[:1]:
                    return str(hit.parent)
                if (volume / MODEL_DIR_NAME / "config.json").exists():
                    return str(volume / MODEL_DIR_NAME)
    return str(Path.home() / MODEL_DIR_NAME)


TRUNK_BYTES = 12 * GiB           # resident non-expert weights of V4.1 Flash
ALL_EXPERTS_BYTES = 15_360 * 18_800_640   # every routed expert resident
MIN_EXPERT_BUDGET_BYTES = 8 * GiB
WIRED_HEADROOM_BYTES = 6 * GiB
AVAILABLE_HEADROOM_BYTES = 12 * GiB   # transient slots + activations (~5 GiB) and a free floor for the OS


def device_memory() -> tuple[int, int]:
    """(total unified memory, Metal recommended working set) in bytes."""
    try:
        import mlx.core as mx

        info = mx.device_info()
        return int(info["memory_size"]), int(info["max_recommended_working_set_size"])
    except Exception:  # pragma: no cover - non-Apple hosts
        import psutil

        total = int(psutil.virtual_memory().total)
        return total, int(total * 0.8)


def available_memory() -> int | None:
    """
    Memory the runtime can take right now without pushing other processes
    into swap (macOS `vm_stat`): free + speculative + purgeable pages plus
    the file cache (file-backed pages, bounded by the inactive list), which
    the kernel drops without swapping. Anonymous memory of other processes
    is not counted. None when unknown.
    """
    try:
        import subprocess

        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # pragma: no cover - non-macOS hosts
        return None
    page = 16384
    counts: dict[str, int] = {}
    for line in out.splitlines():
        if line.startswith("Mach Virtual Memory Statistics"):
            if "page size of" in line:
                page = int(line.split("page size of")[1].split()[0])
            continue
        if ":" in line:
            name, _, value = line.partition(":")
            value = value.strip().rstrip(".")
            if value.isdigit():
                counts[name.strip().strip('"')] = int(value)
    if "Pages free" not in counts:
        return None
    reclaimable_cache = min(counts.get("File-backed pages", 0), counts.get("Pages inactive", 0))
    pages = counts["Pages free"] + counts.get("Pages speculative", 0) + counts.get("Pages purgeable", 0) + reclaimable_cache
    return pages * page


def resolve_expert_budget(config: RuntimeConfig) -> int:
    """Concrete expert budget in bytes (auto-sized when config value is 0)."""
    if config.expert_cache_budget_bytes > 0:
        return int(config.expert_cache_budget_bytes)
    total, recommended = device_memory()
    by_total = total - TRUNK_BYTES - config.mlx_cache_limit_bytes - config.system_reserve_bytes
    # Leave room inside the wired/recommended set for transient slots (~2.4 GiB),
    # prefix snapshots and activations; 64 GiB of experts on this 96 GB machine
    # produced a Metal out-of-memory during prefill.
    by_wired = recommended - TRUNK_BYTES - config.mlx_cache_limit_bytes - WIRED_HEADROOM_BYTES
    budget = min(by_total, by_wired, ALL_EXPERTS_BYTES)
    # Other processes already hold what they hold: never plan to wire more
    # than what is free or reclaimable now, minus the trunk, the MLX cache and
    # AVAILABLE_HEADROOM (transient slots, activations, a free floor for the
    # OS). On a machine whose other applications use 20 GB this trims the
    # 96 GB formula budget by a few GiB instead of pushing them into swap.
    available = available_memory()
    if available is not None:
        by_available = available - TRUNK_BYTES - config.mlx_cache_limit_bytes - AVAILABLE_HEADROOM_BYTES
        budget = min(budget, by_available)
    return int(max(budget, MIN_EXPERT_BUDGET_BYTES))


def resolve_wired_limit(config: RuntimeConfig, expert_budget: int) -> int:
    """Wired-memory limit in bytes (auto when config value is 0)."""
    _, recommended = device_memory()
    if config.mlx_wired_limit_bytes > 0:
        return int(min(config.mlx_wired_limit_bytes, recommended))
    return int(min(TRUNK_BYTES + expert_budget + config.mlx_cache_limit_bytes, recommended))


DEFAULT_CONFIG = RuntimeConfig()


def load_config() -> RuntimeConfig:
    return DEFAULT_CONFIG.with_env_overrides()
