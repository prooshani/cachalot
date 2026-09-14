from dataclasses import dataclass

GiB = 1024**3


@dataclass(frozen=True)
class RuntimeConfig:
    mlx_memory_limit_bytes: int = 64 * GiB
    mlx_cache_limit_bytes: int = 2 * GiB
    expert_cache_budget_bytes: int = 40 * GiB
    io_workers: int = 8

    model_path: str = "/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash"


DEFAULT_CONFIG = RuntimeConfig()
