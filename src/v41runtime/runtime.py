import mlx.core as mx

from v41runtime.config import DEFAULT_CONFIG, RuntimeConfig


def initialize_runtime(config: RuntimeConfig = DEFAULT_CONFIG) -> int:
    previous_limit = mx.set_memory_limit(config.mlx_memory_limit_bytes)
    mx.reset_peak_memory()
    return previous_limit


def memory_stats() -> dict[str, int]:
    return {
        "active": mx.get_active_memory(),
        "cache": mx.get_cache_memory(),
        "peak": mx.get_peak_memory(),
    }
