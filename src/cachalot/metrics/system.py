import platform
import psutil
import mlx.core as mx


def system_snapshot() -> dict[str, object]:
    vm = psutil.virtual_memory()

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "physical_memory_bytes": vm.total,
        "available_memory_bytes": vm.available,
        "mlx_device": str(mx.default_device()),
        "mlx_active_memory": mx.get_active_memory(),
        "mlx_cache_memory": mx.get_cache_memory(),
        "mlx_peak_memory": mx.get_peak_memory(),
    }
