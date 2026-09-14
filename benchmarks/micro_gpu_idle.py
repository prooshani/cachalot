"""Does GPU work run slower right after an idle gap (clock ramp)? Back-to-back vs sleep-interleaved."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    MODEL_PATH,  # noqa: E402
    load_expert_standalone,  # noqa: E402
)
from cachalot.model.expert_metal import routed_expert_forward  # noqa: E402
from cachalot.storage.index import build_expert_index  # noqa: E402
from cachalot.storage.reader import ExpertReader  # noqa: E402


def main():
    index = build_expert_index(MODEL_PATH)
    reader = ExpertReader()
    experts = [load_expert_standalone(reader, e) for e in (index[(10, i)] for i in range(6))]
    x = mx.random.normal((5120,)).astype(mx.bfloat16)
    # a bigger GPU chunk resembling one layer: 6 experts + a 129k x 5120 head-like matmul
    head = mx.random.normal((16384, 5120)).astype(mx.bfloat16)
    mx.eval(x, head)

    def layer_work():
        outs = [routed_expert_forward(x, w1_packed=e.w1_weight, w1_scales=e.w1_scale, w2_packed=e.w2_weight,
                                      w2_scales=e.w2_scale, w3_packed=e.w3_weight, w3_scales=e.w3_scale, weight=0.3)
                for e in experts]
        y = mx.sum(mx.stack(outs), axis=0)
        z = head.astype(mx.float32) @ y
        mx.eval(z)

    for _ in range(5):
        layer_work()

    tiny = mx.zeros((64,))
    mx.eval(tiny)

    def gap_sleep(ms):
        time.sleep(ms / 1000)

    def gap_spin(ms):
        t_end = perf_counter() + ms / 1000
        while perf_counter() < t_end:
            pass

    def gap_gpu_heartbeat(ms, period_ms=2.0):
        t_end = perf_counter() + ms / 1000
        while perf_counter() < t_end:
            mx.eval(tiny + 1.0)
            time.sleep(period_ms / 1000)

    def gap_gpu_busy(ms):
        # keep GPU loaded with a mid-size op (~0.3 ms each) back to back
        t_end = perf_counter() + ms / 1000
        while perf_counter() < t_end:
            mx.eval(head[:2048].astype(mx.float32).sum())

    for name, gap in (("sleep", gap_sleep), ("cpu-spin", gap_spin), ("gpu-heartbeat-2ms", gap_gpu_heartbeat), ("gpu-busy", gap_gpu_busy)):
        for gap_ms in (0, 60):
            times = []
            for _ in range(12):
                if gap_ms:
                    gap(gap_ms)
                t0 = perf_counter()
                layer_work()
                times.append(perf_counter() - t0)
            times.sort()
            print(f"{name:20s} gap {gap_ms:3d} ms -> layer work median {times[6] * 1e3:6.2f} ms, min {times[0] * 1e3:6.2f}, max {times[-1] * 1e3:6.2f}", flush=True)


if __name__ == "__main__":
    main()
