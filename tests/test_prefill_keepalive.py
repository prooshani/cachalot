import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from cachalot.model import text_decode_runtime as tdr


def test_waiting_on_a_slow_read_keeps_the_gpu_queue_busy(monkeypatch):
    # HANDOFF section 15.11: a multi-second wait for a chunk's Engram rows let
    # macOS un-wire the model; the wait now evaluates a probe every period.
    monkeypatch.setattr(tdr, "PREFILL_KEEPALIVE_S", 0.02)
    rt = SimpleNamespace(prefill_keepalives=0)
    release = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(lambda: release.wait(5) and "rows")
        threading.Timer(0.2, release.set).start()
        assert tdr.TextDecodeRuntime._await_keeping_gpu_awake(rt, future) == "rows"
    assert rt.prefill_keepalives >= 3


def test_keepalive_disabled_just_waits(monkeypatch):
    monkeypatch.setattr(tdr, "PREFILL_KEEPALIVE_S", 0.0)
    rt = SimpleNamespace(prefill_keepalives=0)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(lambda: 7)
        assert tdr.TextDecodeRuntime._await_keeping_gpu_awake(rt, future) == 7
    assert rt.prefill_keepalives == 0
