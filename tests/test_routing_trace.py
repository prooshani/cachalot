import numpy as np

from cachalot.metrics.routing_trace import PHASE_DECODE, PHASE_PREFILL, RoutingTracer, load_trace


def test_tracer_records_prefill_and_decode(tmp_path):
    t = RoutingTracer()
    t.mark("turn1")
    t.record("prefill", 3, 0, np.array([[1, 2, 3], [4, 5, 6]]))
    t.record("decode", 3, 2, np.array([7, 8, 9]))
    assert t.records == 3

    path = t.save(tmp_path / "x.trace.npz")
    arrays, segments = load_trace(path)

    assert arrays["phase"].tolist() == [PHASE_PREFILL, PHASE_PREFILL, PHASE_DECODE]
    assert arrays["layer"].tolist() == [3, 3, 3]
    assert arrays["position"].tolist() == [0, 1, 2]
    assert arrays["experts"].tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert segments == [{"label": "turn1", "at": 0}]


def test_empty_tracer_saves(tmp_path):
    t = RoutingTracer()
    arrays, segments = load_trace(t.save(tmp_path / "e.trace.npz"))
    assert arrays["experts"].shape[0] == 0
    assert segments == []
