import threading

import pytest

from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.io.resident_prefetch import ResidentExpertPrefetcher
from fakes import EXPERT_BYTES, FakeReader, make_index

N_LAYERS = 4
N_EXPERTS = 8


@pytest.fixture
def index():
    return make_index(N_LAYERS, N_EXPERTS)


def make_store(slots: int, latency: float = 0.0):
    reader = FakeReader(latency_s=latency)
    return ResidentExpertStore(budget_bytes=slots * EXPERT_BYTES, reader=reader), reader


def test_hit_miss_and_byte_accounting(index):
    store, reader = make_store(slots=4)
    e = index[(0, 1)]
    a = store.get(e)
    b = store.get(e)
    assert a is b
    stats = store.stats()
    assert (stats.cache_hits, stats.cache_misses) == (1, 1)
    assert stats.ssd_bytes_read == EXPERT_BYTES
    assert store.current_bytes == EXPERT_BYTES
    assert len(store) == 1


def test_lru_eviction_respects_budget(index):
    store, _ = make_store(slots=2)
    store.get(index[(0, 0)])
    store.get(index[(0, 1)])
    store.get(index[(0, 0)])  # refresh 0 -> 1 is LRU
    store.get(index[(0, 2)])  # evicts 1
    assert len(store) == 2
    assert store.current_bytes <= store.budget_bytes
    with store._lock:
        keys = set(store._items)
    assert keys == {(0, 0), (0, 2)}


def test_oversized_expert_rejected(index):
    store, _ = make_store(slots=1)
    store.budget_bytes = EXPERT_BYTES - 1
    with pytest.raises(ValueError):
        store.get(index[(0, 0)])


def test_concurrent_get_loads_once(index):
    store, reader = make_store(slots=8, latency=0.02)
    e = index[(1, 3)]
    results = []

    def worker():
        results.append(store.get(e))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r is results[0] for r in results)
    assert store.stats().cache_misses == 1
    assert store.stats().cache_hits == 7


def test_prefill_admission_is_deterministic_and_quota_bound(index):
    # 8 slots over 4 layers -> 2 per layer
    store, _ = make_store(slots=8)
    entries = [index[(0, i)] for i in range(5)]
    store.prepare_prefill_layer(0, entries, num_layers=N_LAYERS)
    for e in entries:
        store.get_prefill(e)
    with store._lock:
        resident = sorted(k for k in store._items if k[0] == 0)
    assert resident == [(0, 0), (0, 1)]
    stats = store.stats()
    assert stats.cache_misses == 5  # bypassed misses still cost SSD
    assert stats.ssd_bytes_read == 5 * EXPERT_BYTES


def test_prefill_reclaims_decode_overflow_from_other_layers(index):
    store, _ = make_store(slots=8)
    # decode fills the store with layer-3 experts beyond its quota of 2
    for i in range(8):
        store.get(index[(3, i)])
    assert len(store) == 8
    store.prepare_prefill_layer(0, [index[(0, 0)], index[(0, 1)]], num_layers=N_LAYERS)
    store.get_prefill(index[(0, 0)])
    store.get_prefill(index[(0, 1)])
    assert store.current_bytes <= store.budget_bytes
    with store._lock:
        layer3 = [k for k in store._items if k[0] == 3]
    assert len(layer3) <= 6


def test_prefetcher_dedups_and_skips_resident(index):
    store, reader = make_store(slots=8, latency=0.01)
    pf = ResidentExpertPrefetcher(store, workers=2)
    try:
        e = index[(2, 2)]
        store.prepare_prefill_layer(2, [e], num_layers=N_LAYERS)
        pf.prefetch(e)
        pf.prefetch(e)
        assert pf.pending_count() == 1
        r = pf.get(e)
        assert pf.pending_count() == 0
        pf.prefetch(e)  # already resident -> no job
        assert pf.pending_count() == 0
        assert store.get_prefill(e) is r
    finally:
        pf.close()
