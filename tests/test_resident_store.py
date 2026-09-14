import threading

import pytest

from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.io.resident_prefetch import ResidentExpertPrefetcher
from fakes import EXPERT_BYTES, FAKE_TENSOR_SIZES, FakeReader, make_index

N_LAYERS = 4
N_EXPERTS = 8


@pytest.fixture
def index():
    return make_index(N_LAYERS, N_EXPERTS)


def make_store(slots: int, latency: float = 0.0, transient: int = 4):
    reader = FakeReader(latency_s=latency)
    store = ResidentExpertStore(
        budget_bytes=slots * EXPERT_BYTES, reader=reader,
        tensor_sizes=FAKE_TENSOR_SIZES, transient_slots=transient,
    )
    return store, reader


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


def test_budget_below_one_expert_rejected():
    with pytest.raises(ValueError):
        ResidentExpertStore(budget_bytes=EXPERT_BYTES - 1, reader=FakeReader(), tensor_sizes=FAKE_TENSOR_SIZES)


def test_loaded_bytes_match_source(index):
    store, reader = make_store(slots=2)
    e = index[(0, 3)]
    r = store.get(e)
    import numpy as np

    from cachalot.storage.store import ExpertPayload
    from cachalot.storage.tensors import extract_expert_tensors

    ref = extract_expert_tensors(e, ExpertPayload(chunks=FakeReader().read_expert(e)))
    for name in ("w1.weight", "w2.scale", "w3.weight"):
        assert np.array_equal(np.array(r.as_model_dict()[name]), np.frombuffer(ref[name].data, dtype=np.uint8))


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
    # three bypass loads hold transient slots until released
    assert store.transient_free() == store.transient_slots - 3
    store.release_transients([(0, 2), (0, 3), (0, 4)])
    assert store.transient_free() == store.transient_slots


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


def test_get_many_parallel_misses_admit_in_order(index):
    store, reader = make_store(slots=8, latency=0.02)
    entries = [index[(0, i)] for i in range(6)]
    import time

    t0 = time.perf_counter()
    got = store.get_many(entries)
    elapsed = time.perf_counter() - t0

    assert [(e.layer, e.expert) for e in got] == [(0, i) for i in range(6)]
    # 6 x 20 ms serial would be >= 120 ms; parallel should be well under.
    assert elapsed < 0.09
    stats = store.stats()
    assert (stats.cache_hits, stats.cache_misses) == (0, 6)
    with store._lock:
        assert list(store._items) == [(0, i) for i in range(6)]  # logical order

    again = store.get_many(entries)
    assert all(a is b for a, b in zip(got, again, strict=True))
    assert store.stats().cache_hits == 6


def test_get_many_respects_budget_and_mixed_hits(index):
    store, _ = make_store(slots=3)
    store.get(index[(1, 0)])
    got = store.get_many([index[(1, 0)], index[(1, 1)], index[(1, 2)], index[(1, 3)]])
    assert len(got) == 4
    assert store.current_bytes <= store.budget_bytes
    assert len(store) == 3
    store.close()


def test_transient_slots_block_until_released(index):
    import threading
    import time

    store, _ = make_store(slots=8, transient=2)
    entries = [index[(1, i)] for i in range(4)]
    store.prepare_prefill_layer(1, [], num_layers=N_LAYERS)  # nothing admitted
    store.get_prefill(entries[0])
    store.get_prefill(entries[1])
    assert store.transient_free() == 0

    done = threading.Event()

    def blocked():
        store.get_prefill(entries[2])
        done.set()

    threading.Thread(target=blocked, daemon=True).start()
    time.sleep(0.05)
    assert not done.is_set()
    store.release_transients([(1, 0)])
    assert done.wait(1.0)
    store.release_all_transients()
    assert store.transient_free() == 2


def test_slot_reuse_after_eviction(index):
    store, _ = make_store(slots=2, transient=1)
    a = store.get(index[(0, 0)])
    slot_a = a.slot
    store.get(index[(0, 1)])
    c = store.get(index[(0, 2)])  # evicts (0,0); its slot must be reused by someone
    with store._lock:
        assert (0, 0) not in store._items
    assert c.slot is slot_a or store.pool.free_count >= 1
