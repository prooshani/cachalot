import threading

import pytest

from cachalot.cache import resident_store
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


def test_get_many_miss_budget_skips_lowest_priority(index):
    store, _ = make_store(slots=8)
    entries = [index[(2, i)] for i in range(6)]
    prio = [0.5, 0.1, 0.9, 0.2, 0.05, 0.3]
    got = store.get_many(entries, max_misses=2, priorities=prio)
    loaded = [i for i, e in enumerate(got) if e is not None]
    assert loaded == [0, 2]  # highest priorities
    assert store.skipped_experts == 4
    assert store.stats().cache_misses == 2
    # now residents count as hits regardless of budget
    got2 = store.get_many(entries, max_misses=0, priorities=prio)
    assert [i for i, e in enumerate(got2) if e is not None] == [0, 2]


def test_prepare_keeps_needed_speculative_transients_and_releases_the_rest(index):
    store, _ = make_store(slots=4, transient=4)
    # layer 0 planned; then two speculative loads for layer 1 land in transient slots
    store.prepare_prefill_layer(0, [index[(0, 0)]], num_layers=N_LAYERS)
    store.get_prefill(index[(0, 0)])
    spec_needed = store.get_prefill(index[(1, 0)])
    store.get_prefill(index[(1, 1)])
    assert spec_needed.transient and store.transient_free() == 2
    store.prepare_prefill_layer(1, [index[(1, 0)], index[(1, 2)]], num_layers=N_LAYERS)
    # (1, 0) promoted into layer 1's quota, (1, 1) released
    assert store.get_prefill(index[(1, 0)]) is spec_needed
    assert not spec_needed.transient
    assert (1, 1) not in store._transients and store.transient_free() == 4
    assert store.stats().cache_misses == 3


def test_speculative_candidates_prefer_frequent_non_resident(index):
    store, _ = make_store(slots=4, transient=4)
    entries = [index[(2, e)] for e in range(N_EXPERTS)]
    store.use_counts[(2, 5)] = 3
    store.use_counts[(2, 3)] = 1
    store.get(index[(2, 5)])            # resident now -> excluded
    picks = [e.expert for e in store.speculative_candidates(entries, 3)]
    assert picks == [3, 0, 1]
    assert store.speculative_candidates(entries, 0) == []


def test_slru_protects_a_reused_expert_over_a_colder_one(index, monkeypatch):
    monkeypatch.setattr(resident_store, "EVICT_POLICY", "slru")
    store, _ = make_store(slots=3)
    store.get(index[(0, 0)])
    store.get(index[(0, 0)])  # second request promotes (0, 0) to protected
    store.get(index[(0, 1)])
    store.get(index[(0, 2)])
    store.get(index[(0, 3)])  # probation LRU (0, 1) goes, not (0, 0)
    with store._lock:
        keys = set(store._items)
    assert keys == {(0, 0), (0, 2), (0, 3)}


def test_slru_demotes_instead_of_evicting_when_protection_is_full(index, monkeypatch):
    monkeypatch.setattr(resident_store, "EVICT_POLICY", "slru")
    monkeypatch.setattr(resident_store, "SLRU_PROTECTED_FRACTION", 0.5)
    store, _ = make_store(slots=4)
    for expert in range(4):
        store.get(index[(0, expert)])
        store.get(index[(0, expert)])  # every expert asks twice
    with store._lock:
        # the cap is two of four slots, so the two coldest were demoted, and
        # demotion must never drop a resident
        assert len(store._protected) == 2
        assert len(store._items) == 4


def test_slru_forgets_the_segment_of_an_evicted_expert(index, monkeypatch):
    monkeypatch.setattr(resident_store, "EVICT_POLICY", "slru")
    store, _ = make_store(slots=2)
    store.get(index[(0, 0)])
    store.get(index[(0, 0)])  # protected, and the cap is one of two slots
    store.get(index[(0, 1)])
    store.get(index[(0, 1)])  # promoting (0, 1) demotes (0, 0) to probation
    store.get(index[(0, 2)])  # which is then the victim
    with store._lock:
        assert set(store._items) == {(0, 1), (0, 2)}
        assert (0, 0) not in store._protected
        assert set(store._protected) == {(0, 1)}


# ----------------------------------------------------------------------
# Startup hotlist preload
# ----------------------------------------------------------------------


def test_preload_admits_residents_without_touching_hit_rate(index):
    store, reader = make_store(slots=8)
    wanted = [index[(0, 0)], index[(0, 1)], index[(1, 0)]]

    admitted = store.preload(wanted)

    assert admitted == 3
    assert len(store) == 3
    stats = store.stats()
    # A preload is not a miss: the session has not asked for anything yet, and
    # a preloaded session's hit rate has to stay comparable with one without.
    assert (stats.cache_hits, stats.cache_misses) == (0, 0)
    assert store.preloaded_experts == 3
    assert store.preload_bytes == 3 * EXPERT_BYTES

    # The point of the exercise: these are now hits.
    store.get(index[(0, 1)])
    assert store.stats().cache_hits == 1


def test_preload_leaves_room_for_the_prompt(index):
    """A hot set that filled the cache would evict the prefill it is meant to help."""
    store, _ = make_store(slots=8)
    everything = [index[(layer, expert)] for layer in range(N_LAYERS) for expert in range(N_EXPERTS)]

    admitted = store.preload(everything, reserve_fraction=0.25)

    assert admitted == 6  # 8 slots, a quarter held back
    assert len(store) == 6


def test_preload_respects_an_explicit_cap(index):
    store, _ = make_store(slots=8)
    everything = [index[(layer, expert)] for layer in range(N_LAYERS) for expert in range(N_EXPERTS)]

    assert store.preload(everything, max_experts=2) == 2
    assert len(store) == 2


def test_preload_skips_what_is_already_resident(index):
    store, reader = make_store(slots=8)
    store.get(index[(0, 0)])
    before = reader.reads if hasattr(reader, "reads") else None

    admitted = store.preload([index[(0, 0)], index[(0, 1)]])

    assert admitted == 1
    assert len(store) == 2
    if before is not None:
        assert reader.reads == before + 1


def test_preload_of_nothing_is_a_no_op(index):
    store, _ = make_store(slots=4)
    assert store.preload([]) == 0
    assert store.preloaded_experts == 0


# ---------------------------------------------------------------------------
# Predicted-load lifetime.
#
# Until 2026-09-19 `_sweep_inflight_locked(keep=requested)` released every
# *completed* in-flight prediction the current layer had not asked for, so with
# CACHALOT_PREDICT_AHEAD=2 an L+2 read that finished before the L+1 acquisition
# was discarded before L+2 could consume it. A prediction was punished for
# finishing early, and a poor ahead-two result could not be read as evidence
# about lead time. HANDOFF section 9.13.
# ---------------------------------------------------------------------------


def _drain_predictions(store):
    """Block until every in-flight speculative read has finished."""
    with store._lock:
        futures = [f for f, _, _ in store._inflight.values()]
    for future in futures:
        future.result()


def _inflight_keys(store):
    with store._lock:
        return set(store._inflight)


def test_a_finished_prediction_for_a_later_layer_survives_an_earlier_layer(index):
    """The defect, stated as the behaviour it should have."""
    store, _ = make_store(slots=8, transient=24)

    store.prefetch_decode([index[(1, 3)], index[(2, 7)]])
    _drain_predictions(store)
    assert _inflight_keys(store) == {(1, 3), (2, 7)}

    # Layer 1 runs and routes to expert 3 only. The layer-2 prediction has
    # finished, and it is not for this layer -- the old sweep released it here.
    store.get_many([index[(1, 3)]])

    assert (2, 7) in _inflight_keys(store), "an early-finishing L+2 load was discarded"
    assert store.predicted_expired == 0

    # And layer 2 consumes it without a second read.
    reads_before = store.stats().cache_misses
    store.get_many([index[(2, 7)]])
    assert _inflight_keys(store) == set()
    assert store.predicted_used == 2   # layer 1's own expert was predicted too
    assert store.stats().cache_misses == reads_before + 1  # counted once, read once


def test_a_mispredicted_expert_is_released_when_its_own_layer_runs(index):
    """Keeping later layers must not turn genuine waste into a leak."""
    store, _ = make_store(slots=8, transient=24)

    store.prefetch_decode([index[(1, 3)]])
    _drain_predictions(store)

    store.get_many([index[(1, 5)]])  # layer 1 routed somewhere else

    assert _inflight_keys(store) == set()
    assert store.predicted_expired == 1
    assert store.predicted_wasted_bytes == EXPERT_BYTES


def test_a_prediction_still_in_flight_is_never_swept(index):
    store, _ = make_store(slots=8, latency=0.2, transient=24)

    store.prefetch_decode([index[(2, 7)]])
    store.get_many([index[(1, 3)]])           # sweeps while the read is running

    assert (2, 7) in _inflight_keys(store)
    _drain_predictions(store)


def test_a_new_token_expires_the_previous_walk(index):
    """Layers ascend once per token, so a layer that does not advance is a new one."""
    store, _ = make_store(slots=8, transient=24)

    store.get_many([index[(3, 1)]])
    store.prefetch_decode([index[(3, 6)]])    # never consumed this token
    _drain_predictions(store)

    store.get_many([index[(0, 1)]])           # layer went backwards: next token

    assert _inflight_keys(store) == set()
    assert store.predicted_expired == 1


def test_the_same_layer_twice_is_a_new_walk(index):
    """A repeated layer is not an advance, so it must not be read as one."""
    store, _ = make_store(slots=8, transient=24)

    store.get_many([index[(2, 1)]])
    store.prefetch_decode([index[(3, 4)]])
    _drain_predictions(store)
    store.get_many([index[(2, 1)]])

    assert _inflight_keys(store) == set()
    assert store.predicted_expired == 1


def test_a_sequence_reset_expires_predictions_explicitly(index):
    """A reset need not produce a backwards layer, so the runtime says so."""
    store, _ = make_store(slots=8, transient=24)

    store.get_many([index[(0, 1)]])
    store.prefetch_decode([index[(2, 5)], index[(3, 6)]])
    _drain_predictions(store)

    assert store.expire_predictions() == 2
    assert _inflight_keys(store) == set()
    assert store.predicted_expired == 2


def test_predicting_a_resident_or_an_inflight_expert_is_a_no_op(index):
    store, _ = make_store(slots=8, transient=24)

    store.get_many([index[(0, 1)]])                       # (0,1) now resident
    assert store.prefetch_decode([index[(0, 1)]]) == 0

    assert store.prefetch_decode([index[(2, 2)]]) == 1
    assert store.prefetch_decode([index[(2, 2)]]) == 0    # already in flight
    assert store.predicted_loads == 1
    _drain_predictions(store)


def test_prefetch_stops_before_exhausting_the_transient_pool(index):
    """Backpressure is what bounds a longer prediction lifetime."""
    store, _ = make_store(slots=4, transient=resident_store.PREDICT_SLOT_RESERVE + 2)

    submitted = store.prefetch_decode(
        [index[(3, e)] for e in range(N_EXPERTS)]
    )

    assert 0 < submitted <= 2
    assert store._transient_count <= store.transient_slots - resident_store.PREDICT_SLOT_RESERVE
    _drain_predictions(store)


def test_a_prediction_consumed_by_its_own_layer_leaves_no_deadline_behind(index):
    store, _ = make_store(slots=8, transient=24)

    store.prefetch_decode([index[(1, 2)]])
    _drain_predictions(store)
    store.get_many([index[(1, 2)]])

    with store._lock:
        assert store._inflight_deadline == {}
        assert store._inflight == {}
    assert store.predicted_used == 1
    assert store.predicted_expired == 0
