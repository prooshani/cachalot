"""GLM prefix snapshots on disk (HANDOFF 17.1): the cache objects survive a round trip bit for bit."""

import mlx.core as mx
import pytest

from cachalot.glm.model import Snapshot, _NoProjectedCache
from cachalot.glm.snapshots import (
    GlmSnapshotStore,
    decode_cache,
    encode_cache,
    read_snapshot,
    write_snapshot,
)
from cachalot.third_party.mlx_vlm.models.cache import ArraysCache, CacheList, KVCache, PoolingCache


def _cache():
    mx.random.seed(0)
    kda = ArraysCache(size=2)
    kda.cache = [mx.random.normal((1, 3, 8)), mx.random.normal((1, 2, 4, 4))]
    kv1, kv2 = KVCache(), KVCache()
    kv1.update_and_fetch(mx.random.normal((1, 1, 5, 6)), mx.random.normal((1, 1, 5, 6)))
    kv2.update_and_fetch(mx.random.normal((1, 1, 5, 2)), mx.random.normal((1, 1, 5, 2)))
    pool = PoolingCache(4)
    pool.pooled = mx.random.normal((1, 1, 2, 3))
    pool.remainder = 1
    mla = CacheList(kv1, kv2, pool, _NoProjectedCache())
    return [kda, mla]


def _same(a, b):
    assert type(a) is type(b)
    if isinstance(a, mx.array):
        assert a.dtype == b.dtype and a.shape == b.shape and bool(mx.array_equal(a, b))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    elif hasattr(a, "__dict__"):
        assert set(a.__dict__) == set(b.__dict__)
        for k in a.__dict__:
            if isinstance(a, KVCache) and k in ("keys", "values") and a.keys is not None:
                _same(a.__dict__[k][..., : a.offset, :], b.__dict__[k])
            else:
                _same(a.__dict__[k], b.__dict__[k])
    else:
        assert a == b


def test_encode_decode_round_trip():
    cache = _cache()
    arrays, skeleton = encode_cache(cache)
    _same(cache, decode_cache(arrays, skeleton))


def test_kvcache_is_saved_to_its_offset_and_keeps_growing():
    cache = _cache()
    kv = cache[1].caches[0]
    assert kv.keys.shape[2] == 256  # step-allocated
    arrays, skeleton = encode_cache(cache)
    back = decode_cache(arrays, skeleton)[1].caches[0]
    assert back.keys.shape[2] == 5
    new = mx.random.normal((1, 1, 3, 6))
    k1, _ = kv.update_and_fetch(new, new)
    k2, _ = back.update_and_fetch(new, new)
    assert bool(mx.array_equal(k1, k2))


def test_file_round_trip_and_identity(tmp_path):
    snap = Snapshot((1, 2, 3), _cache(), None)
    path = tmp_path / "s.safetensors"
    write_snapshot(snap, path, "id-a")
    back = read_snapshot(path, "id-a")
    assert back.tokens == (1, 2, 3) and back.logits is None
    _same(snap.cache, back.cache)
    assert read_snapshot(path, "id-b") is None


def test_store_persists_and_fetches(tmp_path):
    store = GlmSnapshotStore(tmp_path, "id")
    store.persist(Snapshot((5, 6, 7, 8), _cache(), None))
    again = GlmSnapshotStore(tmp_path, "id")
    assert [s.tokens for s in again.load_all()] == [(5, 6, 7, 8)]
    got = again.fetch((5, 6, 7, 8, 9), 0)
    assert got is not None and got.tokens == (5, 6, 7, 8)


def test_refuses_foreign_classes():
    class Foreign:
        pass

    with pytest.raises(TypeError):
        encode_cache([Foreign()])
