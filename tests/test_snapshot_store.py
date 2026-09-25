"""HANDOFF section 15.12: the disk snapshot store keeps blocks by last use and loads the rest on demand."""

import dataclasses

from cachalot.model import snapshot_store
from cachalot.model.prefix_cache import PrefixCache
from test_system_boundary_snapshots import _random_snapshot, _same


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


def _block(first: int, n: int = 12):
    """A system block of its own: `first` makes its tokens unlike every other block's."""
    snap = _random_snapshot(n, logits=False)
    return dataclasses.replace(snap, tokens=(first,) + tuple(range(1, n)), position=n, image_spans=())


def _store(tmp_path, clock, **kw):
    return snapshot_store.SnapshotStore(tmp_path, "id", clock=clock, **kw)


def _on_disk(tmp_path):
    return sorted(snapshot_store.read_tokens(p)[0] for p in tmp_path.glob("prefix-*.safetensors"))


def test_read_tokens_matches_the_file(tmp_path):
    snap = _block(7, 20)
    path = snapshot_store.save(snap, tmp_path, "id")
    assert snapshot_store.read_tokens(path) == snap.tokens


def test_a_block_in_use_outlives_newer_blocks_nobody_reuses(tmp_path):
    # the main agent's block, then a batch of subagents' blocks, each used only by its own conversation
    clock = Clock()
    store = _store(tmp_path, clock, keep=4)
    main = _block(1)
    store.persist(main)
    for k in range(2, 5):
        store.persist(_block(k))
    # the main agent's next turn reuses its block
    store.on_find(main.tokens + (99,), [main])
    for k in range(5, 8):
        store.persist(_block(k))
    assert _on_disk(tmp_path) == [1, 5, 6, 7]


def test_a_restart_preloads_the_most_recently_used_and_fetches_the_rest(tmp_path):
    clock = Clock()
    store = _store(tmp_path, clock)
    blocks = [_block(k) for k in range(1, 6)]
    for b in blocks:
        store.persist(b)
    store.on_find(blocks[0].tokens, [blocks[0]])  # block 1 is now the most recently used

    store = _store(tmp_path, clock, preload=2)
    cache = PrefixCache()
    loaded = store.load_all()
    assert [s.tokens[0] for s in loaded] == [5, 1]  # oldest first
    for s in loaded:
        cache.add(s, boundary=True)
    cache.persist, cache.on_find, cache.fetch = store.persist, store.on_find, store.fetch
    assert len(store.tokens) == 5

    request = blocks[2].tokens + (50, 51)
    hit = cache.find(request)
    assert hit is not None and hit.tokens == blocks[2].tokens
    _same(hit, blocks[2])
    assert store.fetched == 1
    # now in memory: the next request finds it without the disk
    assert cache.find(request).tokens == blocks[2].tokens
    assert store.fetched == 1


def test_fetch_never_replaces_a_longer_match_in_memory(tmp_path):
    clock = Clock()
    store = _store(tmp_path, clock)
    block = _block(3)
    store.persist(block)
    cache = PrefixCache()
    cache.fetch = store.fetch
    longer = dataclasses.replace(block, tokens=block.tokens + (40, 41), position=len(block.tokens) + 2)
    cache.add(longer)
    assert cache.find(longer.tokens + (42,)).tokens == longer.tokens
    assert store.fetched == 0


def test_a_file_of_another_runtime_is_indexed_but_never_served(tmp_path):
    clock = Clock()
    block = _block(4)
    snapshot_store.save(block, tmp_path, "other-bank")
    store = _store(tmp_path, clock, preload=0)
    store.load_all()
    assert store.fetch(block.tokens + (1,), 0) is None
    assert store.tokens == {}


def test_a_failing_store_does_not_fail_the_request():
    cache = PrefixCache()

    def boom(*_):
        raise OSError("disk gone")

    cache.on_find = boom
    cache.fetch = boom
    snap = _block(5)
    cache.add(snap, boundary=True)
    assert cache.find(snap.tokens + (1,)).tokens == snap.tokens
