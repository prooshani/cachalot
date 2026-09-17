
from cachalot.storage.index import merge_contiguous_ranges
from cachalot.storage.store import ExpertPayload
from cachalot.storage.tensors import extract_expert_tensors
from fakes import EXPERT_BYTES, SCALE_BYTES, WEIGHT_BYTES, FakeReader, make_entry


def test_contiguous_ranges_merge_into_two_reads():
    entry = make_entry(2, 7)
    ranges = merge_contiguous_ranges(entry)
    assert len(ranges) == 1  # fake layout is fully contiguous
    assert ranges[0].size == EXPERT_BYTES


def test_split_layout_yields_separate_ranges():
    entry = make_entry(0, 0)
    # Move the weights far away to emulate the real shard layout.
    moved = tuple(
        t if t.name.endswith("scale") else type(t)(t.name, t.shard, t.start + 10_000_000, t.end + 10_000_000)
        for t in entry.tensors
    )
    entry = type(entry)(layer=0, expert=0, tensors=tuple(sorted(moved, key=lambda t: t.start)))
    assert len(merge_contiguous_ranges(entry)) == 2


def test_extract_tensors_slices_payload():
    entry = make_entry(1, 1)
    payload = ExpertPayload(chunks=FakeReader().read_expert(entry))
    tensors = extract_expert_tensors(entry, payload)
    assert set(tensors) == {"w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale"}
    assert tensors["w1.scale"].size == SCALE_BYTES
    assert tensors["w2.weight"].size == WEIGHT_BYTES
    assert payload.size == EXPERT_BYTES


def test_reader_split_buffers_at_byte_offset():
    from cachalot.storage.reader import ExpertReader

    a, b, c = bytearray(10), bytearray(6), bytearray(4)
    head, tail = ExpertReader._split_buffers([memoryview(a), memoryview(b), memoryview(c)], 13)
    assert [m.nbytes for m in head] == [10, 3]
    assert [m.nbytes for m in tail] == [3, 4]
    head, tail = ExpertReader._split_buffers([memoryview(a), memoryview(b)], 10)
    assert [m.nbytes for m in head] == [10] and [m.nbytes for m in tail] == [6]
    for m in tail:
        m[:] = b"\x07" * m.nbytes
    assert bytes(b) == b"\x07" * 6


def test_reader_readahead_defaults_follow_the_page_cache(monkeypatch):
    """F_RDAHEAD is off when the page cache is bypassed and on when it is used."""
    from cachalot.storage.reader import ExpertReader

    monkeypatch.delenv("CACHALOT_RDAHEAD", raising=False)

    assert ExpertReader(bypass_page_cache=True).readahead is False
    assert ExpertReader(bypass_page_cache=False).readahead is True


def test_reader_readahead_can_be_disabled_with_the_page_cache_on(monkeypatch):
    """CACHALOT_RDAHEAD=0 stops the kernel reading ahead past a scattered piece."""
    from cachalot.storage.reader import ExpertReader

    monkeypatch.setenv("CACHALOT_RDAHEAD", "0")
    assert ExpertReader(bypass_page_cache=False).readahead is False

    monkeypatch.setenv("CACHALOT_RDAHEAD", "1")
    assert ExpertReader(bypass_page_cache=True).readahead is True

    # an explicit argument still wins over the environment
    monkeypatch.setenv("CACHALOT_RDAHEAD", "0")
    assert ExpertReader(bypass_page_cache=False, readahead=True).readahead is True
