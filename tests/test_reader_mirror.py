"""Mirror striping on multi-piece experts (HANDOFF 18.3): same bytes as the primary alone, whichever mode."""

import os
import shutil

import pytest

from cachalot.storage.index import ExpertEntry, TensorRange
from cachalot.storage.reader import ExpertReader

# nine pieces like a MiniMax expert: three large weights, six small scales/biases, not contiguous
SIZES = {"w1.weight": 16384, "w1.scales": 2048, "w1.biases": 2048,
         "w2.weight": 16384, "w2.scales": 2048, "w2.biases": 2048,
         "w3.weight": 16384, "w3.scales": 2048, "w3.biases": 2048}


def _checkpoint(root):
    root.mkdir()
    shard = root / "model-00001-of-00001.safetensors"
    data = os.urandom(sum(SIZES.values()) + 4096 * (len(SIZES) + 1))
    shard.write_bytes(data)
    tensors, pos = [], 4096
    for name, size in SIZES.items():
        tensors.append(TensorRange(f"layers.0.experts.{name}", shard, pos, pos + size))
        pos += size + 4096  # a gap: every piece is its own read range
    return data, ExpertEntry(0, 0, tuple(tensors))


def _read(reader, entry):
    views = {k: bytearray(n) for k, n in SIZES.items()}
    assert reader.read_expert_into(entry, views) == sum(SIZES.values())
    return {k: bytes(v) for k, v in views.items()}


@pytest.mark.parametrize("mode", ["pieces", "split"])
def test_mirror_reads_are_the_primary_bytes(tmp_path, monkeypatch, mode):
    _, entry = _checkpoint(tmp_path / "primary")
    shutil.copytree(tmp_path / "primary", tmp_path / "mirror")
    monkeypatch.setenv("CACHALOT_MIRROR_MODE", mode)
    plain = _read(ExpertReader(mirror_path=None), entry)
    mirrored = ExpertReader(mirror_path=tmp_path / "mirror", mirror_fraction=0.10)
    assert _read(mirrored, entry) == plain
    views = {k: bytearray(n) for k, n in SIZES.items()}
    from cachalot.storage.index import merge_contiguous_ranges

    jobs = mirrored._piece_jobs(merge_contiguous_ranges(entry), views)
    assert any(j[5] for j in jobs), "some bytes must come from the mirror"
    if mode == "pieces":
        # whole small pieces, smallest first, within 10 % of the expert's bytes: 3 of the 2 KiB pieces
        assert sum(j[3] for j in jobs if j[5]) == 3 * 2048


def test_a_mirror_without_the_shard_falls_back_to_the_primary(tmp_path):
    _, entry = _checkpoint(tmp_path / "primary")
    (tmp_path / "empty").mkdir()
    plain = _read(ExpertReader(mirror_path=None), entry)
    reader = ExpertReader(mirror_path=tmp_path / "empty", mirror_fraction=0.10)
    assert _read(reader, entry) == plain
    assert reader.mirror_path is None


def test_a_failing_mirror_read_is_retried_on_the_primary(tmp_path, monkeypatch):
    _, entry = _checkpoint(tmp_path / "primary")
    shutil.copytree(tmp_path / "primary", tmp_path / "mirror")
    plain = _read(ExpertReader(mirror_path=None), entry)
    reader = ExpertReader(mirror_path=tmp_path / "mirror", mirror_fraction=0.10)
    mirror_fd = reader._fd(tmp_path / "mirror" / "model-00001-of-00001.safetensors")
    real = os.preadv

    def preadv(fd, bufs, off):
        if fd == mirror_fd:
            raise OSError(5, "Input/output error")
        return real(fd, bufs, off)

    monkeypatch.setattr(os, "preadv", preadv)
    assert _read(reader, entry) == plain
    assert reader.mirror_path is None
