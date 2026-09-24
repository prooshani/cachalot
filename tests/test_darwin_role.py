import sys

import pytest

from cachalot.darwin_role import apply_darwin_role


def test_zero_leaves_role_alone(monkeypatch):
    monkeypatch.setenv("CACHALOT_DARWIN_ROLE", "0")
    assert apply_darwin_role() is None


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin roles are macOS only")
def test_sets_and_reads_back_role(monkeypatch):
    import ctypes

    libc = ctypes.CDLL(None)
    before = libc.getpriority(6, 0)
    monkeypatch.setenv("CACHALOT_DARWIN_ROLE", "1")
    try:
        assert apply_darwin_role() == 1
    finally:
        libc.setpriority(6, 0, before)
