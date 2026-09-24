from cachalot.cli import default_fast_synch


def test_defaults_on(monkeypatch):
    monkeypatch.delenv("MLX_METAL_FAST_SYNCH", raising=False)
    assert default_fast_synch() == "1"


def test_environment_wins(monkeypatch):
    monkeypatch.setenv("MLX_METAL_FAST_SYNCH", "0")
    assert default_fast_synch() == "0"
