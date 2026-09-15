from dataclasses import replace

import cachalot.config as cfg
from cachalot.config import GiB, RuntimeConfig, resolve_expert_budget, resolve_wired_limit


def _fake_memory(total_gb: float):
    total = int(total_gb * 1000**3)
    return lambda: (total, int(total * 0.81))


def test_auto_budget_scales_with_memory(monkeypatch):
    monkeypatch.setattr(cfg, "available_memory", lambda: None)
    c = RuntimeConfig()
    expected = {64: (8, 20), 96: (40, 50), 128: (70, 80), 512: (268, 269.5)}
    for total_gb, (lo, hi) in expected.items():
        monkeypatch.setattr(cfg, "device_memory", _fake_memory(total_gb))
        budget = resolve_expert_budget(c) / GiB
        assert lo <= budget <= hi, (total_gb, budget)
        wired = resolve_wired_limit(c, int(budget * GiB))
        assert wired <= _fake_memory(total_gb)()[1]


def test_explicit_budget_and_wired_respected(monkeypatch):
    monkeypatch.setattr(cfg, "available_memory", lambda: None)
    monkeypatch.setattr(cfg, "device_memory", _fake_memory(96))
    c = replace(RuntimeConfig(), expert_cache_budget_bytes=40 * GiB, mlx_wired_limit_bytes=56 * GiB)
    assert resolve_expert_budget(c) == 40 * GiB
    assert resolve_wired_limit(c, 40 * GiB) == 56 * GiB


def test_all_experts_cap(monkeypatch):
    monkeypatch.setattr(cfg, "available_memory", lambda: None)
    monkeypatch.setattr(cfg, "device_memory", _fake_memory(1024))
    assert resolve_expert_budget(RuntimeConfig()) == cfg.ALL_EXPERTS_BYTES


def test_env_overrides():
    c = RuntimeConfig().with_env_overrides(
        {"CACHALOT_EXPERT_CACHE_BUDGET_GIB": "48", "CACHALOT_MAX_SEQ_LEN": "8192", "CACHALOT_MODEL_PATH": "/x"}
    )
    assert c.expert_cache_budget_bytes == 48 * GiB
    assert c.max_seq_len == 8192
    assert c.model_path == "/x"


def test_auto_budget_capped_by_available_memory(monkeypatch):
    monkeypatch.setattr(cfg, "device_memory", lambda: (96 * GiB, 72 * GiB))
    monkeypatch.setattr(cfg, "available_memory", lambda: 70 * GiB)
    c = RuntimeConfig(expert_cache_budget_bytes=0)
    # 70 - 12 trunk - 2 cache - 12 headroom = 44 GiB, below the 50 GiB formula
    assert resolve_expert_budget(c) == 44 * GiB
    monkeypatch.setattr(cfg, "available_memory", lambda: 90 * GiB)
    assert resolve_expert_budget(c) == 50 * GiB
    monkeypatch.setattr(cfg, "available_memory", lambda: None)
    assert resolve_expert_budget(c) == 50 * GiB
