"""Proves that the slow-path environment is the path actually taken.

Three times this project has measured nothing: a compiler gate that never read
the compiler's exit status (HANDOFF 7.4.1), predicted loads with no lifetime
(9.13), and every timing number taken on a bank that was not mounted (7.4.6).
The 2026-09-20 bisection is worth a week of machine time and it rests entirely
on a set of environment variables, so the variables get a test.

Nearly every switch is read at *module import time*, which has a consequence
that is easy to get wrong: setting os.environ inside a test that has already
imported the module changes nothing, and monkeypatch.setenv cannot help either.
The only honest check is a fresh interpreter, which is what this does.

CPU only. No model is loaded and no GPU work is dispatched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx.core", reason="the switches live in modules that import mlx")

ROOT = Path(__file__).resolve().parents[1]

#: The slow path of HANDOFF section 2 / next-session prompt v13 Job 2: every
#: optimization off, nothing predicted, nothing fused, nothing striped.
SLOW_PATH = {
    "CACHALOT_ASYNC_MOE": "0",
    "CACHALOT_FUSED_MOE": "0",
    "CACHALOT_FUSED_DECODE": "0",
    "CACHALOT_FUSED_FP8": "0",
    "CACHALOT_WOA_GEMM": "0",
    "CACHALOT_BF16_HEAD": "0",
    "CACHALOT_PREDICT_TOPK": "0",
    "CACHALOT_PREDICT_WORKERS": "1",
    "CACHALOT_PREFETCH_DEPTH": "1",
    "CACHALOT_SPECULATIVE_PREFILL": "0",
    "CACHALOT_PREFILL_SGMM": "0",
    "CACHALOT_PREFILL_BF16_GEMM": "0",
    "CACHALOT_PREFILL_BATCHED_ATTN": "0",
    "CACHALOT_PREFILL_BATCHED_HC": "0",
    "CACHALOT_PREFILL_BATCHED_ENGRAM": "0",
    "CACHALOT_ENGRAM_PREFETCH": "0",
    "CACHALOT_RDAHEAD": "0",
    "CACHALOT_EVICT": "lru",
}

#: module attribute -> the value the slow path must produce. Each one is a
#: constant assigned at import from os.environ.
IMPORT_TIME = {
    "cachalot.model.moe_layer_metal:ASYNC_MOE": False,
    "cachalot.model.moe_layer_metal:PREDICT_TOPK": 0,
    "cachalot.model.decode_fused_metal:FUSED_DECODE": False,
    "cachalot.model.fp8_linear_metal:FUSED_FP8": False,
    "cachalot.model.moe_prefill_grouped:PREFILL_SGMM": False,
    "cachalot.model.moe_prefill_grouped:SPECULATIVE_PREFILL": False,
    "cachalot.model.attention_prefill_batched:WOA_GEMM": False,
    "cachalot.model.moe_prefill_batched:PREFILL_BF16_GEMM": False,
    "cachalot.cache.resident_store:EVICT_POLICY": "lru",
    "cachalot.model.moe_layer_metal:PRELAUNCH_SHARED": 0,
}

_READER = """
import json, importlib, sys
out = {}
for spec in json.loads(sys.argv[1]):
    module, _, attr = spec.partition(":")
    out[spec] = getattr(importlib.import_module(module), attr)
print(json.dumps(out))
"""


def _read_constants(env_overrides: dict[str, str]) -> dict:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("CACHALOT_"):
            del env[key]
    env.update(env_overrides)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, "-c", _READER, json.dumps(sorted(IMPORT_TIME))],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_slow_path_environment_is_honoured():
    got = _read_constants(SLOW_PATH)
    wrong = {k: (got[k], v) for k, v in IMPORT_TIME.items() if got[k] != v}
    assert not wrong, (
        "a slow-path variable was set and ignored; the bisection would have "
        f"measured the default instead: {wrong}"
    )


def test_the_defaults_are_the_configuration_that_produced_the_corruption():
    """Every switch above is on by default, so every one of them is a suspect.
    If a default ever changes, the bisection order in the next-session prompt is
    stale and this test is where that gets noticed."""
    got = _read_constants({})
    assert got["cachalot.model.moe_layer_metal:ASYNC_MOE"] is True
    assert got["cachalot.model.moe_layer_metal:PREDICT_TOPK"] == 6
    assert got["cachalot.model.decode_fused_metal:FUSED_DECODE"] is True
    assert got["cachalot.model.fp8_linear_metal:FUSED_FP8"] is True
    assert got["cachalot.model.moe_prefill_grouped:PREFILL_SGMM"] is True
    assert got["cachalot.model.moe_prefill_grouped:SPECULATIVE_PREFILL"] is True
    assert got["cachalot.model.attention_prefill_batched:WOA_GEMM"] is True
    assert got["cachalot.model.moe_prefill_batched:PREFILL_BF16_GEMM"] is True
    assert got["cachalot.cache.resident_store:EVICT_POLICY"] == "lru"


def test_predict_topk_zero_actually_disables_prediction():
    """moe_layer_metal gates the whole predictor on PREDICT_TOPK > 0, so 0 is
    not 'predict nothing well', it is 'do not predict'. The slow path depends
    on that being true rather than on it looking true."""
    source = (ROOT / "src/cachalot/model/moe_layer_metal.py").read_text()
    assert "if PREDICT_TOPK > 0 else None" in source


def test_mirror_striping_is_off_only_when_the_path_is_unset():
    """CACHALOT_MIRROR_FRACTION=0 does not disable striping: reader.py reads the
    fraction only after the path is set. v12 of the next-session prompt got this
    wrong and would have measured a striped run as an unstriped one."""
    source = (ROOT / "src/cachalot/storage/reader.py").read_text()
    path_at = source.index("CACHALOT_MIRROR_PATH")
    fraction_at = source.index("CACHALOT_MIRROR_FRACTION")
    assert path_at < fraction_at


def test_the_bf16_head_switch_still_guards_the_head_kernel():
    """CACHALOT_BF16_HEAD=0 has to take the head off the hand-written Metal
    GEMV and back onto chunked MLX arithmetic. That kernel is the last
    arithmetic before a token is chosen and it has never been parity-tested."""
    source = (ROOT / "src/cachalot/model/model_boundary_mlx.py").read_text()
    guard = source.index('os.environ.get("CACHALOT_BF16_HEAD", "1") != "0"')
    call = source.index("return bf16_gemv_f32(")
    assert guard < call, "the switch must gate the kernel, not follow it"

def test_the_shared_expert_prelaunch_is_off_by_default_and_precedes_the_routing_sync():
    """CACHALOT_PRELAUNCH_SHARED submits the shared expert before the layer
    blocks on mx.eval(route.indices, ...), so the GPU has work to do during a
    round trip that costs about 0.20 ms whatever it evaluates
    (benchmarks/micro_eval_floor.py). The whole mechanism is source order:
    written after that eval the flag would measure as a null, which is the
    shape of mistake HANDOFF section 11 keeps recording. It also has to be off
    by default until a live arm says otherwise."""
    source = (ROOT / "src/cachalot/model/moe_layer_metal.py").read_text()
    submit = source.index("mx.async_eval(prelaunched_shared)")
    sync = source.index("# Routing is needed on the CPU to address the resident store.")
    assert submit < sync, "the shared expert must be submitted before the routing eval"
    assert _read_constants({})["cachalot.model.moe_layer_metal:PRELAUNCH_SHARED"] == 0
