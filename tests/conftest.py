from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store_true",
        default=False,
        help="run tests that load the real DeepSeek checkpoint",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs the real checkpoint (slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--model"):
        return
    skip = pytest.mark.skip(reason="needs --model")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
