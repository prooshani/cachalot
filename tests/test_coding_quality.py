"""Pins the coding corpus and the per-reply accounting. CPU only, no model.

The corpus this project judged three expert banks on was one three-turn
conversation, and a task that produced no code or answered in the wrong
language simply left the denominator. These tests pin that it cannot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

pytest.importorskip("mlx.core", reason="coding_quality imports the runtime")

from coding_quality import _language_of, classify  # noqa: E402

CPP_TASK = {"id": "t", "language": "cpp", "kind": "program", "expect_compiles": True}
PY_TASK = {"id": "t", "language": "python", "kind": "program", "expect_compiles": True}


def test_the_corpus_is_well_formed():
    corpus = json.loads((BENCHMARKS / "coding_tasks.json").read_text())
    tasks = corpus["tasks"]

    assert len(tasks) >= 20, "four scoreable blocks per arm is what got us here"

    ids = [t["id"] for t in tasks]
    assert len(set(ids)) == len(ids)

    for task in tasks:
        assert task["language"] in {"cpp", "python"}
        assert task["kind"] in {"program", "snippet"}
        assert isinstance(task["expect_compiles"], bool)
        assert task["prompt"].strip()

    languages = {t["language"] for t in tasks}
    kinds = {t["kind"] for t in tasks}
    assert languages == {"cpp", "python"}, "one language cannot be the whole corpus"
    assert kinds == {"program", "snippet"}, "snippets and programs are scored differently"


def test_language_tags_map_to_families():
    assert _language_of("cpp") == "cpp"
    assert _language_of("c++") == "cpp"
    assert _language_of("py") == "python"
    assert _language_of("rust") == "other"
    assert _language_of("") == "other"


def test_a_reply_with_no_code_is_a_failed_task_not_a_missing_one():
    shape = classify("I would rather not write that program.", CPP_TASK)

    assert shape["blocks"] == 0
    assert shape["has_code"] is False
    assert shape["answered_as_asked"] is False


def test_the_wrong_language_does_not_count_as_answering():
    shape = classify("```python\nprint(1)\n```\n", CPP_TASK)

    assert shape["has_code"] is True
    assert shape["blocks_in_requested_language"] == 0
    assert shape["wrong_language_blocks"] == 1
    assert shape["answered_as_asked"] is False


def test_an_untagged_fence_is_counted_rather_than_dropped():
    shape = classify("```\nint main() { return 0; }\n```\n", CPP_TASK)

    assert shape["blocks"] == 1
    assert shape["untagged_blocks"] == 1
    assert shape["wrong_language_blocks"] == 0   # untagged is not wrong-language
    assert shape["answered_as_asked"] is False


def test_a_correct_reply_is_recognised():
    shape = classify(
        "Here you go.\n\n```cpp\n#include <cstdio>\nint main() { return 0; }\n```\n",
        CPP_TASK,
    )

    assert shape["answered_as_asked"] is True
    assert shape["blocks_in_requested_language"] == 1
    assert shape["any_truncated_block"] is False


def test_a_reply_cut_off_mid_block_is_flagged():
    shape = classify("```python\ndef f():\n    return 1", PY_TASK)

    assert shape["answered_as_asked"] is True
    assert shape["any_truncated_block"] is True


def test_several_blocks_in_one_reply_are_all_counted():
    text = (
        "First attempt:\n```cpp\nint main(){}\n```\n"
        "Sorry, corrected:\n```cpp\nint main(){return 0;}\n```\n"
        "And the build line:\n```bash\nclang++ a.cpp\n```\n"
    )

    shape = classify(text, CPP_TASK)

    assert shape["blocks"] == 3
    assert shape["blocks_in_requested_language"] == 2
    assert shape["wrong_language_blocks"] == 1
