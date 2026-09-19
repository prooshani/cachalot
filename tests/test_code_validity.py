"""Pins the code-validity gate against the false success that hid in it.

Until 2026-09-19 `check_cpp` defined a block's errors as the stderr lines
containing `": error: "` and never read the compiler's exit status. Clang
reports a missing header as `": fatal error: "` and stops, so a block whose
only defect was a mangled `#include` produced an empty error list and was
scored clean. One of the four saved FP4 blocks was counted clean on exactly
that path.

These tests are CPU-only and load no model.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from code_validity import (  # noqa: E402
    check_cpp,
    check_python,
    describe_run,
    fenced_blocks,
    resolve_run,
    score,
    summarise,
)

needs_clang = pytest.mark.skipif(
    shutil.which("clang++") is None, reason="clang++ is not on PATH"
)


# ----------------------------------------------------------------- C and C++


@needs_clang
def test_missing_header_is_not_a_clean_block():
    """The exact shape that scored clean before: a fatal, no ordinary error."""
    result = check_cpp("#include <cachalot_no_such_header_123.h>\nint main() { return 0; }\n")

    assert result.valid is True
    assert result.compiled is False
    assert result.returncode != 0
    assert result.errors == []          # nothing matches ": error: "
    assert len(result.fatals) == 1      # and that is precisely the trap
    assert result.aborted is True


@needs_clang
def test_a_clean_block_compiles():
    result = check_cpp("int main() { return 0; }\n")

    assert (result.valid, result.compiled, result.returncode) == (True, True, 0)
    assert result.errors == [] and result.fatals == []
    assert result.aborted is False


@needs_clang
def test_ordinary_syntax_errors_are_counted_and_not_fatal():
    result = check_cpp("int main() { int x = ; return y; }\n")

    assert result.valid is True
    assert result.compiled is False
    assert result.errors, "an ordinary syntax error must still be reported"
    assert result.fatals == []
    assert result.aborted is False


def test_a_missing_compiler_is_an_invalid_measurement(monkeypatch):
    """Infrastructure failure must never look like a model syntax error."""
    monkeypatch.setattr("code_validity.shutil.which", lambda _name: None)

    result = check_cpp("int main() { return 0; }\n")

    assert result.valid is False
    assert result.compiled is False
    assert result.errors == [] and result.fatals == []
    assert "clang++" in result.note


def test_a_timeout_is_an_invalid_measurement(monkeypatch):
    import subprocess

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="clang++", timeout=1)

    monkeypatch.setattr("code_validity.shutil.which", lambda _name: "/usr/bin/clang++")
    monkeypatch.setattr("code_validity.subprocess.run", _timeout)

    result = check_cpp("int main() { return 0; }\n")

    assert result.valid is False
    assert "timed out" in result.note


# ---------------------------------------------------------------------- Python


def test_python_parses_and_fails_as_expected():
    assert check_python("x = 1\n").compiled is True

    bad = check_python("def f(:\n")
    assert bad.valid is True and bad.compiled is False and len(bad.errors) == 1


# ------------------------------------------------------------------ accounting


def test_untagged_and_unknown_fences_stay_in_the_denominator():
    text = "```\nwho knows\n```\n\n```rust\nfn main() {}\n```\n"

    blocks = score(text)

    assert len(blocks) == 2
    assert [b["language_known"] for b in blocks] == [False, False]
    assert [b["measured"] for b in blocks] == [False, False]
    assert blocks[0]["language"] == "(untagged)"


def test_an_unterminated_block_is_kept_and_marked_truncated():
    language, source, truncated = fenced_blocks("```cpp\nint main() {\n")[0]

    assert (language, truncated) == ("cpp", True)
    assert "int main" in source


def test_density_excludes_aborted_truncated_and_invalid_blocks():
    """The stratification that makes two arms comparable at all.

    A block that aborted on a fatal contributes one diagnostic where a fully
    diagnosed block contributes every diagnostic in the file, and a block cut
    off by the token cap produces errors the model did not make. Neither can
    sit in the same density as a block that reached the end.
    """
    blocks = [
        # fully diagnosed: the only one the density may use
        {"measured": True, "language_known": True, "compiled": False, "errors": 10,
         "aborted": False, "truncated": False, "lines": 100},
        # aborted on a mangled include: one diagnostic, not ten
        {"measured": True, "language_known": True, "compiled": False, "errors": 1,
         "aborted": True, "truncated": False, "lines": 200},
        # cut off at the token cap
        {"measured": True, "language_known": True, "compiled": False, "errors": 5,
         "aborted": False, "truncated": True, "lines": 50},
        # checker could not run
        {"measured": False, "language_known": True, "compiled": None, "errors": None,
         "aborted": None, "truncated": False, "lines": 30},
    ]

    stats = summarise(blocks)

    assert stats["blocks"] == 4
    assert stats["measured"] == 3
    assert stats["invalid_measurements"] == 1
    assert stats["compiled"] == 0 and stats["compile_rate"] == 0.0
    assert stats["aborted_on_fatal"] == 1
    assert stats["truncated"] == 1
    assert stats["density_blocks"] == 1
    assert stats["density_lines"] == 100
    assert stats["density_errors"] == 10
    assert stats["errors_per_100_lines"] == pytest.approx(10.0)


def test_compile_rate_is_none_rather_than_zero_when_nothing_was_measured():
    stats = summarise(
        [{"measured": False, "language_known": True, "compiled": None, "errors": None,
          "aborted": None, "truncated": False, "lines": 5}]
    )

    assert stats["compile_rate"] is None
    assert stats["errors_per_100_lines"] is None


def test_python_density_is_marked_as_a_floor():
    """`ast.parse` stops at the first error, so its count is not a defect count."""
    blocks = [
        {"measured": True, "language_known": True, "compiled": False, "errors": 1,
         "aborted": False, "truncated": False, "lines": 50},
    ]

    python = summarise(blocks, "python")
    cpp = summarise(blocks, "cpp")

    assert python["checker_stops_at_first_error"] is True
    assert python["density_is_comparable"] is False
    assert cpp["checker_stops_at_first_error"] is False
    assert cpp["density_is_comparable"] is True


# ------------------------------------------------------- run-directory handling


def _write_run(tmp_path, *, complete, planned=2, completed=2):
    import json

    run = tmp_path / "run"
    (run / "replies").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "bank": "fake-bank", "corpus": "t", "corpus_sha256_16": "abc123",
        "planned_cases": planned, "completed_cases": completed, "complete": complete,
        "git": {"sha": "deadbeefcafe", "dirty": False},
        "sampling": {"temperature": 0.6, "frequency_penalty": 0.2, "max_new_tokens": 2000},
        "seeds": [1, 2],
    }))
    (run / "replies" / "a_seed1.txt").write_text("```cpp\nint main() { return 0; }\n```\n")
    return run


def test_a_bare_directory_of_replies_still_works(tmp_path):
    plain = tmp_path / "replies"
    plain.mkdir()
    (plain / "bank_seed1.txt").write_text("```cpp\nint main(){}\n```\n")

    replies, manifest, problem = resolve_run(str(plain))

    assert replies == plain
    assert manifest is None and problem is None


def test_a_complete_run_reports_no_problem(tmp_path):
    run = _write_run(tmp_path, complete=True)

    replies, manifest, problem = resolve_run(str(run))

    assert replies == run / "replies"
    assert manifest["bank"] == "fake-bank"
    assert problem is None
    assert "fake-bank" in describe_run(manifest)
    assert "2/2 cases" in describe_run(manifest)


def test_an_unfinished_run_is_reported_as_a_problem(tmp_path):
    run = _write_run(tmp_path, complete=False, completed=1)

    _replies, _manifest, problem = resolve_run(str(run))

    assert problem is not None and "complete=false" in problem


def test_a_run_short_of_its_planned_cases_is_a_problem_even_if_marked_complete(tmp_path):
    """The marker and the counts must agree; either disagreeing is a refusal."""
    run = _write_run(tmp_path, complete=True, planned=4, completed=2)

    _replies, _manifest, problem = resolve_run(str(run))

    assert problem is not None and "2 cases completed of 4" in problem


def test_a_snippet_is_kept_out_of_the_compile_rate_and_the_density():
    """The corpus asks some tasks for a fragment; scoring it as a program
    measures the instruction rather than the model."""
    blocks = [
        {"measured": True, "language_known": True, "compiled": False, "errors": 4,
         "aborted": False, "truncated": False, "lines": 100, "expect_compiles": True},
        {"measured": True, "language_known": True, "compiled": False, "errors": 18,
         "aborted": False, "truncated": False, "lines": 20, "expect_compiles": False},
    ]

    stats = summarise(blocks, "cpp")

    assert stats["blocks"] == 2
    assert stats["measured"] == 1            # only the program is rated
    assert stats["snippet_blocks"] == 1
    assert stats["snippet_lines"] == 20
    assert stats["density_errors"] == 4      # the snippet's 18 stay out
    assert stats["errors_per_100_lines"] == pytest.approx(4.0)


def test_a_bare_directory_scores_every_block_as_before():
    """No manifest means no expectations, and the old behaviour stands."""
    blocks = [
        {"measured": True, "language_known": True, "compiled": True, "errors": 0,
         "aborted": False, "truncated": False, "lines": 10},
    ]

    stats = summarise(blocks, "cpp")

    assert stats["measured"] == 1 and stats["snippet_blocks"] == 0


def test_run_expectations_are_loaded_from_rows(tmp_path):
    import json

    run = _write_run(tmp_path, complete=True, planned=1, completed=1)
    (run / "rows.json").write_text(json.dumps([
        {"file": "a_seed1.txt", "task": "snip", "expect_compiles": False},
    ]))

    _replies, manifest, _problem = resolve_run(str(run))

    assert manifest["_expectations"]["a_seed1.txt"]["expect_compiles"] is False
