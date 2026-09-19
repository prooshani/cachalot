"""Pins the include/import integrity check. CPU only, no model.

The point of this check is that it is mechanical: a line is well formed by the
grammar or it is not. These tests pin the grammar, so the check cannot quietly
drift into a heuristic fitted to one model's failures.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from include_integrity import classify_include, scan_text  # noqa: E402


def test_well_formed_lines_are_not_counted():
    text = (
        '```cpp\n'
        '#include <iostream>\n'
        '#include <unordered_map>\n'
        '#include "local_header.h"\n'
        '#include <sys/stat.h>\n'
        'import os\n'
        'import os.path\n'
        'from pathlib import Path\n'
        'from collections import defaultdict, Counter\n'
        'import numpy as np\n'
        '```\n'
    )

    got = scan_text(text)

    assert got["include_bad"] == 0 and got["include_total"] == 4
    assert got["import_bad"] == 0 and got["import_total"] == 5


def test_the_observed_failures_are_all_caught():
    """Every one of these is a real line from a saved FP4 or 3-bit reply."""
    text = (
        '```cpp\n'
        '#include escaping>\n'
        '#include stdio.h>\n'
        '#include unordered_map>\n'
        '#includequeue>\n'
        '#include>\n'
        '#include <ioman double>\n'
        '#include <condition_vqueue>\n'
        '```\n'
    )

    got = scan_text(text)

    assert got["include_total"] == 7
    # The last one is a plausible header name and only the checkpoint knows it
    # is wrong, so six of seven is the honest count for a grammar check.
    assert got["include_bad"] == 6, got["examples"]


def test_malformations_are_classified_the_way_the_prediction_states():
    assert classify_include("#include escaping>") == "missing opening delimiter"
    assert classify_include("#includequeue>") == "missing opening delimiter"
    assert classify_include("#include <iostream") == "missing closing delimiter"
    assert classify_include("#include iostream") == "no delimiters at all"
    assert classify_include("#include <ioman double>") == "whitespace inside the header name"


def test_a_missing_delimiter_is_not_confused_with_a_wrong_name():
    """The prediction turns on this distinction, so it must not be fuzzy.

    A wrong header name is ordinary blur; a dropped delimiter is not.
    """
    assert scan_text("```cpp\n#include <condition_vqueue>\n```\n")["include_bad"] == 0
    assert scan_text("```cpp\n#include condition_variable>\n```\n")["include_bad"] == 1


def test_broken_imports_are_caught_as_the_control():
    text = "```python\nimport  os,\nfrom  import json\nimport\n```\n"

    got = scan_text(text)

    assert got["import_total"] == 3
    assert got["import_bad"] == 3


def test_prose_outside_a_fence_is_not_an_import():
    """A sentence beginning "from" is not a broken import line."""
    text = (
        "The data comes from the upstream service, and we import it nightly.\n"
        "from there it is written to disk.\n\n"
        "```python\nimport os\n```\n"
    )

    got = scan_text(text)

    assert got["import_total"] == 1
    assert got["import_bad"] == 0
