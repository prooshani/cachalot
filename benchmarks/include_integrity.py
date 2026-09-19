"""
Are the `#include` and `import` lines well formed?

**Pre-registered 2026-09-19, part-way through the first FP4 corpus run.** At the
time of writing, 10 of the 40 cases had been generated and read -- all C++, all
from seed 20260919. The remaining 30, which include every Python task and the
whole of the second seed, had not. This file is committed before they exist so
that they are a clean test set for the prediction below, rather than a pattern
fitted on its own data. HANDOFF sections 7.4.4 and 8.3.

## Why a separate check at all

`code_validity.py` reports whether a block compiles, which is the right primary
metric and has two blind spots this does not share:

- **Truncation.** A block cut off at the token cap cannot be compiled fairly, so
  it leaves the density entirely. Its include lines are still perfectly
  readable.
- **Fatal abort.** Clang stops at the first bad include, so a file with one
  mangled include and thirty other defects scores one diagnostic. This counts
  every include line independently.

It is also mechanical rather than heuristic. An `#include` line is well formed
or it is not, by the grammar; there is no phrase list and no threshold fitted to
this model. That is the difference between this and a screen, and it is why a
retry-loop detector built out of apology phrases was deliberately *not* added
(section 7.4.3).

## The prediction being registered

On the 30 unseen cases:

1. **C++ `#include` lines will be malformed at a materially higher rate than
   Python `import` lines** in the same run. Observed so far on the seen cases:
   26 of 70 includes malformed, 0 of 0 imports -- no Python task had run yet,
   so the control is genuinely untested.
2. **The dominant malformation will be a missing `<`**, not a misspelled header
   name. Observed so far: the great majority.
3. On the saved 2026-09-18 arms this separates the banks -- FP4 5 of 27 against
   the 3-bit bank's 28 of 43. **That comparison was fitted after the fact and is
   not part of this registration.** It is repeated here only so a later reader
   knows it was already known.

**What would falsify it.** If Python `import` lines are malformed at a similar
rate, the fault is not about this construction and the "dropped `<`" framing is
wrong. If the malformations are mostly wrong header *names* rather than missing
delimiters, it is ordinary blur and section 7.4.4's argument for a systematic
fault weakens considerably.

This answers "how often is the output wrong here", never "why". The why is
`benchmarks/token_rank_probe.py`, which reads the logits directly.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/include_integrity.py \
      benchmarks/results/coding/<run>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code_validity import fenced_blocks  # noqa: E402

#: A well-formed C preprocessor include: a bracketed or quoted header path and
#: nothing else on the line. Deliberately strict -- a trailing comment is rare
#: in generated code and a stray token is exactly what is being counted.
INCLUDE_OK = re.compile(r'^\s*#\s*include\s*(?:<[A-Za-z0-9_./+-]+>|"[A-Za-z0-9_./+-]+")\s*$')
INCLUDE_ANY = re.compile(r'^\s*#\s*include')

#: A well-formed Python import. The control: same replies, same prompts, same
#: sampling, a different construction.
IMPORT_OK = re.compile(
    r'^\s*(?:from\s+[A-Za-z_][\w.]*\s+)?import\s+'
    r'[A-Za-z_][\w.]*(?:\s+as\s+[A-Za-z_]\w*)?'
    r'(?:\s*,\s*[A-Za-z_][\w.]*(?:\s+as\s+[A-Za-z_]\w*)?)*\s*$'
)
#: Anything that is trying to be an import. Deliberately loose: `from  import
#: json` matches neither a well-formed import nor a well-formed `from ...
#: import`, and a line that matches nothing at all would leave the denominator
#: -- which is the exact failure that made the compiler gate wrong.
IMPORT_ANY = re.compile(r'^\s*(?:from|import)\b')


def classify_include(line: str) -> str:
    """Why this include line is malformed. The categories are the prediction."""
    body = line.split("include", 1)[1]
    has_open = "<" in body or '"' in body
    has_close = ">" in body or body.count('"') >= 2
    if not has_open and has_close:
        return "missing opening delimiter"
    if has_open and not has_close:
        return "missing closing delimiter"
    if not has_open and not has_close:
        return "no delimiters at all"
    if re.search(r"[<\"][^>\"]*\s", body):
        return "whitespace inside the header name"
    return "other"


def scan_text(text: str) -> dict:
    """Scan the fenced code blocks of one reply.

    Only inside the fences: a prose sentence beginning "from" is not a broken
    import, and counting it as one would be the same mistake in the other
    direction.
    """
    inc_total = inc_bad = imp_total = imp_bad = 0
    kinds: Counter = Counter()
    examples: list[str] = []
    lines = [
        line
        for _tag, source, _truncated in fenced_blocks(text)
        for line in source.splitlines()
    ]
    for line in lines:
        if INCLUDE_ANY.match(line):
            inc_total += 1
            if not INCLUDE_OK.match(line):
                inc_bad += 1
                kinds[classify_include(line)] += 1
                if len(examples) < 40:
                    examples.append(line.strip()[:70])
        elif IMPORT_ANY.match(line):
            imp_total += 1
            if not IMPORT_OK.match(line):
                imp_bad += 1
                if len(examples) < 40:
                    examples.append(line.strip()[:70])
    return {
        "include_total": inc_total, "include_bad": inc_bad,
        "import_total": imp_total, "import_bad": imp_bad,
        "kinds": dict(kinds), "examples": examples,
    }


def scan_dir(directory: str) -> dict:
    path = Path(directory)
    replies = path / "replies" if (path / "replies").is_dir() else path
    agg = {"include_total": 0, "include_bad": 0, "import_total": 0, "import_bad": 0}
    kinds: Counter = Counter()
    examples: list[str] = []
    files = 0
    for reply in sorted(replies.glob("*.txt")):
        got = scan_text(reply.read_text())
        files += 1
        for key in agg:
            agg[key] += got[key]
        kinds.update(got["kinds"])
        examples.extend(got["examples"][:3])
    agg["files"] = files
    agg["kinds"] = dict(kinds.most_common())
    agg["examples"] = examples[:25]
    manifest = path / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        agg["bank"] = data.get("bank")
        agg["complete"] = data.get("complete")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directories", nargs="+")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    print("| arm | files | #include bad | rate | import bad | rate |")
    print("|---|---:|---:|---:|---:|---:|")
    results = {}
    for directory in args.directories:
        got = scan_dir(directory)
        results[directory] = got
        name = got.get("bank") or Path(directory).name
        if got.get("complete") is False:
            name += " [INCOMPLETE]"
        inc_rate = got["include_bad"] / got["include_total"] if got["include_total"] else None
        imp_rate = got["import_bad"] / got["import_total"] if got["import_total"] else None
        print(f"| {name} | {got['files']} | "
              f"{got['include_bad']}/{got['include_total']} | "
              f"{f'{inc_rate:.0%}' if inc_rate is not None else 'n/a'} | "
              f"{got['import_bad']}/{got['import_total']} | "
              f"{f'{imp_rate:.0%}' if imp_rate is not None else 'n/a'} |")

    for directory, got in results.items():
        if not got["kinds"]:
            continue
        print(f"\n{Path(directory).name}: how the include lines fail")
        for kind, count in got["kinds"].items():
            print(f"  {count:4d}  {kind}")
        print("  examples:")
        for line in got["examples"][: args.examples]:
            print(f"    {line}")

    print("\nRegistered prediction (see the module docstring, committed before these")
    print("cases existed): includes fail materially more often than imports, and the")
    print("dominant failure is a missing opening delimiter. A similar import rate, or")
    print("failures that are mostly wrong header names, falsifies it.")


if __name__ == "__main__":
    main()
