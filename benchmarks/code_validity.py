"""
Does the code the model writes actually parse?

The two quality gates this project has both miss what a user notices first.
`nll_expert_precision.py` is teacher-forced and cannot see a free-running
failure at all. `repetition_quality.py` catches runaway loops but says nothing
about the artefacts that survive them -- "std std::string", "par parsing",
"jsonFile.is.open()", "for (size_t i =  i < n; ++i)". Those are what make a
generated file useless, and they are exactly what a top-1 deficit looks like.

This scores saved replies with a compiler. It extracts fenced code blocks and
runs a syntax check -- `ast.parse` for Python, `clang++ -fsyntax-only` for C and
C++ -- and reports errors per block.

Read it as a **paired** comparison between banks on identical prompts, not as an
absolute score. A model legitimately writes snippets that omit includes, and
those count as errors here; what is meaningful is that both arms face the same
omissions, so a difference between them is the banks' doing.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py <dir-of-replies>
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS_DIR  # noqa: E402

FENCE_OPEN = re.compile(r"```([A-Za-z0-9+#_-]*)[ \t]*\n")
CPP = {"cpp", "c++", "cxx", "cc", "c"}
PYTHON = {"python", "py", "python3"}


def check_python(source: str) -> list[str]:
    try:
        ast.parse(source)
        return []
    except SyntaxError as exc:
        return [f"line {exc.lineno}: {exc.msg}"]


def check_cpp(source: str) -> list[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as handle:
        handle.write(source)
        path = handle.name
    try:
        done = subprocess.run(
            ["clang++", "-std=c++17", "-fsyntax-only", "-ferror-limit=0", path],
            capture_output=True, text=True, timeout=60,
        )
        return [
            line.split(": error: ", 1)[-1]
            for line in done.stderr.splitlines()
            if ": error: " in line
        ]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"checker unavailable: {exc}"]
    finally:
        Path(path).unlink(missing_ok=True)


def fenced_blocks(text: str) -> list[tuple[str, str, bool]]:
    """(language, source, truncated) for every fenced block.

    A reply cut off at the token cap leaves its last block unterminated, which a
    closing-fence-required regex drops silently -- and that is exactly the block
    a long code generation is judged on. Those are kept and marked truncated.
    """
    out = []
    pos = 0
    while True:
        opened = FENCE_OPEN.search(text, pos)
        if opened is None:
            return out
        language = opened.group(1).lower()
        body_start = opened.end()
        closed = text.find("```", body_start)
        if closed == -1:
            out.append((language, text[body_start:], True))
            return out
        out.append((language, text[body_start:closed], False))
        pos = closed + 3


def score(text: str) -> list[dict]:
    blocks = []
    for tag, source, truncated in fenced_blocks(text):
        if tag in PYTHON:
            errors = check_python(source)
        elif tag in CPP:
            errors = check_cpp(source)
        else:
            continue
        blocks.append(
            {
                "language": tag,
                "lines": source.count("\n") + 1,
                "errors": len(errors),
                "truncated": truncated,
                "first_error": errors[0] if errors else None,
            }
        )
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directories", nargs="+")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    by_bank: dict[str, list[dict]] = defaultdict(list)
    rows = []

    for directory in args.directories:
        for path in sorted(Path(directory).glob("*.txt")):
            bank = path.name.split("_seed")[0]
            label = f"{Path(directory).name}/{path.name}"
            for block in score(path.read_text()):
                rows.append({"file": label, "bank": bank, **block})
                by_bank[f"{Path(directory).name} :: {bank}"].append(block)

    if not rows:
        raise SystemExit("no fenced Python or C++ blocks found in those directories")

    print("| arm | blocks | clean | total errors | errors per 100 lines |")
    print("|---|---:|---:|---:|---:|")
    summary = {}
    for arm, blocks in sorted(by_bank.items()):
        clean = sum(1 for b in blocks if b["errors"] == 0)
        errors = sum(b["errors"] for b in blocks)
        lines = sum(b["lines"] for b in blocks)
        summary[arm] = {
            "blocks": len(blocks), "clean": clean, "errors": errors, "lines": lines,
            "errors_per_100_lines": errors / lines * 100 if lines else 0.0,
        }
        print(f"| {arm} | {len(blocks)} | {clean} | {errors} | "
              f"{errors / lines * 100 if lines else 0:.1f} |")

    print("\nfirst error in each block:")
    for row in rows:
        if row["first_error"]:
            print(f"  {row['file']} [{row['language']}] {row['errors']:3d} errors | "
                  f"{row['first_error'][:80]}")

    out = Path(args.out) if args.out else RESULTS_DIR / "code_validity.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
