"""
Does the code the model writes actually compile?

The two other quality gates this project has both miss what a user notices
first. `nll_expert_precision.py` is teacher-forced and cannot see a free-running
failure at all. `repetition_quality.py` catches runaway loops but says nothing
about the artefacts that survive them -- "std std::string", "par parsing",
"jsonFile.is.open()", "for (size_t i =  i < n; ++i)". Those are what make a
generated file useless, and they are exactly what a top-1 deficit looks like.

This scores saved replies with a compiler. It extracts fenced code blocks and
runs a syntax check -- `ast.parse` for Python, `clang++ -fsyntax-only` for C and
C++ -- and reports, per block, whether the check *succeeded* and what it said.

Read it as a **paired** comparison between banks on identical prompts, not as an
absolute score. A model legitimately writes snippets that omit includes, and
those count against it here; what is meaningful is that both arms face the same
omissions, so a difference between them is the banks' doing.


## Why the primary metric is the return code, not the error count

Until 2026-09-19 this script defined a block's errors as the stderr lines
containing `": error: "`, and never looked at the compiler's exit status. Clang
reports a missing header as `": fatal error: "` and then **stops**, so a block
whose only defect was a mangled `#include` came back with an empty error list
and was scored clean. One of the four saved FP4 blocks was counted clean on
exactly that path; it contains `#include <s>` and does not compile.

The deeper problem is that the truncation biases the *comparison*. A block that
aborts on its first mangled include contributes one diagnostic; an otherwise
identical block whose includes happened to survive contributes every diagnostic
in the file. Error density then measures which arm aborted early rather than
which arm writes better code. Across the two saved arms, three of FP4's four
blocks aborted on a fatal and only one of q3g64act's three did, which is most of
the apparent difference between 8.9 and 31.6 errors per 100 lines.

So this script now reports, in order of how much they can be trusted:

1. **compile success** -- the check ran and the compiler exited zero. This is
   the only metric that means the same thing for every block.
2. **the fatal rate** -- blocks whose compilation aborted on a fatal diagnostic.
   A mangled `#include` is itself a token-level artefact, so this is a real
   quality signal, and it is reported on its own rather than folded into a
   count.
3. **error density over fully diagnosed blocks only** -- blocks that reached the
   end of the file, so their diagnostics are comparable with each other. Blocks
   that aborted on a fatal, and blocks truncated by the token cap, are excluded
   from the density and counted separately, because a file cut off mid-function
   produces errors the model did not make.

A missing compiler, a timeout or a crashed checker is an **invalid
measurement**, not a model syntax error. Those blocks are reported in their own
column and excluded from every rate, so an infrastructure failure cannot look
like a quality result.

Languages are never pooled. `ast.parse` reports at most one syntax error per
file where clang reports many, so the two densities are not the same quantity
and averaging them is meaningless.

Every fenced block is accounted for, including ones whose language tag is
missing or unrecognised, and every reply file is accounted for including ones
that contain no code at all. A model that answers a coding prompt with prose is
failing the task, and it must not vanish from the denominator.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py <dir-of-replies>
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FENCE_OPEN = re.compile(r"```([A-Za-z0-9+#_-]*)[ \t]*\n")
CPP = {"cpp", "c++", "cxx", "cc", "c"}
PYTHON = {"python", "py", "python3"}

CLANG_ARGS = ["-std=c++17", "-fsyntax-only", "-ferror-limit=0"]
CLANG_TIMEOUT_S = 120

#: Languages whose checker stops at the first error, so their error counts are
#: a presence/absence signal and cannot be compared against clang's densities.
AT_MOST_ONE_ERROR = {"python"}


@dataclass
class CheckResult:
    """What a syntax checker said about one block.

    `valid` is about the *measurement*: False means the checker could not be
    run, which is an infrastructure failure and never evidence about the model.
    `compiled` is about the code: True means the checker ran and accepted it.
    """

    valid: bool
    compiled: bool
    errors: list[str] = field(default_factory=list)
    fatals: list[str] = field(default_factory=list)
    returncode: int | None = None
    note: str = ""

    @property
    def aborted(self) -> bool:
        """The compiler gave up before reaching the end of the file."""
        return self.valid and bool(self.fatals)


def check_python(source: str) -> CheckResult:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return CheckResult(
            valid=True,
            compiled=False,
            errors=[f"line {exc.lineno}: {exc.msg}"],
            returncode=1,
        )
    except (ValueError, RecursionError) as exc:
        # A null byte or a pathologically nested expression: the source is not
        # parseable, which is a real defect, but it is not a SyntaxError.
        return CheckResult(
            valid=True, compiled=False, errors=[f"{type(exc).__name__}: {exc}"], returncode=1
        )
    return CheckResult(valid=True, compiled=True, returncode=0)


def check_cpp(source: str) -> CheckResult:
    """Syntax-check one C or C++ block.

    Success is defined as the compiler exiting zero, not as an empty error list.
    A missing header produces `": fatal error: "` and a non-zero exit with no
    `": error: "` line anywhere in stderr; scoring that as clean is the bug this
    function was rewritten to remove.
    """
    if shutil.which("clang++") is None:
        return CheckResult(valid=False, compiled=False, note="clang++ not on PATH")

    handle = tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False)
    try:
        handle.write(source)
        handle.close()
        done = subprocess.run(
            ["clang++", *CLANG_ARGS, handle.name],
            capture_output=True,
            text=True,
            timeout=CLANG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            valid=False, compiled=False, note=f"clang++ timed out after {CLANG_TIMEOUT_S}s"
        )
    except OSError as exc:
        return CheckResult(valid=False, compiled=False, note=f"clang++ could not run: {exc}")
    finally:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)

    errors, fatals = [], []
    for line in done.stderr.splitlines():
        if ": fatal error: " in line:
            fatals.append(line.split(": fatal error: ", 1)[-1])
        elif ": error: " in line:
            errors.append(line.split(": error: ", 1)[-1])

    if done.returncode != 0 and not errors and not fatals:
        # Clang refused for a reason it did not phrase as a diagnostic we match
        # -- an internal error, a signal, an unreadable temporary. Do not
        # silently score it either way.
        return CheckResult(
            valid=False,
            compiled=False,
            returncode=done.returncode,
            note=f"clang++ exited {done.returncode} with no recognised diagnostic",
        )

    return CheckResult(
        valid=True,
        compiled=done.returncode == 0,
        errors=errors,
        fatals=fatals,
        returncode=done.returncode,
    )


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
    """One row per fenced block, including ones no checker covers.

    A block whose fence carries no language tag, or a tag this script has no
    checker for, is returned with `language_known` False rather than dropped.
    It is excluded from every rate, but it stays visible: an arm that answers
    with unlabelled fences is not thereby scoring better than one that labels
    them.
    """
    blocks = []
    for tag, source, truncated in fenced_blocks(text):
        row = {
            "language": tag or "(untagged)",
            "lines": source.count("\n") + 1,
            "truncated": truncated,
        }
        if tag in PYTHON:
            result = check_python(source)
            row["language_family"] = "python"
        elif tag in CPP:
            result = check_cpp(source)
            row["language_family"] = "cpp"
        else:
            blocks.append(
                {
                    **row,
                    "language_known": False,
                    "language_family": "other",
                    "measured": False,
                    "compiled": None,
                    "errors": None,
                    "fatals": None,
                    "aborted": None,
                    "first_error": None,
                    "note": "no checker for this fence tag",
                }
            )
            continue

        blocks.append(
            {
                **row,
                "language_known": True,
                "measured": result.valid,
                "compiled": result.compiled if result.valid else None,
                "errors": len(result.errors) if result.valid else None,
                "fatals": len(result.fatals) if result.valid else None,
                "aborted": result.aborted if result.valid else None,
                "first_error": (result.fatals + result.errors or [None])[0],
                "note": result.note,
            }
        )
    return blocks


def summarise(blocks: list[dict], family: str = "") -> dict:
    """Rates for one arm and one language family.

    `density_*` covers only blocks that were measured, reached the end of the
    file and were not cut off by the token cap, because those are the only
    blocks whose diagnostic counts mean the same thing.

    For a language whose checker stops at the first error -- Python, where
    `ast.parse` raises on one `SyntaxError` and never sees the rest of the file
    -- the density is a presence/absence signal dressed as a count, and
    `density_is_comparable` says so. A saved FP4 block containing five separate
    token artefacts (`c_csv_file_path`, `utfutf-8`, `csv.Dreader`, `__name __`)
    scores exactly 1 there, which is how "Python is not the discriminator"
    survived in the handoff for a day. Compare Python arms on the parse rate.
    """
    expected = [b for b in blocks if b.get("expect_compiles", True)]
    snippets = [b for b in blocks if not b.get("expect_compiles", True)]
    measured = [b for b in expected if b["measured"]]
    invalid = [b for b in expected if b["language_known"] and not b["measured"]]
    diagnosed = [b for b in measured if not b["aborted"] and not b["truncated"]]
    density_lines = sum(b["lines"] for b in diagnosed)
    density_errors = sum(b["errors"] for b in diagnosed)
    return {
        "blocks": len(blocks),
        "measured": len(measured),
        "invalid_measurements": len(invalid),
        "compiled": sum(1 for b in measured if b["compiled"]),
        "compile_rate": (
            sum(1 for b in measured if b["compiled"]) / len(measured) if measured else None
        ),
        "aborted_on_fatal": sum(1 for b in measured if b["aborted"]),
        "truncated": sum(1 for b in blocks if b["truncated"]),
        "total_lines": sum(b["lines"] for b in blocks),
        "density_blocks": len(diagnosed),
        "density_lines": density_lines,
        "density_errors": density_errors,
        "errors_per_100_lines": (
            density_errors / density_lines * 100 if density_lines else None
        ),
        "density_is_comparable": bool(diagnosed) and family not in AT_MOST_ONE_ERROR,
        "checker_stops_at_first_error": family in AT_MOST_ONE_ERROR,
        # Blocks the corpus asked for as fragments. They legitimately omit
        # includes, so they are reported but kept out of every rate above.
        "snippet_blocks": len(snippets),
        "snippet_lines": sum(b["lines"] for b in snippets),
    }


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "   -- "
    return f"{numerator}/{denominator}"


def resolve_run(directory: str) -> tuple[Path, dict | None, str | None]:
    """Accept either a bare directory of replies or a coding_quality run.

    A run directory holds `manifest.json` and `replies/`, and the manifest says
    whether the run finished. Comparing an arm that the memory guardian killed
    against one that completed is what put a wrong number in the handoff for a
    day (HANDOFF section 7.4), so a run that says it is incomplete is refused
    here rather than quietly averaged.
    """
    path = Path(directory)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return path, None, None

    manifest = json.loads(manifest_path.read_text())
    replies = path / "replies"
    if not replies.is_dir():
        replies = path

    # Per-case rows say which tasks were asked for a complete program. A
    # snippet is asked to omit its includes, so scoring it on whether it
    # compiles alone measures the instruction rather than the model.
    rows_path = path / "rows.json"
    if rows_path.is_file():
        manifest = dict(manifest)
        manifest["_expectations"] = {
            row["file"]: row for row in json.loads(rows_path.read_text())
        }

    problem = None
    if not manifest.get("complete"):
        problem = (
            f"{path}: manifest says complete=false "
            f"({manifest.get('completed_cases')} of {manifest.get('planned_cases')} cases)"
        )
    elif manifest.get("completed_cases") != manifest.get("planned_cases"):
        problem = (
            f"{path}: {manifest.get('completed_cases')} cases completed of "
            f"{manifest.get('planned_cases')} planned"
        )
    return replies, manifest, problem


def describe_run(manifest: dict) -> str:
    git = manifest.get("git", {})
    sampling = manifest.get("sampling", {})
    return (
        f"    bank {manifest.get('bank')} | corpus {manifest.get('corpus')}"
        f" [{manifest.get('corpus_sha256_16')}]"
        f" | git {(git.get('sha') or '?')[:8]}{'+dirty' if git.get('dirty') else ''}"
        f" | temp {sampling.get('temperature')}"
        f" freq_pen {sampling.get('frequency_penalty')}"
        f" cap {sampling.get('max_new_tokens')}"
        f" | seeds {manifest.get('seeds')}"
        f" | {manifest.get('completed_cases')}/{manifest.get('planned_cases')} cases"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compile the code in saved replies and compare arms.",
    )
    ap.add_argument(
        "directories",
        nargs="+",
        help="a directory of *.txt replies, or a coding_quality.py run directory "
             "holding manifest.json and replies/",
    )
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="score a run whose manifest says it did not finish. Its arm is labelled "
             "INCOMPLETE and must not be compared against a complete one.",
    )
    args = ap.parse_args()

    by_arm: dict[tuple[str, str], list[dict]] = defaultdict(list)
    files_by_arm: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "without_code": 0})
    rows = []
    manifests: dict[str, dict] = {}
    problems: list[str] = []

    for directory in args.directories:
        replies_dir, manifest, problem = resolve_run(directory)
        if problem:
            problems.append(problem)
        paths = sorted(Path(replies_dir).glob("*.txt"))
        if not paths:
            print(f"warning: no *.txt replies in {replies_dir}", file=sys.stderr)
        for path in paths:
            bank = manifest["bank"] if manifest else path.name.split("_seed")[0]
            arm = f"{Path(directory).name} :: {bank}"
            if problem and not args.allow_incomplete:
                arm = f"{arm} [INCOMPLETE]"
            if manifest:
                manifests[arm] = manifest
            label = f"{Path(directory).name}/{path.name}"
            expectation = (manifest or {}).get("_expectations", {}).get(path.name, {})
            blocks = score(path.read_text())
            for block in blocks:
                # Default True: a bare directory of replies has no per-task
                # expectation, and the previous behaviour was to score
                # everything.
                block["expect_compiles"] = expectation.get("expect_compiles", True)
                block["task"] = expectation.get("task", "")
            files_by_arm[arm]["files"] += 1
            if not blocks:
                files_by_arm[arm]["without_code"] += 1
            for block in blocks:
                rows.append({"file": label, "bank": bank, "arm": arm, **block})
                by_arm[(arm, block["language_family"])].append(block)

    if not rows:
        raise SystemExit("no fenced blocks found in those directories")

    if problems and not args.allow_incomplete:
        for line in problems:
            print(f"warning: {line}", file=sys.stderr)
        raise SystemExit(
            "refusing to score an unfinished run: a guarded arm killed at its timeout "
            "leaves a result that looks whole. Pass --allow-incomplete to override, and "
            "do not compare the result against a complete arm."
        )

    summary: dict[str, dict] = {}
    print("Primary: did it compile. Density covers only fully diagnosed, untruncated blocks.")
    print()
    print("| arm | lang | blocks | compiled | aborted on fatal | truncated | invalid | "
          "density blocks | lines | errors | errors/100 lines | snippets |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (arm, family), blocks in sorted(by_arm.items()):
        stats = summarise(blocks, family)
        summary[f"{arm} [{family}]"] = stats
        if stats["errors_per_100_lines"] is None:
            density = "n/a"
        elif stats["checker_stops_at_first_error"]:
            density = f"{stats['errors_per_100_lines']:.1f} (floor)"
        else:
            density = f"{stats['errors_per_100_lines']:.1f}"
        print(
            f"| {arm} | {family} | {stats['blocks']} | "
            f"{_rate(stats['compiled'], stats['measured'])} | "
            f"{_rate(stats['aborted_on_fatal'], stats['measured'])} | "
            f"{stats['truncated']} | {stats['invalid_measurements']} | "
            f"{stats['density_blocks']} | {stats['density_lines']} | "
            f"{stats['density_errors']} | {density} | {stats['snippet_blocks']} |"
        )

    if manifests:
        print()
        print("runs:")
        for arm, manifest in sorted(manifests.items()):
            print(f"  {arm}")
            print(describe_run(manifest))

    print()
    for arm, counts in sorted(files_by_arm.items()):
        summary.setdefault(f"{arm} [files]", {}).update(counts)
        note = (
            f"  {arm}: {counts['files']} replies, "
            f"{counts['without_code']} with no fenced code at all"
        )
        print(note)

    thin = [k for k, s in summary.items() if s.get("density_blocks", 1) < 3]
    if thin:
        print()
        print("warning: fewer than three fully diagnosed blocks in "
              + ", ".join(sorted(thin))
              + " -- the density there is one or two files, not a rate.")
    if any(s.get("checker_stops_at_first_error") for s in summary.values()):
        print()
        print("note: a density marked (floor) comes from a checker that stops at the first "
              "error, so it counts broken files rather than defects. Compare those arms on "
              "the compile column.")

    print("\nfirst diagnostic in each block:")
    for row in rows:
        if row["first_error"]:
            kind = "FATAL" if row["aborted"] else "error"
            count = row["errors"] if row["errors"] is not None else 0
            print(f"  {row['file']} [{row['language']}] {kind} {count:3d} diagnostics | "
                  f"{row['first_error'][:80]}")
        elif not row["measured"]:
            print(f"  {row['file']} [{row['language']}] NOT MEASURED | {row['note']}")

    if args.out:
        out = Path(args.out)
    else:
        from _common import RESULTS_DIR  # noqa: PLC0415  -- keeps MLX out of a CPU-only run

        out = RESULTS_DIR / "code_validity.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
