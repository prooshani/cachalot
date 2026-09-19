"""
Generate the coding corpus against one expert bank, with a run manifest.

Why this exists rather than another `repetition_quality.py --conversation`
file. The coding evidence this project judged three expert banks on was a
single three-turn CSV-to-JSON conversation. It produced four scoreable C++
blocks per arm, two of them cut off at the token cap, and on one arm all three
were. When the compiler gate was repaired on 2026-09-19 the shortage became
visible: stratified onto blocks that could be compared at all, FP4 had one and
the candidate bank had none. The gate was wrong, and the corpus behind it was
too thin to have settled anything either way. HANDOFF section 7.4.1.

What is different here:

**Independent tasks.** Every task is its own single-turn conversation with
`rt.reset()` between, so a collapse on one cannot poison the next and every
bank enters every task from a byte-identical context. The multi-turn
conversation test is a different measurement and `repetition_quality.py` keeps
it; a controlled per-task comparison is this one.

**Twenty of them, across four shapes** -- complete programs, snippets, edits to
supplied code, and bug fixes -- in two languages, rather than one file-format
conversion repeated. `benchmarks/coding_tasks.json` carries them and says for
each one which language it must be answered in and whether it is a complete
program that has to compile on its own.

**Every task is accounted for.** A task that produces no code, answers in the
wrong language, or stops at the token cap is recorded as that. The planned and
completed counts both go in the manifest, and the `complete` marker is written
last, so a run the memory guardian killed is explicitly invalid rather than a
short result file that looks whole. That failure cost this project a wrong
number for a day (HANDOFF section 7.4).

**The run records what produced it.** Git SHA and dirty state, the bank and its
format, bytes per expert, budget, mirror settings, every sampling parameter,
the corpus hash, and the seeds. A result whose configuration has to be inferred
from a launch command is not evidence.

This script only generates. `benchmarks/code_validity.py` scores, and the two
are separate so a corpus can be re-scored without generating it again.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh \
      --budget-gib 24 --max-seconds 10800 --tag cq-fp4 -- env \
      CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts \
      CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
      benchmarks/coding_quality.py --seeds 2 --max-new-tokens 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR  # noqa: E402
from cachalot.model.generation import (  # noqa: E402
    SamplingParams,
    load_official_encoding,
    stream_tokens,
)
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from code_validity import CPP, PYTHON, fenced_blocks  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "coding_tasks.json"


def _git_state() -> dict:
    root = Path(__file__).resolve().parents[1]

    def _run(*args: str) -> str:
        try:
            done = subprocess.run(
                args, cwd=root, capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return done.stdout.strip() if done.returncode == 0 else ""

    return {
        "sha": _run("git", "rev-parse", "HEAD"),
        "dirty": bool(_run("git", "status", "--porcelain")),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
    }


def _language_of(tag: str) -> str:
    if tag in PYTHON:
        return "python"
    if tag in CPP:
        return "cpp"
    return "other"


def classify(text: str, task: dict) -> dict:
    """What the reply contains, before anything is compiled.

    Kept separate from `code_validity.py`'s scoring: this answers "did the model
    do the task as asked", which is upstream of "does the code build" and must
    not be conflated with it. A reply with no fenced block is a failed task, not
    a missing measurement.
    """
    blocks = fenced_blocks(text)
    wanted = task["language"]
    matching = [b for b in blocks if _language_of(b[0]) == wanted]
    return {
        "blocks": len(blocks),
        "blocks_in_requested_language": len(matching),
        "untagged_blocks": sum(1 for tag, _, _ in blocks if not tag),
        "wrong_language_blocks": sum(
            1 for tag, _, _ in blocks if tag and _language_of(tag) != wanted
        ),
        "has_code": bool(blocks),
        "answered_as_asked": len(matching) >= 1,
        "any_truncated_block": any(truncated for _, _, truncated in blocks),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--first-seed", type=int, default=20260919)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=2000,
                    help="high enough that a complete program is not cut off: a "
                         "truncated block cannot be scored against an untruncated one")
    ap.add_argument("--frequency-penalty", type=float, default=0.2)
    ap.add_argument("--penalty-window", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--only", default="", help="comma-separated task ids, for a smoke run")
    ap.add_argument("--out-dir", default="", help="defaults to results/coding/<stamp>_<bank>")
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    tasks = corpus["tasks"]
    if args.only:
        wanted = {t.strip() for t in args.only.split(",") if t.strip()}
        unknown = wanted - {t["id"] for t in tasks}
        if unknown:
            raise SystemExit(f"unknown task ids: {sorted(unknown)}")
        tasks = [t for t in tasks if t["id"] in wanted]

    corpus_hash = hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()[:16]
    planned = len(tasks) * args.seeds

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=args.max_seq_len) as rt:
        bank = Path(rt.expert_bank_path).name
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "coding" / f"{stamp}_{bank}"
        replies_dir = out_dir / "replies"
        replies_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "corpus": corpus.get("corpus", "unknown"),
            "corpus_sha256_16": corpus_hash,
            "bank": bank,
            "bank_path": str(rt.expert_bank_path),
            "bank_format": str(getattr(rt, "expert_format", None)),
            "expert_bytes": getattr(rt, "expert_bytes", None),
            "model_path": str(MODEL_PATH),
            "expert_budget_bytes": rt.expert_cache_budget_bytes,
            "max_seq_len": args.max_seq_len,
            "sampling": {
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "frequency_penalty": args.frequency_penalty,
                "penalty_window": args.penalty_window,
            },
            "seeds": [args.first_seed + i for i in range(args.seeds)],
            "planned_cases": planned,
            "completed_cases": 0,
            "git": _git_state(),
            "env": {
                k: v for k, v in sorted(os.environ.items()) if k.startswith("CACHALOT_")
            },
            "python": platform.python_version(),
            "started_utc": datetime.now(UTC).isoformat(),
            "complete": False,
        }
        # Written now so an interrupted run leaves a manifest saying it is
        # incomplete, rather than leaving nothing and looking like it never ran.
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        print(f"runtime ready (bank {bank}, budget "
              f"{rt.expert_cache_budget_bytes / 2**30:.1f} GiB)", flush=True)
        print(f"corpus {manifest['corpus']} [{corpus_hash}], {len(tasks)} tasks "
              f"x {args.seeds} seeds = {planned} cases -> {out_dir}", flush=True)

        encoding = load_official_encoding(MODEL_PATH)
        rows = []

        for seed_index in range(args.seeds):
            seed = args.first_seed + seed_index
            params = SamplingParams(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=seed,
                frequency_penalty=args.frequency_penalty,
                penalty_window=args.penalty_window,
            )
            for task in tasks:
                rt.reset()
                messages = [{"role": "user", "content": task["prompt"]}]
                prompt = encoding.encode_messages(
                    messages, thinking_mode="chat", reasoning_effort=None
                )
                ids = list(rt.tokenizer.encode(prompt))

                produced: list[int] = []
                finish = None
                t0 = perf_counter()
                for event in stream_tokens(rt, ids, params, use_prefix_cache=False):
                    if event.kind == "token":
                        produced.append(event.token)
                    elif event.kind == "done":
                        finish = event.finish_reason
                seconds = perf_counter() - t0

                text = rt.tokenizer.decode(produced)
                name = f"{task['id']}_seed{seed}.txt"
                (replies_dir / name).write_text(text)

                shape = classify(text, task)
                row = {
                    "task": task["id"],
                    "language": task["language"],
                    "kind": task["kind"],
                    "expect_compiles": task["expect_compiles"],
                    "seed": seed,
                    "file": name,
                    "prompt_tokens": len(ids),
                    "tokens": len(produced),
                    "finish": finish,
                    # A reply that stopped at the cap has a block the model did
                    # not choose to end, and it cannot be compared with one that
                    # did. This is the flag that says so.
                    "hit_token_cap": finish == "length",
                    "seconds": seconds,
                    "tok_per_s": len(produced) / seconds if seconds else 0.0,
                    **shape,
                }
                rows.append(row)
                (out_dir / "rows.json").write_text(json.dumps(rows, indent=2))

                print(
                    f"  {task['id']:28s} seed {seed} | {len(produced):5d} tok | "
                    f"{finish or '?':10s} | blocks {shape['blocks']:2d} "
                    f"({shape['blocks_in_requested_language']} {task['language']}) | "
                    f"{'CAP' if row['hit_token_cap'] else '   '} | "
                    f"{row['tok_per_s']:5.2f} tok/s",
                    flush=True,
                )

        manifest["completed_cases"] = len(rows)
        manifest["finished_utc"] = datetime.now(UTC).isoformat()
        manifest["complete"] = len(rows) == planned
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    answered = sum(1 for r in rows if r["answered_as_asked"])
    capped = sum(1 for r in rows if r["hit_token_cap"])
    print()
    print(f"| {bank} | cases | answered as asked | hit the token cap | no code |")
    print("|---|---:|---:|---:|---:|")
    print(f"| planned {planned} | {len(rows)} | {answered} | {capped} | "
          f"{sum(1 for r in rows if not r['has_code'])} |")
    if not manifest["complete"]:
        print("\nINCOMPLETE RUN: do not compare this against another arm.")
    if capped:
        print(f"\n{capped} replies stopped at --max-new-tokens. Their last block is "
              "truncated and cannot be scored against an untruncated one; raise the "
              "cap or exclude them, and say which.")
    print(f"\nwrote {out_dir}")
    print("score it with:")
    print(f"  PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py "
          f"{replies_dir}")


if __name__ == "__main__":
    main()
