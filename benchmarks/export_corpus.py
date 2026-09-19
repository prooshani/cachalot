"""
Export the coding corpus so another runtime can be scored against Cachalot.

The point of the export is that the two arms must be **paired**: same prompts,
same sampling, same scoring code. A reference run whose prompts or settings
drifted proves nothing, and the one thing this project keeps relearning is that
an unmatched comparison reads as a result (HANDOFF sections 7.4, 7.4.1).

Writes a self-contained pack:

    prompts.jsonl      one task per line: id, language, kind, expect_compiles, prompt
    settings.json      the sampling Cachalot used, to be matched exactly
    RUN.md             what the operator has to do and what must come back
    run_reference.py   runs the harness-free arm against an OpenAI-compatible endpoint
    score.sh           the command that scores the returned replies here

The operator runs the 20 prompts through whatever runtime is under test, saves
each reply verbatim, and returns a directory. Nothing else is needed: the
replies are scored by `code_validity.py` and `include_integrity.py`, the same
code that scored the Cachalot arm.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/export_corpus.py --out /tmp/corpus-pack
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CORPUS = Path(__file__).resolve().parent / "coding_tasks.json"

#: What the Cachalot arm used. The reference arm has to match these or the
#: comparison is not paired. penalty_window has no OpenAI-API equivalent; see
#: RUN.md for what to do about that.
SETTINGS = {
    "temperature": 0.6,
    "max_new_tokens": 2000,
    "frequency_penalty": 0.2,
    "penalty_window": 128,
    "top_p": 1.0,
    "seeds": [20260919, 20260920],
    "system_prompt": None,
    "turns_per_conversation": 1,
    "prefix_cache": False,
}

RUN_MD = """\
# Coding-corpus reference run

Twenty independent single-turn coding tasks. The Cachalot arm has already been
run on them; this pack exists so another runtime can be scored **against** it
with the same prompts, the same sampling and the same scoring code.

## Why this comparison is worth running

Cachalot streams the routed experts of DeepSeek V4.1 Flash from SSD. Its expert
bank is the 40 expert-bearing shards of the official checkpoint, **copied byte
for byte and verified identical**. The checkpoint's own `quantization_config`
reads `{{quant_method: fp8, expert_dtype: fp4}}`, so the official weights ship
FP4 routed experts.

A hosted endpoint serving the official weights is therefore running **the same
expert bytes**. If its output is clean and Cachalot's is not, the fault is in
the runtime, the kernels or the serving path — not in the quantization, and not
in the model. That is the question this pack is built to answer.

If the endpoint's weights are *not* the official ones (a redistilled, requantized
or differently-served variant), say so in `notes`: the comparison is still
useful but it no longer isolates the runtime.

## What Cachalot measured, so you know what you are comparing against

Forty cases, two seeds, one shot each, no retries and no tooling:

| | result |
|---|---|
| C++ programs that compile | **0 of 42** |
| Python blocks that parse | 5 of 26 |
| `#include` lines malformed | **49 of 154 (32 %)**, 78 % of them a dropped `<` |
| `import` lines malformed | 4 of 57 (7 %) |
| replies that loop by self-correcting | 2 of 40 |

## Two runs, and the order matters

**Run A, the diagnostic: no harness at all.** Bare API calls, no system prompt,
no tools, no retries on content. This is the run that isolates the runtime, and
it only works if nothing sits between the prompt and the model. A default
harness profile injects its own system prompt and settings, so a clean result
under one would be indistinguishable from a clean runtime -- which is the whole
question.

`run_reference.py` in this pack does run A. Standard library only, no
dependencies, tested on Python 3.9:

```bash
export OPENROUTER_API_KEY=sk-or-...
python3 <repo>/benchmarks/run_reference.py \
  --pack <this-pack> \
  --out ~/cachalot-reference-openrouter \
  --model deepseek/deepseek-v4.1-flash \
  --provider <provider-slug>
```

**Pin the provider.** OpenRouter load-balances, and different providers serve
different quantizations of the same model name. Unpinned, forty requests can
land on three stacks and the arm measures nothing. Check the model's page on
OpenRouter for the provider list; the script records which one served each case
and exits non-zero if more than one appears.

Smoke it on two tasks first: add `--only cpp-csv-to-json,py-csv-to-json`.

**Run B, the product question: with the harness.** Same prompts, through Hermes
or whatever agent loop is being evaluated, with its tools and retries switched
on. This answers "can a harness make this model usable", which is a different
and also useful question. Keep its output in a *separate* directory and say in
`notes.json` what the harness was allowed to do.

Do A before B. If A is clean the fault is in Cachalot's runtime, and there is a
bug to fix rather than a workaround to build.

## Doing it by hand instead

If the script cannot be used:

1. Read `prompts.jsonl`. One task per line, with `id`, `language`, `kind` and
   `prompt`.
2. Send **each prompt as its own single-turn conversation.** No system prompt,
   no shared history, no prior context. A task must not see another task's
   output; that is what makes the cases independent.
3. Match `settings.json` as closely as the endpoint allows:
   - `temperature` 0.6, `top_p` 1.0, `max_tokens` 2000.
   - `frequency_penalty` 0.2. Cachalot applies it over a **128-token sliding
     window**; the OpenAI API applies it over the whole context and has no
     window parameter. Use 0.2 and record in `notes` that the window differs.
     This is the one setting that cannot be matched exactly.
   - Run **both seeds** if the endpoint supports seeding. If it does not, run
     each prompt twice anyway and name them `seed1` / `seed2` — two samples per
     task is the point, reproducibility is a bonus.
4. Save each reply **verbatim**, including any prose outside the code fences and
   including replies that look broken or empty. A reply that is thrown away
   silently changes the denominator, which is exactly the failure that made
   Cachalot's own gate wrong for two sessions.

## What must come back

A directory of `.txt` files, one per case, named:

    <task-id>_seed<seed>.txt

for example `cpp-thread-pool_seed20260919.txt`. The `_seed` separator matters —
the scorer splits on it to group an arm.

Alongside them, a `notes.json` (free form, but please include):

    {{
      "endpoint": "openrouter/deepseek/deepseek-v4.1-flash",
      "harness": "hermes <version>",
      "weights_are_official_checkpoint": true,
      "settings_actually_used": {{...}},
      "unsupported_settings": ["penalty_window"],
      "cases_planned": 40,
      "cases_completed": 40,
      "finish_reasons": {{"stop": 38, "length": 2}},
      "retries_or_tooling": "none"
    }}

**`cases_planned` and `cases_completed` are not optional.** If a case failed,
timed out or was skipped, it must appear in the counts. A short run that looks
complete is the single most expensive mistake this project has made.

## Scoring

Run `score.sh <directory>` here. It runs the same two checks that scored the
Cachalot arm:

- `code_validity.py` — did it compile, by the compiler's exit status. Reports
  the fatal-abort rate and truncation separately, and the error density only
  over blocks that reached the end of the file.
- `include_integrity.py` — are the `#include` and `import` lines well formed.
  The prediction it tests was **pre-registered** before the Cachalot cases that
  confirmed it existed, and is stated in that file's docstring.

## What the outcome will mean

- **Reference clean, Cachalot corrupt** — the runtime is at fault. The dropped
  `<` becomes a bug to find, and the standing "quality over speed" trade is
  built on a defect rather than on FP4.
- **Both corrupt at a similar rate** — it is the model or the FP4 expert format
  itself, Cachalot is faithful, and the honest conclusion is that this
  checkpoint does not one-shot compiling C++ at these settings.
- **Reference corrupt but much less so** — a runtime contribution on top of a
  model tendency. Worth splitting further with the greedy run below.

## One extra case worth running, and it is cheap

Run `cpp-csv-to-json` and `cpp-thread-pool` at **temperature 0** on both sides.
Cachalot's own reply retried eighteen times and produced `#include>` on all
eighteen; if sampling were the cause some attempt would have landed. Greedy
decoding removes sampling from the question entirely. Name those
`<task-id>_seed0.txt` and keep them in a separate directory.
"""

SCORE_SH = """\
#!/bin/bash
# Score a returned reference run against the Cachalot arm.
#
# This script lives in the pack, not in the repository, so it cannot find the
# repository by walking up from its own location. CACHALOT_REPO overrides the
# default below.
set -euo pipefail

REPO=${{CACHALOT_REPO:-{repo}}}
PY=${{CACHALOT_PYTHON:-{python}}}

REF=${{1:?usage: score.sh <directory-of-returned-replies>}}
REF=$(cd "$REF" && pwd)

[ -d "$REPO" ] || {{ echo "repository not found at $REPO; set CACHALOT_REPO" >&2; exit 1; }}
cd "$REPO"

MINE=$(ls -dt benchmarks/results/coding/*/ 2>/dev/null | head -1 || true)

echo "=== reference: $REF ==="
echo "=== cachalot:  ${{MINE:-<none found, scoring the reference alone>}} ==="
echo

PYTHONPATH=src "$PY" benchmarks/code_validity.py "$REF" ${{MINE:+"$MINE"}}
echo
PYTHONPATH=src "$PY" benchmarks/include_integrity.py "$REF" ${{MINE:+"$MINE"}}
""".format(
    repo=Path(__file__).resolve().parents[1],
    python="~/venvs/deepseek-v41/bin/python",
)



HERMES_MD = """\
# Run B — the harness arm

Twenty coding tasks, run through the agent **with its normal scaffolding on**.
This is the product question: *can a harness make this model usable?* It is not
the diagnostic. Run A (`run_reference.py`, no harness) answers whether the model
is clean, and should be done first.

## The rules that keep this comparable

1. **One task per conversation.** Start a fresh conversation for each prompt in
   `prompts/`. A task must never see another task's output.
2. **Do not edit the prompts.** Paste each one exactly as written, including the
   final sentence about the fenced block. Cachalot and the reference arm both
   saw that wording; changing it breaks the pairing.
3. **Two passes over all twenty**, so there are two samples per task.
4. **Let the harness do whatever it normally does** — tools, retries, compiler
   feedback, self-correction. That is the thing being measured here. But write
   down in `notes.json` exactly what it was allowed to do.
5. **Save the agent's final answer for each task**, verbatim, including any
   prose around the code fences.

## Saving

One file per task per pass, in a single directory:

    <task-id>_seed1.txt        first pass
    <task-id>_seed2.txt        second pass

The task id is the filename in `prompts/` without its extension, for example
`cpp-thread-pool_seed1.txt`. The `_seed` separator matters -- the scorer splits
on it.

**Save every task, including ones that fail, loop, produce no code, or are cut
off.** A case that is quietly dropped changes the denominator, and that exact
mistake put a wrong number in this project's own records for two sessions.

## notes.json, alongside the replies

```json
{
  "arm": "run B, harness",
  "harness": "hermes <version>, <profile name>",
  "endpoint": "openrouter deepseek/deepseek-v4.1-flash",
  "provider": "<the provider slug, if it can be pinned>",
  "system_prompt": "<verbatim, or null if none>",
  "tools_enabled": ["..."],
  "retries_allowed": "<e.g. up to 3 on compile failure, or none>",
  "compiler_feedback": true,
  "temperature": 0.6,
  "cases_planned": 40,
  "cases_completed": 40,
  "notes": "anything the harness did that the prompts did not ask for"
}
```

`cases_planned` and `cases_completed` are not optional.

## What this arm can and cannot show

It can show that a harness produces compiling code where one-shot generation
does not, which is a real product result.

It cannot show that the model is fine. Cachalot's own output already contains
eighteen self-corrections in a single reply, all of which reproduced the same
corruption, so plain "try again" is known not to work here. If this arm
succeeds, the interesting question is *what* the harness fed back -- a compiler
diagnostic changes the context, a bare retry does not.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/corpus-pack")
    ap.add_argument("--corpus", default=str(CORPUS))
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "prompts.jsonl").open("w") as handle:
        for task in corpus["tasks"]:
            handle.write(json.dumps({
                "id": task["id"],
                "language": task["language"],
                "kind": task["kind"],
                "expect_compiles": task["expect_compiles"],
                "prompt": task["prompt"],
            }) + "\n")

    (out / "settings.json").write_text(json.dumps(
        {**SETTINGS, "corpus": corpus.get("corpus"), "tasks": len(corpus["tasks"])},
        indent=2,
    ))
    (out / "RUN.md").write_text(RUN_MD)
    (out / "HERMES-RUN-B.md").write_text(HERMES_MD)

    # One plain-text file per task, so the operator can drop them into a desktop
    # agent one at a time without hand-extracting them from JSON.
    per_task = out / "prompts"
    per_task.mkdir(exist_ok=True)
    for task in corpus["tasks"]:
        (per_task / f"{task['id']}.txt").write_text(task["prompt"].rstrip() + "\n")
    # Ship the runner inside the pack so the operator needs nothing else.
    runner = Path(__file__).resolve().parent / "run_reference.py"
    (out / "run_reference.py").write_text(runner.read_text())
    score = out / "score.sh"
    score.write_text(SCORE_SH)
    score.chmod(0o755)

    print(f"wrote {out}")
    for name in sorted(p.name for p in out.iterdir()):
        print(f"  {name}")
    print(f"\n{len(corpus['tasks'])} tasks x {len(SETTINGS['seeds'])} seeds = "
          f"{len(corpus['tasks']) * len(SETTINGS['seeds'])} cases")
    print("Hand the whole directory over. RUN.md is written for the operator.")


if __name__ == "__main__":
    main()
