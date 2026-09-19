# Next-session prompt — **v11**, written 2026-09-19

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v11** | 2026-09-19 | an outside review, acted on | the coding gate was broken and every C++ syntax-error figure is withdrawn; predicted loads had no lifetime and now do; non-cumulative lookahead measured and closed, and it prices the timing term at ~24 ms/token |
| v10 | 2026-09-19 | the FP4 bank finally profiled | mirror striping shipped; speculation, eviction, prefetch lead time and prefetch precision all closed; dispatch count demoted |
| v9 | 2026-09-18 | the searched, weighted 3-bit bank built and gated | lever 0 closed; FP4's 0.6 corrected to 8.9; dispatch count named the top lever |
| v8 | 2026-09-18 | FP4 vs 3-bit vs 2-bit, matched | 3-bit retired; FP4 on the internal SSD |
| v7 | 2026-09-18 | 3-bit bank built and gated | q3g64 adopted (later retired) |
| v6 | 2026-09-18 | chat-collapse investigation | quality over speed; repetition and code-validity gates |

---

You are continuing performance engineering on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B
parameters, 40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming
routed experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

## Read `docs/HANDOFF.md` in full first

**Section 7.4.1 is the most important thing in the document and the reason this version exists.** Section 2
carries the standing decision. Section 7.4.2 is the corpus built to replace the evidence 7.4.1 withdrew.
Section 6.1 is the FP4 anatomy. Section 9 opens with the ranking; 9.12.1 and 9.13 are new. Section 10 lists
which older conclusions have expired.

## What happened last session, because it governs what you trust

An outside review of the repository (`docs/REVIEW-SPEED-QUALITY-2026-09-19.md`, with
`docs/MISS-REDUCTION-ASSESSMENT-2026-09-19.md`) was read closely and checked rather than accepted. Two of its
findings were real, reproduced here, and acted on.

**1. The coding quality gate scored failed compilations as clean.** `check_cpp` grepped stderr for
`": error: "` and never read clang's exit status. Clang reports a missing header as `": fatal error: "` and
then stops, so a block whose only defect was a mangled `#include` came back with an empty error list.

Re-scored from the saved replies: **no bank this project has built has produced a single compiling C++
block** — FP4 is 0 of 4 and the searched 3-bit bank 0 of 3. Worse, the old density was largely measuring
which arm aborted first: three of FP4's four blocks aborted on a fatal and only one of the 3-bit bank's
three did, so 39 of FP4's 41 errors came from its one fully diagnosed block. **8.9 against 31.6, "3.5x",
"6.6x", "4.6x", 19.6, 25.3, 22.7 and 4.9 are all withdrawn.** `ast.parse` has the same flaw, so "Python is
not the discriminator, both banks score 2.2" is withdrawn too — on the parse rate FP4 is 2 of 10 and the
3-bit bank 0 of 5.

**The standing decision survives and its size does not.** It never rested on this metric: FP4 collapses on
0 of 9 free-running replies against 2 of 9, wins top-1 by 5.3 points and wins the paired sign test beyond
5 sigma. Quality over speed stands. "3.5x fewer syntax errors" does not.

**2. Predicted expert loads had no lifetime.** `_sweep_inflight_locked(keep=requested)` released every
*completed* in-flight prediction the current layer had not asked for, so an L+2 read that finished before
the L+1 acquisition was discarded before L+2 could use it — a prediction was punished for finishing early.
Fixed: each prediction now records the walk and layer it was issued for. A null on decode at lead 1, which
is correct, because at lead 1 no deadline is ever tested.

**3. With that fixed, non-cumulative lookahead was measurable, and it loses by 2.0 %.** `PREDICT_LEAD=2`
predicts L+2 instead of L+1. 328.5 against 335.0 ms per token, four runs a side, non-overlapping ranges.
**Its premise was wrong in an instructive way**: the same prediction *count* is not the same *bytes*, because
precision falls from 55 % to 45 % and traffic rises 171 MiB per token on a drive 80.5 % busy.

**But it produced the most useful number of the session.** 171 MiB is about 30 ms of drive time and the arm
loses only 6.5, so the extra layer of lead recovers roughly **24 ms per token** — 7 % of decode. Lead time
converts; recall is what is missing. Section 9.12's "different signal" now has a measured prize attached.

## The standing decision, already made

**Quality over speed.** Hamed will trade decode speed for code that compiles. Do not re-open it. Do not
build a third quantized bank above 2 bits expecting quality: bits, not fit, are binding (section 9.0).

## Job 1 — finish the coding evidence, because everything else is judged by it

Section 7.4.2 built what was missing: `benchmarks/coding_tasks.json`, twenty independent tasks across C++ and
Python covering complete programs, edits and bug fixes; `benchmarks/coding_quality.py`, which runs them in
one process with a manifest recording git SHA, bank, budget, sampling, corpus hash and planned-against-
completed cases; and a `code_validity.py` that refuses to score a run whose manifest says it did not finish.

**An FP4 baseline run was started at the end of the session** (20 tasks x 2 seeds, about 1.7 hours, at a
24 GiB budget). Check `benchmarks/results/coding/` for it and whether its manifest says `complete: true`
before using it. **The first case is alarming and needs a second look**: `cpp-csv-to-json` returned 36 tokens
containing `#include escaping>`, `#include>` and an empty `main()`. If that shape recurs across the corpus it
is the strongest artefact evidence this project has; if it is confined to prompts ending "and nothing after
it", the prompt is inducing it and the corpus needs rewording. **Decide which before quoting anything.**

What is still missing, and it is the honest gap: **behavioural tests on the programs that compile**, and
**successful tasks per wall-clock hour** as the product metric. Compiling is necessary and not sufficient.

## Job 2 — the expensive levers, and there is now a price on one

**(a) A routing predictor with a different signal.** Section 9.12 bounds the shipped one at 71.5 % recall,
and the missing 28.5 % is the router's selection boundary. Section 9.12.1 now says what closing it is worth:
about 24 ms per token, 7 % of decode, for a predictor that reaches two layers ahead at lead-1 recall.
Recovering it needs an input carrying part of layer L's own update, and whatever it costs must come out of
the layer it runs ahead of. Days, no quality risk, and it is the best-argued item on this list.

**(b) Fewer bytes per expert at FP4 quality.** Decode is ~96 % drive-bound and bytes are the only term that
matters. MLX offers nothing between 3 bits and FP4's 4.25; two affine banks above 2 bits have been gated and
rejected. The one untried shape is **mixed precision by expert popularity** — FP4 for the hot experts, lower
for the cold tail — blocked by design in `storage/index.py`. Gate it on the new corpus, not on a screen.
High payoff, real quality risk, days.

**(c) Dispatch fusion, for prefill.** 84.6 ms of compute hides under 320 ms of drive time during decode, so
it buys nothing there. **Prefill has never been profiled at the layer level on FP4** and cold prefill is 27 s.
Profile before assuming the decode breakdown transfers; that assumption is what the 09-19 morning had to undo.

## Cheap things that are left

- **Re-run the cumulative `CACHALOT_PREDICT_AHEAD=2` arm** on the fixed store. It will stay a null, but its
  recorded precision and wasted-byte figures were taken while the store discarded its own L+2 work, so those
  two numbers should not be quoted until it is re-run. An hour.
- **36 against 44 GiB on FP4 at matched settings.** The CPU replay in the miss assessment says 44 GiB is
  worth 8.2 fewer misses per token, 11.4 %. Nothing has measured it on the runtime.
- **The frequency penalty and window against the new corpus**, so the setting is chosen on whether code
  compiles rather than only on whether it loops (section 9.9). An hour, and now it has a gate worth tuning
  against.

## What the review said that was *not* acted on, and why

Judged, not ignored. Reopen any of these with an argument.

- **Independent reference fixtures against the official implementation.** Correct in principle and the most
  expensive item on the review's list. Deferred: the runtime already reproduces the USB copy bit-identically
  on the same seeds, and no observed failure points at trunk numerics. Do it when one does.
- **The 4.59 tok/s traffic bound.** The arithmetic is right and it bounds *this* workload under unchanged
  routing and cache membership. It is not a ceiling on the design. Do not quote it as one.
- **CED / bounded-replay prefill.** The review is right that `README.md` claims a decoder replay shortcut the
  prefill path does not implement — worth correcting in the README. The optimization itself is a research
  branch and is approximate by DeepSeek's own account. Not this session.
- **`tool_choice` accepted by HTTP and dropped before `ChatRequest`.** Real, small, and untouched. Either
  implement it or reject it explicitly.

## Ground rules

1. **Memory safety is not optional.** Never an automatic budget; every benchmark through
   `benchmarks/guarded_run.sh`; one runtime at a time. Check `pgrep` before starting anything.
2. **Measure before changing behaviour**, one change at a time, interleaved both ways, `settle.sh` between.
3. **Two arms per side is not an A/B**, and a *stochastic* failure needs a rate, not an A/B. Four seeds
   minimum. And **check that each arm finished** — the new manifest does this for coding runs; for timing
   runs read the guarded log's exit lines, not the JSON.
4. **Do not read a winner out of overlapping ranges.** Mirror fractions 0.08 / 0.10 / 0.12 are a plateau.
5. **A simulation is not a prediction of the runtime.** `simulate_policies.py` projected +0.9 points of decode
   hit for segmented LRU; the runtime delivered −0.25.
6. **A teacher-forced gate cannot see a free-running failure.** Numerics and sampling changes need
   `repetition_quality.py` and `coding_quality.py` as well as `nll_expert_precision.py`.
7. **Compare arms on matched generations**, and never across languages.
8. **Judge a quality arm by the paired median and the sign test, never the mean NLL** (paired SE ≈ 0.04 nats).
9. **A screen is for choosing what to gate, never for predicting what a gate will say.** Established five
   times now; the newest is 9.12.1, where an offline recall table gave a count and hid its price in bytes.
10. **A gate needs a fixture it is known to fail.** `check_cpp` had no test, returned a list whose emptiness
    read as success, and produced a headline number in a standing decision for two sessions. This is the
    lesson of the whole session; apply it to the next gate you write.
11. **Nothing timing-sensitive is valid while anything else is on the GPU.** `kill -STOP` a background build.
12. **Watch your own helper processes.** `pgrep -f foo` matches the shell running it (use `[f]oo`). Put
    anything long-running in a script file and kill by recorded PID. Write for `/bin/bash` 3.2: `set -u` with
    a bare empty `"${arr[@]}"` aborts the run; use `${arr[@]+"${arr[@]}"}`.
13. **Every repository edit goes through shell commands**; every command copy-paste ready with absolute `cd`,
    `PYTHONPATH=src` and the full interpreter path.
14. **Nothing writes to `/Volumes/X10Pro/Flash4-1`.** It holds the only complete copy of the checkpoint.
15. **After each production patch:** byte-compile, focused test, `ruff check src tests benchmarks`,
    `git diff --check`, inspect the diff. The lint baseline is clean as of 2026-09-19 — keep it that way.

## What is measured and what is assumed

Measured on FP4 this session, four runs a side, interleaved, `settle.sh` between:

| | result |
|---|---|
| `CACHALOT_PREDICT_LEAD` 1 / 2 | **328.5 / 335.0 ms/token**, ranges non-overlapping, 2.0 % worse |
| precision and bytes at lead 2 | 55 % → 45 %, 1,866 → 2,037 MiB/token |
| predicted-load lifetime fix, at lead 1 | a null on decode, as expected |
| saved C++ replies, re-scored | FP4 0 of 4 compile, 3-bit 0 of 3 |
| saved Python replies, re-scored | FP4 2 of 10 parse, 3-bit 0 of 5 |
| corpus smoke, 3 tasks on FP4 | all finished on `stop`, one correct-language block each, none truncated |

Assumed, stated as assumptions:

- **The extra layer of lead is worth about 24 ms per token.** Inferred by subtracting the observed 6.5 ms
  loss from the 30 ms the extra bytes should cost. It assumes those bytes are fully exposed, which on a drive
  at its knee is close to true but was not measured directly.
- **The cumulative `PREDICT_AHEAD=2` null survives the lifetime fix.** Very likely — it doubles the predicted
  count where lead 2 only moved it — but not re-measured.
- **Mirror striping buys less at 44 GiB with the hotlist than the 5 % measured at 36 GiB.** Still unmeasured.
- **Prefill's component breakdown resembles decode's.** It may well not. Profile before believing it.

## How to start

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && git log --oneline -5 && git status --short && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q tests && ~/venvs/deepseek-v41/bin/python -m ruff check src tests benchmarks && ls -dt benchmarks/results/coding/*/ 2>/dev/null | head -3 && /bin/df -g /System/Volumes/Data | tail -1
```

The command Hamed runs. Keep it working; hand it back verbatim when asked. `scripts/chat.sh` now does the
same thing and cannot be split by a wrapping terminal, so prefer it.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && pgrep -fl "deepseek-v41/bin/python|cachalot" || CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json CACHALOT_HOTLIST_GIB=8 CACHALOT_MIRROR_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_MIRROR_FRACTION=0.10 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat --expert-budget-gib 44 --max-seq-len 32768 --max-new-tokens 4096 --temperature 0.6 --frequency-penalty 0.2 --penalty-window 128
```

Check the `expert bank:` line at startup — a wrapped copy-paste that loses the variable silently serves FP4
from the USB drive at a quarter of the speed. The frequency penalty is not decoration: without it 62 % of
long code replies collapse into a repeating loop.
