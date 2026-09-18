# Next-session prompt — **v10**, written 2026-09-19

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v10** | 2026-09-19 | the FP4 bank finally profiled | mirror striping shipped; DSpark closed; dispatch count demoted; prefetch precision is the top lever |
| v9 | 2026-09-18 | the searched, weighted 3-bit bank built and gated | lever 0 closed; FP4's 0.6 corrected to 8.9; dispatch count named the top lever |
| v8 | 2026-09-18 | FP4 vs 3-bit vs 2-bit, matched | 3-bit retired; FP4 on the internal SSD |
| v7 | 2026-09-18 | 3-bit bank built and gated | q3g64 adopted (later retired) |
| v6 | 2026-09-18 | chat-collapse investigation | quality over speed; repetition and code-validity gates |
| v5 | 2026-09-17 | DSpark / compute floor | compute floor 93 ms; DSpark 1.20x; hotlist shipped |

---

You are continuing performance engineering on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B
parameters, 40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming
routed experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

## Read `docs/HANDOFF.md` in full first

Section 2 carries a standing decision that governs everything. **Section 6.1 is new and is the most important
thing in the document**: it is the first anatomy ever taken on the bank actually in use, and it moved four
conclusions. Section 9 opens with the ranking. Section 12.1 is how to read a live session's counters. The
dated logs are derivations only; section 10 lists which of their conclusions have expired — and **six entries
were added on 2026-09-19.**

## The standing decision, already made

**Quality over speed.** Hamed will trade decode speed for code that compiles. Do not re-open it. FP4 is the
bank in use at 8.9 C++ syntax errors per 100 lines against every quantized bank's 19.6 to 31.6, and two
quantized banks above 2 bits have been built, gated and rejected. **Bits, not fit, are binding.** Do not build
a third expecting quality.

## What the last session established, because it governs what you trust

**Every timing number in this document before 2026-09-19 was measured on the 2-bit bank, which is not
mounted.** Measuring the FP4 bank moved four conclusions at once:

1. **The drive is busy 80.5 % of decode, not 45 %, and it is at its knee.** Raising
   `CACHALOT_PREDICT_WORKERS` from 2 to 8 *lowers* achieved bandwidth (5.70 to 5.34 GB/s) and lengthens the
   demand reads on the critical path. `decode_anatomy.py`'s own "at 7.3 GB/s with 8 reads in flight" line is a
   model and the model is wrong on this drive.
2. **Decode on FP4 is drive-bound almost end to end.** 1,858 MiB per token at 5.8-6.0 GB/s is about 320 ms of
   a 341 ms token; the 84.6 ms compute floor hides underneath it.
3. **The compute floor is 84.6 ms on FP4, not the 93 ms this document carried**, and the FP4 expert kernel
   costs 23.5 ms per token against the affine path's 24.6. Compute is bank-independent in fact.
4. **Mirror striping was never harmful, only mis-tuned**, and it is shipped: −5 % decode, −7 % cold prefill.

**The general lesson, which is the one to carry:** a lever's rank is a property of the bank, not of the
runtime. Re-run `decode_anatomy.py` and `profile_decode_components.py` on whatever is mounted *before*
re-ranking anything or trusting a number in section 9.

## Job 1 — prefetch precision (section 9.12), the largest open lever

613 MiB of the 1,858 MiB read per token is predicted and never used — 34.2 wasted loads at 55 % precision,
about 105 ms of drive time on a drive that is the binding resource. **This is not the width knob**, which is
settled at top-6 and swept on FP4; it is the accuracy of the predictor, which runs layer L+1's router on layer
L's input and has never been improved.

**Deciding measurement, and do it before writing a predictor:** `analyze_trace.py` over a recorded routing
trace, to bound how much of the 45 % error is recoverable at all. The predictor's input is one layer stale by
construction, so part of the gap belongs to the model. If L+1's router on L+1's own input is itself only 70 %
right, there are 15 points available and not 45. Free to bound, days to exploit, and no quality risk —
prediction is a cache hint and cannot change what the model computes.

## Job 2 — `CACHALOT_PREDICT_AHEAD`, one interleaved sweep

Never swept on FP4. The recorded null — "timing is not the problem, coverage is" — was measured where timing
was 2.2 % of blocked time and worth 1.6 ms per token. **On FP4 timing is 20.5 % and 41.4 ms per token.** The
null is unbeaten but its premise has expired. Note the knob is cumulative: `PREDICT_AHEAD=2` predicts L+1
*and* L+2 from layer L's input, so it doubles speculative reads on a drive already at its knee. Cheap to
settle: four runs a side, interleaved, `decode_anatomy.py`.

## Job 3 — dispatch count (section 9.2), demoted but not closed

84.6 ms of compute over roughly 400 GPU dispatches, none dominating, and the model spends longer in
hyper-connections (68.7 ms across 80 sublayers) than in its routed experts (23.5 ms). It is bank-independent
and it is real work. **It is also nearly free money at present**, because that compute already hides under
320 ms of drive time. Do it after bytes come down, not before — or do it for prefill, which is compute-bound
in a way decode is not and has never been profiled at the layer level on FP4.

**Closed since v9:** DSpark speculative decoding (section 9.1). The arithmetic v9 asked for was done in
minutes with no model loaded: 1.03x on measured constants against its own 1.15x bar. Its entire projected
value lived in the gap between the drive's assumed and achieved bandwidth, and mirror striping has taken part
of that gap for hours of work instead of a session. Reopen only if bytes per expert fall.

## Ground rules

1. **Memory safety is not optional.** Never an automatic budget; every benchmark through
   `benchmarks/guarded_run.sh`; one runtime at a time. FP4 arms at a 36 GiB budget wire 52 GiB and peak near
   60; they are safe on an idle machine and were run repeatedly on 2026-09-19. They are not safe with
   applications open.
2. **Measure before changing behaviour**, one change at a time, interleaved both ways, `settle.sh` between.
3. **Two arms per side is not an A/B**, and a *stochastic* failure needs a rate, not an A/B. Four seeds
   minimum. And **check that each arm finished**: a guarded arm killed at its timeout looks like a complete
   result in the JSON.
4. **Do not read a winner out of overlapping ranges.** The mirror fractions 0.08 / 0.10 / 0.12 differ by 3 to
   5 ms with a within-arm spread of 7 to 12; that is a plateau, not a ranking. Say so rather than pick one.
5. **A simulation is not a prediction of the runtime.** `simulate_policies.py` projected +0.9 points of decode
   hit for segmented LRU at 36 GiB; the runtime delivers −0.25. Use it to choose what to measure.
6. **A teacher-forced gate cannot see a free-running failure.** Numerics and sampling changes need
   `repetition_quality.py` and `code_validity.py` as well as `nll_expert_precision.py`.
7. **Compare arms on matched generations**, and never across languages: short Python does not separate banks
   that long C++ separates 3.5x.
8. **Judge a quality arm by the paired median and the sign test, never the mean NLL** (paired SE ≈ 0.04 nats).
9. **A screen is for choosing what to gate, never for predicting what a gate will say.** Established three
   times, most recently with a screen deliberately made more faithful.
10. **Nothing timing-sensitive is valid while anything else is on the GPU.** `kill -STOP` a background build.
11. **Watch your own helper processes.** `pgrep -f foo` matches the shell running it (use `[f]oo`); an inline
    `bash -c` monitor puts the whole benchmark command in its own command line, so `guarded_run.sh`'s
    preflight sees a phantom runtime. **Put anything long-running in a script file and kill by recorded PID.**
    And write those scripts for `/bin/bash` 3.2: `set -u` with an empty `"${arr[@]}"` aborts the run.
12. **Every repository edit goes through shell commands**; every command copy-paste ready with absolute `cd`,
    `PYTHONPATH=src` and the full interpreter path.
13. **Nothing writes to `/Volumes/X10Pro/Flash4-1`.** It holds the only complete copy of the checkpoint. The
    mirror path opens it `O_RDONLY`; keep it that way.
14. **After each production patch:** byte-compile, focused test, `git diff --check`, inspect the diff.

## How to start

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && git log --oneline -3 && git status --short && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q tests && /bin/df -g /System/Volumes/Data | tail -1 && ls -d /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash /Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128
```

The command Hamed runs. Keep it working; hand it back verbatim when asked:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && pgrep -fl "deepseek-v41/bin/python|cachalot" || CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json CACHALOT_HOTLIST_GIB=8 CACHALOT_MIRROR_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_MIRROR_FRACTION=0.10 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat --expert-budget-gib 44 --max-seq-len 32768 --max-new-tokens 4096 --temperature 0.6 --frequency-penalty 0.2 --penalty-window 128
```

`CACHALOT_MIRROR_*` is new on 2026-09-19 and is worth −7 % of cold prefill and −5 % of decode at a 36 GiB
budget; expect less at Hamed's 44 GiB, where the hit rate is 87 to 90 % and there are fewer misses to help.
Swap `CACHALOT_EXPERT_BANK` to `.../DeepSeek-V4.1-Flash-q2g128` for speed over quality, and check the
`expert bank:` line at startup — a wrapped copy-paste that loses the variable silently serves FP4 from the USB
drive at a quarter of the speed. The frequency penalty is not decoration: without it 62 % of long code replies
collapse into a repeating loop.
