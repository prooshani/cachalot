# Next-session prompt — **v17**, written 2026-09-20

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v17** | 2026-09-20 | the gate came back clean | quality is **restored and measured**: 20/20 compiling, 0/101 malformed, every column equal to the reference. The hunt is over. The work now is undoing the damage the defect did to this document's conclusions, and reopening speed. |
| v16 | 2026-09-20 | the root cause found | `hc_post` applied the hyper-connection mix transposed; copying 51 % to 99 %; gate not yet run |
| v15 | 2026-09-20 | the attention measured | attention retrieves at 0.928 and the answer is still rank 17,935 |
| v14 | 2026-09-20 | the probes v13 asked for | the defect is in-context copying; expert path and Engram cleared |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read `docs/HANDOFF.md` section 7.4.8 first.** It is one page and it is why everything else in that
document needs re-reading.

## Where the project stands

The quality gap that defined the last several sessions is closed. `hc_post` applied the hyper-connection
mixing matrix transposed — `comb @ residual` where DeepSeek's `Block.hc_post` does `comb.T @ residual` — in
both the MLX path and the fused Metal kernel. Fixed in `e37b73c`, gated in `e9327e3`.

| 40-case corpus | corrupt | **fixed** | reference |
|---|---:|---:|---:|
| C++ blocks that compile | 0/42 | **20/20** | 20/20 |
| Python blocks that parse | 5/26 | **18/18** | 18/18 |
| `#include` lines malformed | 49/154 (32 %) | **0/101 (0 %)** | 0/102 (0 %) |
| C++ errors per 100 lines | 42.2 | **0.0** | 0.0 |

Throughput during the clean gate was 2.1-2.5 tok/s, unchanged by the fix. **No speed optimization was ever
the cause**, and section 9 is unfrozen.

**Nothing is mid-flight.** No background job, clean tree, 203 tests passing.

## The two lessons that should shape how you work here

**An A/B only finds a defect that differs between its arms.** Both implementations of `hc_post` carried the
same transpose, so `CACHALOT_FUSED_DECODE=0` moved nothing and a screen in this project recorded "all three
kernels cleared" — true of the kernels, false of the code they shared. The same blind spot cleared expert
arithmetic while leaving the shared router untouched. What found the bug was reading against the reference
implementation, which ships with the weights at
`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` and which no session before 2026-09-20 had opened.

**Aggregate metrics were blind to this by construction.** `comb` is sinkhorn-normalized and close to doubly
stochastic, so every residual stream received about the right *total* weight either way. NLL stayed at 2.4
perplexity, top-1 at 88 %, hit rate unchanged — while the runtime could not copy a word it had written ten
tokens earlier. Use `benchmarks/continuation_rank.py`'s **repeat** column as the quality number. It is the
one that moved 51 % to 99 %.

## Job 1 — re-establish what was measured through the defect

This is the real work and it is easy to skip because nothing looks broken. Every quality conclusion in
`docs/HANDOFF.md` predating 2026-09-20 was measured on a runtime with a transposed residual mix, and the
corruption was data-dependent, so it did not hit all arms equally.

1. **The bank comparisons — sections 8, 9.0, 9.3, 7.5.** FP4 against 2-bit against 3-bit, the searched fit,
   the activation-weighted fit. Three sessions ranked quantization formats through the defect and concluded
   quality lived in the weights. The 2-bit bank still exists at `~/DeepSeek-V4.1-Flash-q2g128`; the 3-bit
   banks were deleted and rebuild in under an hour. Re-run the FP4-against-2-bit gate before any of those
   conclusions is quoted again. It is plausible the cheaper bank is now good enough, which would change the
   whole speed picture.
2. **The frequency penalty — section 9.9.** `--frequency-penalty 0.2 --penalty-window 128` was adopted
   because 62 % of long code replies collapsed into a repeating loop. A runtime that could not reliably copy
   its own recent output is an excellent explanation for a repetition loop. Test whether the penalty is
   still needed; if it is not, take it off, because it distorts code that legitimately repeats. This is one
   corpus run with the penalty at 0.
3. **Section 7.4.1's withdrawn figures.** Those were withdrawn for a broken gate, not for this defect, and
   the gate is trustworthy now. They can be re-measured cheaply if anyone still wants them.

## Job 2 — reopen speed, with a quality check that can see

Section 9 is a work queue again and its arithmetic about the machine is still correct. Decode on FP4 is
drive-bound almost end to end: 1,858 MiB per token, the drive busy 80.5 % of decode, an 84.6 ms compute
floor under a 341 ms token. Section 6.1.

Two conditions. Any lever whose case rests on a quality argument needs that argument re-established under
Job 1 first. And a lever's quality check must report the repeat-copy number, not NLL or top-1.

## Job 3 — hunt for the same class of defect

One transposed contraction survived months because it preserved aggregate statistics. Look for others in
places nothing pins:

- **Prefill was never read against the reference.** The decode path was read at about forty points and
  matches; prefill has not been touched. One difference is already known to live there: the reference masks
  with `where(idxs < compress_lens, idxs + offset, -1)`, which cannot bite at decode but can in prefill,
  where `compress_lens` varies per query.
- Any matrix applied where the index order is not pinned by a test.
- The MoE gate returns experts ascending where the reference returns descending. Benign over an
  order-independent sum, but the same *kind* of mismatch.

**Write a test when a comparison finds a match, not only when it finds a bug.** `hc_post` had no test, and
no A/B could have caught it.

## Rules that still hold

- **Never quote a screen as a gate.** Wrong six times, most recently the fused-decode screen that cleared a
  kernel pair sharing the actual bug.
- **Never drop a case from a denominator.** `code_validity.py` now refuses to score an unfinished run; pass
  `--allow-incomplete` only for a progress read and label it as partial.
- **Check a confound before reporting an effect.** A 40-against-77 distance split last session was rarity
  masquerading as distance; the controlled rerun was flat.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once. `guarded_run.sh`'s `pgrep` preflight also trips on a wrapping shell whose
  command line contains the pattern, so launch each run as its own command, not from a loop or a heredoc
  that mentions the script name.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/continuation_rank.py` | first-vs-repeat copy rank — **the number that found the bug** | instant, reads a probe JSON |
| `benchmarks/token_rank_probe.py` | where the correct token ranks, teacher-forced | ~10 min / 1,500 positions |
| `benchmarks/attention_mass_probe.py` | where attention goes at one position, and whether the value is delivered | ~3 min |
| `benchmarks/nll_expert_precision.py` | production expert path vs dense fp32 reference math | 6 and 11 min |
| `benchmarks/coding_quality.py --resume` | the 40-case corpus, restartable, refuses to merge configurations | ~2.5 h |
| `benchmarks/code_validity.py`, `include_integrity.py` | the gate, against the reference arm | seconds |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | **the official implementation** — `model.py`, `engram.py`, `kernel.py`, and `config.json`, which is the file to check constants against |
| `~/cachalot-runA-relace-fp4-scored/` | the reference arm, 40/40, no harness, provider pinned |
| `benchmarks/results/coding/hcfix/` | **the clean run**, 40/40, equal to the reference |
| `benchmarks/results/coding/20260919-085931_*/` | the corrupt arm, kept as the before picture |
| `benchmarks/results/rankprobe-cont*.json` | the copy probes, before and after the fix |
