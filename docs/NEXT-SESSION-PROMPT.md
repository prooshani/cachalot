# Next-session prompt — **v16**, written 2026-09-20

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v16** | 2026-09-20 | the root cause found and fixed | `hc_post` applied the hyper-connection mixing matrix **transposed**, in both the MLX path and the Metal kernel. Copying went 51 % to **99 %**, overall top-1 85 % to **95 %**. The quality hunt is over; what remains is confirming it on the corpus and re-auditing everything that was measured through the defect. |
| v15 | 2026-09-20 | the attention measured | attention retrieves the right token at 0.928 and the answer is still rank 17,935: the fault is downstream of attention |
| v14 | 2026-09-20 | the probes v13 asked for | the defect is in-context copying; the expert path and Engram cleared; the checkpoint ships the official implementation |
| v13 | 2026-09-20 | v12's knob table re-read | corrected knob table, cheap probes first, `--resume` |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read `docs/HANDOFF.md` section 7.4.8 first.** It is the root cause and it changes what the rest of the
document means.

## What was wrong

`Block.hc_post` writes a sub-layer's output back into the four residual streams and mixes the incoming
streams through `comb`. The official implementation contracts `comb` over its **first** index:

```python
torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)   # out[j] = sum_i comb[i,j] * residual[i]
```

Both of our implementations contracted the second index — `comb @ residual` instead of
`comb.T @ residual` — identically, in `hyper_connection_mlx.hc_post` and in
`decode_fused_metal._hc_post_kernel`. Fixed in `e37b73c`, pinned by `tests/test_hyper_connection.py`.

| | before | after |
|---|---:|---:|
| copying a word already spelled out | 51 % | **99 %** |
| its worst rank in 1,500 positions | 23,989 | **2** |
| everything else | 88 % | **95 %** |
| the token after `#include` | 84 % | **100 %** |

**Why it hid**, and the lesson worth carrying: `comb` is sinkhorn-normalized and close to doubly stochastic,
so each stream received about the right total weight either way and the model stayed fluent. A transpose
changes *which* stream information lands in, not how much flows. Every aggregate metric here — NLL,
perplexity, top-1, hit rate — was blind to it. It also defeated two A/B screens, because both
implementations shared the error and switching between them moved nothing. **An A/B can only find a defect
that differs between its arms.**

## Job 1 — finish the gate

A 40-case corpus run was started at the end of the previous session:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/hcfix/ /Users/hamedprooshani/cachalot-runA-relace-fp4-scored /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/
```
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/include_integrity.py /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/hcfix/ /Users/hamedprooshani/cachalot-runA-relace-fp4-scored /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/
```

If it was interrupted, re-issue the same generate command with `--resume` and the same `--out-dir`; it
refuses to merge rows written under a different configuration and names the field that differs.

**Clean means reference-equal compile rate and 0 malformed includes.** The reference arm is
`~/cachalot-runA-relace-fp4-scored` at 20/20 compiling and 0/102 malformed. If the fixed runtime does not
reach that, the remaining gap is a *new* question and the probes that found this one are the way to ask it:
`token_rank_probe.py` plus `continuation_rank.py` for a number in ten minutes, and
`attention_mass_probe.py` when a specific position needs opening up.

## Job 2 — re-audit everything measured through the defect

This is the part that is easy to skip and should not be. Every quality comparison this project has made was
made on a runtime with a transposed residual mix, and the defect is data-dependent, so it did not affect all
arms equally.

1. **The bank comparisons.** FP4 against 2-bit against 3-bit, the searched fit, the activation-weighted fit
   (sections 9.0, 9.3, 7.5, 8.1). All of them ranked quantization formats through a corrupted runtime. The
   2-bit bank still exists at `~/DeepSeek-V4.1-Flash-q2g128`; the 3-bit banks were deleted and rebuild in
   under an hour. At minimum, re-run the FP4 against 2-bit gate before any of those conclusions is quoted
   again.
2. **The repetition and collapse work** (section 9.9). `--frequency-penalty 0.2 --penalty-window 128` was
   adopted because 62 % of long code replies collapsed into a loop. A runtime that could not reliably copy
   its own recent output is an excellent explanation for a repetition loop. Re-measure whether the penalty
   is still needed; if it is not, it should come off, because it distorts code that legitimately repeats.
3. **The speed levers.** Section 9 is frozen only because quality was broken. It can now be unfrozen, but
   every lever closed on a quality argument needs re-reading, and any re-audit must use the repeat-copy
   number rather than NLL, which was blind to this.

## Job 3 — hunt for the same class of defect

One transposed contraction survived months because it preserved aggregate statistics. Look for others:

- Any place where a matrix is applied and the index order is not pinned by a test.
- `hc_pre` is a plain weighted sum and has no transpose to get wrong; it is already pinned.
- The prefill counterparts of the decode path, which were never read against the reference. The decode path
  was read at ~40 points and matches; prefill has not been.
- The MoE gate returns experts in ascending score order where the reference returns descending. That is
  benign over an order-independent sum, but it is the same *kind* of mismatch and the next one might not be
  benign.

**Write a test whenever a comparison finds a match, not only when it finds a defect.** `hc_post` had no test
and no A/B could have caught it.

## Rules that still hold

- **Never quote a screen as a gate.** Wrong six times now, and the fused-decode screen in the previous
  session cleared a kernel pair that shared the actual bug.
- **An A/B only finds defects that differ between its arms.** Shared code is invisible to it. Compare
  against the reference implementation in `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/`.
- **Never drop a case from a denominator.**
- **Check a confound before reporting an effect.** A 40-against-77 distance split last session was rarity
  masquerading as distance and the controlled rerun was flat.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once. `guarded_run.sh`'s `pgrep` preflight also trips on a wrapping shell whose
  command line contains the pattern, so launch each run as its own command, not from a loop or a heredoc
  that mentions the script name.

## The instruments, all of them written and working

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/token_rank_probe.py` | where the correct token ranks, teacher-forced | ~10 min for 1,500 positions |
| `benchmarks/continuation_rank.py` | splits first occurrences from repeats — **the number that found this bug** | instant, reads a probe's JSON |
| `benchmarks/attention_mass_probe.py` | where attention goes at one position, and whether the value is delivered | ~3 min |
| `benchmarks/nll_expert_precision.py` | production expert path against dense fp32 reference math | 6 and 11 min |
| `benchmarks/coding_quality.py --resume` | the 40-case corpus, restartable | ~2 h |
| `benchmarks/code_validity.py`, `include_integrity.py` | the gate, against the reference arm | seconds |
