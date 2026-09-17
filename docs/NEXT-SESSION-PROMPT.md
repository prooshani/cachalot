# Prompt for the next session

Paste everything below the line into a fresh Claude Code session started in
`/Users/hamedprooshani/Projects/deepseek-v41-mac`.

---

You are continuing performance engineering on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B
parameters, 40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming
routed experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

## What to read, in this order

1. **`docs/HANDOFF.md`, in full.** It is the authoritative state as of 2026-09-17: the machine, the storage
   layout, the configuration in use, the operating rules, every measured baseline, the ranked lever catalogue
   with the measurement that decides each one, the retired premises, the null results and the pitfalls. Treat
   its numbers as established fact and do not re-derive them.
2. The dated logs **only when you need a derivation**: `docs/HANDOFF-2026-09-16.md` and
   `docs/HANDOFF-2026-09-17.md`. Both are superseded as briefings and both contain conclusions whose premises
   have since expired — section 10 of `HANDOFF.md` lists exactly which, and it matters, because several of them
   read as settled nulls and are not.

Do not skip straight to code. The three highest-value things this project has done were each preceded by a
measurement that cost less than an hour and changed what was worth building.

## State in one paragraph

Cachalot 0.4.0, tag `v0.4.0`, `main` clean and pushed, 60 tests passing. Decode runs at **182.5 ms per token,
5.48 tok/s** at a 36 GiB budget, against 275 ms and 3.64 tok/s at the start of 2026-09-17; interactive chat at
a 44 GiB budget runs at **6.0 to 7.5 tok/s** against 4.3 to 5.6 before, with an 87.3 % session hit rate and
sub-second follow-up prefills. That came from two changes: a 2-bit routed-expert bank built here from the FP4
checkpoint, which took bytes per token from 1030 to 486 MiB, and `CACHALOT_PREDICT_TOPK` raised from 3 to 6,
which only became worth doing once experts got smaller.

**Decode is now compute-bound**: 120 ms of compute per token, 73 ms of drive time hiding underneath it, the
drive idle 55 % of the decode, and 62 ms of exposed wait in between. Every ranking this project carried before
2026-09-17 assumed the opposite.

## The open decision, which is Hamed's and not yours to settle by measurement

The speed cost quality. Against the 3-bit bank, at 512 teacher-forced tokens on the production path, the 2-bit
g128 bank in use costs **+0.019 nats, 1.9 % of perplexity and 6.3 points of top-1** (50.8 % to 44.5 %). It is
visible in output as dropped characters inside words and renamed entities. Three measured options exist — the
2-bit g128 bank (fastest, on the internal SSD), the 2-bit g64 bank (4 % slower, 2 points of top-1 back, also on
the internal SSD), and a 3-bit bank built here rather than downloaded, which beat the download by 0.030 nats in
dense math at identical size and would need 221.5 GiB and one 2-bit bank deleted. Section 7.3 of `HANDOFF.md`
has the full table. Raise it with him before doing anything that assumes an answer.

## The three levers worth your session, with the first action for each

Section 9 of `HANDOFF.md` is the full catalogue with evidence and costs. In short:

1. **MTP speculative decoding.** The FP4 checkpoint contains three complete next-token-prediction layers with
   their own 128 routed experts each, 7.39 GiB in total, unused by the runtime. This is the only lever whose
   ceiling is another 1.5x. The 2026-09-16 log calls speculative decoding a null result because verification
   multiplies bytes per token; that was true when bytes were the constraint and is not true now.
   **First action, half a day and no runtime changes:** load the MTP layers, run them over real decoded
   prefixes, and measure per-position draft acceptance for k = 1, 2, 3. That number alone sets the ceiling. If
   it is below about 60 %, write the null down and move to lever 2.
2. **Compute, 120 ms per token,** now 66 % of decode and never attacked because it was never binding. The
   component split on record is stale — it was measured on 15.48 MiB experts with the 3-bit kernel, which
   micro-benchmarks 2.7x slower per row than the 2-bit one. **First action:** settle the inconsistency in
   section 9.2 — the 2026-09-16 log claims all-resident decode is 68 ms per token, which would mean 50 ms of
   the current 120 ms is neither arithmetic nor expert wait — then re-profile with
   `profile_decode_components.py` and `profile_decode_gpu.py`.
3. **Recover the 2-bit quality,** which costs build time only: no runtime change, no risk to the decode path,
   one 33-minute rebuild plus one gate run per attempt. **First action:** screen a finer search grid and an
   AWQ-style per-channel rescale folded into the stored scales with `expert_requant_error.py`, which takes
   seconds, before spending a rebuild on either.

## Ground rules, condensed from section 5 of `HANDOFF.md`

1. **Memory safety is not optional.** Never an automatic expert budget; every benchmark through
   `benchmarks/guarded_run.sh`; one runtime process at a time. If preflight refuses a budget, lower it.
2. **Measure before changing behaviour**, one change at a time, arms interleaved in both orders with
   `benchmarks/settle.sh` between them.
3. **Two arms per side is not an A/B.** Throughput's run-to-run spread reaches 7 %; a 5 % effect needs four per
   side at least.
4. **Gate quality on the production path**, not only the dense reference math — the two disagree in sign
   between the 3-bit and 2-bit banks — and use 512 tokens for any top-1 claim.
5. **Judge numerics by NLL and top-1, never by comparing greedy text.**
6. **Every repository edit goes through shell commands**, never prose asking Hamed to edit a file.
7. **Every command you give him is complete and copy-paste ready**: absolute `cd`, `PYTHONPATH=src`, the full
   interpreter path `~/venvs/deepseek-v41/bin/python`. Never a bare `python`, never an ellipsis.
8. **After each production patch:** byte-compile, run the focused test, `git diff --check`, inspect the diff.

## How to start

Confirm the machine is in the expected state:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && git log --oneline -3 && git status --short && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q tests && df -h /System/Volumes/Data | tail -1 && ls -d /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g64
```

Then propose a plan for the lever you and Hamed agree on, with the measurement that will decide it stated
before any code is written.

This is the command he runs to use the model. Keep it working, and give it back verbatim whenever he asks:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && pgrep -fl "deepseek-v41/bin/python|cachalot" || CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat --expert-budget-gib 44 --max-seq-len 8192 --max-new-tokens 1024 --temperature 0.6
```

With other applications open that becomes `CACHALOT_MLX_WIRED_LIMIT_GIB=64` and `--expert-budget-gib 36`. To
try a different bank, change `CACHALOT_EXPERT_BANK` and nothing else — and check the `expert bank:` line the
runtime prints at startup, because a wrapped copy-paste that loses the variable silently serves FP4 from the
USB drive at a quarter of the speed with no error.
