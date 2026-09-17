# Prompt for the next session

Paste everything below the line into a fresh Claude Code session started in
`/Users/hamedprooshani/Projects/deepseek-v41-mac`.

---

You are continuing performance engineering on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B
parameters, 40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming
routed experts from SSD.

**Read `docs/HANDOFF-2026-09-17.md` first, then `docs/HANDOFF-2026-09-16.md`, both in full, before running
anything or proposing any change.** The 2026-09-17 document supersedes the older one wherever they differ. It
covers three sessions run on that date: sections 1 to 9 are the first, 10 to 15 the second, 16 to 19 the third,
and later sections win wherever they disagree. **Section 19 is the current lever ranking.** Sections 16 to 18
are the session that changed the machine's configuration and inverted the ranking, so read them closely:
between them they retire the assumption that a smaller bank was blocked on GGUF k-quants, retire the value of
imatrix calibration, reopen the prediction-width sweep that section 13 closed, and establish that decode is now
compute-bound rather than bytes-bound. Treat the numbers as established fact and do not re-derive them.

## Ground rules

1. **Memory safety is not optional.** This machine kernel-panicked twice on 2026-09-15 because the runtime ran
   with an automatic expert budget and wired about 67 GiB while other applications were open. Never run the
   runtime with an automatic budget. Every benchmark goes through `benchmarks/guarded_run.sh`, one runtime
   process at a time, never two in parallel. If its preflight refuses a budget, lower the budget rather than
   forcing it: 36 GiB needs 65 GiB available and other applications can hold 8 GiB wired on their own.
2. **Measure before changing behaviour.** One architectural change at a time, and A/B every optimization with
   the arms run sequentially, interleaved in both orders, with `benchmarks/settle.sh` between arms. Two arms
   started back to back share memory and produce outliers. Decode throughput's run-to-run spread reaches 7 %,
   so two arms per side is not enough to call a 5 % effect; use at least four.
3. **Quality is gated, not assumed, and gated on the production path.** Any change touching expert or Engram
   numerics must pass `benchmarks/nll_expert_precision.py` before it is adopted. The dense reference arm
   (`--experts fp4`, `--experts requant`) ranks weights; the production arm (`--experts runtime`) ranks what
   the model actually computes, and section 18.2 shows the two disagree **in sign** between the 3-bit and
   2-bit banks. Run the production arm. Judge numerics by teacher-forced NLL and top-1, never by comparing
   greedy text: this model's greedy decoding flips tokens on changes as small as one floating-point unit. Use
   512 tokens, not 160, for any top-1 comparison — the binomial standard deviation at 160 tokens is 3.9
   points.
4. **Every repository edit goes through shell commands**, never prose asking the user to edit a file by hand.
5. **Every command you give the user is complete and copy-paste ready**: absolute `cd`, `PYTHONPATH=src`, the
   full interpreter path `~/venvs/deepseek-v41/bin/python`. Never a bare `python`, never an ellipsis. Repeat the
   full command in every message that asks for a test to be run.
6. **After each production patch**: byte-compile, run the focused test, `git diff --check`, inspect the diff.
   Keep benchmark scripts out of runtime code.
7. **Chat replies terse.** Prose in files, commits and documents stays normal and complete.

## Where the work stands

Version 0.3.0 plus ten commits, `main` clean, 60 tests passing.

Decode runs at **182.5 ms per token, 5.48 tok/s** at a 36 GiB budget, against 275 ms and 3.64 tok/s at the
start of 2026-09-17. Both halves of that came from the third session: a 2-bit expert bank built here from the
FP4 checkpoint, which cut bytes per token from 1030 to 486 MiB, and `CACHALOT_PREDICT_TOPK` raised from 3 to 6,
which the smaller expert made worth 5.2 %.

**Decode is now compute-bound**: 120 ms of compute per token against 73 ms of drive time that hides under it,
with 62 ms of exposed wait in between. Every ranking this project carried before section 19 assumed the
opposite. The compute split itself is stale — it was measured on 15.48 MiB experts with the 3-bit
`quantized_matmul`, which micro-benchmarks 2.7x slower per row than the 2-bit one — so the first thing the next
session should do with a lever in mind is re-profile where those 120 ms go.

**One decision is open and it is Hamed's, not the gate's.** The speed cost quality: against the 3-bit bank, at
512 tokens on the production path, the 2-bit g128 bank costs +0.019 nats, 1.9 % of perplexity and 6.3 points of
top-1 (50.8 % to 44.5 %). The 2-bit g64 bank costs +0.033 nats but only 4.3 points of top-1, at 4 % less speed.
If he wants the quality back, the measured alternative is a **3-bit bank built here**: re-quantizing FP4 to
3-bit with plain `mx.quantize` beat the oQ3e download by 0.030 nats in dense math at identical size, so it
should be better than this morning's quality with roughly this morning's speed. Building it needs 221.5 GiB and
one of the 2-bit banks deleted first (190 GiB free).

## Storage

| what | where | size | speed |
|---|---|---|---|
| FP4 checkpoint — **only complete copy** | `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash` | 475.2 GiB | 1.0 GB/s, USB 3.2 Gen 2 |
| 3-bit oQ3e bank — **only copy** | `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-oQ3e-mtp` | 331 GB | 1.0 GB/s |
| 2-bit g128 bank, in use | `/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128` | 142.4 GiB | 6.6-6.8 GB/s cold |
| 2-bit g64 bank | `/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g64` | 158.2 GiB | same |
| free space, internal | | 190 GiB | |

The X10Pro must stay connected: it holds the only FP4 copy, the only 3-bit copy, and the Engram tables the
runtime reads on every prefill. The internal copy of the 3-bit bank was deleted on 2026-09-17 to make room;
restoring it is a 331 GB copy from that drive.

## How to start

Confirm the machine is in the expected state, then propose a plan for the highest-ranked lever you and the user
agree on, with the measurement that will decide it stated before any code is written.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && git log --oneline -3 && git status --short && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q tests && df -h /System/Volumes/Data | tail -1 && ls -d /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g64
```

The user runs interactive chat himself, in a separate terminal, with this command. Keep it working, and give it
back to him verbatim whenever he asks to try the model:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && pgrep -fl "deepseek-v41/bin/python|cachalot" || CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat --expert-budget-gib 44 --max-seq-len 8192 --max-new-tokens 1024 --temperature 0.6
```

With other applications open, that becomes `CACHALOT_MLX_WIRED_LIMIT_GIB=64` and `--expert-budget-gib 36`. To
try a different bank, change `CACHALOT_EXPERT_BANK` and nothing else.
