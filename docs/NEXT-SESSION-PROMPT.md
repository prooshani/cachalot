# Prompt for the next session

Paste everything below the line into a fresh Claude Code session started in
`/Users/hamedprooshani/Projects/deepseek-v41-mac`.

---

You are continuing performance engineering on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B
parameters, 40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming
routed experts from SSD.

**Read `docs/HANDOFF-2026-09-17.md` first, then `docs/HANDOFF-2026-09-16.md`, both in full, before running
anything or proposing any change.** The 2026-09-17 document supersedes the older one wherever they differ. It
covers two sessions run on that date: sections 1 to 9 are the first, sections 10 to 14 the second, and where
they disagree the later sections win. Between them they retire the Engram lever, close the eviction lever,
close the read/compute overlap lever and the `F_RDAHEAD` lever, and establish that decode's floor is set by
bytes rather than by concurrency or by overlap. Section 13 is the current lever ranking; section 6 is kept only
because sections 10 to 12 argue against it. The document is the complete measured state of the project:
storage layout, the configuration in use, every performance curve, the open levers in priority order, the null
results that must not be repeated, and the pitfalls in the code. Treat its numbers as established fact and do
not re-derive them.

## Ground rules

1. **Memory safety is not optional.** This machine kernel-panicked twice on 2026-09-15 because the runtime ran
   with an automatic expert budget and wired about 67 GiB while other applications were open. Never run the
   runtime with an automatic budget. Every benchmark goes through `benchmarks/guarded_run.sh`, one runtime
   process at a time, never two in parallel.
2. **Measure before changing behaviour.** One architectural change at a time, and A/B every optimization with
   the arms run sequentially, interleaved in both orders, with `benchmarks/settle.sh` between arms. Two arms
   started back to back share memory and produce outliers.
3. **Quality is gated, not assumed.** Any change touching expert or Engram numerics must pass
   `benchmarks/nll_expert_precision.py` against the stored FP4 reference of 2.3004 nats before it is adopted.
   Judge numerics by teacher-forced negative log-likelihood, never by comparing greedy text: this model's greedy
   decoding flips tokens on changes as small as one floating-point unit.
4. **Every repository edit goes through shell commands**, never prose asking the user to edit a file by hand.
5. **Every command you give the user is complete and copy-paste ready**: absolute `cd`, `PYTHONPATH=src`, the
   full interpreter path `~/venvs/deepseek-v41/bin/python`. Never a bare `python`, never an ellipsis. Repeat the
   full command in every message that asks for a test to be run.
6. **After each production patch**: byte-compile, run the focused test, `git diff --check`, inspect the diff.
   Keep benchmark scripts out of runtime code.
7. **Chat replies terse.** Prose in files, commits and documents stays normal and complete.

## Where the work stands

Version 0.3.0 plus six commits, `main` clean, 53 tests passing. Interactive chat runs at 4.3-5.6 tok/s with
sub-1.2-second follow-up prefills, reached on 2026-09-16 by landing five independent changes: a calibrated
3-bit expert bank, a larger budget, an idle heartbeat that stops macOS un-wiring the working set between turns,
typing-time prefill, and the OS page cache as a second-level cache.

Nothing has moved throughput since. Both sessions on 2026-09-17 produced knowledge instead: the levers ranked
first, second and fourth at the start of the day are all worth close to nothing, and the reason decode is slow
is neither the one the project assumed in the morning nor the one it assumed at noon.

The floor is bytes. At a 36 GiB budget a token needs 988 MiB of experts, which is 133 ms of drive time at the
7.79 GB/s the drive can deliver, against 134 ms of compute. Perfect waste-free overlap therefore cannot take
decode much below 6.2 tok/s, and 96.6 % of the time decode spends blocked is waiting on misses that were never
predicted, which neither more prediction width nor more prediction lead time can fix at an affordable byte
cost. Section 13 of the handoff ranks what is left. The first lever is reading fewer bytes per expert; the
second is closing the gap between the 6.25 GB/s the runtime achieves and the 7.79 GB/s the drive gives, where
the page cache under a large wired set is the one candidate that has never been tested under realistic
conditions.

## How to start

Confirm the machine is in the expected state, then propose a plan for the highest-ranked lever you and the user
agree on, with the measurement that will decide it stated before any code is written.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && git log --oneline -3 && git status --short && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q tests && df -h / | tail -1 && ls -d /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash /Users/hamedprooshani/DeepSeek-V4.1-Flash-oQ3e-mtp
```

The user runs interactive chat himself, in a separate terminal, with this command. Keep it working, and give it
back to him verbatim whenever he asks to try the model:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && pgrep -fl "deepseek-v41/bin/python|cachalot" || CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-oQ3e-mtp CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat --expert-budget-gib 44 --max-seq-len 8192 --max-new-tokens 1024 --temperature 0.6
```

With other applications open, that becomes `CACHALOT_MLX_WIRED_LIMIT_GIB=64` and `--expert-budget-gib 36`.
