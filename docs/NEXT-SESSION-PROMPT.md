# Prompt for the next session

Paste everything below the line into a fresh Claude Code session started in
`/Users/hamedprooshani/Projects/deepseek-v41-mac`.

---

You are continuing performance engineering on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B
parameters, 40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming
routed experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

## What to read, in this order

1. **`docs/HANDOFF.md`, in full.** It is the authoritative state: the machine, the storage layout, the
   configuration in use, the operating rules, every measured baseline, the ranked lever catalogue with the
   measurement that decides each one, the retired premises, the null results and the pitfalls. Treat its
   numbers as established fact and do not re-derive them.
2. The dated logs **only when you need a derivation**: `docs/HANDOFF-2026-09-16.md` and
   `docs/HANDOFF-2026-09-17.md`. Sections 20 to 27 of the second are the most recent session and carry the
   tables behind everything below. Both are superseded as briefings, and section 10 of `HANDOFF.md` lists
   exactly which of their conclusions have expired — several read as settled nulls and are not.

Do not skip straight to code. The highest-value things this project has done were each preceded by a
measurement that cost less than an hour and changed what was worth building. The most recent session spent its
first two hours entirely on measurements and closed two levers, reopened one, and halved the expected value of
the one everybody thought was largest.

## State in one paragraph

Cachalot 0.4.0, `main` clean, tests passing. Decode runs at **182.5 ms per token, 5.48 tok/s** at a 36 GiB
budget; interactive chat at a 44 GiB budget runs at **6.0 to 7.5 tok/s** with an 87.3 % session hit rate. The
expert bank in use is 2-bit affine group 128, 9.49 MiB per expert, 486 MiB read per decoded token.

**The structural fact to hold on to: a decode token is 93 ms of compute and 89 ms of expert streaming.** The
93 ms was measured directly — decode the same tokens twice and read the second pass at a 100 % hit rate — and
it is dispatch-bound, roughly 400 GPU dispatches at 0.2 ms each, with more time in hyper-connections than in
routed experts. The 89 ms is 62 ms of exposed wait plus 27 ms that appears only while experts are being
fetched and is not accounted for anywhere. Anything you read that says the floor is 120 ms, or that bytes hide
under compute, predates this and is wrong.

## The three levers worth your session

Section 9 of `HANDOFF.md` is the full catalogue. In short, and in the order I would take them:

1. **Dispatch count (lever 2).** The largest lever that depends on nothing else: 93 ms of compute spread over
   about 400 dispatches, none of which dominates. `mx.compile` over a whole layer, or one kernel for the
   hyper-connection triple (which costs 74 ms of isolated time across 80 sublayers against the routed experts'
   25 ms), is where to start. Measure with `benchmarks/decode_resident.py`, which is the only clean read of
   the floor, and `benchmarks/profile_decode_components.py`.
2. **The 27 ms nobody has looked at.** It is the difference between the all-resident floor and
   `decode_anatomy`'s "rest", it exists only when experts are being fetched, and eviction-policy work already
   measured store bookkeeping at 1.0 ms per token — so it is probably GPU stalls against concurrent DMA rather
   than CPU time. Nobody has measured it. It is 15 % of a token.
3. **Tune lever 5, which is built and measured.** The startup hotlist is implemented, off by default
   (`CACHALOT_HOTLIST`, `CACHALOT_HOTLIST_GIB`) and A/B'd four runs a side: cold prefill −5.7 %, first-turn
   hit rate +4.0 points, 6.4 GiB less read, whole cold session −1.1 % which is inside noise. What is untuned
   is the size — 8 GiB of a 36 GiB budget is a large static reservation, and `hotlist_coverage.py` says 4 GiB
   gives two thirds of the coverage for half of it. One more A/B settles it. Also worth recording a hot set
   over more than the five prompts the trace holds.

**DSpark (lever 1) is measured and is smaller than it looked.** The `mtp.*` layers are not plain
multi-token-prediction layers; they are DSpark, and they draft five tokens per main forward at 72.7 %
acceptance at depth 1 and 2.85 tokens accepted per forward. But verifying W positions to accept T tokens reads
W/T times the bytes, and bytes are already half a token, so the projection is 1.07x to 1.17x with
confidence-gated width — and verifying all five drafted positions is a loss at every budget measured. If you
touch it, first re-measure `benchmarks/dspark_draft_cost.py` **on a quiet machine**, because every projection
moves with that number and the recorded one was taken while a bank build was running.

## Ground rules, condensed from section 5 of `HANDOFF.md`

1. **Memory safety is not optional.** Never an automatic expert budget; every benchmark through
   `benchmarks/guarded_run.sh`; one runtime process at a time. If preflight refuses a budget, lower it.
2. **Measure before changing behaviour**, one change at a time, arms interleaved in both orders with
   `benchmarks/settle.sh` between them.
3. **Two arms per side is not an A/B.** Throughput's run-to-run spread reaches 7 %; a 5 % effect needs four per
   side at least. And nothing timing-sensitive is valid while anything else is on the GPU — suspend a
   background build with `kill -STOP` rather than measuring through it.
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
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && git log --oneline -3 && git status --short && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q tests && /bin/df -g /System/Volumes/Data | tail -1 && ls -d /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g64 /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128-lsq
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
