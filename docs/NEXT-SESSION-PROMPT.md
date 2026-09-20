# Next-session prompt — **v18**, written 2026-09-21

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v18** | 2026-09-21 | the re-audit finished and speed reopened | the 2-bit bank passes the gate equal to FP4 and ships at **7.6 tok/s**; the frequency penalty and mirror striping both come off; 1-bit is closed with a number; **compute is now 69 % of the token** and section 9's ranking needs redoing |
| v17 | 2026-09-20 | the gate came back clean | quality restored and measured, 20/20 compiling; the work was undoing the damage to this document's conclusions |
| v16 | 2026-09-20 | the root cause found | `hc_post` applied the hyper-connection mix transposed |
| v15 | 2026-09-20 | the attention measured | attention retrieves at 0.928 and the answer is still rank 17,935 |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read `docs/HANDOFF.md`'s opening block and then section 6.2.** The first tells you what ships; the second
tells you why the work queue below is ordered the way it is.

## Where the project stands

The quality defect of 2026-09-20 — `hc_post` contracting the hyper-connection mixing matrix over the wrong
index — is fixed, gated and pinned by tests. **The 2026-09-20/21 session re-established everything that had
been measured through it, and the answer moved the product.**

| | before 2026-09-20 | **now** |
|---|---|---|
| bank | FP4, 17.93 MiB/expert | **2-bit g128, 9.49 MiB/expert** |
| interactive decode | 2.1-2.6 tok/s | **7.55-7.62 tok/s** |
| session hit rate | 87.3 % | **90.2 %** |
| C++ blocks compiling, 40-case corpus | 0/42 | **20/20**, equal to the hosted reference |
| `--frequency-penalty` | 0.2 / 128, mandatory | **off** |
| mirror striping | on | **off** — an 18 % loss at this expert size |

**Nothing is mid-flight.** Clean tree, 214 tests passing, no background jobs.

## The lesson this session added to the two from last time

The first two still hold: **an A/B only finds a defect that differs between its arms**, and **aggregate
metrics were blind to this by construction — use `continuation_rank.py`'s repeat column**.

The third: **a lever's value is a property of the bank, not of the runtime.** Four levers flipped when the
expert got smaller — dispatch count, prefetch timing, mirror striping, and the entire quantization ranking.
Mirror striping is the clearest case and worth reading (section 9.11.1): it is worth −5 % on FP4 and +18 %
*cost* on 2-bit, and a fraction sweep from 0.10 down to 0.02 is **flat**, which proves the penalty is the
second drive's fixed round-trip latency rather than the tail's transfer time. The threshold is whether the
head read is long enough to hide that latency: 9.37 ms on FP4 hides it, 4.15 ms on 2-bit does not. Any future
bank smaller than FP4 inherits this, and the check is one comparison rather than a sweep.

## Job 1 — re-rank section 9 against a 131 ms token

**This is the main work and it is mostly arithmetic, not code.** Every ordering in section 9 was computed on
a 341 ms token that read 1,858 MiB and kept the drive 80.5 % busy. The token is now 131 ms reading 804 MiB
with the drive 57.1 % busy and **69 % of the token in `rest`** — compute, against a 93.0 ms all-resident
floor for this bank.

Start by re-running the anatomy on the shipped configuration, which has never been profiled: 44 GiB with the
hotlist, not the 32 GiB the 2026-09-20 pair used.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 44 --max-seconds 3600 --tag anatomy-q2-44 -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json CACHALOT_HOTLIST_GIB=8 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64
```

Then re-read these three, each of which has an expired premise:

1. **Lever 2, dispatch count.** Demoted with "worth close to nothing until bytes come down: 84.6 ms is
   already hidden". Bytes came down 57 % and the compute is no longer hidden. Section 9.2.
2. **Section 9.12 and 9.12.1, prefetch precision and lookahead.** Priced against a drive at 80.5 % busy.
   It is at 57.1 %, and the live session still wastes **67.4 % of predicted loads — 311 GB of 716.8** —
   which is section 9.10's 42.7 % of all bytes, unchanged across two banks and a runtime fix.
3. **Section 9.4, expert budget.** Its curve is 2-bit and still valid, but 44 GiB now holds 4,480-4,746
   experts at a 90.2 % live hit rate, so the marginal point has moved.

## Job 2 — the compute floor, which nobody has ever attacked

`profile_decode_components.py` on the shipped bank. On FP4 the hyper-connection machinery was the largest
block — `hc_mixes` 27.8 ms, `hc_pre + rms_norm` 22.5 ms, `hc_post` 18.4 ms, **68.7 ms of a 84.6 ms token** —
and compute was measured to be bank-independent. Nothing in that profile has ever been optimized; `hc_post`
was only *corrected* last session. It is now the largest addressable term in the product.

Note the 2-bit all-resident floor is **93.0 ms against FP4's 84.6**, measured 2026-09-17, so the affine
expert path costs about 8 ms more per token than the FP4 kernel. Re-measure that before trusting it: it
predates the fix and predates mirror striping.

## Job 3 — two loose threads from the live session, both cheap

1. **A stray character.** Turn 2 of the 2026-09-21 session returned `wHi! How can I help you today?` — one
   spurious `w` at the first position of a reply. `chat_turns.py` decodes the identical prompt greedily and
   returns it clean, and the 40-case gate is at 0/94 malformed includes, so this is probably the sampler at
   temperature 0.6. **It is not established.** Four greedy repeats of those two turns, plus
   `token_rank_probe.py` on a transcript containing them, settles it in ten minutes. Section 7.2.2.
2. **Typing-time prefill.** 44 runs spent 20.18 s on 81 tokens — 249 ms per token against the 90-116 ms a
   batched turn prefill costs. It hides behind human typing so it costs nothing observable, but nobody has
   asked why it is 2.5x more expensive per token.

## What is closed, so nobody reopens it

- **1 bit per weight.** MLX refuses it outright — *"supported bits are 2, 3, 4"* — so it needs custom Metal
  kernels for the fit and the matmul. And the weights do not survive: on 24 real experts a 1-bit affine fit
  gives a **relative output error of 16.1 against 2-bit's 1.05, a factor of 15.4**, which is noise rather
  than degradation. `benchmarks/onebit_screen.py` reproduces it in under a minute. Even at zero quality cost
  it could not win more than about 16 % of a 131 ms token. Section 9.8.
- **Mirror striping on this bank**, at any fraction. Section 9.11.1.
- **The frequency penalty.** Section 9.9. The knobs still exist; a real loop would be a genuine finding,
  since 0 of 12 bounds the rate at 22 % rather than at zero.
- **Lever 3**, recovering the quality the 2-bit bank cost. There was none to recover. Section 9.3.

## Rules that still hold

- **Never quote a screen as a gate.** Wrong six times. `benchmarks/onebit_screen.py` above is a screen and is
  quoted as one: it decides whether to spend a day, not whether something ships.
- **An A/B only finds defects that differ between its arms.** Compare against
  `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/`.
- **Write a test when a comparison finds a match, not only when it finds a bug.** Prefill was read at three
  points on 2026-09-20 and all three matched; the eleven tests recording that are what makes the next change
  there safe.
- **Never drop a case from a denominator.** `--allow-incomplete` is for progress reads, labelled partial.
- **Check a confound before reporting an effect.** The mirror result named one — the X10Pro had just absorbed
  a 143 GiB copy — and a run 90 minutes later killed it.
- **Memory.** `guarded_run.sh` needs `budget + 29` GiB available. A 44 GiB budget needs 73 and is genuinely
  marginal: it was refused four times on 2026-09-20 at 71.7-72.1 GiB with Firefox, Slack, Mail and Stream
  Deck already closed. Never pass `--force`; lower the budget instead. If the terminal emulator is the last
  1.6 GiB, hand Hamed the command rather than killing his session.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once. Launch each run as its own command, never from a loop or heredoc that
  mentions `guarded_run.sh`.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/continuation_rank.py` | first-vs-repeat copy rank — **the number that found the bug** | instant, reads a probe JSON |
| `benchmarks/token_rank_probe.py --probe benchmarks/probe_repeat_identifiers.txt --prefill 8 --tokens 1500` | where the correct token ranks, teacher-forced; the probe is in the repo now | ~5 min |
| `benchmarks/chat_turns.py --max-new-tokens 160` | six chat turns exactly as the CLI runs them, with the text | ~3 min |
| `benchmarks/decode_anatomy.py` | where a token's time goes | ~6 min |
| `benchmarks/profile_decode_components.py` | where the compute floor goes, kernel by kernel | ~4 min |
| `benchmarks/repetition_quality.py --seeds 4 --frequency-penalty 0` | free-running collapse rate | ~50 min FP4, ~30 min 2-bit |
| `benchmarks/onebit_screen.py` | quantization output error at 1, 2 and 3 bits on real experts | ~1 min |
| `benchmarks/coding_quality.py --resume` | the 40-case corpus, restartable | ~1.5 h on 2-bit |
| `benchmarks/code_validity.py`, `include_integrity.py` | the gate, against the reference arm | seconds |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | **the official implementation** — `model.py`, `engram.py`, `kernel.py`, `config.json` |
| `~/cachalot-runA-relace-fp4-scored/` | the hosted reference arm, 40/40, no harness, provider pinned |
| `benchmarks/results/coding/q2g128-v17/` | **the 2-bit gate**, 40/40, equal to the reference |
| `benchmarks/results/coding/hcfix/` | the FP4 gate, 40/40 |
| `benchmarks/results/coding/20260919-085931_*/` | the corrupt arm, kept as the before picture |
| `benchmarks/results/rankprobe-v17-*.json` | the copy probes, FP4 and 2-bit, both at 99 % repeat |
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128/` | the 143 GiB mirror copy — **no longer used**, delete it if the drive is wanted |
