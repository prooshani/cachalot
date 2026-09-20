# Cachalot — Engineering Handoff

**Authoritative state as of 2026-09-20, end of the session that found and fixed the quality defect.** This document supersedes
`HANDOFF-2026-09-16.md` and `HANDOFF-2026-09-17.md` wherever they differ. Those two remain as the session
logs: they carry the derivations, the discarded attempts and the raw tables behind the numbers quoted here,
and section 14 indexes them. Read this document in full before running anything or proposing any change.

**Version:** Cachalot 0.5.0, tag `v0.5.0`. `main` is clean; **203 tests pass**, including
`tests/test_hyper_connection.py`, which pins the contraction that section 7.4.8 is about. Not pushed since
the fix — the last push predates it.

**The defect is found, fixed and gated: `hc_post` applied the hyper-connection mixing matrix transposed.**
`comb @ residual` where DeepSeek's `Block.hc_post` does `comb.T @ residual`, in both the MLX path and the
fused Metal kernel. On the 40-case corpus this runtime now compiles **20 of 20** C++ blocks, parses **18 of
18** Python blocks, and emits **0 of 101** malformed `#include` lines — every column equal to the hosted
reference arm, against 0/42, 5/26 and 49/154 before. Section 7.4.8.

**The checkpoint ships the official implementation and no session before 2026-09-20 had opened it.**
`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` holds `model.py`, `engram.py`, `kernel.py`,
`convert.py`, `generate.py` and `encoding/`. Every sentence in this document that says no reference exists
is wrong, including the ones that justified three bank-building sessions. Section 7.4.7.

**Start at section 7.4.8.** It is the root cause and the fix, and it is why most of this document needs
re-reading rather than trusting. In short: a hosted endpoint serving the same model compiled 20 of 20 C++
blocks where Cachalot compiled 0 of 42 (section 7.4.6, which proved the fault was in this runtime rather
than in FP4, the model or the sampler); the defect turned out to be a transposed hyper-connection residual
mix; and with it fixed Cachalot compiles **20 of 20** and malforms **0 of 101** include lines — every column
equal to the reference. Section 7.4.7 is the evidence trail between those two points, and section 7.4.1 is
why every C++ syntax-error figure older than 2026-09-19 is withdrawn for an unrelated reason.

**Section 9 was frozen until quality was restored. That condition is now met** — the gate in section 7.4.8
is clean at 20/20 compiling and 0/101 malformed, equal to the reference — **so section 9 is unfrozen.** Two
conditions attach to reopening it. Every quality conclusion this document reached before 2026-09-20 was
measured through the transposed residual mix and has to be re-established before it is quoted, which is most
of sections 8, 9.0, 9.3 and 7.5. And any re-audit must use the repeat-copy number from
`benchmarks/continuation_rank.py`, not NLL or top-1, both of which were blind to the defect by construction.

**No speed optimization was ever the cause of the quality gap.** The fix reorders a contraction and cost
nothing: throughput during the clean gate was 2.1-2.5 tok/s, the same as before.

**What the 2026-09-19 session did, in one paragraph.** It profiled the bank that is actually mounted. Every
timing number this document carried had been measured on the 2-bit bank while FP4 is what runs, and correcting
that moved six conclusions, shipped one lever and closed four. Decode on FP4 is drive-bound almost end to end;
the drive is busy 80.5 % of decode rather than 45 % and is at its concurrency knee; the compute floor is
84.6 ms rather than 93; mirror striping across both drives was never harmful, only mis-tuned, and is now on;
and speculation, eviction policy, prefetch lead time and prefetch precision were each measured and closed for
a few hours of machine time and no runtime code.

**What the second half of 2026-09-19 did, and it is a correction rather than a measurement.** An outside
review of the repository found the coding quality gate scoring failed compilations as clean: `check_cpp`
grepped stderr for `": error: "` and never read the compiler's exit status, so a block aborting on a mangled
`#include` came back with an empty error list. Re-scored, **no bank this project has built has produced a
single compiling C++ block**, and the 8.9-against-31.6 that justified the standing decision's size was
largely measuring which arm aborted first. Section 7.4.1. The same review found that in-flight predictions
have no lifetime and are discarded for finishing early (section 9.13), which puts one cheap lever back on
the list.

**What 2026-09-20 did, in one paragraph.** It ran the independent reference arm that sections 7.4.4 and
7.4.5 had been asking for, and the answer is unambiguous. Two hosted arms — one FP4, one FP8, each with the
provider pinned and fallbacks off, no harness, matched sampling, 40 of 40 cases — compile every C++ block,
parse every Python block and emit not one malformed `#include` or `import`. Greedy on the same six tasks is
6 of 6 against Cachalot's 0 of 6. A separate Hermes harness arm was also scored, and it turns out to prove
nothing about harnesses: the corruption never appeared in any of its turns, including first drafts before a
tool ran, so there was nothing for it to repair. The runtime is the fault. Sections 7.4.6 and 2.

---

## 1. What this project is

Cachalot is an MLX runtime that runs **DeepSeek V4.1 Flash** — 552B parameters, 40 layers, 384 routed experts
per layer, top-6 routing — on a single **96 GiB Mac Studio M3 Ultra**, by holding the trunk wired in memory and
streaming routed experts from SSD on demand. The model does not fit in memory; the design question is therefore
always the same one: how few expert bytes can a token be made to cost, and how much of the reading can be
hidden under computation.

## 2. Headline state

| | 2026-09-16 | now |
|---|---|---|
| decode, 36 GiB budget, 512-token prompt | 275 ms/token, 3.64 tok/s | **182.5 ms/token, 5.48 tok/s** |
| interactive chat, 44 GiB budget | 4.3–5.6 tok/s | **6.0–7.5 tok/s** |
| expert bank | 3-bit affine, 14.77 MiB/expert | 2-bit affine, 9.49 MiB/expert |
| bytes read per decoded token | 1030 MiB | 486 MiB |
| expert hit rate, benchmark / chat session | 73.4 % / — | 80.9 % / 87.3 % |
| quality, production path, 512 tokens | 2.4997 nats, 50.8 % top-1 | 2.5187 nats, 44.5 % top-1 |

Two changes produced that, both from 2026-09-17: a 2-bit expert bank built here from the FP4 checkpoint, and
`CACHALOT_PREDICT_TOPK` raised from 3 to 6, which only became worth doing once experts got smaller.

**That table describes the 2-bit bank, which is not the bank in use.** FP4 is, and until 2026-09-19 no
timing measurement in this document had been taken on it. Measured on FP4 at a 36 GiB budget with a
512-token prompt, `decode_anatomy.py`:

| | 2-bit g128 | **FP4, the bank in use** |
|---|---:|---:|
| decode | 182.5 ms/token, 5.48 tok/s | **341.5 ms/token, 2.93 tok/s** |
| with mirror striping (section 9.11) | not measured | **325 ms/token, 3.08 tok/s** |
| cold prefill, 512 tokens | 16.4 s | **29.1 s**, 27.1 s with mirror striping |
| expert hit rate | 80.9 % | 71.1 % |
| bytes read per token | 486 MiB | 1,858 MiB |
| all-resident compute floor | 93.0 ms | **84.6 ms** |

The shape of the problem is different on the two banks and the ranking of levers follows the shape, not the
other way round. On the 2-bit bank streaming is 49 % of a token and the drive is busy 45 % of decode. On FP4
streaming is 73 % of a token, **the drive is busy 80.5 % of decode**, and 1,858 MiB at the achieved
5.8-6.0 GB/s is about 320 ms of drive time inside a 341 ms token. **Decode on FP4 is drive-bound almost
end to end**, and the 84.6 ms of compute is very nearly free underneath it. Section 6.1.

**None of those speed numbers is the headline any more.** Measured on the 40-case coding corpus against a
pinned hosted arm serving the same model, 2026-09-20:

| | Cachalot, corrupt | **Cachalot, fixed** | Cachalot, fixed, **2-bit bank** | hosted reference, FP4, no harness |
|---|---:|---:|---:|---:|
| C++ blocks that compile | 0 / 42 | **20 / 20** | **20 / 20** | 20 / 20 |
| Python blocks that parse | 5 / 26 | **18 / 18** | **18 / 18** | 18 / 18 |
| `#include` lines malformed | 49 / 154 (32 %) | **0 / 101 (0 %)** | **0 / 94 (0 %)** | 0 / 102 (0 %) |
| decode during the run, median of 40 | — | 2.28 tok/s | **4.47 tok/s** | — |

**The fourth column is the 2026-09-20 result that changes the work queue.** The 2-bit g128 bank, re-gated
through the fixed runtime, is equal to FP4 and to the hosted reference on every column of the gate while
generating at nearly twice the rate. Three sessions ranked it below FP4 on NLL and top-1, and section 7.4.8
proved both of those blind to the defect that was actually producing the artefacts. Section 7.6.

Section 7.4.6 opened that gap and **section 7.4.8 closed it the same day**: the cause was a transposed
hyper-connection residual mix, and with it fixed this runtime matches the reference on every column of the
gate. The decision quoted below is kept because it is the decision that produced the fix, and because the
premise it retired — that quantization set the quality ceiling — was wrong in a way worth remembering.

> ### The standing decision, replaced on 2026-09-20: reproduce reference quality first, at any speed
>
> **The old decision was "quality over speed", and it was built on a premise that is now false.** It assumed
> the ceiling was the quantized weights: that FP4 was the best quality this hardware could serve and the job
> was to pay for it in tokens per second. Section 7.4.6 killed that. A hosted endpoint serving the same model
> at FP4, with no harness and matched sampling, compiles **20 of 20** C++ blocks and emits **0 of 102**
> malformed `#include` lines. Cachalot compiles **0 of 42** and malforms **49 of 154**. The gap is not a
> quantization cost that had to be bought. It is a defect in this runtime, and it can be fixed rather than
> traded for.
>
> **The decision now: correctness is the only gate, and throughput is not a constraint while it is being
> found.** Hamed's instruction, 2026-09-20: it does not matter if the runtime produces 0.1 tok/s — the goal
> is to reproduce the model's original quality. A configuration that is slow and correct is a success; a
> configuration that is fast and corrupt is the bug.
>
> **How the culprit gets named: strip, then reinstate one at a time.** Every speed optimization this project
> has shipped is a suspect, because none of them was ever re-audited against a clean reference. The order of
> work is:
>
> 1. Build the slowest, most obviously correct path that exists — no prefetch, no prediction, no striping,
>    no speculation, no eviction cleverness, one expert read at a time, synchronous — and run the 40-case
>    corpus on it. If that is clean, the defect is in an optimization. If it is still corrupt, the defect is
>    in the expert kernel, the dequantization or the tokenizer, and none of section 9 matters.
> 2. Reinstate one optimization. Re-run the corpus. Score it.
> 3. Repeat until the corpus breaks. The optimization that broke it is the culprit.
>
> **This is a bisection, and it only works if the gate is trusted.** It is, now: `code_validity.py` reads the
> compiler's exit status (section 7.4.1), `include_integrity.py` tests a pre-registered prediction
> (section 7.4.4), and both have a clean reference arm to compare against (section 7.4.6). A step is "clean"
> when it matches the reference on compile rate and scores 0 malformed includes — not when it looks better
> than the step before.
>
> **What this retires.** "FP4 is the quality bank and 3.3 tok/s is what it costs" is no longer a trade that
> was made; it is a symptom that was mistaken for a trade. The three bank-building sessions that chased
> quality through quantization — the 3-bit bank, the searched fit, the activation weighting — were all
> answering the wrong question. None of them is wrong about quantization; all of them were aimed at a defect
> that is not in the weights. Sections 9.0 and 9.3.

## 3. The machine

```
repository   /Users/hamedprooshani/Projects/deepseek-v41-mac      (github.com/prooshani/cachalot)
interpreter  ~/venvs/deepseek-v41/bin/python                      (3.14.7)
always       PYTHONPATH=src
MLX          0.32.2
hardware     Mac Studio M3 Ultra, 96 GiB unified memory, Darwin 27.2.0
Metal recommended working set  77.8 GiB
```

### 3.1 Storage — read this before running anything

| what | where | size | speed |
|---|---|---|---|
| FP4 checkpoint — **only complete copy** | `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash` | 475.2 GiB | 1.0 GB/s, USB 3.2 Gen 2 |
| 3-bit oQ3e bank — **only copy** | `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-oQ3e-mtp` | 331 GB | 1.0 GB/s |
| FP4 experts, **in use for quality** | `/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts` | 275.4 GiB | 6.6–6.8 GB/s cold |
| 2-bit g128 bank, the fast alternative | `/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128` | 142.4 GiB | same |
| free space, internal | | 68 GiB | |
| free space, X10Pro | | 871 GiB | |

`fp4-experts` holds **only** the 40 expert-bearing shards (numbers 3 to 42), copied from the USB checkpoint
and never modified. The other 8 shards carry embeddings, the MTP stages and Engram, and are still read from
`CACHALOT_MODEL_PATH` on the USB drive. `build_expert_index` globs `model-*.safetensors` and needs no config,
tokenizer or Engram shard, which is what makes the split legal — no code change was required. Verified: all
40 shards match the source byte count exactly, 15,360 experts index, and the runtime produces text identical
to the USB copy on the same seeds.

The X10Pro must stay connected. It holds the only copy of the FP4 checkpoint, the oQ3e download, and the
Engram tables the runtime reads on every prefill. The 2-bit g64 bank was deleted on 2026-09-18 to make room
for the 3-bit bank; it was strictly dominated and unused, and `build_affine_bank.py` rebuilds it in about half
an hour if it is ever wanted. Restoring the oQ3e download is *not* something to plan for: our own 3-bit bank
ties it on the production path (section 7.5).

The runtime reads its trunk, Engram tables, head and tokenizer from `CACHALOT_MODEL_PATH` and its routed
experts from `CACHALOT_EXPERT_BANK`. Those are independent, which is why switching banks is one environment
variable and no code.

### 3.2 Checkpoint composition, measured from the shard headers

The FP4 checkpoint is 48 shards totalling 475.2 GiB: trunk 10.4 GiB smeared across 46 shards that also hold
experts, routed experts 275.7 GiB across 43 shards, and Engram 189.1 GiB in exactly two pure shards, 47 and 48.
Engram is therefore trivially separable and the trunk is not. Engram layers are 1 and 14; each holds
`embed.weight` as F8_E4M3 of shape [384006168, 256] plus `embed.scale` as F8_E8M0 of shape [384006168, 8].

It also contains **three complete DSpark draft stages**, `mtp.0`, `mtp.1` and `mtp.2`: 2,401 tensors,
7.39 GiB in total, of which 6.72 GiB is routed experts. Each stage has **128 routed experts** of its own
(ids 0 to 127, the same six-tensor FP4 layout as a trunk expert at 18,800,640 B each) plus its own attention,
norms and hyper-connections; stage 0 also carries `main_proj` and `main_norm`, and stage 2 carries the final
norm, a rank-256 Markov head and a confidence head. Together they draft **five** tokens per main forward, not
one per stage — section 21 of `HANDOFF-2026-09-17.md` describes the mechanism, and
`src/cachalot/model/dspark_draft.py` implements it. Nothing in the decode path uses them today.

### 3.3 Expert formats

A fourth format was built and rejected on 2026-09-18: **3-bit g64 with the searched, activation-weighted
fit**, 15,482,880 B per expert, the same bytes as the columns below but the best affine fit this project can
produce. It collapsed on 2 of 6 long turns where FP4 collapsed on none (section 9.0), which is why the
table below still has four columns and not five.

| | **FP4, in use** | 2-bit g128, the fast option | 3-bit g64 (retired) | 3-bit oQ3e |
|---|---|---|---|---|
| bytes per expert | **18,800,640** | 9,953,280 | 15,482,880 | 15,482,880 |
| MiB per expert | **17.93** | 9.49 | 14.77 | 14.77 |
| experts per GiB of budget | **57.1** | 107.9 | 69.3 | 69.3 |
| bank total | **275.4 GiB** (experts only) | 142.4 GiB | 221.5 GiB | 331 GB |
| encoding | E2M1 nibbles, UE8M0 scales, group 32 | affine, searched fit | `mx.quantize`, bf16 scales | MLX affine, bf16 scales |
| C++ blocks that compile | **0 of 4** | not re-scored | not re-scored | not measured |
| where | `~/DeepSeek-V4.1-Flash-fp4-experts` | `~/...-q2g128` | **deleted** | `/Volumes/X10Pro/...` |

Deleted on 2026-09-18: the 2-bit g64 bank (strictly dominated, unused), the 3-bit g64 bank (built, gated,
adopted and retired the same day — see section 7.5) and the searched, activation-weighted 3-bit bank (built,
gated and rejected the same day — see section 9.0). All three rebuild from the FP4 checkpoint in under an
hour, and each one's evidence outlives it.

The oQ3e download is **not worth restoring**: our own 3-bit bank tied it on the production path (paired median
+0.0017, sign z −1.68), and a searched 3-bit fit now beats it on the screen (0.2600 against 0.2996, section
9.0).

## 4. The configuration to use

**Interactive chat, machine otherwise idle.** This is the command Hamed runs himself, in his own terminal.
Keep it working and hand it back verbatim whenever he asks to try the model.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && pgrep -fl "deepseek-v41/bin/python|cachalot" || CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json CACHALOT_HOTLIST_GIB=8 CACHALOT_MIRROR_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128 CACHALOT_MIRROR_FRACTION=0.10 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat --expert-budget-gib 44 --max-seq-len 32768 --max-new-tokens 1024 --temperature 0.6
```

**Two things changed in that line on 2026-09-20 and both are measurements, not preferences.**

**The bank is now the 2-bit one.** Re-gated through the fixed runtime it compiles 20 of 20 C++ blocks and
malforms 0 of 94 include lines, equal to FP4 and to the hosted reference, while reading 804 MiB per token
against 2,080 (sections 7.6 and 6.2). Measured on `chat_turns.py` at a 42 GiB budget with the hotlist and
**no** mirror, it runs the six-turn chat at **6.87 to 7.19 tok/s** against FP4's 2.1-2.6 on the corpus. Its
reply to "write a 100 word story of a small fish living in a greek" — the prompt that produced
"whitewas crumbling houses" and renamed Nikos to "Niks" on 2026-09-17, and that was the stated reason this
bank was abandoned — came back clean, with "Yiannis" spelled correctly throughout and the following turn's
haiku correctly quoting "olive oil" back out of it.

**`--frequency-penalty 0.2 --penalty-window 128` is gone.** It was adopted because 62 % of long code replies
collapsed into a loop; through the fixed runtime the rate is 0 of 12 with the penalty off, on both banks
(section 9.9). It distorts code that legitimately repeats, and nothing measurable pays for that any more.

**`CACHALOT_MIRROR_PATH` now points at the 2-bit copy**, not at the model directory. The mirror is matched by
shard filename inside the *bank*, so pointing it at the FP4 checkpoint while serving the 2-bit bank silently
disables striping with a `lacks model-00001-of-00040.safetensors` line — which is what happened to the first
2-bit corpus run. Drop both mirror variables if that copy is not present.

**To go back to FP4**, set `CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts` and
`CACHALOT_MIRROR_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash`. It is not worse on the gate; it is
1.6x slower for the same result.

**A 44 GiB budget needs about 73 GiB free and that is not always available.** `guarded_run.sh` computes
`need = budget + 29` and refused 44 four times on 2026-09-20 at 71.7-72.1 GiB available, with Firefox, Slack,
Mail and Stream Deck already closed — the last 1.6 GiB was the terminal emulator itself. 42 GiB passes and
costs about one point of hit rate. Never pass `--force`.

**Interactive chat, other applications open.** Identical but `CACHALOT_MLX_WIRED_LIMIT_GIB=64` and
`--expert-budget-gib 36`. At a 44 GiB budget the runtime wires 72 GiB of a 96 GiB machine, which is the
configuration class that panicked this machine twice on 2026-09-15.

The `pgrep` prefix is not decoration: two runtimes at once will exhaust memory. If it prints a process, do not
start another one.

Why each variable is there: `CACHALOT_MODEL_PATH` points at the FP4 checkpoint for trunk, Engram, head and
tokenizer; `CACHALOT_EXPERT_BANK` selects the routed-expert bank; `CACHALOT_PAGE_CACHE=1` leaves the OS page
cache enabled, which is free here and slightly ahead on genuine repeat hits; `CACHALOT_MLX_WIRED_LIMIT_GIB`
sets the Metal residency limit, without which macOS compresses cold expert buffers and decode collapses to
seconds per token; `CACHALOT_HOTLIST` and `CACHALOT_HOTLIST_GIB` preload a recorded hot set in the background
while the runtime finishes starting, worth 5.7 % of a cold prefill and 4.0 points of first-turn hit rate
(section 9.5); `--expert-budget-gib` is always explicit, never automatic.

`CACHALOT_MIRROR_PATH` and `CACHALOT_MIRROR_FRACTION` are new on 2026-09-19 and are the one setting on this
list that pays on *both* phases: the tail 10 % of every expert read is issued to the X10Pro concurrently with
the head on the internal SSD, which cuts the critical-path read from 9.37 ms to 8.10 ms. Measured at a 36 GiB
budget on a 512-token prompt: cold prefill −6.9 %, decode −5 %. The X10Pro is opened read-only and holds the
byte-identical shards the internal copy was made from, so there is no quality question and nothing to keep in
sync. **Expect less than the measured 5 % in Hamed's own configuration** — a 44 GiB budget with a hotlist runs
at an 87 to 90 % hit rate, where there are far fewer misses for the second drive to help with. Section 9.11.
The fraction matters: 0.08 to 0.12 are indistinguishable, and **0.15 is worse than off**.

Dropping the two hotlist variables changes nothing but the first turn, so a copy-paste that loses them is not
a correctness problem — unlike one that loses `CACHALOT_EXPERT_BANK`, which silently serves FP4 from the USB
drive at a quarter of the speed.

`--frequency-penalty` and `--penalty-window` are no longer set: the 62 % collapse rate that justified them
was the transposed residual mix, and through the fixed runtime the rate is 0 of 12 with them off (section
9.9). The CLI still exposes all four sampling knobs if a loop is ever seen again. `--max-seq-len 32768` rather than something enormous:
the caches and the 16-entry prefix cache scale with it, and 262000 costs up to about 13 GB against 0.4 GB at
8192, on a machine already wiring 72 of 96 GiB (section 12).

## 5. Operating rules

1. **Memory safety is not optional.** This machine kernel-panicked twice on 2026-09-15 because the runtime ran
   with an automatic expert budget and wired about 67 GiB while other applications were open: swap grew to 29
   swapfiles, the compressor hit its segment limit, and `kernel_task` spun until the watchdog fired. Never run
   with an automatic budget. Every benchmark goes through `benchmarks/guarded_run.sh`, one runtime process at a
   time, never two in parallel. If its preflight refuses a budget, **lower the budget**; do not force it. A
   36 GiB budget needs 65 GiB available and other applications can hold 8 GiB wired on their own.
2. **Measure before changing behaviour.** One architectural change at a time. A/B every optimization with the
   arms run sequentially, interleaved in both orders, with `benchmarks/settle.sh` between them. Two arms
   started back to back share memory and produce outliers.
3. **Two arms per side is not an A/B.** Decode throughput's run-to-run spread reaches 7 %, so a 5 % effect
   needs at least four runs per side before it is real. The prediction-width decision in section 8.2 was made
   on six per side for this reason, after two per side had produced a misleading table.
4. **A teacher-forced gate cannot see a free-running failure.** Any change that touches numerics or sampling
   also needs `benchmarks/repetition_quality.py`, which measures whether generation falls into a repetition
   loop. The NLL gate feeds the correct prefix at every step and therefore cannot produce one; it scored the
   2-bit bank at 44.5 % top-1 while that bank collapsed on 62 % of long code replies. Judge by `max_run`, not
   by the trigram rate: healthy code repeats trigrams 34 % of the time.
5. **A stochastic failure needs a rate, and a rate needs samples.** Two conclusions in the 2026-09-18 session
   survived five replies and died on the sixth. Four seeds minimum before believing a collapse rate.
6. **Judge a quality arm by the paired median and the sign test, never by the mean NLL.** The 512-token mean
   has a paired standard error of about 0.04 nats and five tokens out of 512 routinely move it further than
   the effect being measured; two different texts have disagreed in its sign while both medians sat at zero.
   The gate prints all of it now. See section 9.3.1.
7. **Quality is gated, not assumed, and gated on the production path.** Any change touching expert or Engram
   numerics must pass `benchmarks/nll_expert_precision.py` before adoption. The dense reference arms
   (`--experts fp4`, `--experts oq3e`, `--experts requant`) rank *weights*; the production arm
   (`--experts runtime`) ranks what the model actually computes. **The two disagree in sign** between the
   3-bit and 2-bit banks (section 7.3). Run the production arm. Use 512 tokens, not 160, for any top-1
   comparison: the binomial standard deviation at 160 tokens is 3.9 points, which is wider than the effects
   being judged.
8. **Judge numerics by teacher-forced NLL and top-1, never by comparing greedy text.** This model's greedy
   decoding flips tokens on changes as small as one floating-point unit.
9. **Every repository edit goes through shell commands**, never prose asking Hamed to edit a file by hand.
10. **Every command given to Hamed is complete and copy-paste ready**: absolute `cd`, `PYTHONPATH=src`, the full
   interpreter path `~/venvs/deepseek-v41/bin/python`. Never a bare `python`, never an ellipsis. Repeat the
   full command in every message that asks for something to be run.
11. **After each production patch**: byte-compile, run the focused test, `git diff --check`, inspect the diff.
   Keep benchmark scripts out of runtime code.
12. **Nothing timing-sensitive is valid while anything else is on the GPU.** Suspend a background build with
    `kill -STOP` and resume it with `kill -CONT` rather than measuring through it.
13. **Chat replies terse. Prose in files, commits and documents stays normal and complete.**

## 6. Where the time goes

`benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64`, 2-bit g128 bank, 36 GiB budget:

    decode 64 tokens: 12.22 s = 5.24 tok/s (191 ms/token)          [top-3 prediction]
      expert hit rate 80.9% | 45.8 misses/token | 486 MiB read/token
      expert wait 4.53 s =  70.8 ms/token (37.1% of decode)
      rest        7.69 s = 120.1 ms/token (62.9% of decode)
      drive busy 5.49 s of 12.22 s decode (45.0%); 2.03 reads in flight while busy
      blocked total           4.53 s =  70.8 ms/token
      a demand read in flight 4.36 s =  68.1 ms/token (96.2% of blocked) -- never predicted
      only a predicted read   0.10 s =   1.6 ms/token ( 2.2%) -- predicted, issued too late
      no read outstanding     0.07 s =   1.0 ms/token ( 1.5%) -- slots, eviction, locks

The arithmetic that should drive every decision below, per token at a 36 GiB budget with top-6 prediction:

    bytes      45.8 misses x 9.49 MiB   = 486 MiB
      at 6.7 GB/s, what the drive gives =  73 ms
    all-resident decode, measured       =  93 ms   <- benchmarks/decode_resident.py, 512-token context
    exposed expert wait                 =  62 ms
    unaccounted, fetch-only overhead    =  27 ms
    measured                            = 182 ms

So the ceiling on a perfect cache is **93 ms per token, 10.8 tok/s**, and 89 ms per token — 49 % — is
attributable to expert streaming. Note that 73 ms of transfer and 62 ms of exposed wait means the drive
essentially does not hide under compute: overlap is not the lever, coverage is.

The 27 ms line is the difference between the all-resident floor and `decode_anatomy`'s 120 ms of "rest". It is
whatever only happens while experts are being fetched — slot acquisition, eviction, promotion, GPU stalls
against concurrent DMA. Eviction-policy work measured total store bookkeeping at 1.0 ms per token, so it is
probably not CPU time. Nobody has measured it.

There is still no fixed per-read latency tax: 3.28 ms for a 9.49 MiB expert is 2.9 GB/s per stream, the same
per-stream rate the 15.48 MiB expert gave at 5.28 ms. Reads got smaller, not slower. While the drive is busy the
runtime moves about 6 GB/s of the 6.7 GB/s available, so there is no bandwidth left to recover at a given
concurrency; what changed is that the drive is busy far less often.

### 6.1 Where the time goes on FP4, the bank actually in use

`benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64`, FP4 bank, 36 GiB budget, top-6,
mirror striping off:

    decode 64 tokens: 22.43 s = 2.85 tok/s (350 ms/token)
      expert hit rate 71.1% | 69.5 misses/token | 1858 MiB read/token
      expert wait 12.93 s = 202.1 ms/token (57.7% of decode)
      rest         9.50 s = 148.4 ms/token (42.3% of decode)
      reader threads 10.86 ms/miss | aggregate 2.58 GB/s | wall-clock 5.56 GB/s
      demand   27.7 reads/token | mean  9.49 ms, p50  8.87, p90 15.22
      predict  76.0 reads/token | mean  6.47 ms, p50  6.07, p90 10.87
      drive busy 18.05 s of 22.43 s decode (80.5%); 2.67 reads in flight while busy
      blocked total            12.93 s = 202.1 ms/token
      a demand read in flight  10.22 s = 159.8 ms/token (79.1% of blocked) -- coverage
      only a predicted read     2.65 s =  41.4 ms/token (20.5% of blocked) -- timing
      no read outstanding       0.06 s =   0.9 ms/token ( 0.5% of blocked) -- store overhead

Three things in that block are different in kind from the 2-bit numbers above, not merely in size.

**The drive is busy 80.5 % of decode, against 45 % on the 2-bit bank.** Several conclusions in this document
rest on the drive having spare capacity during decode. On FP4 it does not. Section 9.10 keeps its conclusion
and loses its stated reason.

**Timing is 20.5 % of blocked time, against 2.2 % on the 2-bit bank.** The 2026-09-17 null "prediction from
an earlier activation: timing is not the problem, coverage is" was measured where timing was worth 1.6 ms per
token. On FP4 it is worth 41.4 ms. The null's premise has expired even though nothing has yet beaten it.

**Store overhead is 0.9 ms per token**, so the 27 ms of unattributed fetch-only overhead the 2-bit anatomy
carried does not appear here. On FP4 the accounting closes: 202 ms of expert wait plus 148 ms of rest, of
which 84.6 ms is the measured compute floor.

**The all-resident floor on FP4 is 84.6 ms, slightly *below* the 2-bit bank's 93.0.**
`profile_decode_components.py --prompt-tokens 512` on FP4:

    hc_mixes (attention HC, sinkhorn kernel)    0.347 ms x 80 = 27.8 ms
    hc_pre + rms_norm                           0.281 ms x 80 = 22.5 ms
    hc_post                                     0.230 ms x 80 = 18.4 ms
    compressed reuse attention                  0.940 ms x 30 = 28.2 ms
    fused routed experts (6, FP4 kernel)        0.589 ms x 40 = 23.5 ms
    shared expert (fp8, 3 gemv)                 0.428 ms x 40 = 17.1 ms
    router route_topk                           0.329 ms x 40 = 13.2 ms
    sliding-window attention                    0.838 ms x  2 =  1.7 ms
    final head + norm                           2.134 ms x  1 =  2.1 ms
    ----
    sum of isolated pieces 154.5 ms; whole token 84.6 ms

The FP4 fused expert kernel costs 23.5 ms per token against the 2-bit affine path's 24.6. **Compute is
bank-independent in fact and not only in principle**, which is worth knowing before spending days on fusion:
the lever is the same size whatever is mounted, and on FP4 it is 84.6 ms of a 341 ms token that is already
hidden under the drive.

**`decode_resident.py` cannot measure the floor on FP4 at a 36 GiB budget.** 36 GiB holds 2,056 FP4 experts,
which is below the probe's own working set, so its repeat pass plateaus at 61 % hit and 141 misses per token
instead of reaching 100 %. The floor above comes from `profile_decode_components.py`, which times an
all-resident token directly. Do not read `decode_resident.py`'s FP4 output as a floor.

### 6.2 Where the time goes on the 2-bit bank, re-measured 2026-09-20

Section 6's 2-bit anatomy is from 2026-09-17 and predates mirror striping, the predicted-load lifetime fix
and the hyper-connection fix. This is a fresh pair, both arms run the same evening at the same budget on the
same machine state, `decode_anatomy.py --prompt-tokens 512 --decode-tokens 64`, 32 GiB budget, mirror
striping off in both (the X10Pro holds the FP4 shards, so the 2-bit bank cannot be mirrored):

| | FP4 | **2-bit g128** |
|---|---:|---:|
| decode | 380 ms/token, 2.63 tok/s | **236 ms/token, 4.23 tok/s** |
| expert hit rate | 68.1 % | **77.9 %** |
| misses per token | 76.6 | **53.1** |
| bytes read per token | 2,080 MiB | **804 MiB** |
| expert wait | 233.3 ms (61.3 % of decode) | **74.1 ms (31.4 %)** |
| rest | 147.1 ms (38.7 %) | 162.3 ms (**68.6 %**) |
| drive busy | 20.57 s of 24.35 s (**84.5 %**) | 8.63 s of 15.13 s (**57.1 %**) |
| prediction precision | 53 % | 49 % |

**The budget is 32 GiB rather than section 6.1's 36 because the preflight refused 36**: 62.1 GiB available
against 65 needed, with other applications open. Rule 1 says lower the budget rather than force it, so both
arms were lowered. That is why the FP4 column here is 380 ms against section 6.1's 341.5 — a smaller budget
holds fewer experts and the machine was not idle. The pair is internally controlled; neither column should be
compared across sections.

**The shape of the problem inverts, and this is the finding.** Section 6.1's central fact was that decode on
FP4 is drive-bound almost end to end: the drive busy 80.5 % of decode, 84.6 ms of compute hiding underneath a
341 ms token, and every lever that saves compute worth nothing. On the 2-bit bank the drive is busy **57.1 %**
of decode and **68.6 % of the token is `rest`** — compute plus whatever is not a blocked expert read. Against
the 93.0 ms all-resident floor recorded for this bank, compute is about 39 % of a 236 ms token and the drive
now has idle capacity to hide things under.

**What that does to section 9's ranking.** Lever 2, dispatch count, was demoted on FP4 with the reasoning
"worth close to nothing until bytes come down: 84.6 ms is already hidden". Bytes have come down, by 61 %, and
the sentence expires with them. Conversely the prefetch-precision and lookahead work, which was priced
against a saturated drive, is worth re-reading against one that is idle 43 % of decode. **Nothing in section 9
below has been re-ranked yet**; this is the measurement that says it must be, and it is the third time this
document has had to re-rank after measuring the bank actually in use rather than the one it was writing about.

**Coverage is still the larger half of the blocked time on both banks.** 64.1 of the 2-bit bank's 74.1 ms of
expert wait, and 185.6 of FP4's 233.3, is a demand read for an expert that was never predicted. Timing is
7.3 ms against FP4's 46.5. So the case section 9.12 makes for a better predictor survives the bank change in
kind, at about a quarter of the size.

## 7. Measured baselines

### 7.1 Decode, 2-bit g128 bank

| budget | prediction | ms/token | tok/s | hit rate | misses/token |
|---|---|---|---|---|---|
| 36 GiB | top-3 | 190–193 (median 192.5) | 5.19–5.26 | 80.9 % | 45.8 |
| 36 GiB | **top-6** | **180–184 (median 182.5)** | **5.43–5.55** | 81.0 % | 45.7 |
| 36 GiB | top-8 | 184–190 | 5.26–5.43 | — | — |
| 36 GiB | top-12 | 202–219 | 4.56–4.95 | 80.8 % | 46.0 |
| 28 GiB | top-3 | 202 | 4.94 | 76.9 % | 55.4 |

2-bit g64 at 36 GiB and top-6: 189–191 ms/token, 81.2 % hit, 45.2 misses. Cold 512-token prefill is 16.4 s with
g128 and 17.9 s with g64; it was not measured on the 3-bit bank at this budget.

For the 3-bit bank, which is no longer on the internal SSD, the recorded curve is in
`HANDOFF-2026-09-16.md` section 4.1: 3.22–3.36 tok/s at 28 GiB rising to 3.89 at 44 GiB, 275 ms/token at
36 GiB, prefill flat at 24.2 s across budgets.

### 7.1.1 Decode and prefill, FP4 bank — the bank in use

All measured 2026-09-19 at a 36 GiB budget with a 512-token prompt, machine idle, four runs per arm
interleaved in both directions with `settle.sh` between. Medians.

| arm | cold prefill, 512 | decode | tok/s |
|---|---:|---:|---:|
| FP4, as it stood on 2026-09-18 | 29.1 s | 341.5 ms/token | 2.93 |
| **FP4, mirror striping at 0.10** | **27.1 s** | **325 ms/token** | **3.08** |
| `CACHALOT_PREDICT_WORKERS=4` | — | 363 ms/token | 2.75 |
| `CACHALOT_PREDICT_WORKERS=8` | — | 369 ms/token | 2.71 |
| `CACHALOT_PREDICT_AHEAD=2` | — | 359.5 ms/token | 2.78 |
| `CACHALOT_EVICT=slru` | — | 326.5 ms/token | 3.06 |

Constant across every arm: hit rate 70.4–71.1 %, 69.5–71.0 misses per token, and — except where the knob
changes what is prefetched — 1,858–1,868 MiB read per token. **Only two of those arms change the bytes**, and
both changed them upward.

**A 44 GiB budget with the hotlist has not been measured on FP4.** Section 12.1's live-session figures are
from the 2-bit bank. Hamed's own configuration runs at an 87–90 % hit rate where this benchmark runs at 71 %,
so every number above is a lower bound on his hit rate and an upper bound on what a storage lever buys him.

### 7.2 Interactive chat, 44 GiB budget, 72 GiB wired

    ready in 16.4 s
    turn 1   prefill   5 tokens (2 reused)   0.8 s    51 tokens   6.03 tok/s
    turn 2   prefill  69 tokens (65 reused)  0.6 s    68 tokens   6.73 tok/s
    turn 3   prefill 157 tokens (152 reused) 0.7 s   117 tokens   7.52 tok/s
    session  87.3 % hit rate, 4,456 resident experts, 108.4 GiB read
             6,495 predicted loads, 2,991 used, 16 prefix-cache entries, 14 hits

A chat session beats the benchmark's hit rate — 87.3 % against 80.9 % — because it reuses experts across turns,
and 44 GiB now holds 4,456 experts where the 3-bit bank held 2,984.

### 7.2.1 Interactive chat re-measured on the fixed runtime, 2026-09-20

`benchmarks/chat_turns.py --max-new-tokens 160`, which replays six chat turns exactly as `cachalot chat`
does — same chat template, same prefix cache. 2-bit g128 bank, **42 GiB** budget, hotlist on, **mirror
striping off** (the copy was still building):

| turn | reply | tok/s | resident experts |
|---|---|---:|---:|
| 1, cold | 9 tok | 4.34 | 2,157 |
| 2 | 9 tok | **7.19** | 3,151 |
| 3, the 100-word story | 127 tok | **6.87** | 4,530 |
| 4, haiku | 20 tok | 6.41 | 4,530 |
| 5, two-sentence explanation | 44 tok | 5.86 | 4,530 |
| 6, Python one-liner | 11 tok | 4.38 | 4,530 |

Section 7.2's 7.52 tok/s was a 117-token turn at a **44 GiB** budget **with** mirror striping. This is a
127-token turn at 42 GiB with neither, so the two are within a few per cent of each other and the remaining
gap is two GiB of budget plus the striping.

**The point of running turn 3 was not the throughput.** "Write a 100 word story of a small fish living in a
greek" is the prompt that, on this same bank on 2026-09-17, produced "whitewas crumbling houses" and renamed
its own character from Nikos to "Niks" — the observation that condemned the 2-bit bank and sent three
sessions after quantization. Through the fixed runtime it returns:

> A tiny fish lived in a Greek taverna's cracked marble sink. Every morning, the owner, Yiannis, poured olive
> oil down the...

and turn 4's haiku quotes "olive oil" back out of it. Copying out of recent context is the operation
`hc_post` destroyed, so that haiku is the cheapest possible end-to-end check that the fix holds in free
generation rather than only under teacher forcing.

**Budget note.** 44 GiB was attempted four times and refused every time at 71.7 to 72.1 GiB available against
the 73 that `guarded_run.sh` requires, with Firefox, Slack, Mail and Stream Deck already closed; the last
1.6 GiB was the terminal emulator hosting the session. 42 GiB is what passed. Section 9.4's curve prices that
step at roughly one point of hit rate.

### 7.3 Quality

All arms are teacher-forced NLL on the same text at a 24 GiB budget. The FP4 reference of 2.3004 nats at 160
tokens is stored in `benchmarks/results/nll_experts_fp4.json`.

**Dense reference math** — ranks the weights, 160 tokens:

| arm | MiB/expert | mean NLL | vs FP4 |
|---|---|---|---|
| FP4 | 17.93 | 2.3004 | — |
| 3-bit g64, oQ3e calibrated | 14.77 | 2.3398 | +0.0394 |
| 3-bit g64, `mx.quantize` from FP4 | 14.77 | 2.3097 | +0.0093 |
| 2-bit g64, searched fit | 10.55 | 2.3207 | +0.0203 |
| 2-bit g128, searched fit | 9.49 | 2.3234 | +0.0230 |
| 2-bit g64, `mx.quantize` | 10.55 | 2.3395 | +0.0391 |
| 2-bit g128, `mx.quantize` | 9.49 | 17.5460 | model destroyed |

**Production path** — ranks what the model computes:

| bank | 160 tokens | top-1 | 512 tokens | ppl | top-1 |
|---|---|---|---|---|---|
| 3-bit g64, oQ3e | 2.3080 | 53.1 % | 2.4997 | 12.178 | 50.8 % |
| 2-bit g128 | 2.3337 | 45.6 % | 2.5187 | 12.413 | 44.5 % |
| 2-bit g64 | 2.3279 | 46.9 % | 2.5327 | 12.587 | 46.5 % |

The absolute level rises with length because the text continues into harder material; only comparisons at equal
length mean anything.

**The cost of the 2-bit bank against the 3-bit one is real, but read it from the median and not the mean.**
Paired per token, the 2-bit g128 bank is worse than the 3-bit one on 62.5 % of the 512 tokens by a median of
+0.035 nats — a sign test beyond 5 sigma — and 6.3 points of top-1, which is 2.9 sigma. The mean difference of
+0.019 nats is only 0.43 sigma and was over-quoted in earlier versions of this document; section 9.3.1
explains why, and the gate now prints the paired statistics. The two metrics still disagree about which 2-bit
bank is better and nothing measured so far separates them confidently.

The cost is visible in output, which matters more than the nats: a 100-word story came back with "whitewas
crumbling houses" and renamed its own character from Nikos to "Niks" in the final sentence. Dropped and mangled
tokens are what a 6-point top-1 loss looks like. Cosmetic in prose; not cosmetic in code or arithmetic.

### 7.4 Quality as a user sees it: three metrics, one conclusion

Teacher-forced NLL and top-1 are the only quality numbers this project carried until 2026-09-18, and both are
blind to how the model behaves when it is generating freely. Two more exist now, and all three agree.

**Free-running repetition** (`benchmarks/repetition_quality.py`). Replays a real multi-turn conversation
through the chat path and reports the longest span covered by a k-gram repeating back to back. Judge by
`max_run`, not by the trigram rate -- healthy code repeats trigrams 34 % of the time. On the 2-bit bank,
generating a long reply from a short prompt collapses into a loop on **62 % of replies**; with
`--frequency-penalty 0.2 --penalty-window 128` that falls to **12 %**. Section 9.9.

**Code that compiles** (`benchmarks/code_validity.py`). Extracts fenced code blocks from saved replies and
syntax-checks them with `ast.parse` or `clang++ -fsyntax-only`. **Every number this metric produced before
2026-09-19 was wrong, and section 7.4.1 is the correction — read it before quoting anything below.**

> **Correction, 2026-09-18.** An earlier version of this section reported "6.6x fewer syntax errors" for the
> 3-bit bank and a "4.6x" figure against the 2-bit one. **Both were confounded and are withdrawn.** The arms
> were not matched on what they generated: one produced three ~107-line C++ programs, the other ten ~19-line
> snippets, and long blocks accumulate errors while short ones do not. The metric was measuring block
> composition as much as correctness. Always compare arms on the same conversation *and* check the average
> block length before believing a ratio.

Matched properly — same three-turn conversation, same sampling, C++ blocks only. **This is the table as the
broken checker produced it, kept because the corrections attached to it are the useful part; section 7.4.1
withdraws every figure in the last column.**

| bank | blocks | avg block | lines | errors | ~~errors per 100 lines~~ |
|---|---:|---:|---:|---:|---:|
| **FP4**, 3 seeds, 2026-09-18 session 4 | 4 | 115 | 461 | 41 | ~~8.9~~ |
| FP4, the 2-seed arm the guard truncated | 3 | 54 | 163 | 1 | ~~0.6~~ |
| 3-bit g64, searched + activation-weighted | 3 | 138 | 414 | 131 | ~~31.6~~ |
| 3-bit g64, `mx.quantize` | 5 | 98 | 491 | 96 | ~~19.6~~ |
| 2-bit g128 | 13 | 18 | 233 | 59 | ~~25.3~~ |

> **Correction, 2026-09-18 session 4.** FP4's **0.6** was real but thin: 163 lines from an arm the memory
> guardian killed after two of three seeds, carried by one 153-line block the checker called clean. Re-run to
> three full seeds on the internal SSD — 461 lines, 4 blocks — FP4 scored **8.9**. Every ratio this document
> quoted against 0.6 is therefore too large by an order of magnitude.
>
> The lesson is the one section 7.4 already carried and did not apply to itself: a code-validity figure is
> only as good as the volume behind it, and a truncated arm is a small sample dressed as a measurement. Check
> that every arm ran to completion before comparing them, and prefer the guarded run's own log over the
> result file, which cannot tell you it is short.

> **Second correction, 2026-09-19, and it supersedes the first.** The 8.9 was not a smaller version of the
> same measurement; it was the same broken measurement. The checker never read clang's return code, so a
> block that aborted on a mangled `#include` scored one error or none. The "clean" block in the 2-seed arm
> and the "clean" block in the 3-seed arm are the same kind of false success. **Re-scored: 0 of FP4's 4
> blocks compile, and the whole column is withdrawn.** Section 7.4.1.

**The quantized banks are not meaningfully different from each other**, which is the finding that matters:
the +5.3 points of top-1 the `mx.quantize` 3-bit bank genuinely bought translated into no usable improvement
in code, and neither did the 28 % of output error the searched, activation-weighted fit bought after it.

> ~~**Python is not the discriminator; long C++ is.** On the same 2026-09-18 session-4 arms, Python blocks
> score **2.2** errors per 100 lines on *both* FP4 and the 3-bit bank.~~ **Withdrawn 2026-09-19.**
> `ast.parse` stops at the first `SyntaxError`, so 2.2 was counting broken files rather than defects. On the
> parse rate FP4 is **2 of 10** and the 3-bit bank **0 of 5**, and one 20-line FP4 block carrying five
> separate artefacts scored 1. Section 7.4.1.

The advice the withdrawn paragraph ended with is still right and is now better supported: **screen on the
hardest thing the model is asked to write.** Long C++ is where blocks run past 100 lines and an artefact
every few hundred tokens is certain to land inside one — but Python is not the clean control this document
took it for, and both languages must be reported on the compile rate rather than on a density.

**The three together.** Top-1 said 50.8 % against 44.5 %. The paired median said the 2-bit bank is worse on
62 % of tokens. The compiler was believed to say 15.2 errors against 2.3, and **it did not say that** — see
7.4.1. The standing decision in section 2 rests on the first two and on the collapse rate.

The repetition loops are a *separate* failure with a *separate* fix: they are sampling dynamics, they happen
on FP4 too, and the frequency penalty handles them. A better bank will not stop loops and the penalty will not
stop artefacts. Both are needed.

### 7.4.1 The compiler gate was broken, and every ratio it produced is withdrawn

**2026-09-19.** `check_cpp` collected the stderr lines containing `": error: "` and never read clang's return
code. Clang reports a missing header as `": fatal error: "` and then **stops**, so a block whose only defect
was a mangled `#include` came back with an empty error list and was scored **clean**. The gate is fixed,
`tests/test_code_validity.py` pins the failure, and the saved arms have been re-scored from the replies
themselves.

**What the saved arms actually are**, `benchmarks/code_validity.py` on `benchmarks/results/replies`:

| arm | lang | blocks | compile | aborted on a fatal | truncated | fully diagnosed | errors / 100 lines |
|---|---|---:|---:|---:|---:|---:|---:|
| FP4 | C++ | 4 | **0 of 4** | 3 of 4 | 2 | 1 | 33.6 |
| 3-bit searched + weighted | C++ | 3 | **0 of 3** | 1 of 3 | 3 | 0 | n/a |
| FP4 | Python | 10 | **2 of 10** | — | 0 | 10 | 2.2 (floor) |
| 3-bit searched + weighted | Python | 5 | **0 of 5** | — | 1 | 4 | 2.0 (floor) |

**Not one C++ block from either bank compiles.** The block this document called FP4's one clean block in four
is the 21-line block in seed 20260919 turn 3; it contains `#include <s>` and fails with
`fatal error: 's' file not found`.

**Three things follow, and the third is the one that matters.**

**The old error density measured which arm gave up first.** A block that aborts on its first mangled include
contributes one diagnostic; a block whose includes happened to survive contributes every diagnostic in the
file. Three of FP4's four blocks aborted and only one of the 3-bit bank's three did — so 39 of FP4's 41
errors came from its single fully diagnosed block, while the 3-bit bank's 131 came from two. **8.9 against
31.6 is very largely that asymmetry and not a quality difference.**

**Stratified onto comparable blocks there is nothing left to compare.** FP4 has exactly one block that
reached the end of the file and was not cut off at the token cap; the 3-bit bank has none, because all three
of its C++ blocks were truncated. The gate cannot separate these two banks on this evidence, in either
direction.

**`ast.parse` has the same flaw and it hid a worse result.** It raises on the first `SyntaxError` and never
sees the rest of the file, so "Python blocks score 2.2 errors per 100 lines on *both* banks, therefore Python
is not the discriminator" was reading a per-file indicator as a defect count. On the parse rate **FP4 is 2 of
10 and the 3-bit bank 0 of 5**. And the FP4 blocks are not marginal: seed 20260919 turn 2 carries
`c_csv_file_path`, `utfutf-8`, `csv.Dreader`, `utfutf-utf8` and `__name __` in twenty lines, and scored 1.
**FP4's own artefact rate is much higher than this document has ever recorded.**

**What survives, and what does not.**

- *Withdrawn:* 8.9, 31.6, 19.6, 25.3, 22.7, 4.9, "3.5x", "6.6x", "4.6x", "15.2 errors against 2.3", and
  "Python is not the discriminator". The 2-bit and `mx.quantize` 3-bit replies were not kept, so their
  numbers cannot be re-scored and must simply not be quoted.
- *Survives:* the standing decision. It does not depend on this metric. FP4 collapses on 0 of 9 free-running
  replies against the searched 3-bit bank's 2 of 9 (section 9.0); it wins top-1 by 5.3 points and the paired
  sign test beyond 5 sigma against every 2-bit bank (section 9.3.1). Two independent metrics, both intact.
- *Survives with its reason replaced:* lever 0 is still closed. Its closing condition was written in terms of
  a number that turns out not to mean anything, but the bank it was testing collapsed on a third of its long
  turns and produced no compiling code, and that is sufficient.

**The general lesson, and it is the fourth time this document has had to write a version of it.** A gate that
cannot fail loudly will fail quietly. `check_cpp` had no test, returned a list whose emptiness was read as
success, and ran for two sessions producing the headline number in a standing decision. **Every gate needs a
fixture that it is known to fail**, and a metric whose denominator is not fixed — errors per 100 lines,
over whichever blocks happened to be scoreable — will drift into measuring its own denominator.

### 7.4.2 The corpus the repaired gate needed, built 2026-09-19

Repairing the checker did not repair the evidence. Items 1 to 3 below were written as a to-do list and then
done the same day; item 4 is still open.

**`benchmarks/coding_tasks.json`** is twenty independent single-turn tasks: ten C++ programs, eight Python
programs and two snippets, covering complete programs, edits to supplied code and bug fixes, on subjects
other than file conversion — an LRU cache, a thread pool, interval merging, an INI parser, a retry decorator,
an atomic-write context manager, an argparse CLI, a mutable-default bug, an off-by-one binary search. Each
task declares the language it must be answered in and whether it is a complete program.

**`benchmarks/coding_quality.py`** runs them against one bank in one process with `rt.reset()` between tasks,
so a collapse cannot poison the next task and every bank enters every task from byte-identical context. The
multi-turn conversation test measures something else and `repetition_quality.py` keeps it.

Four things it does that the old arrangement did not:

1. **Every case is accounted for.** No code, wrong language, an untagged fence, a reply that stopped at the
   token cap — each is recorded as that rather than leaving the denominator.
2. **The run records what produced it.** An immutable directory per run holding the replies, a per-case row
   and a manifest: git SHA and dirty state, bank path and format, expert bytes, budget, every `CACHALOT_*`
   variable, all sampling parameters, the corpus hash, the seeds, and planned against completed cases.
3. **An unfinished run refuses to be scored.** The manifest is written before generation with
   `complete: false` and rewritten at the end. `code_validity.py` refuses a run whose manifest says it did
   not finish, or whose completed count falls short of its plan, unless `--allow-incomplete` is passed — and
   then it labels the arm `INCOMPLETE`. A guarded arm killed at its timeout leaving a result that looks whole
   is exactly what put a 0.6 in this document for a day.
4. **A snippet is scored as a snippet.** Tasks that ask for a fragment are told to omit their includes, so
   compiling one alone yields a page of `use of undeclared identifier 'std'`. They are counted in their own
   column and excluded from the compile rate, the fatal rate and the density. The smoke run showed why: one
   20-line snippet contributed 18 of 29 C++ diagnostics and moved the density from 14.9 to 31.5.

**Smoke-tested end to end on FP4**, three tasks, one seed, at a 24 GiB budget: all three finished on `stop`
with one correct-language block each and nothing truncated. Replies are much shorter than the old
conversation turns — 505, 595 and 127 tokens — because the tasks are scoped, so the full corpus at two seeds
costs about **1.7 hours** on FP4 rather than the nine a worst-case cap suggests.

**Still open, and it is item 4 from the old list.** Behavioural tests on the programs that compile, and
**successful tasks per wall-clock hour** as the product metric. Compiling is necessary and not sufficient,
and tokens per second alone rewards a fast stream of code that does not build. Nothing here measures whether
a program that builds also does what it was asked.

### 7.4.3 A second collapse mode, which `max_run` cannot see

**Found 2026-09-19 while smoke-testing the new corpus, and it touches the one pillar section 7.4.1 said was
intact.**

`repetition_quality.py` calls a reply collapsed when `max_run` -- the longest span covered by a k-gram
repeating **back to back** -- reaches 24 tokens. That detects the failure it was built for,
`"res.res.res.res..."`. It cannot detect a *paraphrased* loop, where the model writes bad code, notices,
apologises, and tries again in slightly different words. Nothing repeats exactly, so `max_run` stays small.

The new corpus produced one on its third task. `cpp-thread-pool`, FP4, 503 tokens, penalty on: **eighteen**
fenced C++ blocks, every one of them a mangled include list, separated by "I need to correct that", "I
apologize for the repeated errors", "I'm clearly stuck in a loop". **Its `max_run` is 7.** The model states
that it is looping and the detector scores it clean.

It is not new to the corpus. Two of the nine saved FP4 replies behind section 7.4's table do the same thing:

| saved reply | fenced blocks | `max_run` | what it is |
|---|---:|---:|---|
| `lc_fp4 … seed20260917_turn2` | 6 | 19 | four self-corrections, ending "I clearly need to reset" |
| `lc_fp4 … seed20260919_turn2` | 7 | 4 | seven, ending "I clearly cannot produce clean code in this response" |

Both are under the threshold of 24. **So "FP4 collapses on 0 of 9 replies" means "0 of 9 exact k-gram
loops", and there were at least two loops of the other kind in the same nine files.**

**What this does and does not change.**

- It does **not** re-rank FP4 against the 3-bit bank. The retry-phrase count is 2 of 9 on FP4 and 0 of 9 on
  the 3-bit bank, which looks bad for FP4 — but the 3-bit bank's replies died early *in exact loops* (two of
  them at 285 and 157 words), so they had far less opportunity to retry. The comparison is confounded in
  FP4's favour and against it at once, and nothing here separates them.
- It does mean the standing decision's free-running evidence is **narrower than recorded**. Section 2 rests
  on collapse rate, top-1 and the paired sign test. The collapse rate now covers one of two known failure
  modes; the other two metrics are teacher-forced and, by operating rule 4, structurally blind to both.
- It does **not** say FP4 is unusable. It says nobody has measured this mode on any bank.

**The frequency penalty does not stop it, and should not be expected to.** A retry loop is semantic, not
lexical: each apology is differently worded, so a per-token frequency penalty has almost nothing to bite on.
Section 9.9's "62 % to 12 %" is about exact loops and stands; it is not a claim about this mode.

**Do not fix this by adding a phrase list to the gate.** Matching "I apologize" is a screen, it is
English-specific, and it will be gamed by the next model that apologises differently — this document has
been wrong five times about screens. The robust signal available today is that the trigram rate separates
these cleanly (57.5 % on the thread-pool reply against 34.8 % on a healthy one from the same run), which
is awkward, because section 7.4 tells the reader to judge by `max_run` and **not** by the trigram rate. That
advice was right for exact loops and is wrong for these. **A collapse metric needs both**, with the
threshold for each set on replies that have been read.

**Measure it before believing any of it.** Three reply pairs is not a rate; rule 5 asks for four seeds. The
corpus run in flight is 40 cases and is the first sample large enough to put a number on this.

### 7.4.4 Where the corruption lands, and a cheap experiment nobody has run

**Diagnosis, not a metric. Read the warning at the end before using any number here.**

The failures in the new corpus cluster somewhere specific. Across the first six C++ replies, **26 of 70
`#include` lines are malformed**, and the dominant shape is a **dropped `<`**:

    #include escaping>          #include stdio.h>        #include unordered_map>
    #includequeue>              #include>                #include <ioman double>

That is fatal in a way an ordinary typo is not: clang stops at the first bad include (section 7.4.1), so one
dropped character costs the whole file and hides every other defect behind it.

Counting the same thing on the saved arms **separates the banks**, on a signal that is immune to the two
problems that broke the compile comparison — it does not care whether the block was truncated at the token
cap, and it does not care whether the compiler gave up:

| arm | `#include` lines | malformed | dropped `<` |
|---|---:|---:|---:|
| FP4 | 27 | **5 (19 %)** | 3 |
| 3-bit searched + weighted | 43 | **28 (65 %)** | 27 |

Python `import` lines are essentially clean on both (0 of 45 and 1 of 24), **which does not mean Python is
safe** — the same FP4 replies carry `utfutf-8`, `csv.Dreader`, `c_csv_file_path` and `__name __` in their
bodies, and 8 of 10 fail to parse (section 7.4.1). The corruption is general. The include line is simply
where it is densest and where one hit is fatal.

> **Why this is not in `benchmarks/` and must not be quoted as a gate result.** The pattern was written
> *after* looking at the replies, and fitted on nine of them per arm. That is a screen built on its own test
> set, and this document has been wrong about screens five times — most recently in section 9.12.1, where an
> offline table gave a count and hid its price. The 19 % against 65 % is a **hypothesis**. To use it,
> pre-register the pattern, then score the 40-case corpus, which was generated before the pattern existed.
> The script lives in the session scratchpad on purpose.

**The cheap experiment this suggests, and it is much cheaper than reference fixtures.** A 37 % failure rate
on one highly predictable token is more consistent with a systematic numerics fault than with quantization
blur, which should degrade everything roughly evenly. That can be tested **without generating anything**:
teacher-force the model over a file containing correct `#include <iostream>` lines and read the logits at
the `<` position. If FP4 ranks `<` far below where a clean path would, the fault is visible in one forward
pass with no sampling, no collapse and no compiler involved — and `nll_expert_precision.py` already has the
teacher-forced machinery to do it. `benchmarks/token_rank_probe.py` **is written and has not been run** -- the GPU was busy generating the
corpus. It teacher-forces a text dense in correctly spelled `#include <...>` lines, reports where the model
ranked the token that should follow `#include`, and carries Python `import` lines as a within-run control:
if `<` is mis-ranked and `import` is not, that is a fact about one token rather than about the whole model.
Run it on the production path and again with `--experts fp4`-style dense substitution to separate the
runtime from the bank.

It is an hour, and it is the first concrete argument this project has for the independent-reference work the
2026-09-19 review asked for: **run it first, and escalate to pinned reference fixtures only if the logits
look wrong.** Note what it cannot tell you: it answers "does the model know", not "does the model emit". A
token ranked 1 that still comes out wrong under sampling is a different bug, in the sampler.

### 7.4.5 Greedy decoding still corrupts, so sampling is not the cause

**Measured 2026-09-19, six C++ tasks at temperature 0 with the frequency penalty
off.** Greedy decoding takes the single most likely token at every step. If the
artefacts were sampling noise — an unlucky draw from a slightly blurred
distribution — greedy would remove them. It does not.

| task | tokens | finish | malformed / total `#include` |
|---|---:|---|---:|
| cpp-lru-cache | 529 | stop | **1 / 4** |
| cpp-matrix-transpose | 427 | stop | **1 / 6** |
| cpp-ini-parser | 355 | stop | **1 / 5** |
| cpp-template-stack | 649 | stop | **1 / 4** |
| cpp-csv-to-json | 2000 | length | 988 / 993 |
| cpp-thread-pool | 2000 | length | 988 / 992 |

**Six of six greedy replies contain at least one malformed include**, and the
two that ran to the token cap were looping on `#include>` itself — which is
what the anomalous 7.2–7.6 tok/s on those two arms was, against a normal 2.7:
a tight loop re-reads the same experts and the hit rate goes up.

**A greedy loop on `#include>` means that after `#include`, the argmax token is
`>`.** Deterministically, with no sampling anywhere in the path. Quantization
blur perturbs a distribution; it does not usually survive `argmax` on a token
this predictable, and it certainly does not do so repeatedly at the same
construction.

**And it is not only the delimiter.** From the greedy `cpp-lru-cache` reply:

    #include <iostream>      correct
    #include <list>          correct
    #include unordered_map>  the '<' is gone
    #include <utility>       correct
    ...
    explicit LRcache(size_t capacity)      LRUCache -> LRcache

Two well-formed includes, one with a dropped delimiter, one well-formed, and a
mangled identifier in the next declaration. The shape is **single-token drops at
high-confidence positions**, not general noise, and the same shape appears in
the 2026-09-18 saved replies as `std std::string`, `utfutf-8`, `csv.Dreader`
and `__name __`.

**What this establishes and what it does not.** It removes sampling from the
question entirely: the fault is in the numerics, the kernels or the weights.
It does **not** yet say which, and it does not say whether the reference
implementation does the same thing — `docs/HANDOFF.md` section 7.4.4's probe
and the OpenRouter reference arm are what separate those. But it does mean the
frequency penalty, the temperature and the sampler are all cleared, and any
future work that starts by tuning them is wasted.

**It also makes the probe trivial to interpret.** `benchmarks/token_rank_probe.py`
teacher-forces correct `#include <...>` lines and reads the rank of the token
that should follow `#include`. Greedy already tells us the argmax is wrong at
that position in real generation; the probe says by how much, and whether the
same happens on a dense-FP4 substitution path that bypasses the expert bank.

### 7.4.6 The reference arms are clean. The fault is in this runtime.

**Measured 2026-09-20. This is the section that closes the question sections 7.4.4 and 7.4.5 opened,
and it is the reason the standing decision in section 2 has changed.**

The 40-case corpus was run against a hosted endpoint serving the same model, with the provider pinned and
fallbacks off, by `benchmarks/run_reference.py`. That script is run A, the diagnostic: bare API calls, no
system prompt, no tools, no content retries, reasoning disabled, and the pack's own settings — temperature
0.6, top_p 1.0, max_tokens 2000, frequency_penalty 0.2, both seeds.

Two reference arms were run rather than one, so that the FP4 expert format could be separated from the
serving stack, and a third arm — the Hermes harness on the same endpoint — was scored alongside them:

| | Cachalot | run A, `relace/fp4` | run A, `deepinfra/fp8` | run B, Hermes harness |
|---|---:|---:|---:|---:|
| C++ blocks that compile | 0 / 42 | **20 / 20** | **20 / 20** | 20 / 20 |
| aborted on a fatal include | 8 / 42 | 0 / 20 | 0 / 20 | 0 / 20 |
| Python blocks that parse | 5 / 26 | **18 / 18** | **18 / 18** | 18 / 18 |
| `#include` lines malformed | 49 / 154 (32 %) | **0 / 102 (0 %)** | **0 / 103 (0 %)** | 0 / 112 (0 %) |
| `import` lines malformed | 4 / 57 (7 %) | **0 / 32 (0 %)** | **0 / 31 (0 %)** | 0 / 44 (0 %) |
| C++ errors per 100 lines | 42.2 | 0.0 | 0.0 | 0.0 |

All four arms are 40 of 40 cases. Both reference arms finished `stop` on every case and were served by
exactly one provider — `{'Relace': 40}` and `{'DeepInfra': 40}` — which is what `--provider` is for. The two
seeds produced twenty distinct replies per arm, so they are two genuine samples per task.

**Greedy, on the same six tasks section 7.4.5 used, at temperature 0:**

| | Cachalot greedy | run A `relace/fp4` greedy |
|---|---:|---:|
| C++ blocks that compile | 0 / 6 | **6 / 6** |
| aborted on a fatal include | 4 / 6 | 0 / 6 |
| `#include` lines malformed | 1980 / 2004 (99 %) | **0 / 34 (0 %)** |

**What this closes.** On RUN.md's own decision table this is "reference clean, Cachalot corrupt", and three
competing explanations are now dead:

- **Not FP4.** The `relace/fp4` arm is FP4 and is clean at 0 %. The checkpoint's `expert_dtype: fp4` is not
  the cause, and every sentence in this document that treated FP4 as an accepted quality cost is wrong.
- **Not the harness.** Run A has no harness at all. Run B's clean result was never the harness repairing
  anything: every assistant turn of all 40 Hermes sessions was scanned, including the first draft before any
  tool ran, and **zero malformed include lines appear anywhere**. Nine of the forty answered in one shot with
  no tool call. The harness had nothing to fix.
- **Not sampling.** Greedy on both sides keeps the gap at 99 % against 0 %, which section 7.4.5 had already
  established one-sidedly.

**What is left.** The defect is in Cachalot's own path: expert streaming, the FP4 expert kernel, the
prefetch and cache machinery, or detokenization. Its shape is a single-token drop at a high-confidence
position — 38 of 49 malformed lines in the sampled run and 1977 of 1980 in the greedy run are "missing
opening delimiter". That is not a quantization blur and it is not a model tendency. It is a bug.

**What this arm does not prove.** Neither Relace nor DeepInfra publishes whether its weights are the
official checkpoint byte for byte, so this is not a byte-identical pairing. It does not need to be: a 32 %
to 0 % gap on one dropped delimiter is not a weights difference.

**Reproducing the tables.**

```bash
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py \
  benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/ \
  ~/cachalot-runA-relace-fp4-scored ~/cachalot-runA-deepinfra-fp8-scored ~/cachalot-runB-hermes-scored
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/include_integrity.py \
  benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/ \
  ~/cachalot-runA-relace-fp4-scored ~/cachalot-runA-deepinfra-fp8-scored ~/cachalot-runB-hermes-scored
```

Each `-scored` directory carries `manifest.json` and `rows.json` so `code_validity.py` treats it as one arm
and honours `expect_compiles`, which keeps `cpp-string-split-snippet` — a task that forbids includes and
`main` — in the snippets column rather than counting it as a failed compile.

**Pinning a provider is fiddly, and the obvious choices do not work.** With `allow_fallbacks: false`,
OpenRouter returns `404 No endpoints found` if the pinned provider does not support every parameter in the
request. The first-party `deepseek` endpoint does not list `seed` and 404s. `sail-research/fp4` lists every
parameter and returned HTTP 429 on every attempt through five retries. `relace/fp4` also omits `seed` from
`supported_parameters` but accepts and serves the request; it is what the FP4 arm used, and the seed is
presumably ignored, which is why the two seeds were checked for being distinct.

**Defects in the returned Hermes run, all corrected before scoring.** Two of forty sessions were never
exported — `cpp-matrix-transpose` pass 1 and `cpp-ini-parser` pass 1 — and were recovered from
`~/.hermes/profiles/test-ds4-1/state.db`. Unrecovered, the denominator would have read 38 and the arm would
have looked complete, which is the third time a short run has nearly put a wrong number in this document.
Two files also carried the wrong task id, and the whole set was named `<task>-seed<n>.json` where the scorer
splits on `_seed` and reads `.txt`.

### 7.4.7 The defect is in-context copying, and six more suspects are cleared

**Measured 2026-09-20, second session. Raw tables and procedure in `HANDOFF-2026-09-20-quality.md`.**

Section 7.4.6 put the fault inside this runtime. This section narrows it, and retires the framing that the
next-session prompt v13 was built on.

**What the defect actually is.** Teacher-forced probes separate predictions that are open-ended from
predictions that are not. Choosing which header a file includes is fair to be uncertain about. Finishing a
word already begun is not, and copying a word the text spelled out ten tokens ago is not remotely.

| class | n | rank 1 | mean logprob | worst rank |
|---|---:|---:|---:|---:|
| continuation, first occurrence | 28 | 86 % | -0.699 | 19 |
| continuation, **repeat of an earlier word** | 101 | **51 %** | -3.578 | **23,989** |
| everything else | 1,371 | 88 % | -0.595 | 3,703 |

First-occurrence continuation is healthy and level with the baseline. Only repeats collapse. The model is
not failing to retrieve, it is retrieving the wrong thing: at `...the file lists allocator_t` it prefers
`' unordered'` and `' iterator'`, words from neighbouring sentences. This is a within-run control and needs
no reference arm — no correct model fails to copy a word it emitted ten tokens ago.

That one statement covers every artefact the project has collected, with no separate story per case:
`#include <stdex>` and `#include iostream>`, `LRUCache` written `LRcache`, `map_._.end()` with a duplicated
`_.`, `std std::string`, `utfutf-8`, `csv.Dreader`, `__name __`. Drops and duplications are both what
mis-retrieval looks like.

**What is cleared, and how.**

| suspect | verdict | evidence |
|---|---|---|
| detokenization | cleared | 34/34 `#include <x>` lines exact: batch, per line, and one token at a time |
| the sampler, again | cleared | greedy manifest records `temperature 0.0`, `frequency_penalty 0.0`; that path is plain `argmax` |
| the prefill path | cleared | 540 tokens batched vs fed one at a time: 60 % vs 65 %, within noise |
| depth and position | cleared | byte-identical block at token 568 scores 100 %; no decay bucketing 640 positions |
| **routed-expert kernel, streaming, cache** | **cleared** | `nll_expert_precision.py` dense fp32 vs production on the same FP4 weights: mean +0.0255 ± 0.0141 nats, **median +0.0000**, 96.0 % greedy agreement, same argmax at every failure |
| **Engram** | **cleared** | nine checks against the shipped reference; prime sums 384,006,168 and 384,016,682 match `engram_num_embeddings` exactly; gate math and `norm_eps` identical |
| fused attention decode, fused FP8, bf16 head | cleared | all three off together: repeat continuation 51 % against 51 %, unchanged |
| copy distance | not the variable | controlled probe, same identifiers, recall at 8/32/96/288 tokens: 73 %, 79 %, 62 %, 75 % against an 88 % run baseline |

**What this does to section 9 and to v13's plan.** Items 1 to 4 of the v13 bisection — the custom Metal
kernels and the prediction and prefetch family — are answered without running the corpus once. The
10-to-20-hour slow-path run that v13 made Job 2 is not the right next move; it was designed to bisect
optimizations, and the optimizations are not the fault.

**What is left, stated without overreach.** The trunk. Precise retrieval of one earlier position is what
attention governs, and V4.1's attention is not plain attention: `index_source_layer_ids = [2, 8, 14, 20, 24,
28, 32, 36]`, `kv_source_layer_ids = [2, 8, 14, 20]`, `candidate_source_layer_id = 20`, `index_topk = 512`,
`sliding_window = 128`, `compress_ratios` of 2 for layers 2-19 and 1 for 20-39. An `Indexer` scores
compressed positions and keeps 512 of them, behind a two-level candidate pre-filter. A selection that is
approximately right leaves fluency intact and breaks exact copying, which is the measured shape — but the
flat distance curve argues against the simplest version of that, because `index_topk` should bite at long
range rather than uniformly. **Which component causes it is not established and should not be asserted.**

**Update, later the same day: that job was done, and then the attention was measured.** The decode path
matches the reference at about forty points — selection, arithmetic, the KV it attends over, its
quantization, the MoE gate, Engram, and the hyper-connection machinery. And at a failing copy,
`benchmarks/attention_mass_probe.py` shows layer 3 putting **0.928 on position 321, the exact token the
model must emit**, alongside 0.845 on the `'_t'` that matches the current one — an induction circuit firing
correctly — while the model ranks that token **17,935th**. Two tokens earlier, at position 331, the same
machinery copies correctly at rank 0.

So retrieval is not the fault; something between the attention output and the logits discards what was
retrieved, at some positions and not others. It is not a static break downstream, not token identity, and
not the cache contents. `docs/HANDOFF-2026-09-20-quality.md` carries the tables and the next instrument.

**The superseded note, kept because it explains the method that worked.** `src/cachalot/model/attention_compressed.py` is 34 KB implementing
that scheme and had never been read beside `inference/model.py`. Engram was cleared by exactly that method
in under an hour. Already compared and matching: `get_window_topk_idxs` in prefill and decode, and the whole
Engram path. Uncompared: the indexer's scoring and top-k, the compressor's partial-group state, the
candidate pre-filter, and the RoPE positions the compressed latents are rotated with — the reference notes a
latent stands for the first token of its group at position `j * ratio`, with a decode-time index of
`start_pos + 1 - ratio`.

### 7.4.8 Root cause: the hyper-connection residual mix was transposed

**Found and fixed 2026-09-20. Tables in `HANDOFF-2026-09-20-quality.md`; the fix is commit `e37b73c`.**

`Block.hc_post` writes a sub-layer's output back into the four residual streams and mixes the incoming
streams through `comb`. The official implementation, `model.py:962`:

```python
torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
```

broadcasts to `elem[i, j, :] = comb[i, j] * residual[i, :]` and sums over `i`, so output stream `j` is
`sum_i comb[i, j] * residual[i]`. **comb is contracted over its first index.** Both of this runtime's
implementations contracted the second index — `comb @ residual` rather than `comb.T @ residual` — and did so
identically, the MLX path by summing the wrong axis and the Metal kernel by indexing `comb[h * hc + j]`
where it needed `comb[j * hc + h]`.

Verified numerically against explicit loops transcribed from `model.py`: ours matched `comb @ residual` to
6e-8 and differed from the reference by 2.0. No compensating transpose exists anywhere — `comb` is built as
`mixes[2*hc + j*hc + k]` with `j` the row in all three files that touch it, and the sinkhorn normalizes rows
then columns in that layout.

**The gate, 40 of 40 cases** — `benchmarks/results/coding/hcfix/`, scored against the reference arm:

| | corrupt | **fixed** | reference |
|---|---:|---:|---:|
| C++ blocks that compile | 0/42 | **20/20** | 20/20 |
| aborted on a fatal include | 8/42 | **0/20** | 0/20 |
| Python blocks that parse | 5/26 | **18/18** | 18/18 |
| `#include` lines malformed | 49/154 (32 %) | **0/101 (0 %)** | 0/102 (0 %) |
| `import` lines malformed | 4/57 (7 %) | **0/31 (0 %)** | 0/32 (0 %) |
| C++ errors per 100 lines | 42.2 | **0.0** | 0.0 |

Throughput was 2.1-2.5 tok/s during the gate, unchanged: the fix reorders a contraction and costs nothing.

**Result of the fix on the probe**, same 1,500 positions:

| | before | after |
|---|---:|---:|
| copying a word already spelled out | 51 % | **99 %** |
| its mean logprob | -3.578 | **-0.027** |
| its worst rank | 23,989 | **2** |
| everything else | 88 % | **95 %** |
| all positions | 85 % | **95 %** |
| the token after `#include` | 84 % | **100 %** |

**Why it survived months of work, which is the part worth remembering.** `comb` comes out of a sinkhorn
normalization and is close to doubly stochastic, so every residual stream receives about the right *total*
weight whichever way the matrix is applied. The model stayed fluent: 2.4 perplexity, 88 % top-1, prose and
code that read correctly. A transpose does not change how much information flows, it changes **which stream
it lands in**, and that only matters for a prediction depending on one specific earlier fact. Copying is
that prediction. Every aggregate metric this project owned — NLL, perplexity, top-1, hit rate — was blind to
it by construction.

It also defeated this project's own screens twice. The fused-decode screen cleared "the attention kernel"
because both implementations carried the same error, so switching between them moved nothing. And the
dense-versus-runtime expert comparison cleared expert arithmetic while leaving the shared code untouched.
A/B switching can only find a defect that differs between the arms.

**What this retires.** Every conclusion in this document that attributed quality loss to FP4, to the expert
bank, or to a speed optimization. Sections 9.0 and 9.3 chased quality through quantization against a defect
that was never in the weights, and section 7.4.6's "the defect is in this runtime" was right for a reason
nobody had guessed.

### 7.5 The 3-bit bank's gate, 2026-09-18

Three metrics, all against the 2-bit g128 bank it replaces, each on the protocol that metric was defined for.

| gate | 2-bit g128 | **3-bit g64** | how it was read |
|---|---:|---:|---|
| top-1, production path, 512 tokens | 44.5 % | **49.8 %** | +5.3 points |
| paired NLL against the 2-bit bank | — | median **−0.0208**, better on 58.4 % | **sign z +3.80** — the mean, at −0.0443 +- 0.0457, is z −0.97 and says nothing |
| free-running collapse rate, penalty on | 1/8 (12 %) | 1/8 (12 %) | **unchanged, and expected** |
| ~~syntax errors per 100 generated lines~~ | ~~22.7~~ | ~~**4.9**~~ | **withdrawn 2026-09-19** — the checker read a grep, not the compiler's exit status. Section 7.4.1. |
| decode, 44 GiB budget, interactive | 6.3–7.1 tok/s | **4.8–5.0 tok/s** | the price |
| expert hit rate, chat session | 90.7 % | 83.3 % | 1.56x the bytes per expert |

**The collapse rate being unchanged is a result, not a null.** It confirms the two failures are independent:
the bank causes the token-level artefacts, sampling dynamics cause the loops, and neither fix substitutes for
the other. Ship both.

**The error counts differ in shape, not only in size.** The 2-bit arm's worst two blocks carried 81 and 50
errors -- files that are write-offs rather than files with typos -- while the 3-bit arm's worst block had 2.

**And our bank ties the download.** Against oQ3e on the same 512 tokens: paired median +0.0017, better on
46.3 %, sign z −1.68. Eight minutes of building matches a 331 GB restore, so section 3.1's "restoring is a
331 GB copy at 1 GB/s" is now a note of historical interest.

### 7.6 The 2-bit bank's gate, re-run through the fixed runtime, 2026-09-20

**The 2-bit bank is not worse than FP4. It never was, and three sessions concluded otherwise because the
transposed residual mix was in every arm.** The 40-case corpus, generated on the 2-bit g128 bank at the same
budget, corpus, sampling and seeds as the FP4 arm in section 7.4.8, scored against the same hosted reference:

| 40-case corpus | FP4, fixed | **2-bit g128, fixed** | hosted reference |
|---|---:|---:|---:|
| C++ blocks that compile | 20/20 | **20/20** | 20/20 |
| aborted on a fatal include | 0/20 | **0/20** | 0/20 |
| Python blocks that parse | 18/18 | **18/18** | 18/18 |
| `#include` lines malformed | 0/101 (0 %) | **0/94 (0 %)** | 0/102 (0 %) |
| `import` lines malformed | 0/31 (0 %) | **0/29 (0 %)** | 0/32 (0 %) |
| C++ errors per 100 lines | 0.0 | **0.0** | 0.0 |
| decode during the run, median of 40 | 2.28 tok/s | **4.47 tok/s** | — |

Every column is equal. The one C++ case that produces diagnostics, `cpp-string-split-snippet`, produces them
in all three arms including the hosted reference — it is a snippet the corpus asks for without headers, and
`code_validity.py` counts it as a snippet rather than a failure in every arm.

`benchmarks/results/coding/q2g128-v17/`, 40 of 40 cases, `code_validity.py` and `include_integrity.py`.

**The screen that justified spending the two hours.** `benchmarks/continuation_rank.py` on the recovered
repeat-identifier probe, the same 1,500 positions on both banks:

| bank | continuation, first | continuation, **repeat** | repeat worst rank | everything else |
|---|---:|---:|---:|---:|
| FP4 | 86 % | 99 % | 2 | 95 % |
| 2-bit g128 | 86 % | **99 %** | **1** | 95 % |

Indistinguishable, and the 2-bit bank is marginally better on the worst case. That is a screen and it was
read as one: it decided whether the corpus run was worth starting, and the corpus run is the gate.

**What this retires.** Section 7.5's "+5.3 points of top-1 and the price is 6.3-7.1 against 4.8-5.0 tok/s"
ranked the 3-bit bank above the 2-bit one on a metric that section 7.4.8 proved blind to the defect that was
actually producing the artefacts. Section 9.3's "recover the quality the 2-bit bank cost" was aimed at a cost
that, on the gate the project now trusts, is zero. Section 8's conclusion that quality lives in the weights
stands only for the NLL numbers it was computed from, and those numbers were never the thing a user sees.

**What it does not retire.** The paired NLL and top-1 differences between banks are real measurements and they
have not changed; what changed is that they do not predict the gate. The 2-bit bank still scores 44.5 % top-1
against the 3-bit bank's 49.8 % (section 7.3), and it still compiles 20 of 20. Both are true, and the second
is the one the standing decision is about.

**Two conditions on this result, stated so nobody quotes it past them.** The replies are slightly shorter on
the 2-bit bank — 1,423 C++ lines against 1,447, 1,128 Python lines against 1,246, 94 include lines against
101 — which the gate does not penalize and which nothing here explains. And this is one corpus at one
sampling setting; the collapse-rate question (section 9.9) is measured separately and was not re-measured
here.

## 8. What was learned about quantization

### 8.1 MLX's affine fit wastes a level at 2 bits

`mx.quantize` does not fit the affine grid to a group's range. Measured by quantizing known groups and reading
back what it produces, its fit is max-abs symmetric:

    scale = +-max|w| / 2**(bits-1)        bias = -2**(bits-1) * scale

so the representable values are `scale * (q - 2**(bits-1))`. At 4 bits this costs little. At 2 bits the four
levels are `{-2, -1, 0, +1} * scale`: one level sits where the data never goes, and one side of the
distribution is clipped at half its range, which is why the sign of `scale` flips from group to group — the fit
puts the longer tail on the side that has three levels.

The stored format is only `w ~= scale * q + bias` per group, and `mx.dequantize` and `mx.quantized_matmul` take
the scales and biases as ordinary arrays, so **a better fit needs no runtime change at all** — only a packer.
`benchmarks/quant_affine.py` has one: `pack_2bit` (sixteen values per `uint32`, least-significant field first,
pinned against `mx.quantize` by `tests/test_quant_affine.py`), `fit_search` (a 5x5 grid over shrunk range ends,
keeping the lowest-squared-error pair per group) and `fit_minmax` (the textbook fit, which is **worse than
MLX's** at 2 bits — with four levels, clipping outliers beats spanning them).

At 2 bits and group 128 the searched fit is the difference between a working model and a destroyed one:
2.3234 nats against 17.5460.

**Generalised to every width on 2026-09-18.** `pack_bits` writes MLX's layout at any width — one contiguous
little-endian bit stream per row, least-significant bit first, no padding, so a level straddles a word
boundary whenever its width does not divide 32 — and `tests/test_quant_affine.py` pins it against
`mx.quantize` at 2, 3, 4, 5, 6 and 8 bits. Every fit takes `bits` and an optional per-column `weights`, and
`quantize_affine` is the general entry point. The searched fit helps *more* at higher widths, not less: 26 %
of output error at 3 bits against 17 % at 2. **None of which was enough to make a 3-bit bank usable** —
section 9.0 is the whole story and it closes the lever.

### 8.2 Prediction width is a function of expert size

Every predicted expert that is not resident is a read, so recall and bytes rise together and the two settle at
a saddle. On the 15.48 MiB 3-bit bank the saddle was at top-3. On the 9.49 MiB 2-bit bank it is at **top-6**,
because the floor the extra reads raise stays under compute: six runs per side, interleaved both ways, top-3
median 192.5 ms/token and top-6 median 182.5, with every top-6 run faster than every top-3 run. `PREDICT_TOPK`
in `src/cachalot/model/moe_layer_metal.py` now defaults to 6 and both saddles are recorded in the comment
there. **Re-measure this whenever the bank changes.**

### 8.3 The cheap screen, and its limit

`benchmarks/expert_requant_error.py` re-quantizes sampled FP4 experts and reports the relative error of the
routed-expert output on random unit inputs. It costs seconds where the gate costs an hour, and it correctly
ranked the formats within one quantizer. **Across quantizers it is wrong**: it puts the calibrated oQ3e
download ahead of our naive 3-bit on both of its metrics (0.296 against 0.352), and the gate puts the naive one
ahead by 0.030 nats. Use it to choose which formats deserve an NLL run; never as a substitute for one.

It also settled two design questions cheaply. Error is spread evenly across layers (5 % spread, no gradient
from layer 0 to 39) and across the three projections (0.276 for w1, 0.309 for w2, 0.276 for w3), so no
per-layer or per-projection mixed bank is indicated; raising w2 alone to 3 bits buys 12 % of error for 13 %
more bytes. That is convenient, because `storage/index.py` raises on mixed expert quantization by design.

**And the limit is worse than "across quantizers", 2026-09-18 session 4.** The screen was made *more*
faithful — `quant_fit_screen.py --activations` scores on the vectors the model really hands its experts
instead of random unit ones — and it still failed. It ranked a searched, activation-weighted 3-bit fit 28 %
better than the fit the retired bank used, correctly as far as output error goes, and the bank built from it
produced no compiling C++ block and collapsed on 2 of 6 long turns against the retired bank's own record.
(The "31.6 against 19.6" this paragraph carried is withdrawn — section 7.4.1.) **Output error is not a proxy
for usable output at this resolution, however it is measured.** Use the screen to decide which formats
deserve a bank; never to predict what a bank will do.

### 8.4 Cost of a gate run

Each gate arm re-quantizes about 20,000 experts, not 15,360, because the 8 GiB packed cache evicts. At
`mx.quantize` speed (19 ms per expert) an arm is 540 s; with a 13x13 search grid (720 ms per expert) the first
attempt was still running when the guard's 5400 s timeout killed it. The 5x5 grid costs 125 ms per expert and
about 2200 s per arm, and buys 0.0005 of relative weight error against the finer grid. If a future arm needs a
better fit, raise the timeout or build the bank and gate it with `--experts runtime`; do not raise the grid.

---

## 9. Open levers, ranked

> **Frozen 2026-09-20, unfrozen the same day.** The freeze existed because the runtime's output did not
> match a reference at any speed. Section 7.4.8 found the cause — a transposed hyper-connection mix, not a
> speed lever — and the 40-case gate is now clean at 20/20 compiling and 0/101 malformed, equal to the
> reference. **The shipped optimizations are cleared: none of them was the defect, and the fix cost no
> throughput.** This is a work queue again.
>
> Two carried conditions. The quality arguments used to open or close levers below were measured through the
> defect and several of them ranked quantization formats, so they are provisional until re-established —
> section 2. And a lever's quality check must use the repeat-copy number rather than NLL or top-1, which
> stayed at 2.4 perplexity and 88 % while the runtime could not copy a word it had just written.
>
> **The first of those conditions is discharged for the bank question, 2026-09-20, and the answer inverts
> this section's premise.** The 2-bit g128 bank was re-gated on the 40-case corpus through the fixed runtime
> and is equal to FP4 and to the hosted reference on every column, at nearly twice the throughput
> (section 7.6). The whole ranking below was computed on a token that reads 1,858 MiB; the cheaper bank reads
> 804 MiB for the same gate result. **Re-measure the anatomy before trusting any ordering here** — section
> 9's own lesson is that a lever's rank is a property of the bank.

Ranked by expected value per unit of work, with the evidence, the cost and — most importantly — the
measurement that decides each one before any code is written.

**The ranking changed again on 2026-09-19, and this time because the bank in use was finally measured.**
Every timing number behind the 2026-09-18 ranking came from the 2-bit bank. On FP4 the token is 341 ms, of
which about 320 ms is drive time and 84.6 ms is compute that hides underneath it (section 6.1). That
demotes dispatch count and closes speculation:

| lever | state on FP4 |
|---|---|
| 11, mirror striping across both drives | **shipped 2026-09-19**: −5 % decode, −7 % cold prefill, no quality change |
| 12, prefetch precision | **bounded and mostly closed**: the missing 28.5 % is the router's selection boundary, and no cheap re-use of the stale scores beats plain top-k |
| 13, predicted-load lifetime | **fixed and shipped 2026-09-19**: a completed prediction for a later layer was released before it could be used. A null on decode at lead 1, as expected, and it made 9.12.1 interpretable |
| 12.1, non-cumulative lookahead | **closed 2026-09-19**: 2.0 % worse, ranges non-overlapping — but it prices the prefetch-timing term at about 24 ms per token, which is now the case for a better predictor |
| 2, dispatch count | open, but worth close to nothing until bytes come down: 84.6 ms is already hidden |
| 1, DSpark speculative decoding | **closed**: 1.03x on measured constants, against its own 1.15x bar |
| 0, a searched fit above 2 bits | closed 2026-09-18, built and gated |
| 3, 4, 10 | closed |
| 5, 9, 11 | shipped |
| 6 | low value for interactive use |
| 7 | robustness only |
| 8 | do not start there |

**The general lesson of the day: a lever's rank is a property of the bank, not of the runtime.** Three
conclusions in this document were correct on 9.49 MiB experts and wrong on 17.93 MiB ones — the drive's spare
capacity, the irrelevance of prefetch timing, and speculation's economics. Re-measure the anatomy whenever the
bank changes, before re-ranking anything. Read sections 6.1 and 9.11 first.

**The state after 2026-09-19 is that nothing cheap is left, and this time the cheap things were looked for.**
The review found one real defect -- predictions with no lifetime -- and it is fixed, tested and a null on
decode, which is what it should be. The lever it unblocked was measured the same day and lost by 2.0 %. What
that experiment bought instead is a **price**: the prefetch-timing term is worth about 24 ms per token, 7 %
of decode, to anything that can reach two layers ahead without losing recall (section 9.12.1). That is the
first number section 9.12's "different signal" has ever had attached to it, and it is the argument for
spending a session on a predictor rather than on a knob. Mirror striping is taken. Speculation,
eviction policy, prefetch width, prefetch lead time and prefetch precision are all measured and closed, four
of them in a single session for a few hours of machine time and no code. What remains is expensive and
honest: **fewer bytes per expert at FP4 quality**, which MLX cannot express (section 9.8) and which no affine
bank above 2 bits has survived (section 9.0); **a genuinely better routing predictor**, which needs a signal
the stale vector does not carry (section 9.12); and **dispatch fusion**, which is real work on a term that is
already hidden under the drive (section 9.2) and pays only after one of the other two lands. A session that
starts by looking for another environment variable will not find one.

### 9.0 Lever 0 — a searched fit above 2 bits — **built, gated and closed, 2026-09-18**

**Where this came from.** FP4 writes C++ that mostly compiles; the 3-bit and 2-bit banks do not. That looked
like a cliff rather than a slope, and a cliff between 4 and 3 bits was suspicious enough to investigate:
section 8.1 had already shown that `mx.quantize`'s fit is max-abs symmetric and **wastes one level at every
width**, and the 3-bit bank that failed was built with it. The question was whether the cliff was the bits or
the fit.

**It is the bits.** The fit was improved as far as it can usefully be improved, the bank was built, and the
compiler said no.

#### What was built

`pack_bits` in `benchmarks/quant_affine.py` writes MLX's layout at any width, which was the blocker. MLX's
layout turned out simpler than the 2-bit special case suggested: the levels of a row are one contiguous
little-endian bit stream, filled least-significant bit first with no padding, so a level straddles a word
boundary whenever its width does not divide 32 and 32 three-bit levels occupy exactly three words.
`tests/test_quant_affine.py` pins it against `mx.quantize`'s own output at 2, 3, 4, 5, 6 and 8 bits and at
every group size the kernels accept — worth doing, because a bank written with the wrong layout fills the
shard header exactly and reads back as noise with no error anywhere.

Everything downstream is width-agnostic now, and that part is worth keeping whatever happens to this lever:
every fit takes `bits` as a keyword, `quantize_affine` is the general entry point, `build_affine_bank.py`
takes a searched fit at any width, and `quant_fit_screen.py --bits` and `nll_expert_precision.py
--experts requant` screen any width.

**Activation weighting was added and it works.** Section 9.0.1 called it the part of the "dynamic
quantization" idea this project can use, and it was untried. The premise had never been checked either, so it
was checked first: `benchmarks/capture_activations.py` wraps `moe_layer_forward` in every block module and
records the vector the decode path actually hands the routed experts. **Inside an average group of 64 columns
the most and least excited column differ by a factor of 7.9 in mean square**; across a whole layer, 73. An
unweighted fit spends the same effort on both ends of that.

Every fit now takes a `weights` array and minimizes the weighted squared error, with `_lsq_step` solving the
weighted normal equations. The weighting is per column, because that is the axis the input is contracted over
and the axis the groups run along: w1 and w3 are weighted by the recorded mean square of the MoE input, a
property of the layer; w2 by the mean square of the SwiGLU hidden that input produces *through this expert's
own FP4 weights*, which makes w2's weighting per expert. A weighting of all ones reproduces the unweighted fit
exactly, and that is pinned.

#### What the screen said, and it was not a small effect

At 3-bit group 64 on 16 real FP4 experts, scored on **recorded MoE inputs** rather than random unit vectors —
random inputs make every column equally important by construction, which is the assumption being tested:

| fit | w-err | y-err | against `search`+lsq |
|---|---:|---:|---:|
| `mlx`, the fit the retired bank used | 0.2108 | 0.3335 | +32.7 % |
| `search` | 0.1614 | 0.2560 | +1.9 % |
| `search`+lsq | 0.1587 | 0.2513 | — |
| **`search`+lsq+act** | 0.1630 | **0.2384** | **−5.1 %** |

Weight error *rises* while output error falls, which is the signature of the trade working rather than of a
generically better fit. Against `mx.quantize` the total is **−28.5 %**. The 9x9 wide grid is a null at 3 bits
(0.2588 against `search`+lsq's 0.2587) for 378 ms per expert against 143, so it was not used.

**And it transfers, which is the standing objection to anything calibrated.** Fitted on an English-prose
recording and scored on activations from a code text it keeps 82 % of what a self-calibrated weighting gets
(0.2347 against 0.2310, where the unweighted fit is 0.2465). A weighting merged from both recordings —
`capture_activations.py --merge`, which adds the second moments and pools the probes without loading the model
— comes within 0.3 % of the self-calibrated number on *both* texts, so the bank was built with that.

#### What the compiler said

`DeepSeek-V4.1-Flash-q3g64-act` — 3-bit group 64, `search-lsq`, activation-weighted from the merged
recording, 15,482,880 B per expert, 221.5 GiB — was built on the X10Pro in 47 minutes at 166 ms per expert,
and 12 sampled experts read back byte for byte through the real index. The internal SSD has 70 GiB free, so
there was nowhere else to put it; the source was read from the internal FP4 copy so the reads stayed off the
drive being written.

Gated against FP4 on the same conversation, the same three seeds and the same sampling
(`--frequency-penalty 0.2 --penalty-window 128`, 1400 tokens, 24 GiB budget), C++ blocks only:

| arm | C++ blocks | avg block | lines | **compile** | aborted on a fatal | truncated |
|---|---:|---:|---:|---:|---:|---:|
| **FP4** | 4 | 115 | 461 | **0 of 4** | 3 of 4 | 2 |
| **3-bit searched + weighted** | 3 | 138 | 414 | **0 of 3** | 1 of 3 | 3 |

Re-scored 2026-09-19 with the repaired gate. The "1/4 clean, 8.9 errors per 100 lines against 31.6" this
table carried is withdrawn: the checker never read clang's exit status, and the density was largely measuring
which arm aborted on a mangled `#include` first. **Neither bank compiles anything, and the C++ blocks cannot
separate them.** Section 7.4.1. What closes the lever is the row below.

Free-running collapse, penalty on: **2 of 9 replies (22 %)** against FP4's **0 of 9**, which is **2 of 6**
long turns against none — the short first turn of each seed never collapses on either bank. Three seeds is
below the four this document asks for a rate (rule 5), so read the long-turn count rather than the
percentage; what is not in doubt is that FP4 produced six full 1400-token replies and the 3-bit bank produced
four, having run two of them into a loop at 279 and 129 tokens.

Block lengths are comparable — 138 against 115 — so this is not the block-composition artefact that withdrew
the "4.6x" earlier the same day. Python parse rates, re-scored 2026-09-19: FP4 **2 of 10**, the 3-bit bank
**0 of 5**. (The "2.2 on both arms" this paragraph carried is withdrawn — section 7.4.1.)

Section 9.0 set its own closing condition before the bank was built: *if it lands near 19.6, bits rather than
fit are binding and this lever closes for good.* **That condition was written in terms of a number that does
not mean anything**, and it cannot be evaluated as written. The lever closes anyway, on the evidence that did
survive: the bank produced no compiling code, parsed 0 of 5 Python blocks against FP4's 2 of 10, and ran 2 of
its 6 long turns into a repetition loop where FP4 ran none. **Closed.**

#### The three things to carry forward

1. **Bits are binding above the fit.** 28 % less routed-expert output error, measured on real activations and
   shown to generalise, changed nothing a compiler can see. There is no reason left to expect a cleverer
   affine fit at 3 bits to reach FP4, and `mx.quantized_matmul` offers nothing between 3 bits and FP4's 4.25
   (section 9.0's own finding that 4-bit affine at group 128 is byte-identical to FP4 and strictly worse).
2. **A more faithful screen is still not the gate.** Section 8.3 said the screen ranks correctly within one
   quantizer and wrongly across them; section 9.3 said it is also wrong within one quantizer when the fit's
   character changes. This adds the strongest version: making the screen *more* faithful, by scoring on the
   model's own activations instead of random vectors, did not make it predictive. The map from output error
   to syntax errors is not merely non-linear, it is not a function of output error at all at this resolution.
3. **The machinery is worth keeping even though the lever is closed.** `pack_bits` unblocks any future width;
   activation weighting is a general capability that costs no bytes and no runtime; `capture_activations.py`
   is the first tool this project has for looking at what the model actually multiplies, and nothing says its
   only use is quantization.

**The bank was deleted on 2026-09-18** — 221.5 GiB for a measured negative, and it rebuilds in 47 minutes
from the command in section 13 if anyone ever wants to re-open the question. Its gated output is kept under
`benchmarks/results/replies/lc_q3g64act`, alongside FP4's matched arm, so the comparison can be re-scored
without generating anything.

### 9.0.1 Two questions asked and answered on 2026-09-18

**Can Unsloth-style dynamic quantization help?** Its premise is that some tensors matter far more than others,
found by calibration, then bits spent unevenly. **That premise is measured absent here** (section 8.3): error
spread across layers is 5 % with no gradient from layer 0 to 39, and across projections it is 0.276 / 0.309 /
0.276 for w1 / w2 / w3. Raising w2 alone to 3 bits buys 12 % of error for 13 % more bytes — break-even. The
one vendor-calibrated bank available, oQ3e, *lost* to naive `mx.quantize` by 0.030 nats at identical bits.
There is also a hard blocker: `mx.quantized_matmul` takes one width per tensor and `storage/index.py` raises
on mixed expert quantization by design. The part worth keeping is **activation-weighted fitting** — the
imatrix idea without the mixed precision — which needs no format change.

**Update, 2026-09-18 session 4: it was tried.** The premise the *mixed-precision* half of the idea rests on
is absent here, as measured above, but the weighting half is not: inside an average group of 64 columns the
mean square of the input spans a factor of 7.9. Weighting the fit by it is worth 5.1 % of routed-expert
output error on top of the searched fit, transfers across texts, and costs no bytes and no runtime — and it
made no difference the compiler could see (section 9.0). So the answer to "can Unsloth-style dynamic
quantization help?" is now fully settled: the mixed-precision premise is absent, the weighting premise is
real and the weighting does not rescue 3 bits.

**Would FP8 experts be more accurate?** No, and the checkpoint settles it: `quantization_config` reads
`{quant_method: fp8, expert_dtype: fp4}`. The trunk ships FP8; the routed experts ship **FP4**. FP4 is the
source of truth, which is why `nll_experts_fp4.json` is the reference arm every other bank is scored against.
An FP8 expert bank would dequantize FP4 and re-store identical values at twice the bytes — 551 GiB, no
quality gain, and it would not fit. The untouched FP8 question is *activations*, not weights.

### 9.1 Lever 1 — DSpark speculative decoding — **closed on FP4, 2026-09-19**

> **Closed before any code was written, which is what the arithmetic is for.** v9 of the next-session prompt
> set the bar: redo the projection on FP4's expert size, and if it still clears about 1.15x it is worth a
> session. Done with `speculation_bytes.py --expert-bytes 18800640` and `speculation_policy.py`, no model
> loaded, in minutes.
>
> On the constants this document carried it projects **1.17x** and would have opened. On the constants this
> session *measured* it projects **1.03x** and closes. The three that moved it:
>
> | constant | carried | measured 2026-09-19 |
> |---|---:|---:|
> | drive rate | 6.7 GB/s | **5.97 GB/s achieved** (section 6.1) |
> | all-resident compute | 93.0 ms (2-bit) | **84.6 ms** (FP4) |
> | baseline to beat | 342 ms | **325 ms** with mirror striping |
>
> Best confidence-gated policy: threshold 2.0, mean width 1.75, 1.62 tokens per forward, 314 ms per accepted
> token against 325. **The entire projected value of speculation lived in the gap between the drive's assumed
> and achieved bandwidth, and mirror striping has taken part of that gap directly, for hours of work instead
> of a session.** Verifying all five drafted positions is 0.67x.
>
> Reopen only if bytes per expert fall, which changes the byte term that dominates every row. The acceptance
> measurements (2.85 tokens per main forward, the confidence head's separation) are unaffected and stand; it
> is the economics that fail. Results in `benchmarks/results/speculation_policy_fp4_measured.json`.

The measurements behind it, which remain valid:


**What it actually is.** Not three multi-token-prediction layers, as this document said before 2026-09-17's
fourth session: the `mtp.*` namespace holds **DSpark**, which the model card describes as "semi-autoregressive
draft generation with confidence-scheduled verification". One main forward drafts **five** tokens, not one to
three. `src/cachalot/model/dspark_draft.py` implements it and section 21 of `HANDOFF-2026-09-17.md` describes
the mechanism.

**Acceptance, measured.** Four prompts, 384 draft blocks, greedy on both sides: 72.7 % at depth 1 falling to
16.8 % at depth 5, mean accepted prefix 1.85, so **2.85 tokens per main forward**. The confidence head
separates accepted from rejected positions cleanly, 1.763 against 0.309. This passed the bar comfortably.

**The draft is cheap: 22.9 ms per block** on a quiet machine with 2-bit draft experts, 25.3 ms with the FP4
experts as shipped, and the spread across 40 blocks is under a millisecond. (Every draft figure taken while a
bank build was running — 75 to 95 ms — was inflated three to four times over. Operating rule 10 exists because
of this.)

**It is worth 1.20x, and the limit is bytes.** Widening a forward is cheap in compute — a fixed cost plus
19.8 ms per extra position — but not in reads: adjacent tokens share only about 29 % of their experts, so
verifying W positions to accept T tokens reads W/T times the bytes, and bytes are already half of a token.
Verifying all five drafted positions is a **loss** at every budget measured. Extending the block while
confidence >= 1.0 gives width 2.30 for 1.94 tokens per forward and projects to **1.20x** at a 36 GiB budget —
153 ms per accepted token against 182.5. Sections 22 to 24 of `HANDOFF-2026-09-17.md` have the tables and the
model's assumptions, the main one being that reads and compute do not overlap, which the anatomy says is close
to true.

**The lever is worth more in interactive use than this says**, because the miss curve behind the projection
comes from a trace running at a 74 to 77 % hit rate while a chat session at 44 GiB runs at 87.3 %.

**One thing already settled, and it is about memory rather than speed.** Quantizing the draft's own experts to
2-bit g128 costs 2.8 % of accepted tokens and saves 2.4 ms per block, which is a wash — both formats project
to 1.20x. What it buys is residency: 1.71 GiB instead of 3.24 GiB of a budget that is the binding constraint
on everything else. Take it for that reason.

**Cost if it proceeds.** The largest on this list: a draft forward, verification of K positions in one pass,
KV rollback on rejection, prefix-cache interaction. A whole session, for a projected 1.1x. Levers 2 and 3
below are cheaper and two of them make this one worth more.

### 9.2 Lever 2 — Dispatch count — **demoted on FP4, 2026-09-19**

> **It is real work on a term that is already free.** The floor is **84.6 ms on FP4** (section 6.1), inside a
> 341 ms token of which about 320 ms is drive time. Fusing compute to nothing would not be visible until
> bytes come down. The lever is unchanged in size and bank-independent — the FP4 expert kernel costs 23.5 ms
> per token against the affine path's 24.6, and hyper-connections cost 68.7 ms across 80 sublayers on both —
> but it is not the place to spend days while decode is drive-bound. **Prefill is the exception**: it is
> compute-bound in a way decode is not and has never been profiled at the layer level on FP4.

The figures below were taken on the 2-bit bank and the shape holds on FP4.

**What.** The all-resident floor is 93 ms per token at a 512-token context, and it is roughly 400 GPU
dispatches at about 0.2 ms each. The inconsistency the previous version of this section asked to settle is
settled (section 20 of `HANDOFF-2026-09-17.md`): 68 ms was a stale number, 120 ms was "rest" with streaming
overhead inside it, and the real floor is 93 ms.

**Where it goes**, profiled against the 2-bit bank with the fused decode entry points, as isolated pieces each
paying its own evaluation barrier — which is why they sum to 162 ms against a 93 ms token:

    hyper-connections (mixes, pre+norm, post)   74.2 ms across 80 sublayers
    compressed reuse attention                  30.7 ms across 30 layers
    routed experts, 2-bit affine g128           24.6 ms across 40 layers
    shared expert (fp8)                         16.7 ms across 40 layers
    router                                      12.7 ms across 40 layers
    final head + norm                            2.1 ms once

**The finding is the shape, not any one row.** No piece dominates, every piece is a fraction of a millisecond,
and the model spends three times longer in hyper-connections than in its routed experts. A faster expert
kernel cannot move this; fewer, larger dispatches can. `mx.compile` over a whole layer, or one kernel for the
hyper-connection triple, is where to start.

**Deciding measurement.** Count and time dispatches directly — `profile_decode_gpu.py` — then fuse the
cheapest-to-fuse group and re-run `decode_resident.py`, which is the only clean read of the floor.

**Cost.** Profiling is hours. Fusion work is days, and it is the largest lever that depends on nothing else.

### 9.3 Lever 3 — Recover the quality the 2-bit bank cost — **closed 2026-09-20: there was no cost to recover**

> **Closed by section 7.6.** This lever existed to buy back "+0.019 nats and 6.3 points of top-1 against the
> 3-bit bank, visible as dropped and mangled tokens in output". The dropped and mangled tokens were the
> transposed residual mix, not the bank: re-gated through the fixed runtime the 2-bit bank compiles 20 of 20
> C++ blocks and malforms 0 of 94 include lines, equal to FP4 and to the hosted reference. The NLL and top-1
> differences below are real and unchanged; what is retired is the belief that they predicted anything a user
> sees. Everything after this line is kept as written, because the null it records about searched fits is
> still a null and still cost a bank build to establish.


**What.** +0.019 nats and 6.3 points of top-1 against the 3-bit bank (section 7.3), visible as dropped and
mangled tokens in output. This lever costs **build time only** — no runtime change, no risk to the decode
path.

**A better fit was built and gated, and it is a null.** A grid search picks the best of 25 shrunk ranges per
group; once the level assignment is fixed, the best (scale, bias) for that assignment is the closed-form
least-squares fit of the weights against the levels, which is not a grid point. Alternating the two converges
in four iterations. On the cheap screen this looked decisive — 24 real FP4 experts at 2-bit group 128:

| fit | mean w-err | mean y-err | vs the bank in use |
|---|---:|---:|---:|
| `search`, the bank in use | 0.3702 | 0.5912 | — |
| `search` + least squares | 0.3518 | 0.5552 | −6.1 % |
| 9x9 wide grid + least squares | 0.3331 | 0.5289 | **−10.6 %** |

A whole bank was built with the last of those (`build_affine_bank.py --fit wide-lsq`, 115 minutes, verified
byte for byte) and gated on the production path at 512 tokens on two different texts. Paired against the bank
in use, per token:

| text | mean | paired SE | median | refined better on |
|---|---:|---:|---:|---:|
| model README | +0.0216 | 0.0386 | +0.0017 | 46.9 % |
| `encoding.py` | −0.1392 | 0.0494 | +0.0001 | 47.9 % |

**The typical token does not move.** Both medians are within 0.002 nats of zero and neither sign test is
significant; the two means disagree in direction and each is driven by about five tokens out of 512. A 10.6 %
reduction in the screen's output error bought nothing the model can be shown to notice.

The bank was deleted after the gate: 142.4 GiB for a measured null, and `--fit wide-lsq` rebuilds it in
115 minutes from the FP4 checkpoint if anyone ever wants to re-open the question. Its per-token results are
kept in `benchmarks/results/nll_experts_runtime_affine2g128_DeepSeek-V4-1-Flash-q2g128-lsq*.json`, which is
what section 9.3.1's paired comparisons are computed from.

So section 8.3's warning — the screen ranks correctly within one quantizer and wrongly across quantizers — is
**too generous**. It is also wrong within one quantizer when the fit's *character* changes, and a searched fit
that clips outliers is a different character from one refined onto its own level assignment.

**Still untried.** Real activation weighting — capture activations from a prefill and weight the per-group fit
by what the model actually multiplies. Note that calibration is worth less here than the 2026-09-16 handoff
assumed: naive `mx.quantize` from FP4 beat the calibrated download by 0.030 nats at identical bits. Given the
above, screen any candidate against the *production* path or not at all.

### 9.3.1 The gate's own statistic was the bigger finding

Every quality claim this project has made rests on a 512-token mean NLL, and that mean has a **paired standard
error of 0.039 to 0.049 nats** — wider than every difference it has been asked to rank. Re-tested paired, on
the same recorded per-token data, against the 3-bit bank:

| bank | mean | z on the mean | median | worse on | top-1 |
|---|---:|---:|---:|---:|---:|
| 3-bit g64 oQ3e | — | — | — | — | 50.8 % |
| 2-bit g128 `search` | +0.0191 | +0.43 | +0.0348 | 62.5 % | 44.5 % |
| 2-bit g64 `search` | +0.0330 | +0.81 | +0.0353 | 62.9 % | 46.5 % |
| 2-bit g128 `wide-lsq` | +0.0407 | +0.94 | +0.0387 | 61.7 % | 46.5 % |

**The conclusion survives but the evidence for it was the wrong evidence.** "+0.019 nats" is 0.43 sigma on the
mean and should never have been quoted as established. What is established, and strongly, is the median and
the sign test: every 2-bit bank is worse than the 3-bit one on about 62 % of tokens by about 0.035 nats — a
sign test at more than 5 sigma — which agrees with the top-1 result and with the visible artefacts.
`nll_expert_precision.py` now prints the paired median, the sign test and the five tokens that move the mean
most, so no future arm is judged on the mean alone.

**Still untried.** Real activation weighting — capture activations from a prefill and weight the per-group fit
by what the model actually multiplies. Note that calibration is worth less here than the 2026-09-16 handoff
assumed: naive `mx.quantize` from FP4 beat the calibrated download by 0.030 nats at identical bits.

### 9.4 Lever 4 — A larger expert budget — **closed**

Re-simulated at the current expert size, which nobody had done since experts got 39 % smaller. Decode hit rate
against budget, first-come admission, LRU, within a point of the measured 80.9 % at 36 GiB:

| budget GiB | 36 | 40 | 44 | 48 | 52 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode hit | 81.9 | 83.8 | 85.1 | 86.5 | 87.7 | 88.9 | 91.0 |

Going from the 44 GiB in use to 52 GiB buys 2.6 points and would wire about 80 GiB of a 96 GiB machine — the
configuration class that kernel-panicked this machine twice. **Not worth it.** Reopen only if speculation
lands, which changes the arithmetic (section 9.1).

### 9.5 Lever 5 — Startup hotlist preload — **measured, and better than it looked**

A session is 16.4 s to ready and its first turn pays full miss cost; later turns run at 87.3 % because they
reuse what the first turn dragged in. The open question was never the cost, it was whether a hot set
generalizes: do the experts a *new* prompt wants resemble the ones old prompts wanted?

`benchmarks/hotlist_coverage.py` answers it leave-one-prompt-out — rank on the other prompts in the trace,
score on the held-out one, so nothing is credited for memorizing its own prompt:

| hot set | experts | of the bank | load time | unseen prefill | unseen decode |
|---:|---:|---:|---:|---:|---:|
| 2 GiB | 215 | 1.4 % | 0.3 s | 15.9 % | 14.5 % |
| 4 GiB | 431 | 2.8 % | 0.6 s | 23.1 % | 20.9 % |
| 8 GiB | 863 | 5.6 % | 1.3 s | 32.4 % | 29.6 % |
| 16 GiB | 1,726 | 11.2 % | 2.6 s | 45.3 % | 41.9 % |

**5.6 % of the bank covers 30 % of an unseen prompt's requests.** Routing is far more concentrated than a
top-6-of-384 router suggests.

**Implemented, and measured end to end.** `CACHALOT_HOTLIST` names a file from
`benchmarks/build_hotlist.py`, `CACHALOT_HOTLIST_GIB` caps what is read (default 8 GiB), and unset the runtime
behaves exactly as before. Four runs a side, interleaved both ways, 512-token cold prompt, 36 GiB budget:

| | off | on | change |
|---|---:|---:|---:|
| ready | 15.3 s | 15.5 s | +0.2 s |
| cold prefill | 16.6 s | 15.7 s | **−5.7 %** |
| first-turn hit rate | 34.2 % | 38.2 % | **+4.0 pts** |
| read during the turn | 117.4 GiB | 111.0 GiB | **−5.5 %** |
| decode | 201.1 ms | 206.1 ms | +2.5 % |
| cold start to end of first turn | 38.3 s | 37.8 s | −1.1 % |

The prefill gain is the reliable part: the four arms do not overlap, 16.5–16.7 s against 15.6–16.2 s. The
decode difference and the whole-session figure are inside the 7 % run-to-run spread.

**The first attempt was a net loss, and the reason is worth keeping.** Reading the hot set in the foreground
cost 1.3 s of startup to save 0.7 s of prefill. The same bytes read *during* prefill hide under prefill's own
compute; read at startup they hide under nothing. Moving the read to a thread that is joined before the first
prompt — so it overlaps RoPE precompute, the Engram reader and the prefetcher — cut the startup cost to 0.2 s
and turned the lever positive.

**Two honest limits remain.** The trace holds five prompts, so leave-one-out ranks on four: indicative, not
tight, and recording a hot set over a wider spread of real sessions is cheap. And 8 GiB of a 36 GiB budget is
a large static reservation; the sweep in `hotlist_coverage.py` suggests 4 GiB gives two thirds of the coverage
for half the reservation and has not been A/B'd.

### 9.6 Lever 6 — Long-prompt prefill

512-token cold prefill is 16.4 s and follow-up prefills in chat are under a second thanks to typing-time
prefill and the prefix cache. A 2048-token prefill has not been measured on any recent bank. Low expected value
for interactive use, real value if long documents become a use case.

### 9.7 Lever 7 — Engram onto the internal SSD

Measured as a speed null on 2026-09-17 (section 2 of that log: a prefill is barely blocked on Engram, and the
faster drive changes nothing), so this is **robustness only** — it would remove one of the three reasons the
X10Pro must stay connected. It is newly affordable: 189.1 GiB of FP4 Engram against 190 GiB free, or 91.9 GiB
if taken from the oQ3e conversion. Do it if disk pressure ever eases further, not for throughput.

### 9.8 Lever 8 — Below 9.49 MiB per expert — **reopened, conditionally**

Closed in the previous version on the grounds that bytes were no longer the constraint. They are: 49 % of a
token is expert streaming (section 6), and under speculation bytes per accepted token rise further (section
9.1), so a smaller expert is worth more now than when this was written.

What has not changed is the cost. Effectively closed inside MLX. `mx.quantized_matmul` accepts group sizes 32, 64 and 128 only, and 2 bits at
group 128 is already in use, so 9.49 MiB is the floor for any format the existing kernels can read. Going lower
means custom Metal kernels for a custom encoding — a much larger piece of work than the 2-bit bank was, and it
would be attacking bytes, which are no longer the constraint. Mentioned for completeness; do not start here.

### 9.9 Repetition collapse — **the defect, not the sampler. Closed 2026-09-20.**

> **The collapse was the transposed residual mix.** Re-measured through the fixed runtime on the protocol
> this section was written from — `repetition_quality.py`, 4 seeds, 900-token cap, short prompts — with the
> frequency penalty **off**:
>
> | arm | bank | penalty | collapses |
> |---|---|---|---:|
> | 2026-09-17, corrupt runtime | q2g128 | off | **5 / 8 (62 %)** |
> | 2026-09-17, corrupt runtime | q2g128 | 0.2 / 128 | 1 / 8 (12 %) |
> | **2026-09-20, fixed runtime** | **q2g128** | **off** | **0 / 12 (0 %)** |
> | **2026-09-20, fixed runtime** | FP4 | off | **0 / 12 (0 %)** |
>
> Against the original's own denominator — turns 2 and 3, the long replies — this is **0 of 8 against 5 of
> 8**, same bank, same seeds, same cap: two-sided Fisher exact **p = 0.026**. The FP4 arm is the control that
> says it is not a bank effect, which is what this section always claimed and now has both directions of.
> The worst repeat across all 24 replies was 7 tokens against a 24-token collapse threshold.
>
> **This was predictable from section 7.4.8 and nobody predicted it.** A repetition loop is a model that
> cannot tell it has already written something. `hc_post` applied the hyper-connection mix transposed, which
> is precisely a failure to read back what the recent context holds — the same defect that put a word the
> text had spelled out ten tokens earlier at rank 23,989. The sampler was treating the symptom.
>
> **What comes off.** `--frequency-penalty 0.2 --penalty-window 128` is removed from section 4's command and
> from the HTTP server defaults. It distorts code that legitimately repeats, which is most code, and the
> number that bought that distortion no longer exists.
>
> **What this does not establish.** 0 of 12 is not a proof of zero: the 95 % upper bound on the rate is
> 22 %, and P(0 of 12) is 0.216 even if the true rate were still the penalty-on 12 %. What is excluded, at
> P = 9.1e-6, is 62 %. If a loop is ever seen again, the knobs are still there and the CLI still exposes all
> four; the claim here is that nothing measurable justifies paying for them by default.

The original section, kept because its diagnosis of the *shape* was right and only its cure was wrong:

Free generation on code prompts falls into a repeating loop. It is **not** a quantization failure -- FP4 does
it too -- and it is not the prefix cache, which was tested and cleared. It tracks **generating a long reply
from a short prompt**: 62 % of replies collapse in that shape, against 0 of 7 when the same turn is generated
from about 1,700 tokens of context. A collapsed turn then poisons the next one.

`frequency_penalty` was adopted as the fix, because it grows with the count; a classic repetition penalty
fires once per unique token and a confident loop rides straight over it. At 0.2 with a 128-token window the
rate fell from 62 % to 12 %. The survivor had period 14, which only puts each of its tokens in the window
about nine times.

Note that "it tracks generating a long reply from a short prompt" and "0 of 7 from 1,700 tokens of context"
is, read through section 7.4.8, a description of the defect rather than of sampling: a long context gives the
model many more routes to the fact it is trying to recall, so a transposed mix hurts it less.

### 9.10 Wasted prefetch: 42.7 % of every byte read

From a real chat session's `/stats`, the byte accounting closes exactly:

    demand misses 30,184 + wasted predicted loads 22,465 = 52,649
    52,649 x 9,953,280 B = 524,030,238,720 B = ssd_bytes_read, to the byte

So **42.7 % of all SSD traffic was read and never used**, at 39.6 % prediction precision, while 48.8 % of real
misses did get a head start. The top-6 width was tuned at an 80.9 % hit rate on the benchmark; a chat session
runs at 90.7 %, where there are far fewer misses to predict and the same width wastes proportionally more --
and those reads occupy the drive and the loader threads during exactly the windows the demand misses need.

**Swept on FP4 and closed, 2026-09-18.** Widths 0/2/3/4/6 at a 36 GiB budget, three passes interleaved:
2.78, 2.83, 2.86, 2.89 and **2.93 tok/s**. Top-6 is optimal on 17.93 MiB experts as well, monotonically, with
non-overlapping ranges; prediction off is the worst setting.

> **Correction, 2026-09-19.** This section explained that result by saying the drive is not saturated during
> decode, so a speculative read is nearly free. **That is false on FP4**: the drive is busy 80.5 % of decode
> (section 6.1), and the predict-worker sweep in section 11 shows it is at its knee — adding in-flight reads
> lowers achieved bandwidth rather than raising it. The width result stands, on a different mechanism: a
> predicted read costs 6.47 ms against a demand read's 9.49 because it is off the critical path, and
> prediction serves 41.8 of the 69.5 misses per token early. Width is settled; **precision is not, and it is
> now the largest open lever** (section 9.12).

**Keep the general lesson, and note it cuts both ways:** a large waste figure is not a lever unless the
wasted resource is the binding one — and when the bank changes, check whether it has become binding before
reusing the conclusion's reasoning for anything else.

### 9.11 Lever 11 — Mirror striping across both drives — **shipped, 2026-09-19**

**What.** The FP4 experts live on the internal SSD; the X10Pro holds the checkpoint they were copied from,
byte for byte, under the same shard names. `reader.py` has carried a mirror path since 2026-09-16: the tail
`CACHALOT_MIRROR_FRACTION` of every expert read is issued to a second drive concurrently, so one expert lands
sooner than either drive alone could deliver it.

> **This is a re-measurement, not a discovery, and the 2026-09-19 session initially wrote it up as one.**
> `HANDOFF-2026-09-16.md` §7.3 is precise about why the mirror was turned off, and section 11 of this document
> compressed it into "harmful; leave it off", which is what misled a later reader. The real finding was:
> **with the 3-bit stacked bank** decode fell from 3.22 to 2.74 tok/s, and the cause was a code interaction
> rather than bandwidth — `ExpertReader._read_pieces_concurrently` is skipped whenever a mirror is configured,
> so that bank's nine pieces per expert fell back to serial reads with USB tail latency on top. **An FP4
> expert is one contiguous range**, so it never takes that path and the interaction cannot bite. 09-16 also
> measured the positive case (10 % striping: 2.86 to 3.00 tok/s, 512-token prefill 32 to 28.5 s) and derived
> the optimum as `bandwidth_of_second_drive / total`, noting that past about 15 % the USB becomes the
> bottleneck. Everything below reproduces that on the current configuration and ships it.
>
> **The lesson is about this document rather than about storage:** a null compressed to its verdict loses the
> condition that made it true. §7.3 said "with the 3-bit bank" and "it could be fixed"; the summary said
> "harmful". Keep the condition in the null.

Decode is drive-bound on FP4 (section 6.1) and the gain is a cliff rather than a plateau. `decode_anatomy.py`,
36 GiB budget, 512-token prompt, four runs per arm interleaved in both directions with `settle.sh` between:

| mirror fraction | ms/token | median | demand read mean | wall-clock GB/s | coverage block |
|---:|---|---:|---:|---:|---:|
| off | 342, 341, 342, 337 | 341.5 | 9.37 ms | 5.80 | 156.6 ms |
| **0.10** | 326, 328, 323, 328 | **327** | **8.10 ms** | **5.97** | **145.8 ms** |
| 0.15 | 352, 352, 352, 352 | 352 | 9.75 ms | 5.54 | 167.5 ms |

The ranges do not overlap. **At 0.15 it is worse than off**, which is what a mirror looks like past its
optimum: the USB carries 280 MiB per token at 1.0 GB/s, about 293 ms inside a 352 ms token, and the slow tail
becomes the long pole. At 0.10 it carries 186 MiB, about 195 ms inside a 327 ms token, and stays off the
critical path.

**Between 0.08 and 0.12 the effect is flat** — medians 328, 324 and 322.5, with a within-arm spread of 7 to
12 ms that swamps the 3 to 5 ms between them. The shipped default of 0.10 sits in the middle of that plateau
and does not need tuning. Do not read a winner out of those three numbers; four runs a side cannot separate
them.

**It is a pure latency win and the diagnostics say so.** Bytes read and hit rate are identical across arms
(1,861 against 1,864 MiB, 71.0 % both). Only the arrival time moves: the demand read, which is the critical
path, drops from 9.37 ms to 8.10 ms.

**And it pays on prefill, which was the thing to check before recommending it.** The X10Pro also serves
trunk, Engram, head and tokenizer, so decode's win might have been taken out of prefill. `decode_throughput.py`,
four runs per arm, interleaved:

| | cold prefill, 512 tokens | median | decode ms/token | median |
|---|---|---:|---|---:|
| off | 29.1, 29.0, 29.1, 29.1 | 29.1 s | 332, 333, 333, 340 | 333 |
| **0.10** | 27.1, 27.6, 27.1, 27.1 | **27.1 s** | 323, 326, 324, 323 | **323.5** |

**Prefill −6.9 %, decode −2.9 %, both with non-overlapping ranges, and no quality question to answer at all**
— the bytes are the same bytes, read from a byte-identical copy, and the X10Pro is opened `O_RDONLY`
(`storage/reader.py:103`).

**Two honest limits.** Everything above was measured at a 36 GiB budget on a 512-token prompt with the
machine idle, where the hit rate is 71 %. Hamed runs 44 GiB with a hotlist and a live conversation at 87 to
90 %, where there are far fewer misses for the second drive to help with, so **expect less than 5 % there**;
it has not been measured. And the whole lever is contingent on the X10Pro staying connected, which section
3.1 already requires for three other reasons.

### 9.12 Lever 12 — Prefetch precision — **bounded and mostly closed, 2026-09-19**

**Where it came from.** A token reads 1,858 MiB, of which **613 MiB is predicted and never used** — 34.2
wasted loads at 55 % precision, about 105 ms of drive time on the binding resource. Width is settled at top-6
and swept on FP4; accuracy had never been touched. It looked like the largest open lever.

**It was bounded before anything was written, offline, with no GPU and no experts loaded.**
`benchmarks/predictor_recall.py` reads `capture_activations.py`'s per-layer MoE inputs — token-aligned across
layers, because every layer sees every token — and the gate tensors straight out of the shard headers, then
scores the predictor the runtime actually uses against the truth it is trying to guess.

| width | one layer early, as shipped | two layers early | previous token |
|---:|---:|---:|---:|
| 6 | **71.5 %** | 65.0 % | 34.2 % |
| 7 | 75.8 % | 69.0 % | 36.5 % |
| 8 | 78.9 % | 72.1 % | 38.4 % |
| 12 | 85.7 % | 79.8 % | 43.9 % |
| 16 | 88.8 % | 83.9 % | 47.4 % |
| 24 | 92.0 % | 88.4 % | 52.2 % |

71.5 % at top-6 reproduces the "~73 %" in `moe_layer_metal.py`'s own comment, which is the check that the
script measures the right thing.

**The missing 28.5 % is the router's selection boundary, not lost information.** Two extra layers of
staleness cost only 6.5 points, so the residual stream drifts slowly and the stale vector is nearly as
informative as the true one. Recall by the router's own ranking says where the loss is:

    rank 1 (of 6)  95.5%      rank 4 (of 6)  69.7%
    rank 2 (of 6)  89.9%      rank 5 (of 6)  54.0%
    rank 3 (of 6)  79.7%      rank 6 (of 6)  40.5%

The predictor nails the expert the router wants most and coin-flips the one it wants least. What it loses are
experts whose scores sit within drift of the cut.

**So the obvious idea was to spend extra predictions only where the stale router is unsure, and it is a
null.** Take top-6 plus every expert within `margin` of the 6th score, `margin` in units of the spread from
the 1st to the 6th so it does not depend on a layer's score scale:

| margin | mean predicted | recall | fixed width at the same cost |
|---:|---:|---:|---|
| 0.00 | 6.00 | 71.5 % | top-6 at 71.5 % |
| 0.05 | 8.82 | 78.7 % | **top-8 at 78.9 %** |
| 0.10 | 12.44 | 83.1 % | **top-12 at 85.7 %** |
| 0.20 | 17.22 | 87.6 % | top-12 at 85.7 % |
| 0.35 | 20.73 | 90.0 % | top-12 at 85.7 % |

**Adaptive width is not better than fixed width per byte, and at 12 predictions it is 2.6 points worse.** The
score gap does not predict which way drift will flip the ordering; the ambiguous layers simply absorb extra
predictions without proportionally more hits.

**What that leaves.** No cheap re-use of the stale scores beats plain top-k, and top-k's own optimum on FP4
is settled at 6 (section 9.10). Recovering the remaining 28.5 % needs a *different signal* — a learned
correction, or an input that includes part of layer L's own update — which is a research project rather than
a session, and it would have to pay for whatever it costs to compute inside the layer it is trying to run
ahead of.

### 9.12.1 Non-cumulative lookahead — **built, measured and closed, 2026-09-19**

This section used to end: *"a non-cumulative version would double the lead time at constant bytes — worth a
code change and a 20-minute A/B."* The knob was built (`CACHALOT_PREDICT_LEAD`, which shifts the prediction
window where `PREDICT_AHEAD` widens it), and the A/B was run on the fixed store. **The premise was wrong and
the lever loses — but by much less than the bytes say, and that is the useful part.**

Four runs a side, interleaved in both directions, `settle.sh` between, FP4 at a 36 GiB budget with mirror
striping on and top-6 both sides:

| arm | ms/token | median | hit rate | MiB read/token | precision | wasted loads/token |
|---|---|---:|---:|---:|---:|---:|
| **lead 1**, predict L+1 (shipped) | 325, 328, 329, 329 | **328.5** | 70.8–70.9 % | 1,866 | **55 %** | 34.3 |
| lead 2, predict L+2 instead | 330, 334, 336, 344 | 335.0 | 70.9–71.0 % | 2,037 | 45 % | 43.8 |

**The ranges do not overlap** — every lead-1 run is at or under 329 ms and every lead-2 run at or over 330 —
so the 2.0 % is real at four runs a side.

**"Constant bytes" was the error, and it was an error about what a prediction costs.** Lead 2 predicts the
same *number* of experts as lead 1. It does not read the same number of *bytes*, because a wrong prediction
is a read that gets thrown away while a right one is either already resident or about to be needed. Precision
falls from 55 % to 45 % and wasted loads rise from 34.3 to 43.8 per token, so traffic rises 171 MiB per token
— **9.1 % more bytes on the resource that is 80.5 % busy.** The offline recall table's "6.5 points" is a
count; its price is bytes, and nothing in the table says so. Section 8.3, a fifth time.

**And yet the lead time is worth roughly 24 ms per token.** 171 MiB at the 5.97 GB/s this configuration
achieves is about 30 ms of drive time, and the arm is only 6.5 ms slower. By subtraction the extra layer of
lead recovers about 24 ms of the 41.4 ms the anatomy attributes to prefetch timing — which is what the recall
table predicted and is the largest confirmation this project has that timing, not only coverage, is worth
attacking on FP4. **Read that 24 ms as an inference, not a measurement**: it assumes the extra bytes are
fully exposed, which on a drive at its knee is close to true but is not measured directly.

**What it tells the next session, and it is worth more than the null.** A predictor that reached L+2 *at
lead-1 recall* would be worth about 24 ms per token, 7 % of decode, on the bank in use. That is no longer a
speculative prize attached to section 9.12's "different signal" — it is a measured lower bound on what one
would pay. Recall is what must improve; lead time is already known to convert.

`CACHALOT_PREDICT_LEAD` ships defaulted to 1, which is exactly the previous behaviour, and is kept because it
is three lines and it is how this was measured.

**The regression check that came with it.** Lead 1 on the fixed store is 328.5 ms against the 327 ms recorded
for the same configuration before the predicted-load lifetime fix (section 9.13) — inside the 7 % run-to-run
spread and inside this arm's own 325–329 range. **The lifetime fix costs nothing at lead 1**, which is what
it should do, since at lead 1 every prediction is aimed at the very next layer and no deadline is ever tested.

### 9.13 Lever 13 — Predicted loads had no lifetime — **fixed and shipped, 2026-09-19**

**Found by an outside review, reproduced on CPU with no model loaded, fixed and measured the same day.**

> **Shipped.** Each in-flight prediction now records the walk it belongs to and the layer it was issued
> for, and expires when that walk ends or when its layer has been requested. `TextDecodeRuntime.reset`
> expires them explicitly. `predicted_expired` counts the releases, so waste from a wrong prediction stays
> visible and is no longer mixed with work thrown away for being punctual. Nine deterministic store tests
> pin it, the first of which fails against the previous sweep.
>
> **It is a null on decode, which is the expected result and not a disappointment.** At the shipped lead of
> 1 every prediction is aimed at the very next layer, so no deadline is ever tested: lead 1 measures 328.5 ms
> per token against the 327 recorded before the fix, inside this arm's own 325-329 range. The fix is not a
> speed lever. It is what makes any experiment beyond lead 1 mean anything, and section 9.12.1 is the first
> one that did.

`ResidentExpertStore.get_many` opens with `self._sweep_inflight_locked(keep=requested)`, and that sweep
releases **every completed in-flight prediction that the current layer did not ask for**:

```python
def _sweep_inflight_locked(self, keep: set[Key]) -> None:
    """Release finished predicted loads that no request has claimed."""
    for key in [k for k, (f, _, _) in self._inflight.items() if k not in keep and f.done()]:
```

There is no target token or layer on an in-flight entry, so the sweep cannot tell a stale prediction from a
correct one that simply belongs to a later layer. With `CACHALOT_PREDICT_AHEAD=2`, layer L predicts both L+1
and L+2; when L+1 is then requested, any L+2 read that has already **finished** is dropped, its slot is
returned to the pool and its 18,800,640 bytes are counted in `predicted_wasted_bytes`. An L+2 read still in
flight survives, because the sweep only takes `f.done()` entries.

**So a prediction is punished for completing early.** That is the opposite of the intended behaviour, and it
falls hardest on exactly the predictions lead time is supposed to buy.

**What it invalidates.** Section 11 records `CACHALOT_PREDICT_AHEAD=2` as 8.8 % worse and attributes it to the
knob being cumulative — more bytes, lower precision, a queue on a saturated drive. That mechanism is real and
measured. But the arm was also discarding its own L+2 work, and the wasted-byte and precision figures it
reported (1,868 to 2,514 MiB, 55 % to 39 %) include those discards. **The null stands as "this knob loses";
it does not stand as evidence about lead time**, and the non-cumulative experiment in section 9.12 cannot be
interpreted until this is fixed.

**The fix.** Each in-flight speculative entry records the walk and the layer it was predicted for. Decode
visits layers in ascending order once per token, so a requested layer that does not advance is a new walk.
A prediction expires when its walk ends or when its layer has been requested -- at which point it was either
consumed, and is no longer in flight, or mispredicted, and is genuinely waste. Slot pressure needed nothing
new: `prefetch_decode` already refuses to start a load once the transient slots are down to
`PREDICT_SLOT_RESERVE`, so a longer lifetime costs prefetch depth rather than the demand path, and a test
pins that.

**Still not re-measured: the cumulative `CACHALOT_PREDICT_AHEAD=2` arm.** Section 11 keeps it as a null and
it will almost certainly stay one — it doubles the predicted count where lead 2 merely moved it, and lead 2
alone cost 171 MiB per token. But its recorded precision and wasted-byte figures were taken while the store
was discarding its own L+2 work, so those two numbers specifically should not be quoted. An hour of machine
time would settle it.

### 9.14 Miss reduction: what the offline replay says about the assessment's five recommendations

`docs/MISS-REDUCTION-ASSESSMENT-2026-09-19.md` proposes five experiments. Its own CPU replay was
**reproduced and is correct**: `simulate_policies.py` on `trace_routing_v7` gives decode hit 69.9 % at 36 GiB
and 73.3 % at 44, which is 72.2 and 64.1 misses per token and the 8.2-miss, 11.4 % gap it reports.

**Recommendation 2, predicted-expert lifetimes, is done** and is section 9.13.

**Recommendation 3, session-aware retention across suffix prefills, was screened and does not justify runtime
work yet.** `prefill_layer` evicts every resident of a layer the current prompt does not route to, so a short
new prompt wipes the previous turn's decode working set. Exempting a bounded, decaying set of
recently-decode-useful experts from that wipe, swept over the reserve per layer:

| reserve per layer | 36 GiB misses/token | 44 GiB misses/token |
|---:|---:|---:|
| 0 (baseline) | 72.319 | 64.106 |
| 1 | 72.206 | 64.013 |
| 2 | 72.006 | 63.919 |
| 4 | 71.969 | 63.663 |
| 8 | **71.731** (−0.59) | **63.138** (−0.97) |

The best case is **0.97 misses per token, 1.5 %**, which is 18 MiB and about **3 ms of a 328 ms token**. The
run-to-run spread is 7 %. Even if the runtime delivered the full simulated gain it could not be measured, and
the simulator has already over-predicted once by more than this whole effect — it put segmented LRU at
+0.9 points where the runtime delivered −0.25 (section 11).

**But the screen is weak in the direction that matters, and that is the finding.** This trace has **32 decode
tokens per segment**. Session retention is a mechanism for carrying a working set *built during decode* across
the next prefill, and 32 tokens barely builds one. A real coding turn generates 256 to 1,024. The assessment
says as much about its own replay. So the honest reading is not "session retention does not work" but **"this
trace cannot tell, and it is the wrong trace to ask."**

**The cheap prerequisite nobody has done: record a longer routing trace.** `src/cachalot/metrics/routing_trace.py`
already records one, and recording it during a real multi-turn chat session costs nothing but the session.
Until a trace exists with realistic decode lengths, recommendations 3, 4 and 5 are all being screened on five
32-token segments, and none of their results will mean much. **Record the trace first; it makes three
experiments interpretable for the price of one chat.**

**Recommendation 1, 36 against 44 GiB on the runtime**, is unscreened because it needs no screen — the
replay's 8.2 misses per token is 154 MiB, about 26 ms, which is large enough to measure. Hamed already runs
44 GiB, so this settles the benchmark's budget rather than his configuration.

---

## 10. Retired premises — conclusions whose reasons expired

These were correct when written and are now misleading. Anyone reading the older logs will meet them.

| claim | where | why it no longer holds |
|---|---|---|
| FP4 collapses on 0 of 9 free-running replies | HANDOFF §2, §9.0, §7.4 | True of the failure `max_run` detects, which is an *exact* k-gram loop repeating back to back. It cannot see a **paraphrased retry loop** -- bad code, an apology, another attempt -- because nothing repeats exactly. Two of those same nine FP4 replies are retry loops scoring `max_run` 19 and 4 against a threshold of 24, and the new corpus reproduced one at `max_run` 7 in which the model writes "I'm clearly stuck in a loop". Does not re-rank the banks; does narrow what the collapse rate covers. §7.4.3. |
| Judge a collapse by `max_run`, never by the trigram rate | HANDOFF §7.4, §9.9, rule 4 | Right for exact loops and wrong for retry loops, where `max_run` stays under 10 and the trigram rate separates cleanly (57.5 % against a healthy 34.8 %). A collapse metric needs both, each thresholded on replies someone has read. §7.4.3. |
| A smaller bank is blocked on GGUF k-quants MLX cannot read | 09-16 §7.5 | A better bank was built here from the FP4 checkpoint in 33 minutes. No GGUF, no k-quants. |
| Their mixed-precision recipe is the quality trick to copy | 09-16 §7.5 | Naive `mx.quantize` from FP4 beats the calibrated download by 0.030 nats at equal bits. Calibration is worth ~16 % of weight error, not a recipe worth copying. |
| Speculative decoding and MTP are a null result | 09-16 §8 | Rejected because verification multiplies bytes and bytes were the constraint. Bytes are no longer the constraint. See lever 1. |
| Do not re-run the prediction-width sweep | 09-17 §13 | True for the 15.48 MiB bank only. On a 9.49 MiB expert the saddle moved from top-3 to top-6 and paid 5.2 %. |
| Reading fewer bytes per expert is the only lever with real room | 09-17 §15.3 | It was, and it was taken. Bytes now hide under compute; the byte lever is spent. |
| Decode's floor is set by bytes | 09-17 §15 | The floor is compute, and it is 93 ms per token. |
| The compute floor is 120 ms per token, 8.3 tok/s | HANDOFF §2, §6, before 09-17 session 4 | 120 ms was `decode_anatomy`'s "rest", which contains streaming overhead as well as arithmetic. The all-resident floor, measured directly, is 93 ms and 10.8 tok/s. |
| Bytes now hide under compute, so the byte lever is spent | HANDOFF §6, before 09-17 session 4 | 73 ms of transfer against 62 ms of exposed wait: the drive essentially does not hide. 49 % of a token is streaming cost. Lever 8 is reopened. |
| The `mtp.*` layers are three MTP layers giving a draft depth of up to three | HANDOFF §3.2, §9.1 | They are DSpark: one block of five drafted tokens, a bidirectional draft block, a rank-256 Markov correction and a confidence head. |
| Speculative decoding's ceiling is another 1.5x | HANDOFF §9.1 | Acceptance is excellent — 2.85 tokens per main forward — but verification reads W/T times the bytes. The projection is 1.07x to 1.17x. |
| A larger expert budget is an open lever | HANDOFF §9.4 | Re-simulated at the current expert size: 44 to 52 GiB buys 2.6 points of hit rate and wires 80 GiB of 96. Closed. |
| FP4 scores 0.6 syntax errors per 100 lines, 30 to 40x better than any quantized bank | HANDOFF §2, §3.3, §7.4 | Twice wrong. First: 163 lines from an arm the guardian truncated after two of three seeds. Then **8.9 was wrong too** — the checker never read clang's exit status. §7.4.1. |
| FP4 writes roughly 3.5x fewer C++ syntax errors than a quantized bank | HANDOFF §2, §7.4, §9.0 | **The gate was broken.** `check_cpp` grepped for `": error: "` and ignored the return code, so a block aborting on a mangled `#include` scored clean. Re-scored: **0 of 4 FP4 blocks compile and 0 of 3 from the 3-bit bank**, and the density was measuring which arm aborted first. No C++ ratio between any two banks is established. The standing decision survives on collapse rate and top-1. §7.4.1. |
| Python is not the discriminator; both banks score 2.2 errors per 100 lines | HANDOFF §7.4, §9.0 | `ast.parse` stops at the first `SyntaxError`, so 2.2 counted broken files, not defects. On the parse rate FP4 is **2 of 10** and the 3-bit bank **0 of 5**; one 20-line FP4 block with five artefacts scored 1. §7.4.1. |
| A finished predicted load survives until the layer that asked for it | implicit in HANDOFF §9.12, §11 | `_sweep_inflight_locked(keep=requested)` releases every *completed* in-flight prediction not requested by the current layer, so an `L+2` read that finishes before `L+1` is discarded before `L+2` can use it. Finishing sooner makes a prediction less useful. This invalidates reading the `PREDICT_AHEAD=2` result as evidence about lead time. §9.13. |
| A better affine fit above 2 bits is the only route left to a smaller quality bank, and the next session's first job | HANDOFF §2, §9.0 | Taken. The fit was improved 28 % on a screen made faithful with real activations, a 221.5 GiB bank was built and gated, and it collapsed on 2 of 6 long turns where FP4 collapsed on none and parsed 0 of 5 Python blocks against FP4's 2 of 10. Bits, not fit, are binding. **Lever 0 closed** — on that evidence, not on the withdrawn C++ densities (§7.4.1). |
| Activation-weighted fitting is the untried part of the calibration idea worth keeping | HANDOFF §9.0.1, §9.3 | Tried. Worth 5.1 % of routed-expert output error at 3 bits, transfers across texts, costs no bytes — and worth nothing a compiler can see. The *capability* is kept; the lever it was meant to open is closed. |
| The drive is not saturated during decode, so a speculative read is nearly free | HANDOFF §9.10 | True on 9.49 MiB experts, where the drive was busy 45 % of decode. On FP4 it is busy **80.5 %** and at its knee: raising `CACHALOT_PREDICT_WORKERS` from 2 to 8 *lowers* achieved bandwidth from 5.70 to 5.34 GB/s. The width conclusion survives on a different mechanism. §6.1, §9.10. |
| Prediction from an earlier activation is free and useless: timing is not the problem, coverage is | 09-17 §13, HANDOFF §11 | Measured where timing was 2.2 % of blocked time and worth 1.6 ms per token. On FP4 timing is **20.5 % and 41.4 ms per token**. Still unbeaten, but the premise has expired and `CACHALOT_PREDICT_AHEAD` has never been swept on FP4. §6.1. |
| Mirror striping is harmful; leave it off | HANDOFF §11 | **The null lost its condition.** 09-16 §7.3 measured it harmful *with the 3-bit stacked bank*, because a mirror disables concurrent piece reads and that bank had nine pieces per expert; an FP4 expert is one contiguous range. 09-16 also measured the positive case and derived the optimum. Reproduced and shipped 2026-09-19 at 0.10: −5 % decode, −7 % cold prefill. §9.11. |
| DSpark is parked pending arithmetic on FP4's expert size | HANDOFF §9.1, prompt v9 | Arithmetic done. **1.03x** on measured constants against its own 1.15x bar. Closed. §9.1. |
| The all-resident compute floor is 93 ms | HANDOFF §6, §9.2 | 93.0 ms is the 2-bit bank's. On FP4 it is **84.6 ms**, and the FP4 expert kernel costs 23.5 ms per token against the affine path's 24.6. Compute is bank-independent in fact. §6.1. |
| Dispatch count is the largest open lever and the only large one depending on nothing else | HANDOFF §9.2, prompt v9 | True by size, misleading by value on the bank in use: 84.6 ms of a 341 ms token that is already hidden under ~320 ms of drive time. It pays after bytes come down, not before. §9 ranking. |
| FP4 is the quality bank, and 3.3 tok/s is what quality costs on this hardware | HANDOFF §2, §3.3, §7.4 | The premise was that the quantized weights set the ceiling. A hosted FP4 arm with no harness compiles 20 of 20 C++ blocks and malforms 0 of 102 includes; Cachalot compiles 0 of 42 and malforms 49 of 154. The quality was never bought with speed — it was lost to a runtime defect. §7.4.6. |
| The corpus results are evidence about the model or about FP4 | HANDOFF §7.4.3, §7.4.4, §7.4.5 | They are evidence about this runtime only. Both a pinned FP4 provider and a pinned FP8 provider are clean on the same 40 prompts at the same settings. §7.4.6. |
| Quality work means building a better bank | HANDOFF §9.0, §9.3, §8 | Three sessions of bank building — 3-bit, searched fit, activation weighting — were aimed at a defect that is not in the weights. The quantization findings stand; the lever they were meant to open never existed. §7.4.6, §2. |
| The next quality step is `benchmarks/token_rank_probe.py` on the production path | HANDOFF §7.4.4 | Still worth running, but it is no longer the first step and no longer the cheapest decisive one. The reference arm answered the question the probe was a proxy for. The probe's remaining value is localising *where* in the forward pass the token is lost, which is step 1 of the bisection in §2. |
| There is no `pack_3bit`, and MLX's packing above 2 bits is the blocker | HANDOFF §9.0, §8.1 | `pack_bits` writes MLX's layout at any width and is pinned against `mx.quantize` at 2, 3, 4, 5, 6 and 8 bits. The layout is one contiguous little-endian bit stream with no padding. |

## 11. Null results — do not repeat these

Each was measured and rejected, and the reasoning still holds. Re-running them costs hours and returns nothing.

**Storage and I/O**
- **Closing the achieved-bandwidth gap.** While the drive is busy the runtime already moves about 6 GB/s of the
  6.6–6.8 GB/s the drive delivers on cold experts.
- **The OS page cache under a large wired set.** A full 2x2 at 2 and 8 concurrent loaders, every arm on disjoint
  cold experts: all eight readings fall between 6.62 and 6.81 GB/s. Both variables null. `CACHALOT_PAGE_CACHE=1`
  stays on because it costs nothing and helps genuine repeat hits.
- **`F_RDAHEAD 0`** alongside `F_NOCACHE`: null.
- **Chunked expert reads**: a single expert read already saturates a stream.
- **Mirror striping with a stacked bank**: harmful, and the condition is the point. A configured mirror disables `_read_pieces_concurrently`, so a bank with nine pieces per expert falls back to serial reads. **An FP4 expert is one contiguous range and this does not apply**; mirror striping is shipped on FP4 at fraction 0.10 (section 9.11). Past about 15 % the USB drive becomes the bottleneck on any bank.
- **Engram on the internal SSD** for speed: null (it survives as lever 7 for robustness only).

**Caching and scheduling**
- **Eviction policy work.** Segmented LRU and decayed frequency are worth 1 to 2 points against 13 % fewer
  bytes from a larger budget, and total store overhead is 1.0 ms per token, so there is nothing to win.
- **LRU prefill admission** as an alternative to the quota planner: identical decode misses, worse prefill.
- **Prediction from an earlier activation** (more lead time): free, and useless — timing is not the problem,
  coverage is.
- **A GPU heartbeat to prevent clock drop**: the clock does not drop during decode.
- **More predict-pool workers on FP4.** `CACHALOT_PREDICT_WORKERS` 2 / 4 / 8 at a 36 GiB budget, four runs
  each interleaved both ways: **341 / 363 / 369 ms per token**, ranges non-overlapping, monotonically worse.
  The drive is already at its knee at 2.67 reads in flight; more concurrency lowers achieved bandwidth
  (5.70 / 5.35 / 5.34 GB/s) and lengthens the demand reads that are on the critical path (9.59 / 11.56 /
  11.78 ms). The timing term does fall as intended, 41.2 to 26.4 ms, and the coverage term rises further than
  that gain, 160.3 to 199.8 ms. **`decode_anatomy.py`'s own "at 7.3 GB/s with 8 reads in flight" line is a
  model, not a measurement, and the model is wrong on this drive.** The default of 2 is correct.
- **`CACHALOT_PREDICT_AHEAD=2` on FP4.** 330.5 against 359.5 ms per token, four runs a side interleaved,
  8.8 % worse. The extra lead time did what it was meant to -- coverage blocking fell 22 ms, from 147.1 to
  125.1 -- and it lost anyway, because the knob is cumulative: bytes rise from 1,868 to 2,514 MiB per token,
  precision falls from 55 % to 39 %, and on a saturated drive the queue pushes the timing term up 40 ms. The
  2026-09-17 null survives for a third distinct reason. **Lead time is worth having; this knob cannot buy it
  without bytes.**
- **Non-cumulative lookahead** (`CACHALOT_PREDICT_LEAD=2`, predict L+2 instead of L+1 at top-6, so the
  same number of experts is predicted). 328.5 against 335.0 ms per token, four runs a side interleaved,
  non-overlapping ranges, 2.0 % worse. **Its premise was wrong**: the same prediction count is not the same
  bytes, because precision falls from 55 % to 45 % and traffic rises 171 MiB per token on a drive that is
  80.5 % busy. The lead time itself works -- 171 MiB is about 30 ms of drive time and the arm loses only
  6.5 -- so a predictor with lead-1 recall two layers early is worth roughly 24 ms per token. Measured on the
  store *after* the lifetime fix, so unlike the `PREDICT_AHEAD=2` arm it is evidence about lead time.
  Section 9.12.1.
- **Adaptive prefetch width by the stale router's own score margin.** Extra predictions only where the 6th
  and 7th scores are close: 78.7 % recall at a mean of 8.82 predicted against fixed top-8's 78.9 % at 8.00,
  and 83.1 % at 12.44 against fixed top-12's 85.7 %. **Not better than fixed width per byte, and worse when
  wide.** Measured offline in minutes with `predictor_recall.py`, no GPU. Section 9.12.
- **Segmented LRU at FP4 expert size.** Re-run because the 2026-09-17 null was measured with a resident set
  1.8x larger, and `simulate_policies.py --expert-bytes fp4` predicted +0.9 points of decode hit at 36 GiB
  and +1.7 at 44. The runtime delivers **−0.25 points** (70.6-70.7 % against LRU's 70.8-70.9 %) and 326.5
  against 327 ms per token, fully overlapping. **The simulator over-predicts SLRU and should not be used as a
  runtime prediction for it**; the 44 GiB figure was never measured and should not be quoted.

**Speculation and drafting**
- **Verifying all five drafted DSpark positions.** A loss at every budget measured, because bytes per accepted
  token rise by the ratio of positions verified to tokens accepted. Confidence-gated widths near 2.3 are the
  optimum.
- **Serving the draft's experts from FP4.** The FP4 path repacks each projection into MLX's 8-bit affine
  layout on every matmul, 0.391 ms on top of a 0.547 ms product. Quantize the draft's experts once at load.

**Numerics and kernels**
- **Expert pruning or pinning a hot subset**: zeroing experts outside a calibrated pinned half costs about 91 %
  of gsm8k accuracy. Fetch-on-miss keeps the computation exact.
- **The miss-budget approximation**: 9.7 tok/s at budget 0, but KL 0.64 and 7 of 40 tokens identical.
- **A fused multi-expert kernel and `gather_qmm` chunking**: both slower than per-expert quantized matmul.
- **A simdgroup FP4 GEMM**: 2.5x less GPU time, zero wall-clock change.
- **`fit_minmax` at 2 bits**: worse than MLX's own max-abs fit.
- **Least-squares refinement of the affine fit** (`--fit wide-lsq`): 10.6 % less routed-expert output error on
  the screen, and nothing the model notices — paired median within 0.002 nats of zero on two texts, neither
  sign test significant. A whole bank was built and gated to find this out. Section 9.3.
- **A searched, activation-weighted 3-bit bank** (`--bits 3 --group 64 --fit search-lsq --importance`):
  28 % less routed-expert output error than the fit the retired 3-bit bank used, measured on the model's own
  recorded activations and shown to transfer across texts, and **2 of 6 long turns collapsing against FP4's
  none**, **0 of 5 Python blocks parsing against FP4's 2 of 10**, and no compiling C++ block on either side.
  A 221.5 GiB bank was built in 47 minutes and gated on matched generations to find this out. Bits, not fit,
  are binding above 2 bits. Sections 9.0 and 7.4.1 — the "31.6 against 8.9" this entry used to quote came
  from a broken checker and is withdrawn. Do not build another affine expert bank above 2 bits expecting
  quality.
- **A 9x9 wide grid for the affine fit at 3 bits**: 0.2588 against the 5x5 grid's 0.2587, for 378 ms per
  expert against 143. The 2-bit finding that a finer grid does not pay holds at 3 bits too.
- **Storing affine scales and biases in fp32 instead of bf16**: 0.5767 against 0.5789 of routed-expert output
  error, inside the noise. Scale precision is not where the 2-bit error lives.
- **A grid finer than 9x9 for the affine fit**: a 17x17 grid plus least-squares refinement buys 0.05 %.
- **Segmented LRU, decayed frequency and popularity-ordered prefill admission, re-run at the 9.49 MiB
  expert**: 0.9 and 0.5 points respectively over plain LRU, and popularity ordering is worse than first-come.
  The 2026-09-17 null survives the smaller bank.

## 12. Pitfalls worth knowing before touching the code

- **`F_NOCACHE` does not reliably keep expert reads out of the page cache.** Repeat reads of the same experts
  through a reader with `bypass_page_cache=True` went 5.55 GB/s, then 9.07, then 9.07 — the second and third
  passes were partly served from memory. `CACHALOT_PAGE_CACHE=0` therefore does not mean what its name
  suggests, and several bandwidths above 7 GB/s recorded in the older logs are partly memory hits.
- **A storage arm is only valid on experts nothing has read yet**, including in a previous process. Warmth
  survives across runs; one discarded matrix reported 17 to 64 GB/s, which is physically impossible for this
  drive and was the tell. `expert_read_scaling.py --expert-offset` slices one seeded shuffle disjointly so each
  arm gets cold experts by construction. Interleave wired and unwired arms, and never compare an arm that ran
  first against one that ran after it on the same experts.
- **MLX streams are thread-local.** Evaluating a runtime graph from a worker thread raises
  `There is no Stream(gpu, N) in current thread`. Typing-time prefill runs on the chat thread for this reason;
  the heartbeat and forward passes share `runtime._gpu_lock`.
- **Free memory is always near zero on macOS** because the file cache fills it. Judge pressure by
  `kern.memorystatus_vm_pressure_level` and compressor growth, not by free pages.
- **A wrapped copy-paste silently drops the expert bank.** If a shell-wrapped chat command loses
  `CACHALOT_EXPERT_BANK`, the runtime serves FP4 experts from the USB drive and everything is four times slower
  with no error. Check the `expert bank:` line the runtime prints at startup; it reports path, format, bits and
  MiB per expert.
- **`tty.setcbreak` flushes pending input** unless called with `TCSANOW`. Without it, anything typed while the
  model was generating is silently discarded.
- **A pseudo-terminal harness must drain the child's output while typing**, or the chat blocks writing to a
  full buffer and the test appears to hang.
- **The occasional stutter in generated text** — "con concurrency", "nib nibbled" — appears with every numerics
  path including the fully unfused baseline, and predates the 2-bit bank. It is model behaviour. The *new*
  artefacts the 2-bit bank adds are different: dropped characters inside words and renamed entities.
- **Nothing under `benchmarks/results` is tracked by git**, and result files are named by arm. The
  `--experts runtime` arm used to write one filename whatever bank it served, and overwrote the 3-bit bank's
  per-token data on 2026-09-17; it now writes `nll_experts_runtime_<kind><bits>g<group>.json`. Check for
  collisions before adding an arm.

## 12.1 Reading a live session's numbers

A healthy chat session at a 44 GiB budget with the hotlist on looks like this, from `/stats` on 2026-09-18:

    expert_hit_rate 90.7 %       better than the 87.3 % on record; the hotlist is part of it
    resident 4,431 experts       93.4 % of the budget, and 4,431 x 9,953,280 B exactly
    decode 6.3 to 7.1 tok/s      the upper half of the recorded 6.0-7.5 range
    prefix cache 19 hits, 1 miss
    mlx peak 59.6 GiB            under the 72 GiB wired limit

Two ways to misread the machine while it runs:

**"RAM is at 81 %, so there is headroom."** There is not. MLX alone peaked at 59.6 GiB and 81 % of 96 GB is
about 77.8 GB. The wired limit is already 72 GiB, and the measured budget curve says 44 to 52 GiB buys
2.6 points of hit rate while wiring about 80 GiB — the configuration class that panicked this machine twice.
Section 9.4.

**"The GPU is only at 56 %, so there is compute headroom."** That idle *is* the expert stall. At 7.08 tok/s a
token is 141 ms and the measured all-resident floor is 93 ms, so about a third of every token is the GPU
waiting on the drive. You cannot convert it by giving the GPU more of the same work. Only two things use it:
removing stalls (sections 9.10 and 9.4) or filling them with speculative work (section 9.1).

## 13. Reference commands

All are copy-paste ready and assume nothing about the current directory.

**The reference arm, run A — the diagnostic that says whether a quality problem is ours**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && set -a && . ~/.hermes/.env && set +a && python3 benchmarks/run_reference.py --pack ~/cachalot-corpus-pack --out ~/cachalot-runA-relace-fp4 --model deepseek/deepseek-v4.1-flash --provider relace/fp4 --attempts 5 --sleep 1.5
```

**Score any arm against Cachalot's, and against the reference**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/ ~/cachalot-runA-relace-fp4-scored ~/cachalot-runA-deepinfra-fp8-scored
```
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/include_integrity.py benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/ ~/cachalot-runA-relace-fp4-scored ~/cachalot-runA-deepinfra-fp8-scored
```

**Generate a Cachalot arm on the same 40 prompts**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 14400 --tag coding -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/coding_quality.py
```

**Full test suite**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q tests
```

**Decode benchmark, current best configuration**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 2400 --tag decode -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_throughput.py --prompt-tokens 512 --decode-tokens 64
```

**Where decode's time goes**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 3600 --tag anatomy -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64
```

**Quality gate, production path — required for any bank or numerics change**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 5400 --tag nll512 -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/nll_expert_precision.py --experts runtime --tokens 512
```

**Quality gate, dense reference math for a format that has no bank yet**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 7200 --tag nll_requant -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/nll_expert_precision.py --experts requant --requant-bits 2 --requant-group 128 --requant-fit search --tokens 160
```

**Screen candidate quantization formats in seconds**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/expert_requant_error.py --experts-per-layer 2 --probes 8
```

**Build a bank** (resumable; skips shards a previous run finished)
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/build_affine_bank.py --out /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 --bits 2 --group 128 --fit search --verify 12
```

**Storage probe on genuinely cold experts**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/expert_read_scaling.py --experts 512 --expert-offset 0 --wire-gib 0
```

**The all-resident compute floor** — the only clean read of what a token costs with no reads in it
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 1800 --tag resident -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_resident.py --prompt-tokens 16 --decode-tokens 16 --passes 4
```

**Where the 93 ms of compute goes**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 1800 --tag components -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/profile_decode_components.py --prompt-tokens 512
```

**DSpark draft acceptance** — the measurement that decides lever 1
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 16 --max-seconds 3600 --tag dspark -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/dspark_acceptance.py --prompt-tokens 128 --decode-tokens 96 --prompts 4 --draft-expert-bits 2
```

**What the draft itself costs** — run this on a quiet machine, nothing else on the GPU
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/dspark_draft_cost.py --blocks 40 --draft-expert-bits 2
```

**Speculation's bytes and the confidence-gated projection** — free, no model
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/speculation_bytes.py benchmarks/results/trace_routing_v7.trace.npz --budgets-gib 36,44 && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/speculation_policy.py --acceptance benchmarks/results/dspark_acceptance_q2.json --budget-gib 36 --draft-ms 30
```

**Budget and eviction-policy sweep** — free, no model
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/simulate_policies.py benchmarks/results/trace_routing_v7.trace.npz --budgets-gib 36,44,52 --expert-bytes 9953280 --orders first-come,popularity --policies lru,slru,lfu
```

**Screen candidate affine fits at a fixed format** — any width, and `--activations` scores on the model's
own recorded inputs rather than random unit vectors
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/quant_fit_screen.py --experts 24 --probes 8 --groups 128,64
```
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/quant_fit_screen.py --bits 3 --experts 16 --groups 64,128 --fits mlx,search,search+lsq --activations benchmarks/results/activations_moe_input_both.npz
```

**Record what the routed experts are multiplied by** — needed for any activation-weighted fit, and the only
tool here that looks at the model's own activations
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 2400 --tag acts -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/capture_activations.py --tokens 128 --prefill 128 --samples 64
```

**Add recordings together**, so a weighting is about the model and not about one document (no model loaded)
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/capture_activations.py --merge benchmarks/results/activations_moe_input.npz,benchmarks/results/activations_moe_input_code.npz --out benchmarks/results/activations_moe_input_both.npz
```

**Does free generation fall into a repetition loop?** — required for any numerics or sampling change
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 5400 --tag rep -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/repetition_quality.py --seeds 4 --max-new-tokens 900 --save-text /tmp/replies
```

**Does the generated code compile?** — the paired bank comparison, no GPU needed. Read the **compile**
column first; the error density is only defined over blocks that reached the end of the file and were not
cut off at the token cap, and a density marked `(floor)` comes from a checker that stops at the first error.
Section 7.4.1 is why.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py /tmp/replies_bank_a /tmp/replies_bank_b
```

**Generate the coding corpus against a bank** — twenty independent tasks, about 1.7 hours on FP4 at two
seeds. Swap `CACHALOT_EXPERT_BANK` for the arm under test and run both arms with `settle.sh` between.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 14400 --tag cq-fp4 -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 CACHALOT_MIRROR_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_MIRROR_FRACTION=0.10 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/coding_quality.py --seeds 2 --max-new-tokens 2000
```

**Score a run, or two against each other** — no GPU. Pass the run directories, not the replies: the manifest
is what lets the scorer refuse an unfinished arm and tell a snippet from a program.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py benchmarks/results/coding/<run-a> benchmarks/results/coding/<run-b>
```

**Is one token mis-ranked, or is everything blurry?** — the section 7.4.4 experiment, one forward pass, no
sampling and no compiler
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 3600 --tag rank-runtime -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/token_rank_probe.py --tokens 600
```

**Smoke the harness in ten minutes** before committing hours to it
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 3600 --tag cq-smoke -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/coding_quality.py --seeds 1 --max-new-tokens 1400 --only cpp-lru-cache,py-retry-decorator,cpp-string-split-snippet
```

**The prefetch-lead A/B** — the 2026-09-19 lever, four runs a side interleaved both ways, `settle.sh`
between. Put it in the background and kill it by the recorded PID, never by a `pgrep` pattern.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && nohup benchmarks/ab_predict_lead.sh > /tmp/ab_predict_lead.log 2>&1 & echo $! > /tmp/ab_predict_lead.pid
```

**Re-score the two saved arms** — no generation, seconds, and the command behind section 7.4.1
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py benchmarks/results/replies/lc_fp4 benchmarks/results/replies/lc_q3g64act
```

**The whole deciding gate for a bank, both arms matched** — this is what closed lever 0. Three seeds, 1400
tokens, the production sampling, `settle.sh` between arms, and the paired comparison at the end. It takes
about four hours with one arm served from the USB drive, and **both arms must run to completion**: FP4's
truncated two-seed arm is what put a 0.6 in this document for a day (section 7.4).
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && for arm in "lcgate-fp4 /Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts /tmp/lc_fp4" "lcgate-cand /path/to/candidate-bank /tmp/lc_cand"; do set -- $arm; benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 10800 --tag "$1" -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK="$2" CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/repetition_quality.py --seeds 3 --max-new-tokens 1400 --frequency-penalty 0.2 --penalty-window 128 --save-text "$3"; benchmarks/settle.sh --budget-gib 24; done && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py /tmp/lc_fp4 /tmp/lc_cand
```

**Build a 3-bit bank** (`mlx` fit: about 19 ms per expert, so I/O bound rather than fit bound)
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/build_affine_bank.py --out /Users/hamedprooshani/DeepSeek-V4.1-Flash-q3g64 --bits 3 --group 64 --fit mlx --verify 12
```

**Build the searched, activation-weighted 3-bit bank** — 166 ms per expert, 47 minutes, 221.5 GiB. **This
bank was built, gated, rejected and deleted on 2026-09-18 (section 9.0);** the command is kept because the
recipe is the evidence, and because it is the template for any future width. The output goes on the X10Pro
because the internal SSD has 68 GiB free, and the source is read from the internal FP4 copy so the reads stay
off the drive being written.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/build_affine_bank.py --model-path /Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts --out /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q3g64-act --bits 3 --group 64 --fit search-lsq --importance benchmarks/results/activations_moe_input_both.npz --verify 12
```

**Mirror striping A/B** — the 2026-09-19 lever. Four runs a side, interleaved both ways, `settle.sh` between;
drop `CACHALOT_MIRROR_*` for the off arm. The X10Pro is read only.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 1800 --tag mir-0.10 -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 CACHALOT_MIRROR_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_MIRROR_FRACTION=0.10 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64
```

**Where the time goes on the FP4 bank** — run this first whenever the mounted bank changes; every timing
conclusion in this document is a property of the bank, not of the runtime
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 3600 --tag fp4-anatomy -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64
```

**The FP4 compute floor** — `decode_resident.py` cannot measure it at 36 GiB (2,056 slots is under its own
working set); `profile_decode_components.py` times an all-resident token directly and reports 84.6 ms
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 2400 --tag fp4-components -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/profile_decode_components.py --prompt-tokens 512
```

**Speculation's economics on any expert size** — free, no model; this is what closed lever 1 on FP4
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/speculation_bytes.py benchmarks/results/trace_routing_v7.trace.npz --budgets-gib 36,44 --expert-bytes 18800640 --out benchmarks/results/speculation_bytes_fp4.json && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/speculation_policy.py --acceptance benchmarks/results/dspark_acceptance_conf.json --bytes benchmarks/results/speculation_bytes_fp4.json --budget-gib 36 --baseline-ms 325 --compute-fixed-ms 84.6 --draft-ms 25.3 --miss-scale 0.8654 --expert-bytes 18800640 --drive-gbps 5.97
```

**Settle memory between arms**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh --budget-gib 36
```

## 14. Files that matter

| path | role |
|---|---|
| `src/cachalot/storage/index.py` | expert bank detection, stacked and FP4 indexing, per-expert byte ranges; raises on mixed quantization |
| `src/cachalot/storage/reader.py` | positional reads into slot buffers, concurrent pieces, mirror striping |
| `src/cachalot/cache/resident_store.py` | residency, LRU, prefill admission planner, decode prefetch |
| `src/cachalot/cache/slots.py` | pre-allocated wired slot pool, any tensor set |
| `src/cachalot/model/expert_affine.py` | routed-expert math for affine banks; takes bits and group from the format |
| `src/cachalot/model/moe_layer_metal.py` | decode MoE path, routing prediction, `PREDICT_TOPK` and its two measured saddles |
| `src/cachalot/model/moe_prefill_grouped.py` | expert-major prefill, format branch |
| `src/cachalot/model/text_decode_runtime.py` | runtime assembly, bank selection, budget and wired limit |
| `benchmarks/quant_affine.py` | the packers (`pack_2bit`, `pack_bits` at any width) and the fits MLX does not provide, all of them width-agnostic and optionally activation-weighted |
| `benchmarks/build_affine_bank.py` | bank builder: one shard per layer, one expert in memory, resumable, self-verifying |
| `benchmarks/expert_requant_error.py` | the seconds-long screen for candidate formats |
| `benchmarks/nll_expert_precision.py` | the quality gate; dense arms and the production arm |
| `benchmarks/decode_anatomy.py` | blocked time split by cause, reads split by worker pool |
| `benchmarks/guarded_run.sh`, `benchmarks/settle.sh` | the memory guardian and the between-arms gate |
| `tests/test_quant_affine.py` | pins the packing against `mx.quantize`'s own layout at 2, 3, 4, 5, 6 and 8 bits, including the word-straddling case |
| `benchmarks/predictor_recall.py` | bounds the routing predictor offline: recall against width, against staleness, and by the router's own ranking; no GPU, no experts |
| `benchmarks/capture_activations.py` | records what the routed experts are multiplied by, per layer; `--merge` adds recordings without loading the model |
| `benchmarks/activation_importance.py` | recorded activations to the per-column weighting a fit takes; w2's is derived per expert from the SwiGLU hidden |
| `tests/test_activation_importance.py` | pins the weighting's orientation, w2's derivation, and that a weighting of all ones is the unweighted fit |
| `src/cachalot/model/dspark_draft.py` | the three DSpark stages, their window caches, the Markov and confidence heads; optional one-time quantization of the draft's own experts |
| `benchmarks/dspark_acceptance.py` | per-depth draft acceptance and per-block confidence |
| `benchmarks/dspark_draft_cost.py` | what one draft block costs, split by stage, head and Markov correction |
| `benchmarks/verify_forward_cost.py` | the cost of a K-position forward, measured on the prefill path |
| `benchmarks/speculation_bytes.py` | misses per forward against verification width, replayed from a trace |
| `benchmarks/speculation_policy.py` | the confidence-gated projection, and every assumption behind it |
| `benchmarks/decode_resident.py` | the all-resident compute floor |
| `benchmarks/quant_fit_screen.py` | candidate affine fits at one format, in minutes |
| `tests/test_dspark_draft.py` | pins the draft's attention index set, whose failure mode is a false null |
| `benchmarks/repetition_quality.py` | free-running collapse rate; canned-context and no-prefix-cache arms |
| `benchmarks/code_validity.py` | compiles generated code blocks; compile rate first, fatal rate second, density only over comparable blocks |
| `tests/test_code_validity.py` | pins the missing-header false success that made the gate report 8.9, and the block accounting behind it |
| `benchmarks/coding_tasks.json` | the coding corpus: twenty independent tasks, each declaring its language and whether it must compile alone |
| `benchmarks/coding_quality.py` | generates the corpus against one bank, one process, with the run manifest that makes a result traceable and an unfinished run refusable |
| `tests/test_coding_quality.py` | pins the corpus shape and the per-reply accounting: no code, wrong language, untagged fence, truncation |
| `benchmarks/ab_predict_lead.sh` | the interleaved prefetch-lead A/B, four runs a side both ways |
| `benchmarks/token_rank_probe.py` | teacher-forced rank of one suspected token, with a within-run control; separates a mis-ranked token from general blur without sampling or a compiler |
| `tests/test_sampling_penalties.py` | pins that the frequency penalty grows with the count and survives greedy |
| `benchmarks/run_reference.py` | run A, the diagnostic: the corpus against a hosted endpoint, no harness, one pinned provider, transport retried and content never retried |
| `benchmarks/export_corpus.py` | writes `~/cachalot-corpus-pack`: the prompts, the settings, the run script and the instructions another runtime is handed |
| `benchmarks/include_integrity.py` | the pre-registered include/import check; the one quality signal that is immune to truncation and to a compiler giving up |
| `~/cachalot-runA-relace-fp4-scored/` | the FP4 reference arm, 40 of 40, with `ANALYSIS.md` and the notes that say how the provider was pinned |
| `~/cachalot-runA-deepinfra-fp8-scored/` | the FP8 reference arm, 40 of 40 — the control that rules the expert format out |
| `~/cachalot-runA-relace-fp4-greedy-scored/` | six tasks at temperature 0, paired with Cachalot's greedy run |
| `~/cachalot-runB-hermes-scored/` | the Hermes harness arm, with the two sessions recovered from the Hermes database and an `ANALYSIS.md` explaining why it answers a different question |
| `benchmarks/quant_affine.py` | the fits; `fit_search`/`refine_lsq` work at any width, `dequantized()` screens without packing |
| `tests/test_bank_writer.py` | pins that the quantizer's output fills exactly what the shard header reserved |

### Session logs, for history

| document | what it holds |
|---|---|
| `docs/HANDOFF-2026-09-16.md` | the five changes that reached 4.3–5.6 tok/s; budget sweeps on the 3-bit bank; the panic postmortem |
| `docs/HANDOFF-2026-09-17.md` §1–9 | concurrency diagnosis, Engram null, prediction-knob nulls, segmented LRU |
| `docs/HANDOFF-2026-09-17.md` §10–15 | what decode blocks on; coverage versus timing; the three measurement traps; the real drive speed |
| `docs/HANDOFF-2026-09-17.md` §16–19 | the 2-bit bank, the fit, the gate, the width sweep, and the ranking this document replaces |
| `docs/HANDOFF-2026-09-17.md` §20–27 | the compute floor, DSpark and its acceptance, speculation's economics, the refined 2-bit fit, and lever 4 closed |
| `docs/HANDOFF-2026-09-20.md` | the reference-arm session: how the arms were run and pinned, the raw tables behind §7.4.6, and the Hermes arm's own analysis |
| `docs/HANDOFF-2026-09-20-quality.md` | the session that found the defect: every suspect eliminated and how, the three reading passes over the decode path against the shipped reference, the attention and value-delivery measurements, the transposed `hc_post`, and the clean gate. Also the two hypotheses that were wrong — a distance effect that was rarity confounded, and a constants mismatch that came from reading the wrong config file |
