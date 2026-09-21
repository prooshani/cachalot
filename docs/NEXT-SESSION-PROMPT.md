# Next-session prompt — **v22**, written 2026-09-21

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v22** | 2026-09-21 | the CPU third was attacked again, the plan in v21 was wrong, and the runtime was then run interactively on both arms | memoising the kernels' scalar parameters ships at **1.2 ms** and the CPU side is **20.7 ms**; **tracing the hyper-connection glue is worth 0.2 ms and the router 0.3**, so v20's and v21's Job 1 is closed; **routing prediction is 11 ms of the 77 ms floor** and is now the largest named item; moving its submission off the decode thread is a null the GIL explains; and two live sessions found that **decode falls 29 % from a 545-token reply to a 1,788-token one**, which every number in this document was blind to |
| v21 | 2026-09-21 | the traced runtime was run interactively, both arms | two full sessions at 7.76 and 7.82 tok/s with reference-class output; the live A/B is a null by construction; the section 4 one-liner does not survive line wrapping, so `./chat.sh` is what gets handed over |
| v20 | 2026-09-21 | the CPU third was attacked with `mx.compile` | the decode MoE block traces once instead of forty times a token: 82-85 ms to **77 ms**, CPU 27 to 21.5 |
| v19 | 2026-09-21 | the compute floor was measured properly | hyper-connections are 4.6 ms, not 68.7; a token is 65 % GPU wait and 35 % CPU graph building |
| v18 | 2026-09-21 | the re-audit finished and speed reopened | the 2-bit bank ships at 7.6 tok/s; frequency penalty and mirror striping off; 1-bit closed |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read `docs/HANDOFF.md`'s opening block, then 7.2.4, then 9.18, then 9.17, then section 11's kernels
block.** The first tells you what ships; the second is the live read and the one new effect this session
found; the third is where the remaining time is and the reason the last two prompts pointed at the wrong
thing; the fourth is what shipped; the fifth is three nulls that were each a ranked job an hour earlier.

## Where the project stands

The shipped configuration is unchanged: the 2-bit g128 bank at a 44 GiB budget with the hotlist, no mirror,
no frequency penalty. Quality is gated at 20/20 compiling C++ blocks and 0/94 malformed includes, equal to
the hosted reference, and the production arm is 2.2356 nats / 52.0 % top-1 at 512 tokens — unchanged again
this session, to four decimals.

**One change shipped, and it is small.** Every fused Metal kernel in the decode path was handed its row
count, its epsilon and its scale as freshly built one-element `mx.array`s, about **1,600 constructions per
token**. They are constants. `src/cachalot/model/kernel_consts.py` builds each distinct one once. Three runs
a side, interleaved with `settle.sh`: CPU outside eval 22.3/22.4/21.6 ms against **21.0/21.1/20.7**, medians
23.5/23.1/21.8 against **21.7/21.7/21.6**, the gap before the router eval 21.1/21.1/20.4 against
**19.6/19.5/20.0** — three non-overlapping statistics at **1.2-1.3 ms**. The whole token overlaps, as a 1 ms
change must. NLL identical to four decimals, 219 tests, live anatomy moved only in `rest` (125.2 to
123.4 ms/token) with the hit rate flat at 73.4 %. Section 9.17.

**The bigger result is a measurement, and it retires the plan the last two prompts carried.**

| | v21 said | **measured 2026-09-21, third session** |
|---|---|---|
| trace the hyper-connection glue | "start here, same shape as the change that shipped" | **0.84 ms per token, of which 0.63 is taken by memoising the constants: 0.2 ms incremental.** v21's screen timed `hyper_connection_mlx`; the runtime is on the fused Metal kernels |
| trace the router pass | not priced | **0.30 ms per token** |
| the prediction submission path | "3.9 ms, moving it off the decode thread is unexplored" | **routing prediction is 11 ms of the 77 ms floor** — 7 of CPU, 4 of GPU, split evenly between computing it and submitting it |
| moving that submission to its own thread | "an unexplored three-figure-millisecond-per-session idea" | **a null, and worse on the median.** The GIL |
| CPU outside eval | 21.5 ms | **20.7 ms** |

The fused kernels were already the fix for graph-construction cost in the glue, so there was no second
helping there. What is left on the CPU is not Python building operations; it is MLX carrying work through
each token's forty evals, and prediction is what puts most of it there.

**It was then run interactively, one session per arm, and it behaves.** `./chat.sh --max-new-tokens 2000`,
same four prompts: 7.86 and 7.95 tok/s on the story turns, 6.94 and 5.61 on the Objective-C turns, hit rate
90.03 % against 89.90 %, prediction precision 33.15 % against 33.09 %. **The live A/B is a null by
construction** — 1.2 ms on a 126-178 ms token is 0.7-1.0 %, and the matched prompt came out 1.1 % in favour
of the arm *without* the change. Quality is the reference class on both: coherent 600-token stories with a
name held from the first line to the last, Objective-C with `#import <Foundation/Foundation.h>` intact, RFC
4180 quote doubling and no malformed identifier. Section 7.2.4.

**Two loose threads closed by those sessions.** Raising the cap to 2000 works — both coding turns finished
on `stop=stop` at 1,483 and 1,788 tokens instead of being cut mid-method — and `chat.sh` now defaults to it.
And session A answering a bare "Hi" in Chinese is **sampling, not an artefact**: `chat.sh` passes no system
prompt, the text is clean Chinese with an emoji, and both second turns are English. Do not chase it.

**Nothing is mid-flight.** Clean tree, 219 tests passing, no background jobs.

## The lessons this session added

The five from before still hold, and the fifth — *a number is only as current as the code path it was
measured on* — cost this session its first hour: `benchmarks/micro_compile_hc.py` had been sitting in the
repository as evidence for a ranked job, and it times a module the runtime stopped calling. **Two prompts
ranked a 0.2 ms lever first on the strength of a screen nobody re-read.** Before ranking a lever off a
script, open the script and check which import it exercises.

The seventh: **a thread does not hide Python from the decode loop.** CPython holds one lock; moving the
prediction submission to a dedicated worker left the same bytecode running in the same interpreter and
added a handoff, and the median got worse. Offloading is only a lever for work that releases the GIL —
reads, and time inside MLX.

## Job 1 — why a long reply decodes 29 % slower than a short one

**This is new, it is large, and every number above section 7.2.3 is blind to it.** Seven long turns across
four sessions, all on the shipped configuration:

| tokens in the reply | context at the last token | tok/s |
|---:|---:|---:|
| 545 | ~600 | **7.95** |
| 602 | ~660 | 7.86 |
| 643 | ~700 | 7.76 |
| 889 | ~950 | 7.82 |
| 1,024 | ~1,700 | 6.95 / 6.76 |
| 1,483 | ~2,160 | 6.94 |
| 1,788 | ~2,410 | **5.61** |

The floor is 77 ms and it was measured at a 512-token context, which is the short end of that table. A 44
GiB interactive turn at 2,400 tokens of context is **178 ms per token**. Two causes are candidates and
nobody has separated them: compressed attention's shapes grow with the context, and a longer turn touches
more experts, so the hit rate falls as the reply runs. **They are separable in three runs** —
`decode_anatomy.py` already splits blocked time from `rest`, so run it at `--prompt-tokens` 512, 1024 and
2048 and read which term grows. If it is `rest`, it is attention and Job 3 is the fix. If it is blocked
time, it is coverage and Job 2 is. **Do this first: it decides which of the other two jobs matters.**

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 3600 --tag anat2048 -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py --prompt-tokens 2048 --decode-tokens 64
```

## Job 2 — make routing prediction cheaper, because it is 11 ms of the floor

Section 9.18 has the anatomy. Prediction is a good trade on the live path — it serves 41.8 of 46 misses
early — but it is not free, and nobody has looked for a cheaper way to buy the same hit rate.

1. **Admit the mispredicted bytes instead of dropping them.** This is v21's Job 2 and it is still the best
   idea on the list, now with a bigger denominator: 67.4 % of predicted loads are wasted and 42.7 % of all
   bytes read, and a mispredicted load is read, completed and then discarded — `prefetch_decode` gives it a
   transient slot and `_sweep_inflight_locked` releases it. The bytes are in memory. **The decisive
   measurement is offline**: instrument the store to record, for each expired prediction, whether the same
   key is requested within the next few tokens. A low rate closes it for the cost of one benchmark.
2. **Predict fewer, better experts.** Top-6 was tuned on a 341 ms token that read 1,858 MiB (section 9.12);
   the token is now 192 ms reading 945, and the width that pays is a function of what a wasted read costs.
   `benchmarks/predictor_recall.py` is offline and free; the A/B is `decode_anatomy.py`.
3. **The submission itself, made cheaper rather than moved.** 3.6 ms per token across 240 `submit` calls,
   240 dict lookups and forty `.tolist()`s. One submit per layer carrying six entries — inside the store,
   not on another thread — is the shape that the GIL does not defeat.

Measure the CPU side with `profile_decode_sync.py` and nothing else; its run-to-run spread is about 1 ms,
which is what made a 1.2 ms change provable when a throughput A/B could not have resolved it. Three runs a
side, interleaved, `settle.sh` between:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 1800 --tag syncprof -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/profile_decode_sync.py --prompt-tokens 512
```

Three env switches exist for bisecting this and they are diagnostic, not configurations:
`CACHALOT_PREDICT_TOPK=0` (no prediction), `CACHALOT_PREDICT_SUBMIT=0` (compute it, throw it away),
`CACHALOT_KERNEL_CONSTS=0` (rebuild every kernel parameter).

## Job 3 — attention, the one piece of the token still unmeasured on the CPU

It is the only remaining `mx.compile` candidate and the only part of a layer nobody has priced on the CPU
side. Its shapes grow with the context, so a plain trace retraces every token; `mx.compile(shapeless=True)`
has never been tried in this repository. **Screen it before writing anything** — the two screens this
session ran cost twenty minutes and closed two ranked jobs, and Job 2 above may say attention is where the
long-turn cost lives. `benchmarks/micro_compile_hc_fused.py` is the template: time construction with no eval in the loop, chain 40 launches for the GPU side, and assert
bit-identical output on all arms.

## Job 4 — the loose threads, still open, still cheap

1. **A stray character.** Turn 2 of an earlier session returned `wHi! How can I help you today?`. It has
   now failed to reproduce in **four** clean sessions. Four greedy repeats plus `token_rank_probe.py`
   settles it in ten minutes, or write it off. Sections 7.2.2, 7.2.3 and 7.2.4.
2. **Typing-time prefill**: **169-288 ms per token** across four sessions, against a batched turn's
   90-116 ms. Measured five times and never explained, and the range is now wide enough that whatever
   causes it is not a constant.
3. **The shipped configuration has still never been profiled.** 44 GiB needs 73 GiB available and was
   refused on two successive days; 40 passes and costs about 3 points of hit rate. Note that a 512-token
   `nll_expert_precision.py` run was killed at 24 GiB on 2026-09-21 and passed at 20.

**Closed by the sessions of 2026-09-21.** `--max-new-tokens 1024` cutting coding turns mid-method: raised
to 2000 in `chat.sh`, and both turns then finished on `stop=stop`. A bare "Hi" answered in Chinese:
sampling with no system prompt, not an artefact.

## What is closed, so nobody reopens it

- **Tracing the fused hyper-connection glue** (0.2 ms incremental) and **tracing the router pass** (0.3 ms).
  Section 11. The v20 and v21 Job 1 is finished, and the answer was "almost nothing".
- **Moving the prediction submission to another thread.** A null and worse on the median; the GIL. Section 11.
- **The eight source and index-source layers as the missing time.** 4.5 ms of excess. Section 6.3.1.
- **Fusing the layer's router with the predictor's.** 0.6 ms per token. Section 11.
- **Dispatch fusion** (4.72 us per dispatch), **a fused multi-expert affine kernel** (≤3 ms), **splitting
  `hc_mixes` across threadgroups** (1 ms), **1 bit per weight**, **mirror striping on this bank**, **the
  frequency penalty**, **lever 3**, **eviction policy**. Sections 9.2, 11, 9.8, 9.11.1, 9.9, 9.3.

## Rules that still hold

- **Never quote a screen as a gate.** Wrong six times.
- **Check that a profile's parts add up to its whole before ranking anything off it.**
- **Check which code path a number was measured on before ranking a lever off it** — including which module
  an existing benchmark script imports. It cost this session an hour and it has now withdrawn three figures.
- **A thread does not hide Python work.** New this session.
- **An A/B only finds defects that differ between its arms.** Compare against
  `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/`.
- **Write a test when a comparison finds a match, not only when it finds a bug.**
  `tests/test_kernel_consts.py` pins the memoised parameters the way `tests/test_expert_bank.py` pins the
  traced MoE block and the slot views.
- **Never drop a case from a denominator.**
- **Check a confound before reporting an effect.**
- **Memory.** `guarded_run.sh` needs `budget + 29` GiB available. Never pass `--force`; lower the budget.
  A 512-token `nll_expert_precision.py` run was killed at a 24 GiB budget on 2026-09-21 and passed at 20.
- **Hand over `./chat.sh`, never the section 4 one-liner.** It does not survive line wrapping and the
  failure is silent — FP4 over USB at 0.6 tok/s. Section 12.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once. Launch each run as its own command.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/profile_decode_sync.py` | **the CPU/GPU split of a token, attributed to each eval site** | ~1 min |
| `benchmarks/profile_decode_layers.py` | per-layer and per-class time with no barrier added | ~1 min |
| `benchmarks/profile_decode_gpu.py` | per-piece GPU time the way a token pays it, chained | ~2 min |
| `benchmarks/profile_decode_cpu.py` | the same token under cProfile | ~1 min |
| `benchmarks/profile_decode_components.py` | one piece against another implementation of it — **not a breakdown** | ~4 min |
| `benchmarks/micro_compile_moe.py` | what `mx.compile` is worth on the MoE block | ~1 min, no model |
| `benchmarks/micro_compile_hc_fused.py` | **the template for a CPU screen**: the fused glue, three arms, construction and chained | ~1 min, no model |
| `benchmarks/micro_compile_router.py` | what tracing the router pass is worth | ~1 min, no model |
| `benchmarks/micro_router_dual_gate.py` | the two router passes, separately and stacked | ~1 min, no model |
| `benchmarks/decode_anatomy.py` | where a live token's time goes, split by blocking cause | ~1 min |
| `benchmarks/nll_expert_precision.py --experts runtime --tokens 512` | the production quality arm | ~2 min |
| `benchmarks/continuation_rank.py` | first-vs-repeat copy rank | instant |
| `benchmarks/chat_turns.py --max-new-tokens 160` | six chat turns exactly as the CLI runs them | ~3 min |
| `benchmarks/coding_quality.py --resume` | the 40-case corpus, restartable | ~1.5 h |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | **the official implementation** |
| `~/cachalot-runA-relace-fp4-scored/` | the hosted reference arm, 40/40 |
| `benchmarks/results/coding/q2g128-v17/` | **the 2-bit gate**, 40/40 |
| `benchmarks/results/guarded/sync-noconst-*`, `sync-const-*` | the memoised-constant A/B, six runs |
| `benchmarks/results/guarded/sync-nopredict-*`, `sync-nosubmit-*` | the prediction pricing, four runs |
| `benchmarks/results/guarded/sync-inline-*`, `sync-async-*` | the off-thread submission null, six runs |
| `benchmarks/results/guarded/nll-kconsts_*`, `anat-kconsts_*` | the quality gate and the live sanity read |
| HANDOFF section 7.2.4 | the two interactive sessions, turn by turn, and the reply-length table |
| `./chat.sh` | **the command to hand Hamed**: the shipped environment, exported, one runtime at a time |
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128/` | the 143 GiB mirror copy — **no longer used**, delete it if the drive is wanted |
