# Next-session prompt — **v19**, written 2026-09-21

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v19** | 2026-09-21 | the compute floor was measured properly | hyper-connections are **4.6 ms of a token, not 68.7** — the old figure was an artifact of the profiler; a token is **65 % GPU wait and 35 % CPU graph building**; dispatch fusion is closed at a 4.72 us floor; the affine view cache shipped at −2.5 ms |
| v18 | 2026-09-21 | the re-audit finished and speed reopened | the 2-bit bank ships at 7.6 tok/s; frequency penalty and mirror striping off; 1-bit closed |
| v17 | 2026-09-20 | the gate came back clean | quality restored and measured, 20/20 compiling |
| v16 | 2026-09-20 | the root cause found | `hc_post` applied the hyper-connection mix transposed |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read `docs/HANDOFF.md`'s opening block and then section 6.3.** The first tells you what ships; the second
is the measurement that this whole work queue is built on, and it withdraws a number the document quoted for
three sessions.

## Where the project stands

The shipped configuration is unchanged: the 2-bit g128 bank at a 44 GiB budget with the hotlist, no mirror,
no frequency penalty, 7.55-7.62 tok/s in Hamed's own session at a 90.2 % hit rate. Quality is gated at 20/20
compiling C++ blocks and 0/94 malformed includes, equal to the hosted reference.

**What changed is the understanding of where the time goes, and it inverts the previous ranking.**

| | v18 believed | **measured 2026-09-21** |
|---|---|---|
| hyper-connections | 68.7 ms per token, "the largest addressable block" | **4.6 ms per token** |
| the compute profile that said so | a breakdown | an artifact: it evaluates each piece behind its own barrier, and its rows sum to 164 ms against a 94 ms token |
| dispatch count (lever 2) | open, reopened when bytes came down | **closed**: the per-dispatch floor is 4.72 us, so the whole prize is ~1.1 ms |
| where a token goes | compute, unspecified | **65 % inside `mx.eval`, 35 % CPU building the next graph with the GPU idle** |
| the all-resident floor, 2-bit | 93.0 ms | 93.6 ms, confirmed |
| production-arm quality, 2-bit | 2.5187 nats / 44.5 % top-1 (measured through the defect) | **2.2356 nats / 52.0 % top-1** |

**One change shipped.** The affine expert path rebuilt nine `.view().reshape()` pairs per expert per layer —
4,320 MLX op constructions per token. They are zero-copy and keep aliasing the slot, so `ExpertSlot.typed`
now holds them. Three interleaved runs a side, non-overlapping ranges on all four statistics: the token falls
2.5-3 ms and the CPU side 2.6 ms. Identical NLL to four decimals. Section 9.15.

**Nothing is mid-flight.** Clean tree, 215 tests passing, no background jobs, and `main` is pushed to
`origin/main` as of 2026-09-21 — the twenty-commit backlog that had stood since before the hyper-connection
fix is cleared, so a session can now be compared against a published history.

## The lesson this session added

The three from before still hold. The fourth: **an instrument that puts a barrier around each piece is
measuring the barrier.** `profile_decode_components.py` prints its own warning — "sum of isolated pieces
164.3 ms; whole token 93.6 ms" — and three sessions read the rows as a breakdown anyway. The repository
already contained `profile_decode_gpu.py`, which measures the same kernels chained, and it disagrees by a
factor of fifteen on hyper-connections. **Before ranking anything off a profile, check that its parts add up
to its whole.**

## Job 1 — the unprofiled half of the eval window

**This is the largest unexamined block and it is pure measurement.** The router eval holds 57.4 ms per token,
and the pieces profiled so far account for roughly 25 ms of it. The rest has never been timed:

1. The four `SOURCE_LAYERS` (2, 8, 14, 20) and the four `INDEX_ONLY_SOURCE_LAYERS` (24, 28, 32, 36), which
   run the indexer. A plain reuse layer's attention is 0.44-0.49 ms; nobody knows what these cost.
2. Engram, the final head, the two sliding-window layers.
3. The predictor's 40 extra `route_topk` launches, which are 6.8 ms by themselves and share their input with
   the layer's own router — **two dispatches over the same vector with two gate matrices.** Fusing them is
   bounded at about 3 ms and is the one dispatch-level change still worth pricing.

Add rows for these to `benchmarks/profile_decode_gpu.py`, which now takes `--prompt-tokens` and handles an
affine bank. It is the right instrument; it just does not cover these layers yet.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 1800 --tag gpuprof -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/profile_decode_gpu.py --prompt-tokens 512
```

## Job 2 — the rest of the CPU third

26 ms per token remain, at 0.65 ms per layer, spent building MLX operations while the GPU has nothing queued.
Section 9.15 has what is known: prediction submission is 3.9 ms of it, measured by turning it off. The
starting instrument is

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 1800 --tag cpuprof -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/profile_decode_cpu.py --prompt-tokens 512 --tokens 20
```

with the caveat that MLX's nanobind calls are invisible to cProfile and their time lands in the calling
Python function's `tottime` — which is why `moe_layer_forward` appears to own 21 ms it does not own alone.
**Measure the CPU gap with `profile_decode_sync.py`, not with cProfile**; its run-to-run spread is about
1 ms, which is what made a 2.5 ms change provable when a throughput A/B could not have resolved it.

Two structural ideas nobody has priced: moving prefetch submission off the main thread, and whether any part
of layer L+1's graph can be built before layer L's router eval returns (MLX is define-by-run, so probably
not, but the question has never been asked).

## Job 3 — the loose threads v18 left, both still open and both cheap

1. **A stray character.** Turn 2 of the 2026-09-21 session returned `wHi! How can I help you today?`.
   `chat_turns.py` decodes the same prompt greedily and returns it clean. Four greedy repeats plus
   `token_rank_probe.py` settles it in ten minutes. Section 7.2.2.
2. **Typing-time prefill**: 249 ms per token against a batched turn's 90-116 ms. It hides behind human
   typing, but nobody has asked why it is 2.5x.

## What is closed, so nobody reopens it

- **Dispatch fusion.** The per-dispatch floor is 4.72 us and a token issues ~230 substantial dispatches.
  Section 9.2.
- **A fused multi-expert affine kernel.** `gather_qmm` on pre-stacked weights beats the shipped loop by
  20-25 %, which is at most 3 ms, and it ignores the cost of making six LRU slots contiguous. Section 11.
- **Splitting `hc_mixes` across threadgroups**: 3.2 ms per token becomes 2.2. Section 11.
- **1 bit per weight**, **mirror striping on this bank**, **the frequency penalty**, **lever 3**. Unchanged
  from v18; sections 9.8, 9.11.1, 9.9, 9.3.

## Rules that still hold

- **Never quote a screen as a gate.** Wrong six times.
- **Check that a profile's parts add up to its whole before ranking anything off it.** New, and it cost three
  sessions.
- **An A/B only finds defects that differ between its arms.** Compare against
  `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/`.
- **Write a test when a comparison finds a match, not only when it finds a bug.** The view cache is correct
  only because MLX's `.view().reshape()` is zero-copy; `tests/test_expert_bank.py` now pins that, so a future
  MLX that copies fails the suite instead of serving one expert's weights under another's name.
- **Never drop a case from a denominator.**
- **Check a confound before reporting an effect.** The first run of the fused-expert screen showed 79 % and
  the next four showed 20-25 %; the first arm was cold.
- **Memory.** `guarded_run.sh` needs `budget + 29` GiB available. **44 was refused again on 2026-09-21** at
  70.6 GiB with nothing left to close, so the shipped configuration has still never been profiled; 40 passes
  and costs about 3 points of hit rate. Never pass `--force`.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once. Launch each run as its own command.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/profile_decode_sync.py` | **the CPU/GPU split of a token, attributed to each eval site** | ~1 min |
| `benchmarks/profile_decode_gpu.py` | per-piece GPU time the way a token pays it, chained | ~2 min |
| `benchmarks/profile_decode_cpu.py` | the same token under cProfile | ~1 min |
| `benchmarks/profile_decode_components.py` | one piece against another implementation of it — **not a breakdown** | ~4 min |
| `benchmarks/decode_anatomy.py` | where a token's time goes, split by blocking cause | ~1 min |
| `benchmarks/continuation_rank.py` | first-vs-repeat copy rank — the number that found the 2026-09-20 bug | instant |
| `benchmarks/nll_expert_precision.py --experts runtime --tokens 512` | the production quality arm | ~4 min |
| `benchmarks/chat_turns.py --max-new-tokens 160` | six chat turns exactly as the CLI runs them | ~3 min |
| `benchmarks/micro_affine_expert_fusion.py`, `micro_hc_mixes_parallel.py`, `micro_affine_cpu.py` | the three screens run on 2026-09-21 | ~1 min each |
| `benchmarks/coding_quality.py --resume` | the 40-case corpus, restartable | ~1.5 h |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | **the official implementation** |
| `~/cachalot-runA-relace-fp4-scored/` | the hosted reference arm, 40/40 |
| `benchmarks/results/coding/q2g128-v17/` | **the 2-bit gate**, 40/40 |
| `benchmarks/results/guarded/syncsite-*`, `ab-base-*`, `ab-cached-*` | the 2026-09-21 CPU-third measurements and the shipped A/B |
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128/` | the 143 GiB mirror copy — **no longer used**, delete it if the drive is wanted |
