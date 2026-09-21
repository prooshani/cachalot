# Next-session prompt — **v28**, written 2026-09-21

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v28** | 2026-09-21 | 0.9.0 was read live at a 52 GiB budget | **the rate moved where a session can see it**: prose 7.78-7.92 → **9.42 tok/s**, Objective-C 6.90-6.93 → **8.53**, session hit rate 90.00 → **92.37 %**, MLX peak 72.7 GiB against a 77.8 GiB limit, so **52 GiB fits**. `simulate_policies.py`'s +2.6-point prediction landed within a quarter of a point — the first time one was checked live. Of the 22-27 ms off a token the budget explains about 6 and the Engram change 16-21, **and the two were not separated**. The coding turn **does not compile**: one wrong method name made twice |
| v27 | 2026-09-21 | the 33 ms found a mechanism | the Engram row reads were serialised on the decode thread; +16 % on the benchmark rate, shipped in 0.9.0; `INDEX_TOPK` is a null |
| v26 | 2026-09-21 | the GPU side closed | attention is 22.5 ms; `mx.compile` on it closed three ways; the prelaunch-shared arm moves 10 ms and costs 13 |
| v25 | 2026-09-21 | the ranking rebuilt on measured sizes | two profiler rows were timing retired paths; the expert kernel is closed |
| v24 | 2026-09-21 | two live sessions on 0.7.0 | the live A/B is a null a third time; 46.9 % of bytes are unused predictions |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**The rate moved for the first time in five sessions and it is visible in a conversation.** Read
`docs/HANDOFF.md`'s opening block, then **7.2.6**, then **7.1.7**, then **9.24**, then 9.25. The first is
the live session and what it does and does not prove; the second is where a streaming token's time goes;
the third is the lever; the fourth is the ranking.

**The shipped configuration has changed by hand and not in code.** 52 GiB is now the budget to run:

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

Section 4's one-liner still says 44 and `chat.sh` still defaults to it. **Decide whether to make 52 the
default** — the evidence is one session (peak 72.7 GiB against 77.8 wired, no pressure event, +2.37 points
of hit rate) and the risk is the configuration class that panicked this machine twice in September, so it
wants a second session before it becomes the default rather than a flag Hamed types.

## Where the time is

A live prose token is **106.2 ms** at 9.42 tok/s (was 128.5 on 0.7.0 at 44 GiB); a coding token is 117.2.
A benchmark token at 40 GiB is 151 ms. The all-resident floor is **76.1-79.6 ms**, about 56 inside
`mx.eval` and 21.5 outside.

| block | size | state |
|---|---:|---|
| blocking on misses no prediction covered | ~19 ms at a 92.4 % session hit rate | every prediction lever closed; **the budget lever is now taken by hand, 44 → 52 GiB.** §9.4, §9.19 |
| **attention, all forty layers** | **22.5 ms** | largest GPU block; `mx.compile` closed three ways; **the shapes themselves have never been attacked** |
| **what a streaming token adds inside `mx.eval`** | **~20 ms** | **the only block with no mechanism.** §7.1.7 |
| routing prediction | 11 ms | closed |
| the 44 `mx.eval` round trips | ~9-12 ms | 0.20 ms each; the overlap arm moves 10 and costs 13. §7.1.6, §9.22 |
| routed experts, traced | 6.9 ms | closed |
| compressor, indexer, compressed-KV write | 5.5 ms | decomposed; `INDEX_TOPK` is a null. §7.1.8 |
| shared expert | 4.8 ms | **never screened** |
| hyper-connection glue | 4.2 ms | closed |
| streaming CPU above the floor | 4 ms | what is left of the old 33 |
| head | 1.6 ms | never attacked |
| Engram row reads | 2.4 ms, was 28.4 | **taken.** §9.24 |

## Job 1 — separate the two causes, because one session moved two things

Section 7.2.6 changed the budget and the Engram path at the same time. The arithmetic attributes about 6 ms
of the 22-27 to the budget and 16-21 to the Engram reads, and arithmetic is not a measurement. **One
conversation settles it**, at the same budget as the good session with the change switched off:

```bash
CACHALOT_ENGRAM_PARALLEL_MIN=1000000 CACHALOT_DECODE_ENGRAM_PREFETCH=0 CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

Ask it the same two long prompts — a 500-word story and the Objective-C JSON-to-CSV program — and compare
the long turns' tok/s and the session hit rate against 9.42, 8.53 and 92.37 %. **A 16-21 ms difference is
five times what a session resolves**, so this is one of the few live A/Bs this project can actually read.
The same A/B as a benchmark needs `52 + 29 = 81` GiB available, which the machine has never had; 44 GiB
needs 73 and was unavailable all of 2026-09-21, so **section 9.24 still has no benchmark at or above the
shipped budget**.

## Job 2 — the ~20 ms a streaming token adds *inside* `mx.eval`

Carried from v27 and now the only block with no mechanism. Section 7.1.7's table: against an all-resident
token, a streaming one at 36 GiB pays +63 ms blocked in the store, +1 in Engram (was +27), +4 of CPU and
**+20 inside `mx.eval`** — 77.3 ms against 56.9 for the same arithmetic.

The candidate is the GPU waiting on memory bandwidth the SSD DMA is also using, and it is testable without
writing a kernel:

1. **Vary the read concurrency.** `io_workers` is 8. If in-eval time falls and blocked time rises as it
   drops, the GPU is competing with the DMA; if the split does not move, it is not.
2. **Serve the same misses from the page cache.** A second streaming pass over the same continuation reads
   the same experts with no device traffic. If in-eval time returns to 57 ms, the device is the cause.
3. **Price the drain first.** A streaming token's evals have more queued behind them, and section 7.1.6's
   queue-depth arm is 0.218 ms behind one launch against 0.552 behind forty. Some of the 20 ms may just be
   the same work arriving later. Rule out before blaming bandwidth.

## Job 3 — attention's shapes, 22.5 ms and the largest GPU block

`mx.compile` is closed in all three of its shapes (§9.21); the shapes have never been touched.
`compressed_attention_decode_reuse` runs 30 times a token at 0.44-0.60 ms and its graph moves every token,
because `attention_kv` grows with the context and `window_slot = start_pos % 128` moves the window slice. A
fixed-capacity shape with a mask, instead of a growing slice, is the only untried idea in the document.
**Screen it before writing anything** with `benchmarks/micro_compile_attention.py` as the template, and
prove numerics with `decode_fingerprint.py` rather than arguing them.

## Job 4 — the shared expert, 4.8 ms and still never screened

Three FP8 gemvs per layer, 0.162 ms per launch. Carried from v26 and v27 unchanged.
`benchmarks/micro_expert_roofline.py` is the template for asking whether it is at the memory wall.

## Job 5 — the quality gate has no Objective-C in it, and both live Objective-C turns are broken

Two live coding turns have now been put through a compiler and both fail on a model-level type error:
0.7.0's declared a helper `NSString *` and handed it `id` values, and 0.9.0's sends `-stringValue` to an
`NSString` twice — once where `clang` catches it and once behind an `id`, where it aborts the program on the
model's own example input (section 7.2.6, `docs/live-turns/2026-09-21-json2csv/`). The 40-case corpus gate
compiles C++ and parses Python and **contains no Objective-C at all**, and nothing in it is executed.

Two cheap things, in order: add a handful of Objective-C cases to `benchmarks/coding_tasks.json` and compile
them in `code_validity.py` the way C++ is compiled; and consider whether the gate should *run* what it
builds on the cases that have no input. Neither is a speed job — but "reading a program by eye is not the
check it was being used as" is now two for two, and the gate is what every quality claim in the document
rests on.

## Job 6 — the last 4 ms of streaming CPU

The MoE eval's own CPU gap grows by about 2 ms under misses (16.4 ms all-resident against 18.3 streaming),
which is the store's admission path — slot acquisition, eviction, the LRU, `slot_views`. And
`kernel_consts.py:39` appears in the streaming arm at 0.5 calls per token with 1.7 ms of store-blocked time
behind it, which nobody has explained.

## What is closed, so nobody spends a session there

- **`INDEX_TOPK`.** 0.485, 0.474, 0.501 and 0.540 ms per source layer at 128, 256, 512 and 1024 — an
  eightfold change is 0.066 ms per layer and not monotone. §7.1.8, §11.
- **`mx.compile` on attention**, in all three of its shapes. §9.21.
- **The expert kernel.** The traced block costs what its three `mx.quantized_matmul` calls cost;
  `gather_qmm` is ≤1.5 ms and needs six LRU slots made contiguous. §7.1.4, §11.
- **Routing prediction**, every named way of making it cheaper. §9.19, §11.
- **Speculative decoding**: a K-position forward is 242.6 ms + 26.9 per extra position. §9.20.
- The hyper-connection glue, the router, dispatch fusion, 1 bit per weight, mirror striping on this bank,
  the frequency penalty, eviction policy. §11.

## Rules that still hold

- **A cost with no row in any instrument hides in plain sight.** The Engram reads survived four sessions of
  ranking because `decode_anatomy.py` buckets them into `rest`, `profile_decode_gpu.py` excludes them by
  construction and `profile_decode_sync.py` charged them to CPU. Before calling a block unexplained, give
  every blocking call on the decode thread its own column. §7.1.7.
- **The drive is shared, so measure I/O-touching code while it is busy.** The same Engram read is 1.7 ms
  all-resident and 28.4 ms streaming.
- **A threshold chosen for prefill can be wrong for decode.** The reader's pool started at 64 rows because
  prefill asks for 12k; decode asks for 24.
- **Check an offline simulation against a live run the first time one is possible.** `simulate_policies.py`
  said 44 → 52 GiB was worth 2.6 points and the session moved 2.37. Four sessions of hit-rate simulation had
  never been validated against anything. New this session. §9.4, §7.2.6.
- **A live session resolves its hit rate, and a change of 20 ms or more.** The old rule — that a session
  cannot see a speed change — was calibrated on 5-8 ms arms and is true for those. It is not a reason to
  skip a live read of something worth 20. Revised this session. §12.1, §7.2.6.
- **Two changes in one session leave arithmetic, not a measurement.** §7.2.6, job 1.
- **A chained profile excludes the per-eval floor by construction.** Add 44 x 0.20 ms back. §7.1.6.
- **Read which function an instrument calls before ranking a lever off it.** Four occurrences.
- **Check that a profile's parts add up to its whole before ranking anything off it.**
- **Never quote a screen as a gate** — and price a loop with one before ranking it.
- **Prove a speed arm's numerics, do not argue them.** `benchmarks/decode_fingerprint.py`. §9.24.
- **Write a multi-arm A/B to a file and run `bash the-file`.** §12.
- **Print what an arm read before attributing a difference to CPU work.**
- **A thread does not hide Python work** — and neither does `mx.async_eval`. §9.22.
- **Compile a live coding turn before calling it reference class.** Two for two now fail. §7.2.6, job 5.
- **An A/B only finds defects that differ between its arms.** Compare against
  `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/`.
- **Write a test when a comparison finds a match, not only when it finds a bug.**
- **Never drop a case from a denominator.**
- **A curve drawn through the turns of a live session is not a curve**, and a short turn is not a rate:
  8 tokens at 5.54 tok/s and 544 at 9.42 in the same session.
- **Read a run's footprint line before its numbers.**
- **Check a confound before reporting an effect.**
- **Memory.** `guarded_run.sh` needs `budget + 29` GiB available. On 2026-09-21, 36 and 40 GiB ran a dozen
  times, 44 was never available, and the machine sat at 67-74 GiB free with 1.25 GiB of stale swap. A *chat*
  session is not bound by that guard and peaked at 72.7 GiB of MLX with a 77.8 GiB wired limit. Never pass
  `--force`; lower the budget and say which one you used.
- **Hand over `./chat.sh`, never the section 4 one-liner.** §12.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once. Launch each run as its own command.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/profile_decode_sync.py --mode both` | **an all-resident token and a streaming one side by side**, each interval split into eval, store-blocked, Engram `pread` and CPU, per call site | ~1 min |
| `benchmarks/decode_anatomy.py` | where a live token's time goes by blocking cause, and the rate | ~1 min |
| `benchmarks/profile_decode_gpu.py` | the GPU side, every layer, the compressor and the indexer, plus what an eval costs to drain | ~1 min |
| `benchmarks/decode_fingerprint.py` | **whether a speed arm changed the numerics** | ~1 min |
| `benchmarks/micro_eval_floor.py` | what one `mx.eval` costs, and whether anything makes it cheaper | instant, no model |
| `benchmarks/micro_compile_attention.py` | the `mx.compile` screen for attention, retraces included | ~3 min |
| `benchmarks/simulate_policies.py --expert-bytes 9953280` | hit rate against budget, offline — **and it is now validated live to a quarter of a point** | instant, no model |
| `benchmarks/micro_expert_roofline.py` | the expert matmuls against `gather_qmm` and 240 distinct experts | instant, no model |
| `benchmarks/micro_compile_hc_fused.py` | **the template for a CPU screen** | ~1 min, no model |
| `benchmarks/nll_expert_precision.py --experts runtime --tokens 512` | the production quality arm | ~2 min |
| `benchmarks/coding_quality.py --resume` | the 40-case corpus, restartable — **C++ and Python only** | ~1.5 h |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | **the official implementation** |
| `benchmarks/results/coding/q2g128-v17/` | **the 2-bit gate**, 40/40 |
| `docs/live-turns/2026-09-21-json2csv/` | **the live coding turn that does not compile**, with the compiler output and the one-line repair |
| `benchmarks/results/guarded/anat40_base_*`, `anat40_new_*` | **the Engram A/B at 40 GiB**: 5.69 → 6.61 tok/s |
| `benchmarks/results/guarded/anateg_base_*`, `anateg_new_*` | the same A/B at 36 GiB |
| `benchmarks/results/guarded/eg_base_*`, `eg_par_*`, `eg_pre_*`, `eg_both_*` | the four arms, per call site |
| `benchmarks/results/guarded/fpeg_base_*`, `fpeg_both_*` | their identical fingerprints |
| `benchmarks/results/guarded/gpu36ci_*` | the compressor, the indexer and four `index_topk` widths |
| HANDOFF section 7.2.6 | **the live session on 0.9.0 at 52 GiB** |
| HANDOFF section 7.1.7 | where a streaming token's time goes |
| HANDOFF section 9.24 | the lever that shipped |
| HANDOFF section 9.25 | **the ranking** |
| `./chat.sh` | **the command to hand Hamed** — with `--expert-budget-gib 52` and `CACHALOT_MLX_WIRED_LIMIT_GIB=80` |
