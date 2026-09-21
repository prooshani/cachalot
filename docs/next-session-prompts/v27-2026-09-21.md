# Next-session prompt — **v27**, written 2026-09-21

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v27** | 2026-09-21 | the 33 ms found a mechanism and the rate moved for the first time in five sessions | **the block three prompts called the largest unexplained cost in the project is the Engram row reads**: a token asks for 24 rows twice, each row is two `pread`s, and the reader only used its worker pool at 64 rows or more, so 96 reads went out one at a time from the decode thread behind a queue the expert stream was filling. Through the pool and issued at the top of the token instead: **5.69 → 6.61 tok/s at 40 GiB and 5.65 → 6.51 at 36**, `rest` down 24 ms, hit rate, bytes, misses and prediction precision identical to the digit, fingerprint identical. Shipped in **0.9.0**. The source layers' 5.5 ms is decomposed and **`INDEX_TOPK` is not a lever** |
| v26 | 2026-09-21 | the GPU side closed, one lever built and measured | the 20 ms of unattributed GPU time is gone; attention is 22.5 ms; `mx.compile` on attention closed three ways; the prelaunch-shared arm moves 10 ms and costs 13 |
| v25 | 2026-09-21 | the ranking rebuilt on measured sizes | two profiler rows were timing retired paths; the expert kernel is closed |
| v24 | 2026-09-21 | two live sessions on 0.7.0 | the live A/B is a null a third time; 46.9 % of bytes are unused predictions |
| v23 | 2026-09-21 | prediction levers closed, shipped budget profiled | admission 4.3 % ceiling, blocklist a loss, submission 0.040 ms |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Something finally shipped.** Read `docs/HANDOFF.md`'s opening block, then **7.1.7**, then **9.24**, then
**7.1.8**, then 9.25. The first is where a streaming token's time actually goes, measured against an
all-resident one in the same process; the second is the lever and its two arms; the third is the source
layers' extra millisecond, decomposed; the fourth is the ranking as it now stands.

**What was wrong with the old picture.** Three analyses in a row said a streaming token spends about 33 ms
more outside the store's blocking calls than an all-resident one and that nothing explained it. Nothing
explained it because no instrument had a row for it: `decode_anatomy.py` buckets it into `rest`,
`profile_decode_gpu.py` excludes the Engram row read by construction, and `profile_decode_sync.py` called
it CPU. It was the two Engram row reads — 1.7 ms per token on an idle drive, 28.4 ms while the expert
stream saturates the same device.

## Where the time is, after this session

A live prose token was **128.5 ms** on 0.7.0 and **0.9.0 has not been run interactively**. A benchmark
token at a 40 GiB budget is **151 ms** (6.61 tok/s), down from 176. The all-resident floor is
**76.1-79.6 ms**, about 56 inside `mx.eval` and 21.5 outside it.

| block | size | state |
|---|---:|---|
| blocking on misses no prediction covered | 55.4 ms at 83.7 % hit, ~25 at a session's 90 % | every prediction lever closed. **A larger budget is the only untried one and needs no code.** §9.4, §9.19 |
| **attention, all forty layers** | **22.5 ms** | largest GPU block; `mx.compile` closed three ways; **the shapes themselves have never been attacked** |
| **what a streaming token adds inside `mx.eval`** | **~20 ms** | new, and the only block left without a mechanism. §7.1.7 |
| routing prediction | 11 ms | closed |
| the 44 `mx.eval` round trips | ~9-12 ms | 0.20 ms each; the overlap arm moves 10 and costs 13. §7.1.6, §9.22 |
| routed experts, traced | 6.9 ms | closed |
| compressor, indexer, compressed-KV write | 5.5 ms | decomposed: two thirds indexer, a tenth compressor. `INDEX_TOPK` is a null. §7.1.8 |
| shared expert | 4.8 ms | **never screened** |
| hyper-connection glue | 4.2 ms | closed |
| **streaming CPU above the all-resident floor** | **4 ms** | what is left of the old 33 after the Engram reads |
| head | 1.6 ms | never attacked |
| Engram row reads on the decode thread | 2.4 ms, was 28.4 | **taken.** §9.24 |

## Job 1 — read 0.9.0 live, and raise the budget while you are there

Two things at once, because a live session resolves neither on its own and this run gives both. The Engram
change has never run interactively, and the budget is still the only lever a live session can see
(section 9.4: 44 → 52 GiB buys 2.6 points of decode hit rate, 52 → 60 another 2.3, and a chat session peaks
at 59.2-59.7 GiB against a 72 GiB wired limit).

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

**Read the session hit rate against 90.0 %, not the tok/s of any one turn** — and expect the Engram gain to
be smaller live than on a benchmark, because it scales with how busy the drive is and a live session misses
less. The rate is measured with `decode_anatomy.py`, not in chat. 44 GiB was not available to the guard on
2026-09-21 (70 GiB free against the 73 it needs), so **neither arm of section 9.24 has been run at the
shipped budget**; that is a benchmark worth having.

## Job 2 — the ~20 ms a streaming token adds *inside* `mx.eval`

This is the new first-among-unexplained. Section 7.1.7's table: against an all-resident token, a streaming
one at 36 GiB pays +63 ms blocked in the store, +27 in Engram (now +1), +4 of CPU and **+20 inside
`mx.eval`** — 77.3 ms against 56.9. The GPU is doing the same arithmetic in both arms.

The candidate mechanism is the one section 7.1.3 named and never tested: the GPU waiting on memory
bandwidth the SSD DMA is also using. It is testable without writing a kernel.

1. **Vary the read concurrency and watch the split.** `io_workers` is 8. If in-eval time falls and blocked
   time rises as the workers drop, the GPU is competing with the DMA; if the split does not move, it is not.
2. **Serve the same misses from the page cache.** A second streaming pass over the same continuation reads
   the same experts from a warm file cache with no device traffic. If in-eval time returns to the
   all-resident 57 ms, the device is the cause.
3. **Check it is not the drain.** A streaming token's evals have more queued behind them; section 7.1.6's
   queue-depth arm says one eval behind 40 launches is 0.552 ms against 0.218 behind one, so part of the
   20 ms may simply be the same work arriving later. Price that before blaming bandwidth.

## Job 3 — attention's shapes, 22.5 ms and the largest GPU block

`mx.compile` is closed in all three of its shapes (§9.21) and the shapes themselves have never been
touched. `compressed_attention_decode_reuse` is called 30 times a token at 0.44-0.60 ms; its graph moves
every token because `attention_kv` grows with the context and `window_slot = start_pos % 128` moves the
window slice. A fixed-capacity shape with a mask, instead of a growing slice, is the only untried idea in
the document. **Screen it before writing anything**, with `benchmarks/micro_compile_attention.py` as the
template, and prove numerics with `decode_fingerprint.py`, not by argument.

## Job 4 — the shared expert, 4.8 ms and still never screened

Three FP8 gemvs per layer, 0.162 ms per launch on the run in section 7.1.8. Carried from v26 unchanged.
`benchmarks/micro_expert_roofline.py` is the template for asking whether it is at the memory wall.

## Job 5 — the last 4 ms of streaming CPU

Small, and now it is the whole of the CPU excess. Two things are visible in the per-site table: the MoE
eval's own CPU gap grows by about 2 ms under misses (16.4 ms all-resident against 18.3 streaming), which is the store's admission path (slot acquisition,
eviction, the LRU, `slot_views`), and `kernel_consts.py:39` appears in the streaming arm at 0.5 calls per
token with 1.7 ms of store-blocked time behind it, which nobody has explained.

## What is closed, so nobody spends a session there

- **`INDEX_TOPK`.** 0.485, 0.474, 0.501 and 0.540 ms per source layer at 128, 256, 512 and 1024 — an
  eightfold change in the width is 0.066 ms per layer and not monotone. §7.1.8, §11.
- **`mx.compile` on attention**, in all three of its shapes. §9.21.
- **The expert kernel.** The traced block costs what its three `mx.quantized_matmul` calls cost;
  `gather_qmm` is ≤1.5 ms and needs six LRU slots made contiguous. §7.1.4, §11.
- **Routing prediction**, every named way of making it cheaper: admission, a re-read blocklist, the
  submission bookkeeping, width, lead, two layers ahead, the GIL switch interval. §9.19, §11.
- **Speculative decoding**: a K-position forward is 242.6 ms + 26.9 per extra position, because the only
  multi-position path is the prefill path. §9.20.
- The hyper-connection glue, the router, dispatch fusion, 1 bit per weight, mirror striping on this bank,
  the frequency penalty, eviction policy. §11.

## Rules that still hold

- **A cost with no row in any instrument hides in plain sight.** The Engram reads survived four sessions of
  ranking because `decode_anatomy.py` buckets them into `rest`, `profile_decode_gpu.py` excludes them by
  construction and `profile_decode_sync.py` charged them to CPU. **Before calling a block unexplained, give
  every blocking call on the decode thread its own column.** New this session, and it is what moved the
  rate. §7.1.7.
- **The drive is shared, so measure I/O-touching code while it is busy.** The same Engram read is 1.7 ms
  all-resident and 28.4 ms streaming. A cost measured on an idle drive is not the cost the runtime pays.
  New this session.
- **A threshold chosen for prefill can be wrong for decode.** The reader's pool started at 64 rows because
  prefill asks for 12k; decode asks for 24. New this session.
- **A chained profile excludes the per-eval floor by construction.** Add 44 x 0.20 ms back before comparing
  a sum of pieces to a token. §7.1.6.
- **Read which function an instrument calls before ranking a lever off it.** Four occurrences.
- **Check that a profile's parts add up to its whole before ranking anything off it.**
- **Never quote a screen as a gate** — and price a loop with one before ranking it.
- **Prove a speed arm's numerics, do not argue them.** `benchmarks/decode_fingerprint.py`, 16 greedy tokens
  with ids and fp32 logit checksums; diff the two arms. §9.22, §9.24.
- **Write a multi-arm A/B to a file and run `bash the-file`.** `settle.sh` matches whole command lines, so a
  script passed as text waits for itself forever. §12.
- **Print what an arm read before attributing a difference to CPU work.** Section 9.24's four arms are
  quotable because all four read 753-754 MiB at a 79.4 % hit rate.
- **A thread does not hide Python work** — and neither does `mx.async_eval`. §9.22.
- **Compile a live coding turn before calling it reference class.**
- **A live session resolves its hit rate and nothing else.** Speed is measured with
  `profile_decode_sync.py`, `profile_decode_gpu.py` or `decode_anatomy.py`.
- **An A/B only finds defects that differ between its arms.** Compare against
  `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/`.
- **Write a test when a comparison finds a match, not only when it finds a bug.**
- **Never drop a case from a denominator.**
- **A curve drawn through the turns of a live session is not a curve.**
- **Read a run's footprint line before its numbers.**
- **Check a confound before reporting an effect.**
- **Memory.** `guarded_run.sh` needs `budget + 29` GiB available. On 2026-09-21, 36 and 40 GiB ran a dozen
  times and 44 was never available; the machine sat at 67-74 GiB free with 1.25 GiB of stale swap. Never
  pass `--force`; lower the budget and say which one you used.
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
| `benchmarks/simulate_policies.py --expert-bytes 9953280` | hit rate against budget, offline | instant, no model |
| `benchmarks/micro_expert_roofline.py` | the expert matmuls against `gather_qmm` and 240 distinct experts | instant, no model |
| `benchmarks/micro_compile_hc_fused.py` | **the template for a CPU screen** | ~1 min, no model |
| `benchmarks/predict_ghost.py` | what happens to a mispredicted expert after it is dropped | ~3 min |
| `benchmarks/nll_expert_precision.py --experts runtime --tokens 512` | the production quality arm | ~2 min |
| `benchmarks/coding_quality.py --resume` | the 40-case corpus, restartable | ~1.5 h |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | **the official implementation** |
| `benchmarks/results/coding/q2g128-v17/` | **the 2-bit gate**, 40/40 |
| `benchmarks/results/guarded/anat40_base_*`, `anat40_new_*` | **the Engram A/B at 40 GiB**: 5.69 → 6.61 tok/s |
| `benchmarks/results/guarded/anateg_base_*`, `anateg_new_*` | the same A/B at 36 GiB |
| `benchmarks/results/guarded/eg_base_*`, `eg_par_*`, `eg_pre_*`, `eg_both_*` | the four arms, per call site |
| `benchmarks/results/guarded/fpeg_base_*`, `fpeg_both_*` | their identical fingerprints |
| `benchmarks/results/guarded/gpu36ci_*` | the compressor, the indexer and four `index_topk` widths |
| `benchmarks/results/guarded/gpu40full3_*` | the GPU side, all forty layers, head and Engram |
| HANDOFF section 7.1.7 | **where a streaming token's time goes** |
| HANDOFF section 9.24 | **the lever that shipped** |
| HANDOFF section 7.1.8 | the source layers' extra millisecond |
| HANDOFF section 9.25 | **the ranking** |
| `./chat.sh` | **the command to hand Hamed** |
