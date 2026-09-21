# Next-session prompt — **v29**, written 2026-09-21

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v29** | 2026-09-21 | the last unexplained block found its mechanism and the GPU side closed | **there is no 20 ms inside `mx.eval` with no mechanism — it is the miss.** A streaming token replayed over the same continuation with everything already resident costs **79.6 ms, the all-resident floor exactly, on every column**; the excess is 0.31 ms per demand miss on top of the 1.41 the store already charges. `io_workers` 2-16 null, `CACHALOT_PAGE_CACHE=0` a 16 ms loss. Then the three unscreened blocks were screened: **attention is 86 % weight streaming**, the FP8 GEMV under it moves 5.14 GB a token at 79 % of the machine, the shared expert is at 80 %, `wo_a` is faster in BF16 than in any FP8 form, and the lanes-per-row policy nobody tuned is within 0.3 %. **Every remaining millisecond is the miss or the floor** |
| v28 | 2026-09-21 | 0.9.0 was read live at a 52 GiB budget | prose 7.78-7.92 → 9.42 tok/s, hit rate 90.00 → 92.37 %, 52 GiB fits; the coding turn does not compile |
| v27 | 2026-09-21 | the 33 ms found a mechanism | the Engram row reads were serialised on the decode thread; +16 % on the benchmark rate, shipped in 0.9.0 |
| v26 | 2026-09-21 | the GPU side closed | attention is 22.5 ms; `mx.compile` on it closed three ways |
| v25 | 2026-09-21 | the ranking rebuilt on measured sizes | two profiler rows were timing retired paths; the expert kernel is closed |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read this first, because it changes what is worth doing.** `docs/HANDOFF.md`'s opening block, then
**7.1.9**, then **7.1.10**, then **9.30**, then 7.2.6. The first is why the last unexplained block was
never a block; the second is where the GPU's time actually goes; the third is the ranking as it now
stands; the fourth is the live session the shipped budget rests on.

**The shipped command, unchanged, and still set by hand rather than in code:**

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

## Where the time is

A live prose token is **106.2 ms** at 9.42 tok/s; a benchmark token at 40 GiB is 141-152 ms. **A streaming
token that does not miss is 79.6 ms, which is the all-resident floor exactly.** Everything between those
two numbers is the miss.

| block | size | state |
|---|---:|---|
| **the miss** | **~1.7 ms each** — 1.41 blocked, 0.31 inside `mx.eval`, ~0.05 CPU | 39.8/token at 83.4 % hit, ~19 ms at a session's 92.4 %. **The budget is the only lever and it is taken by hand.** §7.1.9, §9.4 |
| the FP8 GEMV family | 16.6 ms | 5.14 GB/token, the largest GPU path. **79 % of `mx.sum` over the same bytes.** §7.1.10 |
| routing prediction | 11 ms | closed. §9.19 |
| the 44 `mx.eval` round trips | ~9-12 ms | 0.20 ms each; the overlap arm moves 10 and costs 13. §7.1.6, §9.22 |
| routed experts, traced | 6.9 ms | closed. §7.1.4 |
| compressor, indexer, compressed-KV write | 5.5 ms | decomposed; `INDEX_TOPK` is a null. §7.1.8 |
| `wo_a`, BF16 grouped matmul | 4.3 ms | 625 GB/s, the fastest arm in the model. §9.28 |
| hyper-connection glue | 4.2 ms | closed. §11 |
| sparse attention over the KV itself | ~2.6 ms | **this is what three prompts called "attention's shapes".** §7.1.10 |
| Engram row reads | 1.9 ms | taken. §9.24 |
| head | 1.6 ms | never attacked |

## Job 1 — is the miss at the drive's wall? Nobody has ever asked under decode's memory conditions

**This is now the whole game.** A miss costs about 1.7 ms for 9.49 MiB, which is 5.6 GB/s, and the drive
is rated 6.6-6.8 GB/s cold (§3.1) — but every reading of that number was taken on an idle machine with
nothing wired, while decode holds 55-63 GiB wired and the kernel must reclaim a page for every page it
reads. `benchmarks/expert_read_scaling.py` has had a `--wire-gib` option for that since 2026-09-17 and
**it has never been run at anything like the shipped budget.**

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh --budget-gib 40 && benchmarks/guarded_run.sh --budget-gib 40 --max-seconds 1800 --tag readwall -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/expert_read_scaling.py --bank /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 --loaders 1,2,4,8,16 --page-cache --mlx-slots --wire-gib 50 --experts 512
```

Read the `--expert-offset` help before running it: F_NOCACHE does not reliably keep experts out of the
page cache, and an arm that re-reads what a previous arm read reports a memory hit as a drive read. **If
the answer is 5.6 GB/s, the miss is at the wall and fewer bytes is the only lever there will ever be** —
which makes the budget, and §9.10's 42.7 % wasted prefetch, the whole of the remaining speed work. If it
is 6.6, the reader has a sixth of a millisecond per miss in it, 7 ms per token at a benchmark hit rate.

## Job 2 — separate the budget from the Engram change, carried from v28 and still unrun

Section 7.2.6 moved two things at once and the split between them is arithmetic, not a measurement. One
conversation settles it, at the good session's budget with the change switched off:

```bash
CACHALOT_ENGRAM_PARALLEL_MIN=1000000 CACHALOT_DECODE_ENGRAM_PREFETCH=0 CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

Same two long prompts — a 500-word story and the Objective-C JSON-to-CSV program — against 9.42, 8.53 and
92.37 %. A 16-21 ms difference is five times what a session resolves.

**And the benchmark arm is now possible for the first time.** Section 9.24 has never been benchmarked at
or above the shipped budget because 44 GiB needs 73 GiB free and the machine never had it; on 2026-09-21
it had **74.3 GiB**, and eleven guarded runs at 40 GiB went through without a pressure event. Check
`settle.sh`'s own line before assuming it still does.

## Job 3 — the quality gate has no Objective-C in it, and both live Objective-C turns are broken

Carried from v28 unchanged, and with the speed side closed it is the highest-value job in this document.
Two live coding turns have been put through a compiler and both fail on a model-level type error; the
40-case corpus gate compiles C++ and parses Python, **contains no Objective-C at all**, and executes
nothing. Two cheap things in order: add Objective-C cases to `benchmarks/coding_tasks.json` and compile
them in `code_validity.py` the way C++ is compiled; then consider whether the gate should *run* what it
builds on the cases that have no input. "Reading a program by eye is not the check it was being used as"
is two for two. §7.2.6, `docs/live-turns/2026-09-21-json2csv/`.

## Job 4 — make 52 GiB the default, or decide not to

Section 4's one-liner says 44 and `chat.sh` defaults to it, while the command Hamed is handed says 52. The
evidence is one session — peak 72.7 GiB against a 77.8 GiB wired limit, no pressure event, +2.37 points of
hit rate — and the risk is the configuration class that panicked this machine twice in September. Job 2's
conversation is a second session at 52 GiB and settles this as a side effect: **read its MLX peak before
its tok/s.**

## Job 5 — what is left, and it is small

- **`kernel_consts.py:39`** appears in the streaming arm at 0.5 calls per token with 1.7 ms of
  store-blocked time behind it, and nobody has explained it. It is the last line of the per-site table
  with no account.
- **The 3.48 ms between the FP8 GEMV kernel and `mx.sum` over its own bytes** (§7.1.10). The kernel
  already does `uint4` weight loads, `float4` pre-decoded activations and a tuned lanes-per-row split, so
  whatever is left is not one of the obvious three. Two shapes — `wkv` at 87 GB/s and `wq_a` at 171 — are
  occupancy-bound at 64 and 160 threadgroups, and `mx.sum` over the same buffers only reaches 98 and 212,
  so most of *their* shortfall is the buffer and not the kernel.
- **The `w1`/`w3` fusion in the shared expert**, 0.4 ms per token and **bit-identical by construction**
  (§9.27). Not shipped because 0.4 ms is a tenth of what a live session resolves and this project does not
  ship on a screen. Take it only if you are already in that file for another reason.

## What is closed, so nobody spends a session there

- **The in-eval excess.** It is the miss. A streaming token at a 100 % hit rate is the all-resident floor
  on every column. §7.1.9.
- **`io_workers`** from 2 to 16, and **`CACHALOT_PAGE_CACHE=0`** (a 16 ms loss). §9.26.
- **Attention's shapes.** A reuse layer is 0.377 ms of weight streaming and 0.064 of everything the shape
  idea would touch. §7.1.10.
- **`mx.compile` on attention**, in all three of its shapes. §9.21.
- **Quantizing `wo_a`.** BF16 reads twice the bytes and is faster than every FP8 arm. §9.28.
- **The FP8 GEMV lanes-per-row policy.** Within 0.3 % of the best split per shape. §9.29.
- **The shared expert.** 307 GB/s against a 385 ceiling; the whole headroom is 0.9 ms. §9.27.
- **`INDEX_TOPK`**, **the expert kernel**, **routing prediction** in every named way, **speculative
  decoding**, the hyper-connection glue, the router, dispatch fusion, 1 bit per weight, mirror striping on
  this bank, the frequency penalty, eviction policy. §7.1.8, §7.1.4, §9.19, §9.20, §11.

## Rules that still hold

- **Read which function an instrument calls before ranking a lever off it — five occurrences.** The fifth
  was this session: a four-variant kernel screen was written, run and found bit-identical against
  `fp8_gemv_metal.fp8_gemv_quantized`, which the runtime does not call — `fp8_linear_quantized` dispatches
  to `fp8_fused_metal.fp8_gemv_decoded` whenever `CACHALOT_FUSED_FP8` is set, and it is set by default.
  **Grep for the caller, not the definition.** §12.
- **Two arms must be launched the same way.** `--mode both` leaves 240 experts pinned and changes what the
  continuation evicts; a baseline read off it and compared against `--mode stream` arms is 7 ms out. §9.26.
- **A cost with no row in any instrument hides in plain sight.** §7.1.7.
- **The drive is shared, so measure I/O-touching code while it is busy.** The same Engram read is 1.7 ms
  all-resident and 28.4 ms streaming.
- **A screen that does not reproduce the profiler is measuring something else.** The shared-expert screen
  is quotable because it returns 0.116 ms per layer against the profiler's 0.121. New this session.
- **Check an offline simulation against a live run the first time one is possible.** §9.4, §7.2.6.
- **A live session resolves its hit rate, and a change of 20 ms or more.** §12.1, §7.2.6.
- **Two changes in one session leave arithmetic, not a measurement.** §7.2.6, job 2.
- **A chained profile excludes the per-eval floor by construction.** Add 44 x 0.20 ms back. §7.1.6.
- **Check that a profile's parts add up to its whole before ranking anything off it.**
- **Never quote a screen as a gate** — and price a loop with one before ranking it.
- **Prove a speed arm's numerics, do not argue them.** `benchmarks/decode_fingerprint.py`. §9.24.
- **Write a multi-arm A/B to a file and run `bash the-file`.** §12.
- **Print what an arm read before attributing a difference to CPU work.**
- **Interleave the arms of a sweep and run each twice.** The shipped `io_workers` width spanned 5.5 ms
  across its own two reps, which is the whole range of the sweep. New this session. §9.26.
- **A thread does not hide Python work** — and neither does `mx.async_eval`. §9.22.
- **Compile a live coding turn before calling it reference class.** Two for two now fail. §7.2.6, job 3.
- **An A/B only finds defects that differ between its arms.** Compare against
  `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/`.
- **Write a test when a comparison finds a match, not only when it finds a bug.**
- **Never drop a case from a denominator.**
- **A curve drawn through the turns of a live session is not a curve**, and a short turn is not a rate.
- **Read a run's footprint line before its numbers.**
- **Check a confound before reporting an effect.**
- **Memory.** `guarded_run.sh` needs `budget + 29` GiB available. On 2026-09-21 the machine had 74.3 GiB
  and eleven runs at 40 GiB went through clean; 44 needs 73 and 52 needs 81. A *chat* session is not bound
  by that guard and peaked at 72.7 GiB of MLX against a 77.8 GiB wired limit. Never pass `--force`; lower
  the budget and say which one you used.
- **Hand over `./chat.sh`, never the section 4 one-liner.** §12.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once. Launch each run as its own command.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/profile_decode_sync.py --mode both` | an all-resident token and a streaming one side by side, each interval split into eval, store-blocked, Engram `pread` and CPU, per call site. **`--stream-passes 2` replays the continuation with everything resident; `--io-workers N` sets the device queue depth** | ~1 min |
| `benchmarks/decode_anatomy.py` | where a live token's time goes by blocking cause, and the rate | ~1 min |
| `benchmarks/profile_decode_gpu.py` | the GPU side, every layer, the compressor and the indexer, plus what an eval costs to drain | ~1 min |
| `benchmarks/decode_fingerprint.py` | whether a speed arm changed the numerics | ~1 min |
| **`benchmarks/micro_fp8_gemv_kernel.py`** | **the largest GPU path priced per shape against `mx.sum`, every lanes-per-row split, bit-identity checked** | instant, no model |
| **`benchmarks/micro_shared_expert_roofline.py`** | **the shared expert against the memory wall, and the `w1`/`w3` fusion** | instant, no model |
| **`benchmarks/micro_wo_a.py`** | **`wo_a` BF16 against every FP8 form of the same projection** | instant, no model |
| `benchmarks/expert_read_scaling.py --wire-gib N` | **what the drive gives under decode's memory conditions. Job 1** | ~3 min |
| `benchmarks/micro_eval_floor.py` | what one `mx.eval` costs, and whether anything makes it cheaper | instant, no model |
| `benchmarks/simulate_policies.py --expert-bytes 9953280` | hit rate against budget, offline — validated live to a quarter of a point | instant, no model |
| `benchmarks/nll_expert_precision.py --experts runtime --tokens 512` | the production quality arm | ~2 min |
| `benchmarks/coding_quality.py --resume` | the 40-case corpus, restartable — **C++ and Python only** | ~1.5 h |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | **the official implementation** |
| `benchmarks/results/coding/q2g128-v17/` | **the 2-bit gate**, 40/40 |
| `docs/live-turns/2026-09-21-json2csv/` | **the live coding turn that does not compile** |
| `benchmarks/results/guarded/sync2pass_*` | **the two-pass arm: pass 2 is the all-resident floor** |
| `benchmarks/results/guarded/iow{2,4,8,16}_r{1,2}_*` | the `io_workers` sweep, eight arms |
| `benchmarks/results/guarded/ie_nocache_*`, `ie_w8a_*`, `ie_w8b_*` | `CACHALOT_PAGE_CACHE=0` against two matched baselines |
| `benchmarks/results/guarded/anat40_base_*`, `anat40_new_*` | the Engram A/B at 40 GiB: 5.69 → 6.61 tok/s |
| `benchmarks/results/guarded/gpu40full3_*` | the GPU side, all forty layers, head and Engram |
| HANDOFF section 7.1.9 | **a streaming token at a 100 % hit rate is the floor** |
| HANDOFF section 7.1.10 | **where the GPU's time actually goes** |
| HANDOFF section 9.30 | **the ranking** |
| HANDOFF section 7.2.6 | the live session on 0.9.0 at 52 GiB |
| `./chat.sh` | **the command to hand Hamed** — with `--expert-budget-gib 52` and `CACHALOT_MLX_WIRED_LIMIT_GIB=80` |
