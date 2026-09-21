# Changelog

## 0.9.1 (2026-09-21)

Documentation and one archived artifact; no code and no numerics changed. **0.9.0 was read live at a 52 GiB
expert budget: 9.42 tok/s on prose, 8.53 on 1,493 tokens of Objective-C, a 92.37 % session hit rate.**

### Measurement
- **The live session.** Against 0.7.0's four sessions at a 44 GiB budget — 7.78–7.92 tok/s on prose,
  6.90–6.93 on Objective-C, 89.92–90.00 % hit rate — 0.9.0 at 52 GiB runs at **9.42 and 8.53 tok/s with a
  92.37 % hit rate**, 5,314 resident experts and an MLX peak of 72.73 GiB against the 77.8 GiB the wired
  flag set. That is 22–27 ms off a token; the miss arithmetic gives the larger budget about 6 ms and the
  Engram change of 0.9.0 the remaining 16–21. **The two causes were not separated in one session.**
- **`simulate_policies.py` was validated against a live run for the first time.** It predicted +2.6 points
  of decode hit rate for 44 → 52 GiB; the session moved the session hit rate by **+2.37**.
- **52 GiB fits on a 96 GiB machine** with an 80 GiB wired limit and no memory-pressure event. `chat.sh`
  still defaults to 44 pending a second session.
- **The live coding turn does not compile.** `CSVEscape` sends `-stringValue` to an `NSString`, which the
  class does not declare; repairing that one line compiles the program and it then aborts on the model's
  own example input, because the same mistake appears again behind an `id` from `-allKeys`. Everything else
  in the program — RFC 4180 quoting, nested-value serialisation, the build line, the worked example — is
  correct. It is a model-level type error of the same class as 0.7.0's turn, and not this runtime's defect
  class. The turn, the compiler output and the one-line repair are archived in
  `docs/live-turns/2026-09-21-json2csv/`.

### Documentation
- `docs/HANDOFF.md` section 7.2.6 (the live session, what it proves and what it does not), with section 9.4
  closed live, section 9.24's open live question answered, and section 12.1 rewritten around the 52 GiB
  session shape.
- **Two standing rules revised.** A live session cannot resolve 5–8 ms, which is what the old rule was
  calibrated on, but it resolved 22–27 ms here; and an offline simulation should be checked against a live
  run the first time one is possible.
- README: the interactive numbers now carry both budgets. Next-session prompt v28; v27 archived.

## 0.9.0 (2026-09-21)

**The Engram row reads came off the decode thread: +15.2 % on the benchmark decode rate, with the numerics
bit-identical.** This is the first change to move the rate since 2026-09-19, and what it removes is the
block three successive analyses called the largest unexplained cost in the project. 229 tests.

### Performance
- **Engram row reads go through the reader's worker pool.** `EngramRowReader` only used its sixteen workers
  for batches of 64 rows or more, a threshold chosen for prefill. A decode token asks for 24 rows twice, so
  every decode batch took the serial branch: 96 `pread`s issued one after another from the decode thread,
  each waiting behind a queue the expert stream was filling. `CACHALOT_ENGRAM_PARALLEL_MIN` (default 8) is
  the threshold; every worker writes into its own slice of the output buffer, so the assembled rows do not
  depend on the scheduling.
- **Both Engram layers' reads are issued at the top of the token.** The row ids come from the token being
  decoded, which is known before layer 0 runs, so layer 14's read has thirteen layers of compute to hide
  behind and layer 1's has one. `CACHALOT_DECODE_ENGRAM_PREFETCH` (default 1).
- **Measured.** `decode_anatomy.py`, 36 GiB budget, 96 tokens of continuation: **5.65 → 6.51 tok/s**,
  177 → 154 ms per token, of which 23.7 ms comes out of `rest` — at an identical 81.6 % hit rate, an
  identical 694-695 MiB read per token, 44.1 misses per token either way and 45 % prediction precision
  either way. Repeated at a 40 GiB budget: **5.69 → 6.61 tok/s** (+16.2 %), 176 → 151 ms per token,
  `rest` 120.1 → 95.9, with the hit rate (83.7 %), the bytes (632 MiB), the misses (39.2) and the
  precision (43 %) identical to the digit. `profile_decode_sync.py`, same budget, median streaming token: 188.3 → 162.4 ms, with the
  Engram column falling from 28.4 ms to 2.4 and every other column unchanged. The two mechanisms compose:
  parallel reads alone are 167.8 ms, the prefetch alone 165.3.
- **Numerics unchanged and shown to be.** `decode_fingerprint.py`, 16 greedy tokens: identical ids and
  identical fp32 logit sums and maxima to six decimals against the previous shape. Both flags restore it
  (`CACHALOT_ENGRAM_PARALLEL_MIN=1000000 CACHALOT_DECODE_ENGRAM_PREFETCH=0`) for an A/B.

### Instruments
- **`benchmarks/profile_decode_sync.py --mode both` runs a streaming arm beside the all-resident one** and
  splits every interval between two `mx.eval`s into time inside eval, time blocked in the expert store,
  time inside an Engram `pread` and what is really CPU. This is what found the Engram reads: of the 111 ms
  a streaming token costs over an all-resident one, 63 are the priced miss cost, 27 were Engram, 20 are
  inside eval and 4 are CPU.
- **`benchmarks/profile_decode_gpu.py` times the compressor and the indexer**, the two pieces inside a
  source layer's extra millisecond, the indexer at four `index_topk` widths.

### Measurement
- **A source layer's 5.5 ms per token is decomposed**: about two thirds is the indexer, a tenth the
  compressor and the rest the compressed-KV write.
- **`INDEX_TOPK` is not a lever.** The indexer costs 0.485, 0.474, 0.501 and 0.540 ms per layer at widths
  128, 256, 512 and 1024 — an eightfold change is worth 0.066 ms per layer and is not monotone below the
  shipped 512.

### Tests
- `tests/test_engram_reader_parallel.py`: the parallel and serial row paths against each other and against
  the table itself at seven batch sizes, repeated ids keeping their positions, and the default threshold
  being at or below a decode batch.

### Documentation
- `docs/HANDOFF.md` sections 7.1.7 (where a streaming token's time goes), 7.1.8 (the source layers'
  extra), 9.24 (the lever) and 9.25 (the ranking). Roadmap items 8 and 10 close.

## 0.8.1 (2026-09-21)

Documentation only; no code, no numerics, no version of the runtime's behaviour changed.

### Documentation
- **The changelog now covers every released version.** 0.4.0 through 0.7.0 were tagged and pushed with no
  entry; they are backfilled from the release commits and `docs/HANDOFF.md`, newest first.
- **The README carries the shipped configuration's numbers** instead of the FP4-era ones. Performance,
  Status, Roadmap, the storage table and the mirror-striping guidance were all quoting a runtime that
  decoded at 2.8–2.9 tok/s; the shipped one runs at 7.6–7.9 with an expert hit rate near 90 %, and mirror
  striping is a loss at the expert size that now ships.
- Roadmap re-ordered by measured size in a token, matching `docs/HANDOFF.md` section 9.23.

## 0.8.0 (2026-09-21)

Measurement only: the shipped configuration and every numeric it produces are unchanged. The GPU side of a
decode token is now accounted for layer by layer, a synchronisation has a price, and two levers were
screened and closed. 220 tests.

### Measurement
- **The GPU side of a decode token is fully accounted for.** `profile_decode_gpu.py` now covers the ten
  layers it never did — layer 0 and layer 1's sliding-window attention, the four compressed sources, the
  four index-only sources — plus the head and both Engram forwards, and it prices what an `mx.eval` costs
  to drain. The shipped pieces sum to 41.6 ms against the 54.8 ms a token spends inside `mx.eval`, and the
  remainder is the round trip itself. Attention is 22.5 ms per token, the largest GPU block by a factor of
  three. HANDOFF section 7.1.5.
- **`benchmarks/micro_eval_floor.py`** (new, no model): one `mx.eval` costs about 0.20 ms whatever it
  evaluates, and every way of reading a value back costs the same, so a token's 44 synchronisations are
  about 9 ms of fixed cost. HANDOFF section 7.1.6.
- **`benchmarks/micro_compile_attention.py`** (new): `mx.compile` on decode attention is closed three ways
  — `shapeless=True` cannot infer a custom Metal kernel's output shapes, a plain trace is not
  bit-identical, and it retraces on every token for a net loss. HANDOFF section 9.21.
- **`benchmarks/decode_fingerprint.py`** (new): 16 greedy tokens with their ids and fp32 logit checksums,
  for diffing two arms of a speed change.

### Runtime
- **`CACHALOT_PRELAUNCH_SHARED`** (new, default `0` = off): issues the shared expert before the layer
  blocks on its routing, so the GPU has work during the round trip. `1` submits it with `mx.async_eval`,
  `2` adds it to the routing's own `mx.eval`. Arm 1 moves 10 ms per token out of `mx.eval` and puts 13 ms
  back on the CPU; arm 2 costs nothing on the CPU and 5 ms inside eval. 79.1-79.6 ms shipped against
  81.8-82.6 and 83.5-85.7. Every arm's fingerprint is identical to the shipped one. Off until an arm wins.
  HANDOFF section 9.22.

## 0.7.0 (2026-09-21)

Measurement only: no numerics changed and the shipped configuration is unchanged. Every named way of making
routing prediction cheaper was measured and closed, and the 44 GiB configuration that ships was profiled for
the first time. 219 tests.

### Measurement
- **The shipped budget has a profile.** 44 GiB, 512-token context: 170 ms per token at an 83.5 % expert hit
  rate reading 627 MiB, of which 54.0 ms is blocked on the store and 46.4 of that is misses no prediction
  covered, with the drive idle 45 % of the time. Reproducible to ±0.3 %. HANDOFF section 7.1.3.
- **Three ways to reclaim the prediction's waste, all closed.** Admitting a mispredicted load has a 4.3 %
  ceiling; a blocklist on re-reading a dropped expert trades 8.1 wasted reads for 3.4 demand misses; the
  submission bookkeeping is 0.040 ms per token, not the 3.6 two prompts had carried. HANDOFF section 9.19.
- **Four nulls at the shipped budget**: `CACHALOT_PREDICT_AHEAD=2`, top-4, top-8 and every
  `sys.setswitchinterval`, which rules reader-thread scheduling out of the 33 ms a streaming token spends
  above the all-resident floor.
- **`benchmarks/predict_ghost.py`** (new): what happens to a mispredicted expert after it is dropped, and
  what a blocklist of any lifetime would have done.
- **`benchmarks/micro_predict_submit.py`** (new): the prediction submission bookkeeping, four arms, no model.
- **`profile_decode_gpu.py` corrected**: two of its rows were timing paths the runtime had stopped taking,
  which had put routed experts at 19.8 ms per token instead of 6.7 and the router at 7.5 instead of 0.9.

## 0.6.0 (2026-09-21)

The release that restored quality and took the first ~8 ms off the compute floor. Output is equal to the
hosted reference on every column of the coding gate, and the decode path gained two shipped changes.

### Fixed
- **The hyper-connection residual mix was applied transposed.** `hc_post` computed `comb @ residual` where
  the official implementation does `comb.T @ residual`, in both the MLX path and the fused Metal kernel.
  Re-gated through the fixed runtime, the 2-bit bank compiles **20 of 20** C++ blocks, parses 18 of 18
  Python blocks and emits **0 of 101** malformed `#include` lines — every column equal to the hosted
  reference, against 0/42, 5/26 and 49/154 before. `tests/test_hyper_connection.py` pins the contraction.
  HANDOFF section 7.4.8.
- **The repetition collapse was the same defect, not the sampler.** 0 of 8 against the original 5 of 8 on
  the same bank and protocol, p = 0.026, so `--frequency-penalty 0.2` is no longer needed. HANDOFF 9.9.

### Runtime
- **The decode MoE block is traced once instead of rebuilt forty times a token.** `mx.compile` over the six
  routed experts, the shared expert and their sum takes the all-resident token from 82-85 ms to **77 ms**
  and the CPU side from 27 to 21.5, with mean NLL identical to four decimals. HANDOFF section 9.16.
- **The fused Metal kernels' scalar parameters are built once** (`model/kernel_consts.py`, about 1,600
  constructions per token): **1.2-1.3 ms** of CPU per token, numerics unchanged. HANDOFF section 9.17.
- **Mirror striping is off on this bank.** It is a property of the expert size, not of the runtime: an 18 %
  loss at 9.49 MiB per expert where it was a gain at 17.93. HANDOFF section 9.11.1.
- **`./chat.sh`** (new): exports the shipped environment, refuses a second runtime and passes flags through.
  The multi-variable one-liner it replaces does not survive line wrapping, and the failure is silent — the
  runtime serves FP4 off the USB drive at 0.6 tok/s. The expert-bank banner now prints unconditionally.

### Measurement
- **The compute floor was being measured wrong.** The hyper-connection kernels are **4.6 ms** of a token,
  not 68.7; the 68.7 was an artifact of a profiler that evaluates each piece behind its own barrier. An
  all-resident token is about 65 % GPU wait and 35 % CPU. HANDOFF sections 6.3, 6.3.1.
- **Context length is not what makes a long turn slow.** Worth 4.7 ms of `rest` across a fourfold change
  and not monotone; the live spread between turns is working-set variation. HANDOFF section 6.4.

## 0.5.0 (2026-09-17)

A measurement release: five levers worked, three of them settled against what the handoff expected, and the
quality gate's own statistics were repaired.

### Measurement
- **The compute floor is 93 ms per token, not 120.** 49 % of a decode token is expert streaming and the
  arithmetic half is roughly 400 GPU dispatches rather than anything a faster kernel would fix.
- **The `mtp.*` layers are DSpark**, not plain multi-token prediction: five drafted tokens per forward and
  2.85 accepted, but verification reads W/T times the bytes, so the lever is 1.20x rather than 1.5x.
- **The 512-token mean NLL has a paired standard error wider than every difference it was asked to rank.**
  The gate reports the paired median and the sign test instead, and the operating rules say to judge by them.
- **The coding gate was scoring failed compilations as clean.** `check_cpp` grepped stderr and never read
  the compiler's exit status, so every C++ ratio older than 2026-09-19 is withdrawn. HANDOFF section 7.4.1.
- **In-flight predictions had no lifetime** and were discarded for finishing early; fixed and shipped.
  HANDOFF section 9.13.

## 0.4.0 (2026-09-17)

Decode 275 -> **182.5 ms per token** at a 36 GiB budget, 3.64 -> **5.48 tok/s**, from a 2-bit routed-expert
bank built here out of the FP4 checkpoint and a prediction width the smaller expert made worth raising.
Interactive chat at a 44 GiB budget runs at **6.0-7.5 tok/s** against 4.3-5.6 before, with an 87.3 % session
hit rate and sub-second follow-up prefills.

### Runtime
- **A 2-bit affine g128 expert bank**, 9.49 MiB per expert against FP4's 17.93, built from the checkpoint by
  `benchmarks/build_affine_bank.py`.
- **`CACHALOT_PREDICT_TOPK` raised from 3 to 6**, which only becomes worth doing once experts are small
  enough that the extra reads stay under compute.

### Note added in 0.6.0
The quality cost this release reported against the 3-bit bank — +0.019 nats and 6.3 points of top-1, with
occasional mangled tokens — **was the transposed hyper-connection residual mix, not the bank**. Re-gated
through the fixed runtime the 2-bit bank is equal to FP4 and to the hosted reference on every column.

## 0.3.0 (2026-09-16)

Interactive use on a 96 GB Mac: a chat follow-up turn now answers in about a second instead of ten, and decode
is roughly 40 % faster end to end. Two kernel panics on 2026-09-15 are also addressed at the root.

### Runtime
- **Alternative expert banks.** `CACHALOT_EXPERT_BANK=<dir>` serves the routed experts from a different
  checkpoint than the trunk, Engram and tokenizer. `storage.index.detect_expert_bank` recognises oMLX-converted
  checkpoints (`omlx_deepseek_v41` in `config.json`), whose experts are stacked per layer, and indexes them into
  per-expert byte ranges; `model/expert_affine.py` evaluates them with `mx.quantized_matmul`. Validated against
  `Jundot/DeepSeek-V4.1-Flash-oQ3e-mtp` (calibrated affine 3-bit, 14.77 MiB per expert against FP4's 17.93):
  teacher-forced NLL 2.3004 -> 2.3080 nats, and decode 2.89 -> 3.36 tok/s at an equal 28 GiB budget because the
  same budget holds 1,941 experts instead of 1,599. The FP4 path is unchanged.
- **Idle heartbeat.** Within about six seconds of an idle Metal queue macOS un-wires the whole working set despite
  `mx.set_wired_limit` (wired 50 -> 6 GiB, then 44 GiB in the compressor), and the next turn pays the
  decompression: a 13-token follow-up prefill took 10.5 s after a typing pause against 2.9 s back to back. A daemon
  thread now evaluates a one-element op every `idle_heartbeat_seconds` (default 0.5,
  `CACHALOT_IDLE_HEARTBEAT_SECONDS=0` disables) while no forward pass runs.
- **One-layer-early routing prediction.** Layer L+1's router is applied to layer L's router input and the top-k
  predicted experts load into free transient slots while the GPU runs layer L. Decode 2.26 -> 2.55 tok/s at a
  28 GiB budget; `CACHALOT_PREDICT_TOPK` defaults to 3 (top-2 and top-4 were both worse, the SSD being
  bandwidth-bound).
- **Environment overrides applied when a value is auto.** `TextDecodeRuntime` replaced the environment-loaded
  config unconditionally, so the default 0 discarded `CACHALOT_EXPERT_CACHE_BUDGET_GIB` and
  `CACHALOT_MLX_WIRED_LIMIT_GIB`. Benchmarks asking for 32 GiB ran with the 47-50 GiB auto budget, which wires
  ~67 GiB on a 96 GB machine and, with other applications open, drove it into swap and two kernel panics.
- **Short prefills keep the cache.** Prefill admission evicted every resident of a layer the prompt did not route
  to, so the 2-3-token speculative prefills below shrank the resident set from 1,599 to 453 experts. Quota room a
  prompt's own misses do not fill now keeps that layer's current residents, most recently used first.
- **Concurrent piece reads.** An expert split across several tensors (any stacked bank) has its pieces read in
  parallel: nine scattered reads of a 15.5 MiB expert take 4.2 ms serially, 2.8 ms concurrently, at the same
  aggregate bandwidth. Mirror striping stays on the serial path and should be left off for stacked banks.

### Chat
- **Typing-time prefill.** On an interactive terminal the chat reads keystrokes raw and, after 0.4 s without a key,
  prefills the template head plus the finished words of the message through the prefix cache; Enter then pays only
  for the last word and the template tail. Measured with a pseudo-terminal typing at human pace, the follow-up turn
  reused 50 of 57 prompt tokens instead of 37 and waited 2.0 s instead of 5.5 s. `--no-typing-prefill` disables it
  and `/stats` reports the speculative work.

### Benchmarks
- `benchmarks/guarded_run.sh` runs any benchmark under a memory guardian: it refuses to start when another runtime
  is alive, when memory pressure is not normal, when the boot volume is nearly full or when free plus reclaimable
  memory cannot hold the planned wired set; it samples `vm_stat`, swap, pressure, RSS and the compressor every
  second into a CSV; and it kills the command on critical pressure, sustained warning pressure, swap growth, low
  disk, a timeout, or a runtime that reports a larger budget than requested.
- `benchmarks/nll_expert_precision.py` gates an expert bank on teacher-forced NLL with identical dense reference
  math on both sides; `benchmarks/check_oq3e_shards.py` compares two banks weight by weight without loading the
  model; `benchmarks/chat_turns.py` replays a six-turn chat with optional idle gaps;
  `benchmarks/chat_pty_typing.py` drives the CLI through a pseudo-terminal.

## 0.2.0 (2026-09-15)

First public release as **Cachalot** (package renamed from `v41runtime`).

### Runtime
- `CACHALOT_MIRROR_PATH` / `CACHALOT_MIRROR_FRACTION`: a second identical checkpoint copy on another drive serves
  the tail of every expert read concurrently (byte striping, offsets identical). 10 % on a 1 GB/s USB mirror:
  decode 2.86 -> 3.00 tok/s, 512-token cold prefill 32 -> 28.5 s.
- `TextDecodeRuntime.warmup()` compiles all kernels at load (`V41Model.from_pretrained(warmup=True)`); the first
  decoded token no longer pays ~0.6 s of Metal compilation. Opt-in `CACHALOT_EVICT=lfu` eviction (no measurable gain).
- Prefill loads the next layer's most-used experts speculatively while that layer's router is still being
  computed (cancelled if unneeded, promoted if needed; experts consumed in arrival order) and reads both Engram
  layers' rows in the background from prefill start. 2048-token prefill 55 s -> 44 s cold / 46 s -> 38 s warm,
  512 tokens 32 s / 23 s, i.e. at the SSD floor. `CACHALOT_SPECULATIVE_PREFILL=0` disables speculation.
- The auto expert budget is also capped by memory available at start (free + purgeable + reclaimable file
  cache minus trunk, MLX cache and 12 GiB headroom).
- Prefill attention processes 256-token chunks (`CACHALOT_ATTN_CHUNK`), `wo_a` runs as per-group GEMMs, and FP8
  linears use exact bf16 operands; 2048-token prefill 57 s -> 55 s cold / 49 s -> 46 s warm, 512 tokens 33 s / 25 s.
- Prefill routed experts run on simdgroup-matrix FP4 kernels (`fp4_sgmm_metal.py`: dequantize once, bf16 MMA,
  fp32 split-K) for up to 64 rows per expert, `quantized_matmul` above; 2.5x less GPU time per expert.
  `CACHALOT_PREFILL_SGMM=0` restores the affine-8 path.
- Fused decode path (`decode_fused_metal.py`, `router_fused_metal.py`, `fp8_fused_metal.py`): single-launch
  router top-k, sparse attention, hyper-connection mixes, RoPE, RMSNorm, hc_pre/hc_post, FP8 activation quantization,
  and a vectorized FP8 GEMV; MoE output dispatched asynchronously. All-resident decode token 0.10 s -> 0.068 s.
  Every fused kernel switches off with `CACHALOT_FUSED_DECODE=0` / `CACHALOT_FUSED_FP8=0`.
- Batched compressor/indexer source layers (2/8/14/20, 24/28/32/36); Engram rows via parallel pread instead of
  mmap faults (9-18 s -> 0.2 s per layer); miss-aware prefetch lookahead. 512-token prefill 35 s cold / 29 s warm.
- Routed experts in prefill use an exact FP4 -> affine-8-bit repack and `mx.quantized_matmul`.
- Batched prefill: chunked attention for sliding-window and reuse layers, batched hyper-connection mixes,
  router, routed-expert dequantize+GEMM, shared expert, Engram; `wo_a` dequantized at load.
  512-token cold prefill 86 s -> 55 s on the internal SSD.
- Opt-in decode miss budget (`set_decode_miss_budget`), measured and documented as unusable for quality; off.
- Fused top-k routed-expert Metal kernels (two launches per layer) and a bf16-reading fp32 head GEMV;
  all-resident decode token 0.10 s.
- Checkpoint path discovery (`~/DeepSeek-V4.1-Flash`, `~/models/`, `/Volumes/*`, or `CACHALOT_MODEL_PATH`).
- Pre-allocated, wired expert slot pool; experts are `preadv()`'d from the shard straight into MLX unified memory
  (no per-expert allocation, memcpy, or Metal residency churn).
- `mx.set_wired_limit` keeps trunk + experts resident; macOS no longer compresses cold expert buffers
  (decode 2.3–3.4 s/token → SSD bytes + 0.15 s).
- Expert shard reads bypass the page cache (`F_NOCACHE`).
- Auto-sized expert budget from unified memory (`expert_cache_budget_bytes=0`), `system_reserve_bytes`.
- Prefix cache: multi-turn requests prefill only the new suffix; snapshot/restore of sequence state.
- Decode misses of a layer are loaded concurrently and admitted in logical order.
- Streaming generation core with cancellation; `top_k`/`top_p` layered on the official sampler.
- `max_seq_len` default 32,768.

### Serving and tooling
- OpenAI-compatible server: `/v1/chat/completions` (SSE, tools, `response_format`, thinking mode with
  `reasoning_content`, `stop`, usage with prefix-cache stats), `/v1/completions`, `/v1/models`, `/v1/stats`,
  `/health`, optional Bearer auth.
- `cachalot serve | chat | doctor | bench` CLI; `CACHALOT_*` environment overrides.
- Routing tracer, trace analysis, policy replay simulator, decode profilers under `benchmarks/`.
- Checkpoint-free test suite (store, prefetch, prefix cache, config, server) and GitHub Actions CI.

### Inherited from the pre-release runtime
- Exact DeepSeek V4.1 Flash text path in MLX + Metal (mHC, CSA2 attention, Engram, FP4/FP8 kernels).
- Layer-major prefill with expert-major MoE scheduling and deterministic per-layer admission.
- MLX free-buffer cache capped at 2 GiB.
