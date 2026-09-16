# Changelog

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
