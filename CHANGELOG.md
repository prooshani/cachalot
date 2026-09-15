# Changelog

## 0.2.0 (2026-09-15)

First public release as **Cachalot** (package renamed from `v41runtime`).

### Runtime
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
