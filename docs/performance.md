# Performance notes

Everything here was measured on a Mac Studio M3 Ultra (96 GB) with the checkpoint on a
Crucial X10 Pro over USB 3.2 Gen 2. Scripts live in `benchmarks/`; each finding names the
script that reproduces it.

## 1. The bytes budget

| Quantity | Value |
|---|---:|
| Routed expert (3 × FP4 weights + 3 × E8M0 scales) | 18.8 MB |
| Experts in the checkpoint | 15,360 (289 GB) |
| Experts touched per decoded token | 240 (40 layers × top-6) |
| Distinct experts touched by a 512-token prompt | ~9,500 (62 % of all) |
| USB 3.2 Gen 2 SSD, sequential read (`micro`: `dd`, `pread` at QD1) | 1.0 GB/s, saturated at queue depth 1 |

Consequences:

- A **cold decode token** with 65 % expert hits reads ~84 × 18.8 MB ≈ 1.6 GB: ≥ 1.6 s on this SSD.
- A **cold prompt** of a few hundred tokens or more needs most of the expert set: ≥ 3–4 minutes of SSD time,
  no matter how fast the compute is. Prefix caching makes this a once-per-conversation cost.
- More I/O threads cannot help. Only fewer misses (more resident memory) or faster storage can.

## 2. Routing statistics (`benchmarks/trace_routing.py`, `analyze_trace.py`)

Four realistic 512-token prompts (code review, prose, a second code file, an engineering document)
plus 32 greedy tokens each, then a return to the first prompt:

| Resident budget | Share of experts | Activations covered by the hottest N experts (static bound) |
|---:|---:|---:|
| 40 GiB | 15 % | 61.6 % |
| 50 GiB | 19 % | 66.5 % |
| 64 GiB | 24 % | 74.4 % |
| 100 GiB | 37 % | 86.4 % |

- Per-layer skew is uniform enough that the uniform per-layer quota is within 0.2 points of the global optimum.
- Consecutive decode tokens share only **29 %** of a layer's six experts, so speculative/batched decode
  amortizes far fewer expert loads than on denser MoE models.

## 3. Cache policy (`benchmarks/simulate_policies.py`)

A replay of the trace through a faithful model of `ResidentExpertStore` (per-layer prefill quotas,
decode LRU) reproduces the measured decode hit rate (71 % simulated vs 69–73 % measured).

| Budget | Decode policy | Overall hit | Bytes vs. LRU |
|---:|---|---:|---:|
| 40 GiB | LRU (current) | 40.2 % | – |
| 40 GiB | segmented LRU | 41.2 % | −1.7 % |
| 40 GiB | frequency + decay | 40.9 % | −1.1 % |
| 50 GiB | LRU | 43.8 % | −6.1 % |
| 64 GiB | LRU | 48.3 % | −13.5 % |

Ordering prefill admission by in-prompt popularity instead of first-come changes nothing.
Policy sophistication is worth ~1–2 %; memory is worth 6–14 %. Cachalot therefore keeps the simple,
deterministic LRU and spends effort on the memory budget instead.

## 4. Memory pressure was the hidden decode bottleneck (`profile_decode_timeline.py`)

Before the fix, a cold token spent 0.4–3.7 s in GPU synchronisation on top of its SSD time, and the excess
scaled with the number of expert misses even though the expert kernels themselves ran at 1.5 ms per six experts
(`micro_expert_kernel.py`). `vm_stat` explained it: after a handful of tokens the kernel compressor held
**49 GB** and the system had **0.4 GB free**. macOS was compressing cold MLX buffers, and every GPU access
to a compressed page paid a decompression fault. Ruled out along the way: GPU clock ramp after idle
(real but ~1 ms/layer, `micro_gpu_idle.py`), MLX free-buffer cache size, Metal buffer alloc/free churn,
Python GC, GPU first-touch of new buffers (`micro_eval_slowdown.py`).

Fix: `mx.set_wired_limit(trunk + expert budget + MLX cache)` keeps the working set in the Metal residency
set, and `F_NOCACHE` on the shard descriptors stops the 100+ GB expert stream from pushing everything else
out. Result on the same token sequence:

| | before | after |
|---|---:|---:|
| route-eval waits per token | 0.4–3.7 s | 0.11–0.15 s |
| 110-miss cold token | 3.4 s | 1.2 s |
| all-resident token | 0.25–0.33 s | 0.13–0.16 s |
| compressor after 10 tokens | 49 GB | 4 GB |

Decode is now SSD bytes + ~0.15 s.

## 4b. Wiring made allocation expensive; the slot pool removed allocation

With the wired limit on, every fresh `mx.array` became a Metal residency-set update. Under eight concurrent prefetch
workers, promotion went from 1.4 ms to 35 ms per expert and a cold 512-token prefill from 270 s to 692 s.

The fix is structural: all expert slots are allocated once, and misses are read with `preadv()` directly into a
writable NumPy view of the slot's MLX buffer (`benchmarks/micro_*` verified GPU visibility and byte equality).
The load path now performs zero allocations. Cold decode token with 102 misses: 1.97 s, exactly its SSD bytes
(102 × 18.8 MB at 1.0 GB/s); logits bit-identical to the old loader.

## 5. Allocator cache (inherited finding)

With eight prefetch workers calling `mx.array()` concurrently, MLX's free-buffer cache grew to ~10 GiB and
produced hundreds of 100–380 ms allocation stalls on warm turns. `mx.set_cache_limit(2 GiB)` removed the
tail without changing hit rates or bytes. Do not raise it without a benchmark.

## 6. Prefix cache (`validate_prefix_cache.py`)

Turn two of a conversation reused 28 of 41 prompt tokens: prefill 24.7 s instead of 44.6 s, identical greedy
continuation, argmax-identical first-token logits (max-abs logit difference ≈ 1.0 from the different
accumulation order of layer-major prefill vs. sequential decode).

## 6b. Internal SSD (5.2 GB/s) and fused decode kernels

Moving the checkpoint to the internal drive (`cachalot doctor`: 5,199 MB/s) removed the bandwidth wall: the
loader pulls 102 misses in 0.36 s (5.7 GB/s). Decode became 0.43–0.50 s/token at 73–78 % hit (2.0–2.3 tok/s);
cold 512-token prefill 86 s, warm 75–103 s.

With storage fast, launch overhead showed: ~2,400 kernel launches per token. `moe_fused_metal.py` runs all
top-k experts of a layer in two launches (gate/up + SwiGLU + router weight, then the weighted W2 sum) and matches the
unfused kernels to 2.4e-7; `bf16_gemv_metal.py` reads the bf16 output head directly instead of converting 1.3 GB
to fp32 per token. All-resident decode token: 0.15 s → 0.10 s.

Prefill is now compute-bound: warm prefill with 30 % fewer bytes ran *slower* than cold, because the
token-sequential attention loop, not the SSD, sets the pace. Batched prefill attention is the next lever.

## 6c. Batched prefill

Prefill was compute-bound after the disk upgrade: every attention layer, hyper-connection mix, router,
shared expert and routed expert ran token by token in Python. Batched now (all validated against the sequential
path on real layer weights):

| Component | Before | After |
|---|---|---|
| Sliding-window / reuse attention (32 layers) | per-token loop, ~1.1 ms/token/layer | one chunk pass: window = positions p-127..p over a concatenated key pool, padded per-token compressed top-k; rings exact, outputs within one bf16 ulp; 10-17x faster |
| Hyper-connection mixes | per-token Metal Sinkhorn | batched MLX Sinkhorn |
| Router | per-token | row-wise |
| Routed experts | 3 GEMVs per (token, expert) | dequantize FP4 -> bf16 once per expert, bf16 GEMM over its tokens (official expert linears also return bf16) |
| Shared expert | per-token FP8 GEMV | row-wise FP8 quantization + GEMM |
| Engram | per-token row reads | one deduplicated read, 8 threads |
| `wo_a` dequantization | lazily on first prompt (~4 s) | at load |

512-token cold prefill: 86 s -> 55 s (9.3 tok/s) with a 34 s SSD floor; 128 tokens: 31 s -> 15.6 s.
Source / index-only source layers (8 of 40) still run their attention sequentially (compressor + indexer state).

Routed experts now use an exact repacking of FP4 into MLX's affine 8-bit layout (`fp4_affine8_metal.py`:
q = 2v/2^e + 12, scale 0.5·2^e, bias −6·2^e; bit-exact against dense dequantization on every tested expert) and
`mx.quantized_matmul`, which is 10–30 % faster per matrix than dequantize + bf16 GEMM. Chunking experts into one
`gather_qmm` per layer was measured and rejected: with per-row batches the kernel re-reads weights per row (3.2 ms
vs 2.1 ms for 16 experts), and the padded per-expert batch form needs stacked weights whose copies cost more than
they save (1.8 ms qmm + 1.4 ms stacking). Per-eval sync costs ~0.19 ms and each expert still needs ~12 launches, so
the remaining MoE compute (~0.7 ms/expert GPU) is launch-bound. A fused multi-expert Metal kernel
(`moe_prefill_fused_metal.py`: gate/up/SwiGLU for a 4-expert chunk in one launch, W2 in a second, accumulation
order identical to the decode GEMV) was written and measured: exact to fp32 ulp but 2.75 ms vs 1.83 ms per
4-expert chunk against the qmm path, in every ROWS/TILE configuration and with vectorized loads. It is kept as a
reference; a simdgroup-matrix (tile) implementation would be required to beat MLX's GEMM.

## 6c-2. Source-layer batching, the Engram mmap trap, and miss-aware lookahead

The eight compressor/indexer layers are now batched too (`source_prefill_batched.py`): chunked compressor with
carried partial-group state, indexer scores/top-k/candidate blocks under per-token visibility masks, functional
compressed-KV and index-K updates, attention over window + own top-k. Unit parity against the per-token path on
real weights: caches bit-exact, every top-k set and candidate mask identical, outputs within one bf16 ulp; 5–15×
faster per layer.

The read timeline then exposed the real remaining gap: the two Engram layers. Their 12k random 264-byte row reads
went through `mmap` page faults from several threads, costing 9 s per layer cold and 18 s warm (parallel faulting
on one mapping serialises, and the wired working set leaves little page cache). Reading rows with parallel
`pread()` takes 0.2 s. This one change moved the 512-token cold prefill from 48 s to 35 s and the warm one from
64 s to 29 s; between-layer time fell from 13–37 s to ~1 s.

The prefetch lookahead also became miss-aware (the handoff's TODO A): it keeps N real loads outstanding instead of
scanning N positions, so resident hits inside the window no longer starve the SSD queue on warm prompts.

Result: 512-token prefill 35 s cold (14.6 tok/s, SSD 86 % busy) and 29 s warm (17.8 tok/s), against SSD floors of
~33 s and ~24 s. Prefill is now within 10–20 % of the disk.

## 6c-3. Decode fusion, second pass (`profile_decode_gpu.py`, `profile_attention_reuse.py`)

Isolated timings with one `mx.eval` per call hide a ~0.15 ms eval floor, so this pass measured GPU time by
chaining 40 launches inside one lazy graph. That exposed where an all-resident token (98.9 ms) went:

| piece (per launch, GPU) | before | after | per token |
|---|---:|---:|---:|
| router (`matmul, softplus, argsort, take, ...`) | 0.150 ms | 0.022 ms (`router_fused_metal.py`: GEMV + one-threadgroup top-k with the stable-argsort tie rule) | 40× |
| sparse attention | 0.209 ms | 0.056 ms (`sparse_attention_decode_1d`: one threadgroup per head, simdgroup per key) | 30× |
| FP8 GEMV `wq_b` [32768×1280] | 0.262 ms | 0.113 ms (`fp8_gemv_decoded`: uint4 weight loads, 8 lanes per short row, pre-decoded fp32 activation) | 40× |
| FP8 GEMV `wo_b` [5120×8192] | 0.126 ms | 0.079 ms | 40× |
| shared expert `w2` | 0.070 ms | 0.029 ms | 40× |
| hyper-connection mixes (Sinkhorn) | 0.391 ms* | 0.029 ms (`hc_mixes_1d`) | 80× |
| RoPE, RMSNorm, hc_pre+norm, hc_post, FP8 activation quantization | ~8 launches each | 1 launch each | |

\* isolated measurement. Router top-k indices are identical to `argsort(...)[-k:]` on 200 real-weight trials;
FP8 GEMV results differ from the scalar kernel only in fp32 summation order (max relative difference 1e-7).
The MoE output is dispatched with `mx.async_eval` so the GPU runs the experts while the CPU builds the next
layer's attention graph (median token 74 → 69 ms).

All-resident decode token: **98.9 ms → 68 ms** (min 67 ms, median 68–70 ms over 15 repeats). Teacher-forced NLL
2.283 nats (unfused 2.302; run-to-run band 2.27–2.30). Greedy continuations can differ after ~10 tokens because
the logits differ at fp32-ulp level and greedy decoding is near-tie sensitive; NLL, not text, is the acceptance test.

What remains in the 68 ms: chained GPU pieces sum to ~41 ms (attention 13, experts 11, router 1, shared 5,
HC 6, head 2); the other ~25 ms is 40 router evaluations (each a GPU drain, the CPU decoding the expert ids,
and the launch encoding of the next layer). Removing those syncs needs GPU-side expert addressing, which only
pays when every expert is resident, i.e. not on this machine.

## 6c-4. FP4 GEMM on the simdgroup matrix units (`fp4_sgmm_metal.py`, `micro_fp4_rows.py`)

A 512-token prefill routes ~8 tokens to each expert, so the expert GEMM is a multi-row GEMV whose cost is
streaming 18.8 MB of FP4. The affine-8 path repacks (read 1x, write 2x) and `quantized_matmul` reads the 8-bit
copy: about five times the FP4 bytes. `fp4_sgmm_metal.py` dequantizes each FP4 block once into a bf16
threadgroup tile and multiplies it with `simdgroup_multiply_accumulate` (fp32 accumulation, exact bf16 weights),
split-K across simdgroups with a fixed-order reduce. GPU time per expert (gate/up + down, real weights):

| rows per expert | affine-8 qmm | simdgroup FP4 | row kernel (`moe_prefill_fp4_metal.py`, rejected) |
|---:|---:|---:|---:|
| 1 | 0.17 ms | 0.08 ms | 0.10 ms |
| 8 | 0.22 ms | 0.09 ms | 0.18 ms |
| 32 | 0.33 ms | 0.20 ms | 0.66 ms |
| 64 | 0.45 ms | 0.36–0.43 ms | 1.30 ms |
| 128 | 0.68 ms | 0.71–0.91 ms | 2.57 ms |

The prefill path uses the simdgroup kernels up to 64 rows per expert (`CACHALOT_SGMM_MAX_ROWS`) and
`quantized_matmul` above; `CACHALOT_PREFILL_SGMM=0` restores the affine-8 path. Variants measured and
rejected: staging the activation tile in threadgroup memory (no gain), 16-block groups for longer contiguous
reads (2.5x slower: register pressure). Per-launch time is bounded at ~200-300 GB/s per expert by a ~12 µs
fixed cost plus DRAM latency per simdgroup; batching several experts per launch would amortize it.

**Effect on prefill wall time: none measurable.** 512 tokens: 35 s cold / 26 s warm with either path, i.e.
95-100 % of the SSD floor (34 s / 24 s). 2048 tokens: 57 s / 49 s with either path against a 41 s / 32 s floor.
The expert GEMM is not on the prefill critical path at these lengths. The kernels stay on because they halve
the GPU time of the MoE phase and are validated (bf16-identical at 1 row, bf16-ulp differences otherwise; NLL
2.285 nats).

Two more null results from the same session, recorded so nobody repeats them:

- **Splitting an expert read into concurrent chunks** does not raise SSD throughput: a single 18.8 MB expert
  reads in 3.4 ms (5.5 GB/s) whether issued as one `preadv` or 2-16 concurrent pieces
  (`micro_expert_read_chunks.py`, 150 random experts). Reads that look faster than ~7 GB/s are page-cache hits;
  `F_NOCACHE` stops new pages from being cached but does not evict pages another process already cached.
- **Loader threads do not slow the GPU:** the per-expert MoE step costs 0.37 ms with the SSD idle and 0.38 ms
  with 16 `preadv` workers running at full rate.

Where the 2048-token prefill loses its ~15 s: the SSD is busy only 59-65 % of the wall time, in ~39 idle gaps of
~0.5 s, one per layer boundary. Routing of layer L+1 is unknown until layer L finishes, so the loader idles
through the tail of layer L's compute, the attention of L+1 and the router eval; the fix is overlapping that
tail (next lever, §7).

## 6c-5. Inside the layer-boundary gap (`profile_prefill_timeline.py 2048 1`, `CACHALOT_PROFILE_SYNC=1`)

The timeline script now marks the pre-read phase of every layer. The first SSD read of a layer starts
0.25-0.55 s after its MoE phase begins, and all of that is the router `mx.eval`: it forces the previous layer's
MoE tail, the shared expert, both hyper-connection mixes and the whole batched attention of the new layer, none
of which can overlap with loads because the routing is not known yet. Attribution with a sync after each phase
(2048 tokens, 40 layers):

| phase | first call, in situ | same call repeated (hot GPU) |
|---|---:|---:|
| batched attention (30 reuse + 2 window layers) | 8.6 s | 4.5 s |
| previous layer's MoE tail (lands in the first HC mix) | 2.9 s | |
| compressor + indexer (8 source layers) | 2.5 s | 1.5 s |
| shared expert | 0.9 s | 0.6 s |

Two fixes shipped:

- **Token-chunked attention.** The unchunked formulation gathered all selected keys into a [T, K, D] bf16 tensor
  (1.3 GB at 2048 tokens) and copied it to fp32 (2.7 GB); those two steps alone took 250 ms per layer in situ.
  Processing 256 tokens per chunk (`CACHALOT_ATTN_CHUNK`) keeps the temporaries in the buffer cache; the
  per-token arithmetic is unchanged and the NLL is identical to three decimals at chunk sizes 0/128/256.
  Attention per layer at 2048 tokens: 216 ms → 103 ms (hot).
- **`wo_a` as eight GEMMs** instead of a batched mat-vec over T x 8 groups: 53 ms → ~10 ms per layer. Same
  bf16 products, fp32 accumulation in a different order (max 1 bf16 ulp from an fp32 reference; the old form
  deviated more). NLL: 400 tokens 2.530 → 2.566, 1000 tokens 2.096 → 2.099, i.e. near-tie noise.
- FP8 linears in prefill now multiply bf16 operands (`CACHALOT_PREFILL_BF16_GEMM`); every dequantized E4M3 value
  and power-of-two scale is exact in bf16, so the products are the same as in fp32 and the GEMMs run at bf16
  speed (10 % of the attention time).

Prefill wall time: 2048 tokens 57 s → 55 s cold, 49 s → 46 s warm (floor 42 s / 32 s); 512 tokens 35 s → 33 s cold,
26 s → 25 s warm (at the floor). What remains in the 2048-token gap is ~5 s of attention/indexer arithmetic that
cannot start before the previous layer ends, ~3 s of MoE tail, and a ~2x "cold GPU" factor on the first burst
after an I/O-bound stretch (the repeated call is twice as fast). Overlapping the gap with speculative loads of the
next layer's most popular experts (nearly all 384 are used at 2048 tokens) is the remaining lever.

## 6c-6. Filling the gap: speculative next-layer loads and Engram prefetch

The routing of layer L+1 is unknown while layer L's tail and layer L+1's attention run, but at 2048 tokens
~81 % of a layer's 384 experts are used (64 % at 512 tokens), so the loader no longer waits for the router:

- **Speculative loads.** When layer L's MoE ends and its utilization was ≥ 50 % (`CACHALOT_SPEC_MIN_UTIL`),
  the store's most-requested non-resident experts of layer L+1 are submitted to the load pool, as many as fit the
  measured gap (gap / 3.4 ms, bounded by free transient slots). They land in transient slots. When the routing is
  known, unneeded loads are cancelled (queued ones never start), needed ones are kept and promoted into the
  layer's quota, and the layer consumes experts in arrival order (resident, then speculative in submission order,
  then the rest) so the main thread never waits on a later load while an earlier one is ready. Every (token, slot)
  pair receives exactly one contribution, so the traversal order changes no sum.
- **Engram rows in the background.** Row ids depend on token ids only; both Engram layers' 12k random 5 KB reads
  start at prefill begin instead of on the critical path at layers 1 and 14. Before this, those reads queued
  behind the speculative expert loads and once took 13-18 s instead of 1 s.
- **Budget vs. available memory.** The auto budget is now also capped by what is free or reclaimable at start
  (`available_memory()`, minus trunk, MLX cache and a 12 GiB headroom), so a machine whose other applications hold
  20 GB gives up a few GiB of experts rather than pushing them into swap.

Timeline at 2048 tokens: SSD busy 59 % → 91-93 % of the wall, idle gaps 24 s → 2.3-3.3 s. Prefill wall time now
sits on the SSD floor at both lengths:

| prompt | before this section | now | SSD floor (measured GB/s that run) |
|---|---:|---:|---:|
| 512 tokens cold / warm | 33 s / 25 s | 32 s / 23 s | 33 s / 24 s |
| 2048 tokens cold / warm | 55 s / 46 s | 44-45 s / 37-39 s | 41-47 s / 32-37 s |

Speculation costs ~7 % more bytes at 2048 tokens (cancelled loads that had already started) and nothing at 512
tokens; `CACHALOT_SPECULATIVE_PREFILL=0` disables it.

## 6d. Why 10-20 tok/s decode is out of reach on this machine (exactly)

Per token: ~0.07 s compute (§6c-3) + misses x 18.8 MB / 5.7 GB/s. Measured hit rate 73-78 % at 50 GiB
(the auto budget; 64 GiB produced a Metal out-of-memory). Reaching 10 tok/s needs <= 10 misses per token,
i.e. a 96 % hit rate; the static coverage bound at the largest budget this machine can wire is ~74 %.
Speculative decoding does not help because consecutive tokens share only 30 % of a layer's experts, so
verifying k tokens loads nearly k times the experts.

An opt-in approximation (`V41Model.set_decode_miss_budget(n)`: load at most n non-resident experts per layer,
highest router weight first, drop the rest) was measured on a 40-token greedy continuation:

| miss budget / layer | tok/s | teacher-forced argmax agreement | mean KL | identical greedy prefix |
|---:|---:|---:|---:|---:|
| exact | 2.9 | 40/40 | 0 | 40 |
| 2 | 3.0 | 37/40 | 0.12 | 8 |
| 1 | 3.5 | 32/40 | 0.25 | 4 |
| 0 (never load) | 9.7 | 29/40 | 0.64 | 7 |

Only "never load" reaches the target, and it destroys the output. It stays off by default.
What does reach 10+ tok/s exactly: a machine where the routed experts are resident (all-resident decode measured
0.068 s/token = 14.7 tok/s here after §6c-3; 256-512 GB Macs), or a faster expert path per byte (a second internal-class SSD in
parallel would not help; the loader already runs at the drive's limit).

## 6e. Decode anatomy after the fusion and prefill work (`profile_decode_timeline.py`, `decode_throughput.py`)

After a 512-token prompt, greedy decode runs at 2.8-2.9 tok/s (350 ms/token) with a 77-78 % expert hit rate,
53-55 misses per token. Per layer in steady state: `get_many` (SSD) 5-7 ms when the layer misses, GPU work
2.3-2.9 ms, Python/sync gaps 0.2 ms. The per-miss cost is 4.6 ms against a 3.4 ms single-read floor: a layer's
1-2 misses are the only reads in flight, so the disk runs at its single-stream speed and nothing overlaps the
router sync. Rounded: 55 x 3.4 ms = 190 ms of unavoidable bytes, 68 ms of compute, ~90 ms of latency and sync.

Measured and left off: sampled-LFU eviction (`CACHALOT_EVICT=lfu`, least requested of the 64 least recently
used) gives 77.2 % vs 76.9 % hits, within noise, as the offline replay in §3 predicted. Prefill speculation
(§6c-6) has a useful side effect here: the promoted transients are the layer's most-requested experts, and the
decode hit rate after a prompt rose from 73.5 % to 78 %.

The first decoded token of a process cost ~1 s instead of 0.35 s: Metal compiles each fused kernel on first
use. `TextDecodeRuntime.warmup()` (run by `V41Model.from_pretrained`) does a two-token prefill and one decode
step at load time so the first request does not pay it.

## 7. What would move the needle next

1. ~~Storage~~ done: internal SSD.
2. **Memory.** The auto budget already uses what the machine has; a 128 GB Mac holds 73 GiB of experts (≈ 75 %
   static coverage), a 512 GB Mac holds all of them.
3. ~~Batched prefill~~ shipped for all 40 layers and the whole MoE; prefill runs within 10–20 % of the SSD floor.
4. ~~Decode compute fusion~~ two passes shipped (§6b, §6c-3): all-resident token 0.15 → 0.068 s. The remaining
   ~25 ms per token is the 40 per-layer router syncs; the rest is bandwidth-bound GEMVs.
5. ~~Prefill layer-boundary gaps~~ closed by speculative next-layer loads and Engram prefetch (§6c-6); prefill
   of 512 and 2048 tokens runs at the SSD floor. Faster prefill now needs more resident experts or a faster disk.
6. **Per-miss latency, not bandwidth, bounds decode** (§6e): one 18.8 MB read takes 3.4 ms on this SSD whatever
   the queue depth. Striping each expert across two drives (internal + a Thunderbolt 5 NVMe) would halve it, but
   needs a re-laid-out expert bank file, not symlinked shards; that is the one storage change with a large exact
   decode gain left on a 96 GB machine (~2.9 -> ~4 tok/s).
