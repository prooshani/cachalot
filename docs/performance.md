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
the remaining MoE compute (~0.7 ms/expert GPU) is launch-bound; a fused multi-expert repack+GEMM kernel would be the
way past it.

## 6d. Why 10-20 tok/s decode is out of reach on this machine (exactly)

Per token: ~0.10 s compute + misses x 18.8 MB / 5.7 GB/s. Measured hit rate 73-78 % at 50 GiB
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
0.10 s/token = 10 tok/s here; 256-512 GB Macs), or a faster expert path per byte (a second internal-class SSD in
parallel would not help; the loader already runs at the drive's limit).

## 7. What would move the needle next

1. ~~Storage~~ done: internal SSD.
2. **Memory.** The auto budget already uses what the machine has; a 128 GB Mac holds 73 GiB of experts (≈ 75 %
   static coverage), a 512 GB Mac holds all of them.
3. ~~Batched prefill~~ shipped for 32 of 40 layers and the whole MoE; source-layer attention and the expert GEMM path remain.
4. **Decode compute fusion.** 0.15 s/token of small-op overhead becomes the bottleneck once storage is fast.
