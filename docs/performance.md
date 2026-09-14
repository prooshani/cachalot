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

## 7. What would move the needle next

1. **Storage.** A Thunderbolt NVMe enclosure (3–6 GB/s) or the internal SSD divides every miss cost by 3–6.
2. **Memory.** The auto budget already uses what the machine has; a 128 GB Mac holds 73 GiB of experts (≈ 75 %
   static coverage), a 512 GB Mac holds all of them.
3. **Batched prefill compute.** Per-token Python loops cost ~90 s on a cold 512-token prompt on top of the SSD
   floor; batched attention/MoE GEMM removes most of it and is required for multi-thousand-token prompts.
4. **Decode compute fusion.** 0.15 s/token of small-op overhead becomes the bottleneck once storage is fast.
