# Expert cache miss reduction assessment

The strongest near-term target is 60–65 required expert misses per generated token with a 44 GiB FP4 resident cache, subject to whole-process memory headroom. At 36 GiB, 65–70 is a more defensible initial target. These are engineering targets, not measured improvements or guarantees. Reaching 55–60 at 44 GiB is a stretch goal requiring better retention or admission. The current evidence does not justify promising 30 misses/token or a 90% resident hit rate.

## Definitions

There are 240 routed expert selections per token: 40 layers times six experts. A resident miss means a required expert is not already in the resident store. A successful prefetch can serve such a miss without blocking, but still requires SSD traffic. An unpredicted demand read, a late prefetch, and an unused speculative read are different events and need separate counters.

The previously inspected FP4 anatomy run measured approximately 69.5 resident misses/token, including approximately 41.8 supplied by prediction and 27.7 demand reads. It also incurred approximately 34.2 unused speculative reads/token. Those categories must not be combined into a single cache hit metric.

## CPU-only replay performed for this assessment

Input: `benchmarks/results/trace_routing_v7.trace.npz`. Five 512-token prefill segments each precede 32 decode tokens, for 160 decode tokens total. Four distinct prompts cover code review, prose summary, code explanation, and engineering documentation; the fifth returns to the code-review prompt. This is an older, short trace, not a fresh runtime benchmark.

The existing `Store` class in `benchmarks/simulate_policies.py` was extracted with Python AST and executed with NumPy and Python collections only. No model or MLX execution was used. Replay preserved segment boundaries, simulated prefill, and processed decode requests by token position and layer. FP4 expert size was 18,800,640 bytes.

| Resident budget | Expert slots | LRU misses/token | Future-aware optimistic bound | Two-use admission misses/token |
| --- | ---: | ---: | ---: | ---: |
| 36 GiB | 2,056 | 72.31875 | 55.66250 | 88.29375 |
| 44 GiB | 2,512 | 64.10625 | 50.16250 | 77.80625 |

Each alternative started a decode segment from the same resident keys as the baseline LRU simulation. The oracle tracked all future references within that segment, evicted the resident with the farthest next use, and bypassed admission when the incoming expert's next use was farther away. It cannot be implemented online as specified. Its miss count equaled the number of distinct requested experts absent from the initial resident set in every segment: the oracle eliminated all subsequent reloads in this short horizon. It does not establish a floor for different prefill warming, longer sequences, different routing traces, or a different memory budget.

The two-use experiment counted decode references from the start of each segment. On a miss, it admitted an expert only on its second or later decode request; hits updated LRU recency. This simple admission rule was substantially worse than LRU. It is not an implementation or evaluation of TinyLFU.

Separate end-to-end replays changed only prefill ordering, keeping the existing retain-before-admit behavior:

| Prefill ordering | 36 GiB misses/token | 44 GiB misses/token |
| --- | ---: | ---: |
| First occurrence | 72.31875 | 64.10625 |
| Whole-prompt popularity | 72.40625 | 64.81250 |
| Last 32 prompt tokens' popularity | 73.75000 | 66.49375 |
| Last 128 prompt tokens' popularity | 73.83125 | 66.76250 |
| Most recent occurrence | 73.51250 | 66.83125 |

Thus, neither simple prompt-tail warming nor unconditional two-use admission earned a recommendation. The larger cache reduced misses by 8.2125/token, or approximately 11.4%, in this replay. The ideal retention gap was approximately 16.7/token at 36 GiB and 13.9/token at 44 GiB. A practical policy can recover only an unknown part of that gap. The simulator does not model asynchronous completion, prefetch traffic, GPU execution, or every detail of the current prefill planner; these numbers cannot predict runtime latency directly.

## Recommended experiments, in order

1. Measure 36 versus 44 GiB with identical FP4 weights, prompts, and decoding settings. Keep 44 GiB only if long-context and repeated-session memory tests retain sufficient headroom. This adds 456 expert slots without changing expert mathematics.
2. Fix predicted expert lifetimes before increasing prediction distance. `_sweep_inflight_locked` currently releases finished predictions not requested by the current layer, including valid predictions for later layers. Retain predictions until their intended layer/token deadline, with bounded slots and backpressure. This targets duplicate/wasted reads and stalls, not necessarily resident misses.
3. Test session-aware retention across suffix prefills. Preserve a bounded, decaying set of experts repeatedly useful during recent decode instead of letting a short new prompt immediately replace the working set. Always load and compute every genuinely routed expert. Evaluate fresh requests separately from multi-turn coding sessions.
4. Test adaptive prefill quotas. `prepare_prefill_layer` divides capacity equally among 40 layers. Allocate a bounded amount of capacity using causal per-layer reuse measurements and marginal misses avoided per slot. Keep a minimum allocation and periodically rebalance. Decode already has global LRU; this experiment concerns prefill's initial resident allocation.
5. Test admission based on incoming-versus-victim reuse estimates, retaining a small recency window for newly hot experts. The distinction between admission and replacement is motivated by [TinyLFU](https://arxiv.org/abs/1512.00727); it is not evidence that TinyLFU will win on these expert traces. Reject the policy if held-out replay or runtime measurements do not improve.

## Acceptance criteria

Use held-out coding conversations with 256–1,024 generated tokens and multiple prompt lengths, rather than tuning and validating on the same five short segments. Compare each candidate against the same baseline and route trace before running the full model. Report resident misses/token, demand reads/token, ready and late prefetches, wasted bytes/token, total SSD bytes/token, expert wait milliseconds/token, decode throughput, tail latency, and process/system memory pressure.

Cache changes should preserve all six routed experts and FP4 computation. Require deterministic output agreement with the baseline, then validate compilation and task tests independently; agreement alone does not establish coding quality. Promote only changes that improve elapsed generation time without memory or correctness regressions. At this expert size, ten fewer reads save 188 MB/token, equivalent to about 31 ms of transfer at an assumed sustained 6 GB/s. Actual latency savings depend on overlap and contention.
