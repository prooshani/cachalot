# Cachalot speed and quality review

Date: 2026-09-19. Reviewed checkout: `093d1c6` on `main`.

This is an analysis and experiment proposal, not a runtime patch. The existing untracked `benchmarks/predictor_recall.py` was inspected and left unchanged. An independently running guarded benchmark was present, so no model was loaded and no GPU benchmark or full MLX test suite was started. Compiler checks and small CPU-only reproductions were run.

## Assessment

The underlying design is appropriate for a 96 GiB M3 Ultra: keep the trunk resident, read selected experts into reusable unified-memory buffers, batch prefill by expert, and preserve real routing on cache misses. It already implements MoE. The problems are practical latency and insufficient evidence that the complete numerical/generation path preserves the checkpoint's coding capability.

The most urgent finding is in the quality gate itself. **The saved FP4 comparison has zero compiling C++ blocks out of four, not one out of four.** The checker mistakes a fatal compiler error for success. Consequently, a faster runtime that passes the current gate can still be unusable for coding.

Keep native FP4 as the current quality baseline. Do not spend another large bank build trying to repeat the rejected affine fits. However, distinguish “these candidates failed this evaluation” from a proof that all mixed precision or calibration methods are impossible. The existing small, confounded syntax test cannot support the latter claim.

## 1. Architecture and useful boundaries

The important path is:

1. CLI/library/HTTP requests reach `generation.stream_tokens`, which encodes prompts, restores prefix snapshots, samples, and advances sequence state.
2. `TextDecodeRuntime` owns the resident trunk, expert bank, Engram access, sequence state, layer schedule, and memory limits. Its five block variants cover the sliding, compressed-source, index-source, and reuse cases.
3. Each decode MoE evaluates real routing and speculative future routing. The router evaluation is the CPU/GPU synchronization boundary needed to identify expert slots.
4. `ResidentExpertStore.get_many` resolves hits and predictive loads, issues demand misses, waits for all selected experts, and admits them in deterministic order.
5. `ExpertReader` uses positional reads into preallocated `ExpertSlotPool` buffers. FP4 decode uses fused routed-expert kernels; affine banks use a separate matmul path. The resident shared expert is then combined with routed output.
6. Prefill groups work by expert and uses transient slots plus quota admission. Engram and expert prefetch hide some storage latency.

The storage/cache separation and slot reuse are worth preserving. A broad rewrite or model-generic execution framework is not the first speed improvement. The main structural problem is that configuration, evidence, and policies are less explicit than the numerical execution.

Sources: [runtime](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/model/text_decode_runtime.py:207), [MoE](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/model/moe_layer_metal.py:38), [store](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/cache/resident_store.py:470), [slots](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/cache/slots.py:57), [reader](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/storage/reader.py:149).

## 2. Confirmed quality-gate failure

### Fatal compiler errors are reported as clean

`check_cpp` extracts only stderr lines containing `: error: ` and never checks the compiler return code. Clang's missing-header diagnostic uses `: fatal error: `, so a failed compilation can return an empty error list. `main` interprets that list as a clean block.

CPU reproduction during this review:

```text
Source: #include <cachalot_review_missing_header_123.h>
Existing checker: []
clang++ return code: 1
Diagnostic: fatal error: 'cachalot_review_missing_header_123.h' file not found
```

Recompiling the saved matched comparison with the same C++17 syntax-only flags produced:

| Saved arm | C++ blocks | Previously counted clean | Successful compiler exits | Truncated blocks |
|---|---:|---:|---:|---:|
| lc_fp4 | 4 | 1 | 0 | 2 |
| lc_q3g64act | 3 | 0 | 0 | 3 |

The supposedly clean FP4 block is the 21-line second C++ block in seed `20260919`, turn 3. It contains `#include <s>` and fails with `fatal error: 's' file not found`. This is a direct reproduction, not an inference from reported metrics.

Fix the gate before further quality decisions:

- Define compilation success by successful checker execution **and return code zero**. Capture ordinary and fatal diagnostics separately for diagnosis.
- Treat a missing compiler or timeout as an invalid measurement, not a model syntax error.
- Score every requested task. Missing code, unsupported/unlabelled fences, refusal, incomplete files, and malformed output must not disappear from the denominator.
- Separate complete programs, snippets, and truncated outputs. A fragment that parses is not a completed application.
- Use compile success and behavioral test success as primary outcomes. Diagnostic counts are secondary: one malformed include can suppress later diagnostics, while one syntax mistake can produce dozens of cascading errors.
- Keep language results separate. Python's `ast.parse` currently reports at most one syntax error, whereas Clang can report many; their error densities are not comparable measures.

Sources: [checker](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/code_validity.py:51), [scoring](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/code_validity.py:94), [saved results](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/code_validity_q3g64act.json), [failing FP4 reply](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/replies/lc_fp4/DeepSeek-V4.1-Flash-fp4-experts_seed20260919_turn3.txt).

### Current numerical baseline does not settle runtime fidelity

The expert-precision experiment substitutes expert arithmetic inside Cachalot. The rest of the trunk, attention, indexing, Engram, prompt handling, and state management remains Cachalot. Comparing fused versus unfused Cachalot also shares most of the implementation. Those are useful regressions, but neither is an independent whole-model reference.

Establish pinned reference fixtures from the official implementation using the same checkpoint revision, prompt token IDs, precision semantics, and teacher-forced positions. Compare logits and selected intermediate states. Start with short contexts, then ring-wrap and compression boundaries, chunked prefill, snapshot restore, and long-context indexing. Capture router IDs/weights, selected attention indices, Engram hashes, layer outputs, and final logits so the first divergence is identifiable.

Preserve both an unpenalized fidelity profile and the current usability profile with sampling penalties. Penalties can reduce looping without repairing an incorrect model computation, and can also damage necessary repetition in code. Their effect must be judged on functioning programs. Near-tie greedy divergence explains some different continuations; it does not establish that repeated corruption is inherent to the model.

Sources: [expert gate](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/nll_expert_precision.py:297), [fused comparison](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/validate_fused_decode.py:1), [sampler](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/model/generation.py:70).

## 3. Repair experimental identity and statistics

The current repetition experiment primarily tests one three-turn conversation with a few seeds. In its usual mode, each bank generates its own preceding assistant messages, so later inputs differ. That is a valid end-to-end conversation test, but not a controlled comparison of the same turn. The existing canned-reply option should be used for the controlled arm; retain separately the full conversation test.

The NLL comparison matches target token arrays but does not require matching the complete conditioning prefix, code revision, environment, or checkpoint identity. Its token-level sign z-score treats adjacent tokens as independent and counts exact ties among the non-improvements. Identical differences also create a zero standard error in the printed mean z-score. These are insufficient foundations for the handoff's strong significance claims.

Recommended protocol:

- Compare full prompt/target hashes and resolved configuration, not only suffix targets or bank directory names.
- Keep mean NLL, median paired difference, tails, and top-1. Do not discard the mean merely because a small corpus estimates it noisily: rare large errors can be precisely what breaks code.
- Report ties explicitly, use a tolerance appropriate to the numerical comparison, and handle zero-variance differences.
- Estimate uncertainty across independent tasks/documents, or use block/cluster resampling. A seed is nested within a task; multiple tokens from one document are not hundreds of independent tasks.
- Start with a small smoke corpus, then a fixed held-out suite across code completion, full-file generation, edits, bug fixes, tool calls, and longer multi-turn work. Twenty or more distinct tasks is a practical initial coverage target, not statistical certification.
- For every run, store planned and completed case counts, exit status, finish reason, tokens, timeout/guardian outcome, and a final `complete` marker. Interrupted runs remain explicitly invalid.
- Make run directories immutable and uniquely identified. Include git SHA/dirty state, model/tokenizer/template revisions, bank manifest, package versions, sampling, memory budget, mirror policy, and cache state.

A useful product metric is **successful tasks per wall-clock hour**, accompanied by pass rate and latency distributions. Tokens per second alone rewards a fast stream of unusable code.

Sources: [repetition workload](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/repetition_quality.py:50), [canned context and generation](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/repetition_quality.py:118), [NLL comparison](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/nll_expert_precision.py:376), [guardian](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/guarded_run.sh:119).

## 4. What the FP4 performance evidence actually supports

The saved FP4 anatomy run at 36 GiB and 512 prompt tokens reports 350 ms/token, 69.5 real misses/token, 1,858 MiB read/token, 202.1 ms/token inside expert acquisition, and 148.4 ms/token elsewhere. The interleaved mirror experiment reports approximately 341.5 ms/token without mirroring and 327 ms/token at fraction 0.10. These are different measurements; do not silently combine their counters.

The 84.6 ms repeat-token measurement is a useful resident reference, but the script times one warmed token and does not assert zero misses during that measurement. Add miss/byte deltas and repeated samples before calling it a verified floor.

### “Compute is already free” is not established

`decode_anatomy` computes wall-clock bandwidth as bytes divided by decode wall time. Dividing bytes by that same bandwidth necessarily reconstructs decode wall time; it cannot independently demonstrate complete I/O/compute overlap.

Furthermore, the logged union of read-call intervals is 18.05 s over 22.43 s of decode. Its intersections with expert waits total 12.87 s. About 5.18 s of read intervals therefore overlap the 9.50 s outside `get_many`; roughly 4.32 s of that outside time has no recorded expert read. This does not prove all 4.32 s is GPU work, but it contradicts treating the whole non-wait path as known-hidden computation.

Read-call intervals are also not physical disk utilization: they include software and OS effects and can include page-cache service. The demand/predict overlap buckets do not identify the exact awaited future. An unrelated speculative read can overlap a blocked window. The implementation's measurements are useful; the causal labels are stronger than their instrumentation.

Improve tracing with request IDs and `(request, token, layer, expert)` identity, enqueue/start/end times, requested versus completed bytes, await dependencies, promotion/expiry events, and per-drive counters. Use Metal tracing for actual GPU intervals. Avoid extra synchronization in the throughput arm; profile detailed timing separately and quantify tracing overhead.

Sources: [anatomy reporting](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/decode_anatomy.py:171), [bandwidth calculation](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/decode_anatomy.py:333), [raw FP4 run](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/guarded/fp4-anatomy_20260918-232605.out), [resident probe](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/profile_decode_components.py:63). MLX evaluates lazily, so evaluation placement changes timing attribution: [MLX documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

### A realistic traffic bound

For a transparent scenario, assume an independently sustained 6 GB/s, 69.5 misses/token, unchanged routing/cache membership, and 18,800,640 bytes/expert. Useful expert traffic alone is approximately 1.307 GB/token. Even with zero speculative waste and perfectly hidden compute, that caps this workload at approximately **4.59 tok/s**. This is a conditional bound, not a predicted achieved speed or a universal bound for a 44 GiB chat session.

At the same bandwidth, 5 tok/s permits at most 63.8 full expert misses/token; 10 tok/s permits 31.9, before any waste or exposed compute. Across 240 selections/token those correspond to at least 73.4% and 86.7% resident hits. Prediction alone cannot remove the bytes of genuinely required nonresident experts.

Thus useful intermediate goals are a measured improvement over roughly 3 tok/s and better tail latency. Ten tok/s requires substantially better effective residency, more independently measured storage bandwidth, or another execution strategy. It should not be promised from one predictor change.

## 5. Confirmed lookahead lifetime defect

`get_many` calls `_sweep_inflight_locked(keep=requested)` before resolving the current layer. The sweep releases every completed predicted load that is not requested by this layer. It has no target token/layer deadline.

With `PREDICT_AHEAD=2`, layer L predicts both L+1 and L+2. If an L+2 read finishes before the L+1 acquisition, the L+1 sweep discards it before L+2 can consume it. Earlier successful completion can therefore reduce its usefulness. This invalidates interpreting a poor ahead-two result as evidence against longer lookahead in general.

A CPU-only reproduction executed the actual sweep method with a completed future for `(layer 2, expert 7)` and an intervening request for `(layer 1, expert 3)`. The future-layer slot was released and 18,800,640 bytes counted wasted.

Before tuning prediction depth, give speculative entries explicit target sequence/token/layer lifetimes. Retain valid future-layer work; expire work only after its deadline or cancellation; never reuse a buffer while a read or GPU consumer owns it. Add deterministic tests for early completion, late completion, sequence reset, duplicate predictions, cancellation, short reads, and pool exhaustion.

Sources: [sweep](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/cache/resident_store.py:298), [caller](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/cache/resident_store.py:499), [cumulative lookahead](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/model/moe_layer_metal.py:92).

## 6. Highest-value decode experiments

### A. Improve useful, timely prefetch rather than nominal recall

Keep exact demand fallback. Prediction must never change real expert selection or router weights.

First fix lifetime accounting. Then compare current top-6 with a small policy that considers predicted rank/margin, layer-specific usefulness, resident/inflight membership, recent demand history, time to consumption, and queue pressure. Prioritize demand work and bound speculative bytes; more workers already lost on the measured FP4 setup.

A policy should maximize avoided critical-path wait under a byte budget. Higher route recall can lose if it adds too many useless reads or makes real misses queue longer. Report precision only among actual speculative loads, miss coverage, fraction ready before use, wasted bytes, and p95 token latency. Keep top-6 as the baseline; the existing width sweep is not proof that it remains optimal after changing predictor, budget, or workload.

The proposed predictor ceiling in the handoff needs correction. Applying a deterministic router to its own actual input reproduces its routing, subject to identical numerics/ties. It does not reveal a 70% intrinsic model ceiling. Ordinary routing traces contain selected IDs, not the intermediate activations or prediction scores needed to infer recoverability.

The untracked `predictor_recall.py` is a useful starting screen, not a ceiling estimator. It reports all-route recall rather than cache-filtered precision. Its “previous token” comparison uses adjacent saved samples, although capture can sample every second or later token; merged recordings also introduce artificial document boundaries. It uses MLX without explicitly selecting CPU despite its no-GPU description. Record sequence IDs, positions, full true routes, predicted scores, and residency state; use held-out sessions. A small per-layer linear or low-rank predictor is worth considering only after a measured improvement over these corrected baselines.

Sources: [predictor](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/predictor_recall.py:117), [capture spacing and merge](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/capture_activations.py:73), [trace schema](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/metrics/routing_trace.py:16).

### B. Overlap resident shared-expert work with demand reads

The current decode function calls blocking `get_many` before building `shared_expert_forward`, although the shared expert depends only on the available activation and resident weights. Test constructing and asynchronously dispatching that independent work before waiting for routed experts, without adding an evaluation barrier or changing the final accumulation semantics.

This is a bounded experiment with a clearer dependency argument than a broad kernel rewrite. Measure whole-token latency, including any I/O/GPU contention. The isolated 17.1 ms/token shared-expert total includes repeated evaluation overhead and is not a promised saving.

Later, test whether resident routed experts can execute while missing experts load. That needs explicit slot pinning/fences and careful accumulation order, and can lose the benefit of the fused six-expert kernel. It is a second-stage experiment.

Source: [blocking acquisition and later shared work](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/model/moe_layer_metal.py:143).

### C. Revisit fusion only with a measured exposed compute budget

The current isolated component table sums to 154.5 ms against an 84.6 ms repeat-token measurement. It cannot allocate the latter proportionally. Count actual dispatches and measure GPU intervals before targeting hyper-connections, projections, or expert kernels. A native rewrite is not justified by Python line count.

Keep DSpark deferred on the current measured workload. Its rejection is an economic projection, not a theorem: reopen if useful bytes, cache hit rate, draft residency cost, verification cost, or measured acceptance changes substantially. Speculation must eventually pass rollback and state-equivalence tests as well as speed tests.

## 7. Prefill: investigate CED, not only faster GEMMs

The current prefill iterates all layers over the full prompt and returns only the final position's logits. It therefore follows the full reference-style forward rather than exploiting a decoder replay shortcut. This matters for coding, where uncached repository context and tool output can dominate latency. Typing-time prefill does not hide a large tool response or pasted source file.

DeepSeek's report describes processing the full prompt through the encoder, constructing decoder global KV from encoder output, and replaying only the final 128 positions through the decoder. **This bounded replay is approximate**, because sliding-window dependencies accumulate across layers. The report explicitly distinguishes it from exact reconstruction, which requires a longer suffix. [Official technical report, §3.2.2](https://arxiv.org/html/2609.19969v1#S3.SS2.SSS2).

Treat this as a separate, opt-in research branch. Preserve global KV/index construction for all required prompt positions; test absolute positions, first generation tokens, prefix extension, and restore. Do not merely slice the decoder's input and assume correctness. Compare full prefill, any dependency-proven exact suffix optimization, and bounded replay on task quality and first-token latency. Fewer processed tokens may still touch many of 384 experts, so compute savings do not automatically halve SSD traffic.

The current README claims bounded replay is implemented; the inspected prefill path does not show this decoder shortcut. Clarify which operation that claim refers to. The checkpoint's minimal inference code also runs full layers, so matching it does not deliver the production optimization automatically.

Sources: [full layer loop](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/model/text_decode_runtime.py:1909), [README claim](/Users/hamedprooshani/Projects/deepseek-v41-mac/README.md:42), [local official forward](/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/model.py:1243).

## 8. Memory improvements that can preserve weight quality

Sequence caches are rounded to low precision but stored expanded as bf16. `reset` allocates compressed caches and indexer caches against maximum sequence length. Prefix caching is capped at 16 entries, not a byte budget, and snapshots retain whole-capacity arrays.

From the inspected shapes, compressed KV contributes approximately 2,560 bytes per configured position and indexer K another 640, excluding windows and other state. At 32,768 positions that is about 100 MiB of nominal arrays per distinct complete state; at 262,144 positions it is about 800 MiB. Sharing means multiplying snapshot counts can overestimate physical storage, so measure live allocations rather than summing aliases blindly.

Priority order:

1. Add a prefix-cache byte cap and expose actual retained state and reuse statistics.
2. Retain valid prefixes/chunks instead of full unused capacity; use copy-on-write or immutable chunks where appropriate.
3. Investigate packed FP4 KV and indexer representations, preserving the quantization already performed, with fused unpacking on read.
4. Reassign proven memory savings to expert residency only within the existing whole-system memory guard.

This is more defensible than raising the expert budget until the machine swaps. It is especially relevant for long-context coding, but likely a smaller short-context decode gain than better prefetch.

Sources: [cache allocation](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/model/text_decode_runtime.py:992), [snapshots](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/model/text_decode_runtime.py:1097), [entry-only cap](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/model/prefix_cache.py:66), [expanded low-precision state](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/model/source_prefill_batched.py:1).

## 9. Configuration and operational quality

Several findings make a single resolved profile valuable:

- `scripts/chat.sh` still selects the old internal oQ3e bank and omits the recommended repetition penalty, hotlist, and mirror settings. It conflicts with the current FP4 handoff.
- `HANDOFF.md` still mixes old 2-bit interactive hit rates and “current best” 2-bit benchmark commands with FP4 conclusions. A 90.7% hit rate from a 4,431-expert 2-bit cache is not evidence for the same rate in a much smaller FP4 resident set.
- Mirror striping checks the matching filename exists, but not that checkpoint bytes/layout match. A different checkpoint with the same shard names could silently corrupt experts. Validate a bank/mirror manifest once at startup; avoid full hashing on every read.
- HTTP accepts `tool_choice`, but `build_chat_request` does not carry it into `ChatRequest`. Coding-agent compatibility needs real tool round trips and either implementation or explicit rejection of unsupported controls.
- Prefix reuse, penalties, thinking mode, token cap, and checkpoint identity should be visible in each result, not inferred from launch commands.

Use explicit `fp4-quality` and experimental profiles with resolved immutable settings. Parse environment overrides once. Keep reader, scheduler, sequence state, and model execution as separate responsibilities; do not add descriptor lookups to every expert matmul. Consolidate duplicated documentation by generating baseline summaries from validated run manifests.

Sources: [chat launcher](/Users/hamedprooshani/Projects/deepseek-v41-mac/scripts/chat.sh:18), [mirror handling](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/storage/reader.py:185), [request fields](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/server/app.py:35), [request construction](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/server/app.py:157), [engine request](/Users/hamedprooshani/Projects/deepseek-v41-mac/src/cachalot/server/engine.py:30).

## 10. A gate hierarchy worth enforcing

**Fast CI:** fix the present lint baseline, then require it to stay clean. This review ran the configured `ruff check src tests benchmarks` and found 36 findings. Add CPU tests for compiler failure classification, missing outputs, incomplete manifests, and statistics edge cases. Add deterministic store lifecycle/error-path tests using controlled futures and tiny fake data.

**Small numerical fixtures:** test unpacking, rounding boundaries, router ties, attention visibility, Engram hashes, and state transitions independently. Ensure tests fail on violated tolerances. A script that only prints `bit-identical False` and exits successfully is a diagnostic, not a gate.

**Model fidelity:** run a guarded reference-fixture corpus on production FP4, including prefill/decode equivalence, ring-wrap boundaries, prefix restore, and long-context retrieval. Pure scheduling changes should preserve recorded routes and outputs under the same numerical path; arithmetic changes require explicit measured tolerances.

**Coding quality:** require complete task outputs, successful compilation, and behavioral tests. Include CSV quoting/escaping edge cases for the current example, plus unrelated algorithms, edits, and tool use. Keep isolated fixed-context and full multi-turn suites separate. Report pass rate, truncation, repetition, unsupported output, and uncertainty.

**Performance:** compare exact same task/configuration in randomized interleaved runs. Use at least the existing four-run-per-arm discipline initially, then enough observations to resolve the declared effect. Measure cold and warm startup separately, true first-token latency, decode p50/p95, expert bytes per generated token, timely prefetch coverage, and successful tasks per minute. Include short, 512-token, 2k, and representative repository-context prompts within the memory guard.

**Soak:** long output and multiple turns, cancellation, queued requests, tool-result continuation, memory pressure, and steady-state prefix-cache growth. Do not call a 64-token throughput run representative of a 4k-token coding response.

Source: [current CI](/Users/hamedprooshani/Projects/deepseek-v41-mac/.github/workflows/ci.yml:1), [test model opt-in](/Users/hamedprooshani/Projects/deepseek-v41-mac/tests/conftest.py:12), [print-only validation](/Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/validate_fused_decode.py:37).

## 11. Recommended order and stopping conditions

| Order | Work | Evidence required to continue |
|---|---|---|
| 1 | Repair compiler gate; rescore saved replies; unify active profile and run manifests | No false-success fixtures; every requested case accounted for; no stale-bank ambiguity |
| 2 | Establish independent FP4 reference checks and functional coding baseline | Numerical differences localized; completed programs evaluated; no claim of coding usability from NLL alone |
| 3 | Fix prediction lifetime and dependency tracing | Early future-layer loads survive until target; bytes and awaits reconcile; no slot lifetime regression |
| 4 | Test shared-expert overlap and miss-aware predictive admission separately | Repeated whole-token/p95 improvement at unchanged outputs and safe memory; discard nulls |
| 5 | Evaluate CED prefill strategy and byte-bounded prefix state | Better long-prompt TTFT with an explicitly passed quality gate; approximation stays opt-in |
| 6 | Consider packed KV, targeted fusion, or storage expansion | A measured bottleneck and achievable bound justify the added complexity |

Hardware results from a different residency regime are not targets for this machine. For example, the public M3 Ultra DS4 fork reports its headline V4.1 rates on a **512 GB** Mac. Its kernels and validation ideas can be useful comparisons, but those rates do not transfer to a 96 GiB SSD-streaming setup. [Project's own measurements](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra).

## Evidence and limits

This review used the codebase graph at Verify tier, project `Users-hamedprooshani-Projects-deepseek-v41-mac`, generation `2026-09-18T23:08:21Z`, followed by exact source inspection. Coverage checks for 35 relevant paths reported matching metadata and no recorded parse gaps. The graph is best-effort; its call edges are not proof of dynamic behavior. Ignored benchmark results were read directly.

Reviewed material includes the current handoff and continuation prompt, performance history, relevant dated handoff sections, architecture investigation, runtime/storage/cache/generation/server paths, gate implementations, and selected raw runs and saved replies. This is not a line-by-line proof of every Metal kernel or every historical experiment.

Checks actually performed: seven saved C++ blocks recompiled; missing-header false success reproduced; ahead-two lifetime defect reproduced with the actual sweep function and fake completed futures; configured lint command run. No full-model performance number in this report is a new measurement. The earlier 143-test result remains historical; this review did not rerun the full suite while another benchmark was present. No runtime, checkpoint, or benchmark implementation was modified.
