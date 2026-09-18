# Cachalot — Multi-Model Compatibility Investigation

**Status:** investigation and plan only. No production code has been changed. Nothing here should be
implemented before this document has been reviewed.

**Date:** 2026-09-18. **Baseline:** Cachalot 0.5.0 + the 3-bit bank adoption of 2026-09-18
(`219a77f`), `main` clean.

**Scope:** whether the DeepSeek V4.1 Flash runtime in this repository can become a model-extensible
runtime for very large sparse MoE models on a 96 GiB M3 Ultra, without giving back the decode
performance already measured.

**Reading order.** `docs/HANDOFF.md` remains the authoritative state of the DeepSeek work. This
document does not supersede it and does not change any of its rankings. It adds one axis — model
extensibility — and ends with a recommendation about when to act on it.

---

## Summary of the finding

Three things in the repository decide the answer, and all three were derived from code rather than
assumed.

1. **The streaming core is already model-independent in shape but not in name.** `ExpertReader`,
   `ExpertSlotPool`, `ResidentExpertStore`, `ResidentExpertPrefetcher` and the LRU/quota machinery
   operate on `(layer, expert) -> byte ranges` and on a `{short_tensor_name: size}` map. Nothing in
   them knows about DeepSeek except three literal strings (`w1.weight`, `w2.weight`, `w3.weight`)
   and one default tuple. That is roughly 1,600 lines of the most valuable code in the project and
   it is nearly portable as it stands.

2. **The expert-bank storage layout Cachalot already reads is the same layout every MLX conversion
   of Qwen3.8-Flash-Next and GLM-5.3-Flash uses.** `build_stacked_expert_index()` indexes a 3-D
   `[n_experts, rows, cols]` tensor per projection per layer and slices per-expert contiguous byte
   ranges out of it. MLX's `QuantizedSwitchLinear` (`mlp.switch_mlp.gate_proj.{weight,scales,biases}`)
   is exactly that. The differences are a regex, a config key and a projection-name map.

3. **The execution path is DeepSeek all the way down, and correctly so.** 40 layers, 5120 hidden,
   2304 intermediate, hyper-connections, Engram, the compressed/sliding/source/reuse block taxonomy,
   DSA indexing and DSpark are hard-coded as module-level constants and as a fixed sequence of
   `if layer_id in SOURCE_LAYERS` branches in `_decode_token_impl`. None of that is reusable for
   another architecture and none of it should be made generic. It should be *joined*, not
   *abstracted*: a second model gets a second runtime class beside this one, not a parameterised
   version of it.

The recommendation is **option 2 — limited preparatory refactoring while DeepSeek optimization
continues** — and the preparatory work is much smaller than the brief assumes: it is confined to
`storage/index.py`, `cache/slots.py`, three string constants, and the bank builder. The decode hot
path is not touched at all.

One strategic correction to the brief, backed by arithmetic in section D: **Qwen3.8-Flash-Next will
not exercise Cachalot's SSD streaming machinery, because its entire expert bank fits in memory on
this machine.** At 3-bit group 64 the Qwen expert bank is 49.2 GiB against DeepSeek's 221.5 GiB.
Qwen is still the right first target — lowest execution risk, proves the loader and the tensor
mapping, and is a genuinely useful model on this hardware — but the claim it proves is "Cachalot can
host a second architecture", not "Cachalot's cache machinery generalises". **GLM-5.3-Flash is the
model that proves the second claim**, and it is the one whose working set most resembles DeepSeek's.

---

## A. Current architecture map

Package root `src/cachalot`, 23,782 lines of Python across 92 files. Four layers, bottom up.

### A.1 Storage — `src/cachalot/storage/`

| file | what it owns |
|---|---|
| `index.py` (232 lines) | expert discovery and per-expert byte ranges |
| `reader.py` (268) | positional reads into caller buffers |
| `tensor_index.py` (70) | whole-checkpoint safetensors tensor table |
| `tensor_loader.py` (100) | dtype-preserving load of one resident tensor into MLX |
| `engram_index.py` / `engram_reader.py` | Engram table geometry and sparse row reads |
| `store.py` / `tensors.py` | the older copy-based expert store, superseded by `cache/resident_store.py` |

`storage/index.py` is the single most important file for this investigation.

- `TensorRange` (`index.py:22`) — `(name, shard, start, end)`. Purely positional; no dtype, no model.
- `ExpertEntry` (`index.py:34`) — `(layer, expert, tensors: tuple[TensorRange, ...])`. The atomic
  unit of everything above it.
- `ExpertFormat` (`index.py:41`) — `(kind, bits, group_size, tensor_names, shapes, dtypes)`. This is
  already a miniature capability descriptor for the expert bank, and it is the precedent for the
  descriptor section 11 of the brief asks about.
- `_EXPERT_RE` (`index.py:9`) — `^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$`,
  the shipped FP4 checkpoint, one tensor per expert per projection.
- `_STACKED_RE` (`index.py:14`) —
  `^language_model\.layers\.(\d+)\.ffn\.experts\.(w[123])\.(weight|scales|biases)$`, the
  oMLX/`build_affine_bank.py` layout: one 3-D `[n_experts, rows, cols]` tensor per projection per
  layer.
- `build_stacked_expert_index()` (`index.py:120`) — computes `row_bytes = rows * cols * itemsize`
  and emits `TensorRange(start = base + expert * row_bytes, ...)`. **This is generic
  stacked-expert slicing with a DeepSeek regex bolted to the front.** It also enforces uniform bits
  and group size across the bank and raises on mixed quantization (`index.py:151`).
- `detect_expert_bank()` (`index.py:182`) — reads `config.json`, looks for the key
  `omlx_deepseek_v41`; present means stacked affine, absent means shipped FP4.
- `merge_contiguous_ranges()` (`index.py:208`) — coalesces adjacent `TensorRange`s into one read.
  FP4 collapses nine tensors into two reads; the stacked layout stays at nine.

`storage/reader.py` — `ExpertReader.read_expert_into(entry, views)` (`reader.py:149`) is the I/O hot
path. It `preadv`s each merged range straight into the caller's writable buffers. When an expert is
more than one range (the stacked case) it fans the pieces across a 16-thread pool
(`_read_pieces_concurrently`, `reader.py:238`) — measured 2.8 ms against 4.2 ms serial for nine
15.5 MB pieces. Also owns `F_NOCACHE`/`F_RDAHEAD` policy and optional two-drive mirror striping.
Nothing here is model-specific.

### A.2 Cache and residency — `src/cachalot/cache/`, `src/cachalot/io/`

- `slots.py` (129) — `ExpertSlotPool` allocates `n_slots` fixed slots once at startup, each a dict
  of `mx.array` buffers sized from `tensor_sizes`, with writable NumPy views aliasing the same
  memory. SSD reads land directly in unified memory the GPU already holds. This removes per-expert
  allocation, the memcpy and the Metal residency-set churn that made concurrent promotion 25x
  slower. Model coupling: a module constant `TENSOR_NAMES` (`slots.py:30`) listing the six FP4
  names, and a hard requirement that `{"w1.weight","w2.weight","w3.weight"}` are present
  (`slots.py:65`).
- `resident_store.py` (894) — the heart of the runtime. Two access disciplines:
  - **decode**: `get_many()` (`resident_store.py:470`) resolves a layer's top-k in one locked pass,
    reads all misses concurrently on the load pool, admits in caller order so LRU order does not
    depend on completion order, awaits any predicted load already in flight, and issues the next
    layer's predicted prefetch (`prefetch_decode`, `:191`) into transient slots that never evict a
    resident.
  - **prefill**: `prepare_prefill_layer()` (`:593`) is a deterministic per-layer quota planner run
    before asynchronous prefetch, so SSD completion order cannot change cache membership; unplanned
    misses go to transient slots released after the caller reports evaluation.
  - plus `preload()` (`:216`) for the startup hotlist, three eviction policies behind
    `CACHALOT_EVICT`, per-key locks for duplicate-load suppression, and full byte/time accounting.
  - Model coupling: `tensor_sizes_from_entry()` (`:77`) requires `w1/w2/w3.weight`; one attribute
    `self.format` holding an `ExpertFormat`; `num_layers: int = 40` as a default argument on
    `prepare_prefill_layer`. That is the whole of it.
- `io/resident_prefetch.py` (145) — prefill-side background loads with depth control and
  `discard_pending()` for speculation the router did not need.
- `io/engram_prefetch.py` (189) — the same pattern for Engram rows.

### A.3 Model execution — `src/cachalot/model/` (63 files, ~17k lines)

`text_decode_runtime.py` (2,586 lines) is the assembly point. Module constants at `:119-186` encode
the architecture: `DIM = 5120`, `HC_MULT = 4`, `N_LAYERS = 40`, `HEAD_DIM = 512`, `N_HEADS = 64`,
`ENGRAM_LAYER_IDS = (1, 14)`, `SOURCE_LAYERS = {2:2, 8:2, 14:2, 20:1}`,
`INDEX_ONLY_SOURCE_LAYERS = {24,28,32,36}`, `MTP_TARGET_LAYERS = (37,38,39)`,
`ENGRAM_NUM_EMBEDDINGS`, the two Engram shard filenames (`:485`), and both RoPE parameter sets.

`__init__` (`:225`) does, in order: resolve memory budgets against the machine; build the tensor
index; load the resident trunk; build zero-copy per-layer views; detect the expert bank
(`:372`, honouring `CACHALOT_EXPERT_BANK`); allocate the slot pool sized from one expert entry;
preload the hotlist; start the prefetcher; load the tokenizer; build the Engram token map and
layouts; precompute both RoPE tables; dequantize `wo_a` for all 40 layers; publish
`expert_store.decode_gates` (`:581`) so the MoE layer can predict the next layer's routing; start
the idle heartbeat.

Block implementations are one file per layer *kind*, doubled for decode and prefill:
`block_layer0`, `block_sliding_window`, `block_compressed_source`, `block_compressed_index_source`,
`block_compressed_reuse`. Each is a flat function taking ~40 explicit `mx.array` keyword arguments
supplied by `_common_block_kwargs()` (`:708`). There is no `nn.Module` tree and no per-token Python
object graph — a deliberate choice that keeps the hot path lean.

Numerics live in leaf modules: `fp4_gemv_metal`, `fp8_*`, `hc_sinkhorn_metal`, `decode_fused_metal`
(custom `mx.fast.metal_kernel` implementations of RoPE, RMSNorm, the hyper-connection pre/post/mix
triple and sparse attention), `expert_affine.py` (the affine bank path, `mx.quantized_matmul` on slot
views), `engram_mlx`, `indexer_mlx`, `compressor_mlx`, `dspark_draft.py`.

### A.4 Frontends, metrics, benchmarks

`model/api.py` — `V41Model`, the stable boundary (`from_pretrained`, `chat`, `stats`).
`model/generation.py` — sampling, penalties, chat templating, streaming.
`cli.py` — `serve|chat|doctor|bench`. `server/app.py` + `server/engine.py` — OpenAI-compatible HTTP.
`metrics/routing_trace.py` — columnar `(phase, layer, position, experts)` recording that every
offline policy simulation replays.
`benchmarks/` — 179 files. `guarded_run.sh` (memory guardian and preflight), `settle.sh`,
`decode_throughput.py`, `decode_anatomy.py`, `decode_resident.py`, `nll_expert_precision.py`,
`repetition_quality.py`, `code_validity.py`, `simulate_policies.py`, `build_affine_bank.py`,
`quant_affine.py`, `expert_requant_error.py`, `hotlist_coverage.py`.
`tests/` — 16 files, 1,520 lines, 79 tests; `fake_model.py` and `fakes.py` let the store, index and
server tests run without the checkpoint.

---

## B. Decode hot-path analysis

One token, measured shape from `docs/HANDOFF.md` section 6: **182.5 ms at a 36 GiB budget on the
2-bit bank**, of which 93 ms is the all-resident compute floor, 62 ms exposed expert wait and 27 ms
unaccounted fetch-only overhead.

```
generation.stream_tokens
  └─ TextDecodeRuntime.decode_token                        (text_decode_runtime.py:1673)
       └─ _decode_token_impl                               (:2303)
            ├─ engram_hash.push(token_id)                  CPU, n-gram row ids
            ├─ embed_token_decode                          [HC_MULT=4, DIM=5120] bf16
            ├─ layer 0   _decode_layer0        → block_layer0.layer0_block_decode
            ├─ Engram(1) _apply_engram(1)      → engram row read + engram_forward_decode
            ├─ layer 1   _decode_sliding       → block_sliding_window
            ├─ layers 2..39, three variants selected by set membership:
            │     layer_id in SOURCE_LAYERS {2,8,14,20}    → _decode_source
            │     layer_id in INDEX_ONLY {24,28,32,36}     → _decode_index_source
            │     otherwise                                → _decode_reuse
            │   Engram(14) is applied before layer 14.
            │
            │   every variant ends the same way (block_compressed_reuse.py:236-290):
            │     hc_mixes_decode        one fused Metal kernel, Sinkhorn x20
            │     hc_pre_norm_decode     one fused Metal kernel
            │     ── attention ──
            │     hc_post_decode
            │     hc_mixes_decode / hc_pre_norm_decode
            │     └─ moe_layer_forward                     (moe_layer_metal.py:38)
            │          ├─ route_topk_fused(x, gate_w, gate_b, topk=6)
            │          ├─ route_topk_fused(x, gate_{L+1}, topk=PREDICT_TOPK=6)   ◀ prediction
            │          ├─ mx.eval(route.indices, route.weights, predicted)       ◀ THE SYNC POINT
            │          ├─ indices.tolist()  →  expert_index[(layer, id)]  → [ExpertEntry]
            │          ├─ expert_store.get_many(entries, prefetch=predicted)
            │          │     hit      → OrderedDict.move_to_end, return slot
            │          │     inflight → future.result(), promote slot to resident
            │          │     miss     → _acquire_resident_slot_locked (evict LRU)
            │          │                 → load_pool.submit(read_expert_into)
            │          │                    → ExpertReader preadv into slot views
            │          │     then prefetch_decode(next layer's predicted) into transient slots
            │          ├─ per expert: affine_expert_forward(x, slot.arrays, fmt, w)
            │          │     3 x mx.quantized_matmul on views of the slot bytes — no copy
            │          ├─ shared_expert_forward (fp8, resident)
            │          └─ mx.async_eval(output)                                  ◀ overlap
            │     hc_post_decode
            ├─ final_logits_decode   HC collapse, RMSNorm, chunked fp32 head
            └─ mx.eval(hidden, logits)
```

### B.1 Where abstraction would be dangerous

| site | why it is dangerous |
|---|---|
| `moe_layer_metal.py:103` `mx.eval(route.indices, ...)` | The one mandatory CPU/GPU sync per layer. 40 per token. Anything added between the router and this eval lands on the critical path 40 times per token. A 0.1 ms addition is 4 ms/token, 2 % of decode. |
| `moe_layer_metal.py:171` `fmt.kind == "affine"` | Already the model-type branch the brief warns about, executed 40 times per token. It is currently one attribute read plus a string compare. It must not grow into a dispatch table walked per expert. |
| `resident_store.get_many` `with self._lock:` block (`:499-540`) | A single global lock held while resolving every expert of a layer. Measured total store bookkeeping is 1.0 ms/token. Any per-expert descriptor lookup, dict construction or virtual call inside this section multiplies by 240 per token. |
| `expert_affine.affine_expert_forward` | Three `mx.quantized_matmul` calls plus `.view()/.reshape()` per expert, 240 times per token. The views are constructed from `fmt.dtypes`/`fmt.shapes` dict lookups on every call — already slightly wasteful and a place where a richer descriptor would cost real time. Precompute per-format view metadata once. |
| `_decode_token_impl`'s layer loop | Currently branchless in the cheap sense: two set-membership tests per layer. A generic per-layer descriptor consulted here would add Python work 40 times per token. |
| `ExpertSlotPool` view construction | Done once at startup, but the `dict[str, mx.array]` returned by `as_model_dict()` is passed per expert per layer. Keep it a plain dict. |

### B.2 Where abstraction is free

Everything that runs **once per process**: bank detection, index construction, slot sizing, trunk
loading, tokenizer, RoPE precompute, hotlist preload, gate publication. Also everything in
`prepare_prefill_layer`, which runs once per layer per prefill, not per token.

This is the whole design constraint in one line: **resolve the architecture at construction time,
and hand the hot path plain arrays and plain dicts.**

---

## C. DeepSeek coupling inventory

Category **A** = generic today. **B** = generic concept, DeepSeek-specific implementation.
**C** = fundamentally architecture-specific, should be duplicated rather than abstracted.

| # | file:symbol | responsibility | why DeepSeek-specific | Qwen difference | GLM difference | treatment | perf risk |
|---|---|---|---|---|---|---|---|
| 1 | `storage/index.py:9` `_EXPERT_RE` | per-expert FP4 tensor discovery | literal `layers.N.ffn.experts.E.wK` | n/a — no per-expert layout exists | n/a | **B** — keep as the DeepSeek pattern in a per-architecture table | none (load time) |
| 2 | `storage/index.py:14` `_STACKED_RE` | stacked bank discovery | literal `language_model.layers.N.ffn.experts.wK.field` | `language_model.model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` | same `switch_mlp` scheme | **B** — parameterise the regex + a `{proj_name: slot_name}` map | none |
| 3 | `storage/index.py:120` `build_stacked_expert_index` | 3-D slicing into per-expert ranges | only the regex and the `w1/w2/w3` names | identical maths | identical maths | **A** once #2 is parameterised | none |
| 4 | `storage/index.py:182` `detect_expert_bank` | pick FP4 vs stacked | hard-coded config key `omlx_deepseek_v41` | oMLX would emit a different key; MLX-LM conversions use top-level `quantization` | same | **B** — accept a list of known keys, fall back to `quantization` | none |
| 5 | `storage/index.py:151` mixed-quant guard | refuse mixed bits/group | correct and desirable | **Qwen MLX conversions quantize the n-gram/PLE shards at g32 while experts are g64** — the guard only sees expert tensors, so it holds, but verify | mixed 4/8-bit conversions exist publicly and would trip it | **A**, keep; document that mixed-precision conversions are unsupported by design | none |
| 6 | `cache/slots.py:30` `TENSOR_NAMES` | default FP4 six-name tuple | literal | 9 names, different spellings | 9 names | **B** — drop the module constant, always pass `tensor_sizes` | none |
| 7 | `cache/slots.py:65` `{"w1.weight","w2.weight","w3.weight"}` guard | sanity check | literal projection names | `gate/up/down_proj` | same | **B** — check "three distinct weight tensors" instead | none |
| 8 | `cache/resident_store.py:77` `tensor_sizes_from_entry` | slot sizing | same three literals | same | same | **B**, with #7 | none |
| 9 | `cache/resident_store.py:593` `num_layers: int = 40` | prefill quota denominator | default only; callers pass it | 48 (Qwen) | 42 MoE layers of 45 (GLM-F), 75 of 78 (GLM) | **B** — remove the default, force the caller | none |
| 10 | `cache/resident_store.py` everything else | residency, LRU, SLRU, quotas, transients, prefetch, accounting | nothing | nothing | nothing | **A** | n/a |
| 11 | `storage/reader.py` all | positional reads, NOCACHE, mirror, piece fan-out | nothing | nothing | nothing | **A** | n/a |
| 12 | `io/resident_prefetch.py`, `io/prefetch.py` | background loads | nothing | nothing | nothing | **A** | n/a |
| 13 | `metrics/routing_trace.py` | routing trace | `int16` expert ids | 512 experts — fits | 288/256 — fits | **A** | n/a |
| 14 | `model/expert_affine.py` | affine expert math | assumes `w1=gate, w3=up, w2=down` and SwiGLU with `swiglu_limit` | same SwiGLU, no clamp (`swiglu_limit` absent from Qwen config) | GLM-F config sets `swiglu_limit: 10.0`; GLM-5.3 does not | **B** — parameterise names and the clamp; keep it a flat function | low if resolved at load |
| 15 | `model/moe_layer_metal.py:38` `moe_layer_forward` | decode MoE | `topk=6`, `route_scale=1.5`, `norm_topk_prob`, `sqrt(softplus)` scoring, FP4 fused branch, `N_LAYERS=40` | top-10, 512 experts, sigmoid-family scoring, shared-expert **gate** (`shared_expert_gate`) which DeepSeek does not have | top-8, `sigmoid` scoring, `routed_scaling_factor 2.5`, `noaux_tc` with `n_group=1` | **C** — one MoE-forward per architecture family; share `get_many` underneath | high if made generic |
| 16 | `model/router_mlx.py` `route_topk` | routing | `sqrtsoftplus` scoring is DeepSeek's | different | `sigmoid` + `e_score_correction_bias` | **C** | high |
| 17 | `model/router_fused_metal.py` | fused router kernel | same | same | same | **C** — write a second kernel | high |
| 18 | `model/expert_metal.py`, `fp4_*`, `fp8_*` | FP4/FP8 kernels for the shipped checkpoint | dims 5120/2304 baked in | not applicable — no FP4 checkpoint | not applicable | **C**, and only needed for DeepSeek's own checkpoint | none |
| 19 | `model/attention_compressed.py`, `attention_sliding_window.py`, `attention_layer0.py`, `sparse_attn_mlx.py`, `indexer_mlx.py`, `compressor_mlx.py`, `shared_attention.py` | DeepSeek attention family | entirely | Gated DeltaNet linear attention + Qwen Sparse Attention: **new code** | Kimi-Delta linear attention + DeepSeek sparse attention: **partly reusable** | **C** — new modules per model | n/a |
| 20 | `model/hyper_connection_mlx.py`, `hc_sinkhorn_metal.py`, `decode_fused_metal.hc_*` | hyper-connections | `hc_mult=4`, Sinkhorn 20 iters | **Qwen has them**: `hc_count 4`, `hc_lowrank 320`, tensors `attn_hyper_connection.*` | **GLM-5.3-Flash has them**: `hc_mult 4`, `hc_sinkhorn_iters 20`, `mhc: true`, tensors `attn_hc.{base,fn,scale}` | **B, high value** — near-direct reuse for GLM-F, close for Qwen | none; reuse is a win |
| 21 | `model/engram_*.py`, `storage/engram_*.py`, `io/engram_prefetch.py` | 384M-row n-gram table, SSD row lookup | DeepSeek Engram geometry | **Qwen has an analogue**: sharded n-gram PLE embedding, ~51B params, 128 shards, `ple_layer_ids [2]` | none | **B, high value** — the row-lookup machinery is exactly what Qwen's PLE needs | none |
| 22 | `model/dspark_draft.py` | DSpark speculative draft | entirely DeepSeek | Qwen ships an MTP block (`mtp_num_hidden_layers 1`) — different mechanism | GLM ships `num_nextn_predict_layers 1` | **C** | n/a |
| 23 | `model/text_decode_runtime.py:119-186` constants + `_decode_token_impl` | architecture assembly | entirely | entirely | entirely | **C** — a sibling runtime class per architecture | high if generalised |
| 24 | `config.py:88` `MODEL_DIR_NAME`, `:112` `TRUNK_BYTES`, `:113` `ALL_EXPERTS_BYTES`, `:64` `model_id` | discovery and auto-budget | DeepSeek sizes | different | different | **B** — move to per-architecture values, keep the formula | none |
| 25 | `benchmarks/build_affine_bank.py:60` `HIDDEN, INTER, N_EXPERTS, N_LAYERS` + `tensor_name()` + `omlx_deepseek_v41` | bank builder | literals | needs Qwen geometry and names | needs GLM geometry | **B** — parameterise from a source checkpoint's own config | none (offline) |

**Count.** Category A: ~1,600 lines, portable now. Category B: ~250 lines across 6 files, and every
one of them resolves at load time. Category C: ~14,000 lines, and the correct treatment for all of it
is duplication, not abstraction.

---

## D. Target architecture comparison

All figures below are computed from the authoritative `config.json` of each model, fetched from the
upstream repositories on 2026-09-18, using the byte formula validated against this repository's own
measured banks:

```
bytes_per_expert = P * bits/8 + (P / group) * 4        P = 3 * hidden * moe_intermediate
```

Validation: DeepSeek 3-bit g64 → 15,482,880 B and 2-bit g128 → 9,953,280 B, both exactly the values
in `docs/HANDOFF.md` and in `build_affine_bank.py`.

### D.1 Architecture

| | **DeepSeek V4.1 Flash** | **Qwen3.8-Flash-Next** | **GLM-5.3-Flash** | **GLM-5.3** |
|---|---|---|---|---|
| `model_type` | `deepseek_v41` | `qwen4_exp` | `glm5_next` | `glm_moe_dsa` |
| layers | 40 | 48 | 45 | 78 (+1 MTP) |
| hidden | 5,120 | 2,560 | 4,096 | 6,144 |
| attention | DSA: sliding-window + compressed KV with 4 source layers, 8 index-source layers, reuse layers | hybrid **3 x Gated DeltaNet linear + 1 x Qwen Sparse Attention**, `full_attention_interval 4` | hybrid **34 x Kimi-Delta linear + 11 x DeepSeek sparse attention (MLA)** | **78 x MLA + DSA indexer**, `index_topk 2048` |
| KV / state | window 128 + compressed caches, ratio 2 then 1 | 36 recurrent states (O(1) in length) + 12 KV layers, 2 kv heads x 256 | 34 recurrent states + 11 MLA layers, `kv_lora_rank 512` | 78 MLA layers, `kv_lora_rank 512` |
| hyper-connections | `hc_mult 4`, Sinkhorn 20 | `hc_count 4`, `hc_lowrank 320` | `hc_mult 4`, Sinkhorn 20, `mhc` | **absent** |
| n-gram side table | Engram, 2 layers, 384M rows each | PLE n-gram, `ple_layer_ids [2]`, 128 shards, ~51B params | absent | absent |
| MoE layers | 40 | 48 | 42 (`first_k_dense_replace 3`) | 75 |
| routed experts / layer | 384 | 512 | 288 | 256 |
| experts / token | 6 | 10 | 8 | 8 |
| shared experts | 1 | 1 (**plus a `shared_expert_gate`**) | 1 | 1 |
| moe_intermediate | 2,304 | 640 | 2,048 | 2,048 |
| routing | `sqrtsoftplus`, `noaux_tc`, bias-corrected selection, `norm_topk_prob`, scale 1.5 | softmax-family top-10 of 512, gated shared expert | `sigmoid`, `noaux_tc`, `n_group 1`, scale 2.5 | `sigmoid`, `noaux_tc`, `n_group 1`, scale 2.5 |
| speculation | DSpark, block 5, target layers 37-39 | MTP, 1 hybrid layer | 1 next-n layer | 1 next-n layer (layer 78) |
| vision | yes (excluded) | yes (excluded) | yes (excluded) | text-only config |

### D.2 Storage and streaming arithmetic

Per-expert parameters: DeepSeek 35.39 M, Qwen **4.92 M**, GLM-Flash 25.17 M, GLM-5.3 37.75 M.

| bank format | DeepSeek (15,360 experts) | Qwen (24,576) | GLM-Flash (12,096) | GLM-5.3 (19,200) |
|---|---|---|---|---|
| **4-bit g64** MiB/expert | 17.93 (FP4, shipped) | **2.64** | **13.50** | **20.25** |
| bank total | 275.7 GiB | **63.3 GiB** | **159.5 GiB** | **379.6 GiB** |
| **3-bit g64** MiB/expert | 14.77 | 2.05 | 10.50 | 15.75 |
| bank total | 221.5 GiB | **49.2 GiB** | 124.0 GiB | 295.3 GiB |
| **2-bit g128** MiB/expert | 9.49 | 1.32 | 6.75 | 10.13 |
| bank total | 142.4 GiB | **31.6 GiB** | 79.7 GiB | 189.8 GiB |
| expert reads / token (0 % hit) | 240 | **480** | 336 | 600 |
| worst-case bytes/token @ 4-bit | 4.20 GiB | 1.24 GiB | 4.43 GiB | 11.87 GiB |
| worst-case bytes/token @ 3-bit | 3.46 GiB | 0.96 GiB | 3.45 GiB | 9.23 GiB |

**Residency at a 44 GiB expert budget** — the single number that predicts hit rate:

| model / bank | resident share of bank | expected regime |
|---|---|---|
| DeepSeek 3-bit g64 (in use today) | 19.9 % | measured **83.3 %** chat hit rate |
| DeepSeek 2-bit g128 | 30.9 % | measured **90.7 %** |
| **Qwen 4-bit g64** | **69.5 %** | streaming nearly irrelevant |
| **Qwen 3-bit g64** | **89.4 %** | streaming irrelevant |
| **Qwen 2-bit g128** | **139 % — whole bank resident** | no SSD reads after warm-up |
| **GLM-Flash 4-bit g64** | 27.6 % | DeepSeek-class; the real streaming test |
| **GLM-Flash 3-bit g64** | 35.5 % | better than DeepSeek today |
| **GLM-Flash 2-bit g128** | 55.2 % | comfortable |
| **GLM-5.3 4-bit** | 11.6 % | harder than anything measured |
| **GLM-5.3 2-bit g128** | 23.2 % | roughly DeepSeek 3-bit difficulty, but 600 reads/token |

### D.3 What this means for Cachalot

**Qwen3.8-Flash-Next is a residency problem, not a streaming problem.** Its expert bank at 3-bit is
49.2 GiB, smaller than the cache budget this machine already runs. The value Cachalot adds is
(a) hosting the ~32 GB n-gram PLE table on SSD with row-level reads — which is precisely what the
Engram machinery does — and (b) the wired-residency discipline that keeps macOS from compressing the
working set. Decode will be **dispatch-bound**, and the dispatch count is the risk: 480 routed
expert forwards per token against DeepSeek's 240, over 96 sublayers of hyper-connections against 80.
Naively that is 1,440 `quantized_matmul` dispatches per token for routed experts alone. The measured
DeepSeek floor is 93 ms for roughly 400 dispatches. **Qwen must use a gathered/grouped expert kernel
(`mx.gather_qmm`, i.e. MLX's `QuantizedSwitchLinear` shape) rather than a per-expert loop**, which
turns 30 dispatches per layer into 3. The repository has a null result against `gather_qmm`
(`HANDOFF.md` section 11, "a fused multi-expert kernel and `gather_qmm` chunking: both slower") — but
that was measured on **six experts of 35.4 M parameters each**. Qwen is **ten experts of 4.9 M**.
The regimes are not comparable and the measurement must be redone. *Marked uncertain; resolved by
`benchmarks/micro_gather_qmm.py` re-run at Qwen's shapes, which needs no checkpoint.*

**GLM-5.3-Flash is the architecture that actually stresses the existing machinery**, and it stresses
it slightly harder than DeepSeek: 336 expert reads per token against 240, 13.50 MiB against 14.77,
27.6 % residency against 19.9 %. First-order estimate at a 44 GiB budget and 4-bit: ~85 % hit,
~50 misses/token, ~675 MiB/token, ~100 ms of drive time against DeepSeek's measured 73 ms. Its 34
Kimi-Delta linear-attention layers make KV memory almost free, which frees budget for experts — a
second-order win Cachalot can exploit that DeepSeek cannot offer.

**GLM-5.3 (full) is feasible but slow, and needs no new architecture.** At 2-bit g128 the bank is
189.8 GiB — smaller than the DeepSeek 3-bit bank in use today — at 23.2 % residency with 600 reads
per token. The binding constraints are SSD bytes and 75 MoE layers of compute, not any structural
gap. It is also the *simplest* of the three to execute: pure MLA + DSA, no hyper-connections, no
linear attention, no n-gram table. If GLM-5.3-Flash works, GLM-5.3 is a configuration change plus an
attention variant, not a redesign.

### D.4 Memory budget on this machine

`mx.device_info()` on this host: `memory_size` 96.0 GiB, `max_recommended_working_set_size`
**77.76 GiB**. Today's shipped configuration wires 72 GiB (44 GiB experts + ~12 GiB trunk + 2 GiB MLX
cache) and `HANDOFF.md` section 9.4 closes the "larger budget" lever because 80 GiB wired is the
configuration class that kernel-panicked this machine twice.

| | DeepSeek today | Qwen 3-bit (projected) | GLM-Flash 4-bit (projected) |
|---|---|---|---|
| resident trunk | ~12 GiB | ~5–8 GiB *(uncertain)* | ~10–12 GiB *(uncertain)* |
| SSD-resident side table | Engram 189 GiB, streamed | PLE n-gram ~32 GB, streamed | none |
| expert budget | 44 GiB | 44 GiB holds 89 % of the bank; 52 GiB holds all of it | 44 GiB |
| KV / state @ 32k ctx | ~0.4 GiB | 36 recurrent states (~0.1 GiB, length-independent) + 12 KV layers ~0.8 GiB | 34 states + 11 MLA layers, ~0.4 GiB |
| MLX cache | 2 GiB | 2 GiB | 2 GiB |
| **wired total** | **72 GiB** | **~56 GiB** — comfortably inside the recommended set | **~60 GiB** |

Qwen at 3-bit leaves roughly 20 GiB of headroom against today's configuration. That headroom is the
argument for running Qwen at 4-bit g64 (63.3 GiB bank, 4.25 effective bits) rather than 3-bit: better
quality, and 63.3 + 8 + 2 = 73 GiB wired, the same class as today's shipped DeepSeek configuration.
*Do not exceed that without re-reading `HANDOFF.md` section 5 rule 1.*

### D.5 Which artifact to start from — a correction to the brief

The brief names GGUF artifacts (`unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q6_K_XL ~169 GB, etc.).
**Cachalot cannot execute GGUF k-quants and should not try.**

- MLX's quantization modes in the installed 0.32.2 are `affine`, `mxfp4`, `mxfp8`, `nvfp4`
  (verified via `mx.quantize` docstring). GGUF `Q4_K`/`Q6_K`/`IQ4_XS` are super-block formats with
  per-block mins and secondary scales that none of these decode. `HANDOFF.md` section 10 already
  recorded "blocked on GGUF k-quants MLX cannot read" and then retired the premise by **building a
  bank here instead** — that resolution applies unchanged.
- The right sources are the MLX affine conversions that already exist publicly for both targets, in
  **exactly the stacked layout `build_stacked_expert_index()` already reads**:
  - Qwen: `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit` — verified tensor scheme
    `language_model.model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`,
    default `group_size 64, bits 4`, with the 128 n-gram shards at `group_size 32, bits 4`.
  - GLM-5.3-Flash: `pipenetwork/GLM-5.3-Flash-MLX-4bit` — verified same `switch_mlp` scheme, plus
    `attn_hc.{base,fn,scale}` / `ffn_hc.{base,fn,scale}`, which map onto this repository's
    `hc_attn_{fn,scale,base}` naming almost one for one.
- Also available: `Vontra/GLM-5.3-Flash-MLX-oQ2-MTP`, i.e. an **oMLX oQ2 conversion**, the same
  toolchain that produced the oQ3e DeepSeek bank this repository already reads. oMLX does **not**
  yet support `qwen4_exp` (issue `jundot/omlx#3170`, open).
- 4-bit g64 is 2.64 MiB/expert for Qwen and 13.50 MiB for GLM-Flash. A 6-bit bank is the wrong
  trade for Cachalot: it multiplies both bytes and residency pressure for quality this project has
  repeatedly shown it cannot measure at the top of the range. **Start at 4-bit g64 and use
  `build_affine_bank.py` to descend**, exactly as was done for DeepSeek.

Disk: `/` has 125 GiB free, `/Volumes/X10Pro` has 871 GiB free. A Qwen MLX-4bit checkpoint (~100 GB
including the n-gram table) fits either; GLM-5.3-Flash-MLX-4bit (~170 GB) fits only X10Pro.

**Sources:** upstream `config.json` for
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
[zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3); MLX conversions
[pipenetwork/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-4bit),
[pipenetwork/GLM-5.3-Flash-MLX-4bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-4bit);
reference runtimes [PipeNetwork/qwen38-flash-next-mlx](https://github.com/PipeNetwork/qwen38-flash-next-mlx),
[freddyhaddad/glm53-flash-mlx](https://github.com/freddyhaddad/glm53-flash-mlx),
[mlx-lm PR #1788](https://github.com/ml-explore/mlx-lm/pull/1788); local checkpoint
`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/config.json`.

### D.6 Candidates evaluated and closed, 2026-09-18

The field was searched for a model that rivals DeepSeek V4.1 Flash on capability while being a better
size for this machine. **No successor was found.** That is a result worth recording, because it is
the kind of search a later session will otherwise repeat.

**Selection rule used.** Sparse MoE with routed experts; an MLX affine artifact that exists or is
buildable (not a GGUF k-quant, section D.5); and an expert bank in roughly the 80–160 GiB band, so
that a 44 GiB budget buys 30–60 % residency instead of the 19.9 % V4.1 Flash gives today. All bank
figures below use the byte formula of section D.2, computed from each model's own upstream
`config.json`.

| model | total / active | MoE layers x experts | MiB/expert @4-bit g64 | bank | reads/token | resident @44 GiB | verdict |
|---|---|---|---|---|---|---|---|
| **DeepSeek V4.1 Flash** | 552B / ~16B decode | 40 x 384 = 15,360 | 17.93 FP4 · 14.77 @3b | 275.7 · **221.5 GiB** | 240 | 15.9 % · **19.9 %** | **incumbent, unbeaten on capability** |
| Step-3.5-Flash | 196B / 11B | 42 x 288 = 12,096 | **8.44** | **99.6 GiB** | 336 | **44.2 %** | closed — capability |
| Step-3.7-Flash (VL) | 198B / 11B | 42 x 288 = 12,096 | 8.44 | 99.6 GiB | 336 | 44.2 % | closed — capability |
| DeepSeek V4-Flash | 284B / 13B | 43 x 256 = 11,008 | 12.75 FP4 | 137.1 GiB | 258 | 32.1 % | closed — predecessor |
| GLM-5.3-Flash | 320B / 18B | 42 x 288 = 12,096 | 13.50 | 159.5 GiB | 336 | 27.6 % | **kept — streaming test rig** |
| Qwen3.8-Flash-Next | 125B / 6B | 48 x 512 = 24,576 | 2.64 | 63.3 GiB | 480 | 69.5 % | kept — first port, not a successor |
| MiniMax-M3 | ~430B / ~18B | 57 x 128 = 7,296 | 30.38 | 216.4 GiB | 228 | 20.3 % | closed — no size win |
| GLM-5.3 | ~745B | 75 x 256 = 19,200 | 20.25 | 379.6 GiB | 600 | 11.6 % | closed for now — section D.3 |
| Kimi K3 | 2.8T / ~50B | 896 experts, 16 active | — | ~1.6 TB | — | ~3 % | closed — out of range |

**Why each one closed.**

- **Step-3.5-Flash / Step-3.7-Flash** (`model_type: step3p5` / `step3p7`). Structurally the best fit
  found: 96 % of the checkpoint is routed experts, the trunk is ~3 GiB against V4.1's 12 GiB, a
  64 GiB expert budget would wire only ~69 GiB — *less* than the shipped DeepSeek configuration —
  for 64.3 % residency at full 4-bit, and `mlx-community/Step-3.5-Flash-4bit` (111 GB, 22 shards) is
  verified uniform 4-bit affine g64 on `mlp.switch_mlp.*`, with only the router gates at 8 bits.
  It also has **no hyper-connections**, which is 74.2 ms of V4.1's 93 ms compute floor
  (`HANDOFF.md` section 9.2), and mlx-lm 0.30.6 already implements it, so a reference to diff
  against exists. **Closed on capability, not on engineering:** Step 3.7 Flash scores 47.20 % on
  HLE-with-Tools, ahead of DeepSeek V4 Flash and Gemini 3.5 Flash but below V4.1 Flash, which leads
  Terminal Bench 2.1 at 90.6 and DeepSWE at 74.2. Adopting it would repeat the 2-bit-bank trade the
  standing decision of 2026-09-18 rejected, in a larger denomination.
  *For the record, because it was queried and is easy to misread:* Step-3.5-Flash **is** a sparse
  MoE — `use_moe: true`, `moe_num_experts: 288`, `moe_top_k: 8`, `moe_intermediate_size: 1280`,
  `moe_layers_enum: "3..44"`, and stacked `switch_mlp` expert tensors plus `mlp.gate.router_bias` in
  the MLX conversion. Its `intermediate_size: 11264` is the **dense** FFN of layers 0–2 only.
- **DeepSeek V4-Flash** (`model_type: deepseek_v4`). Half the bank of V4.1 at 137.1 GiB and 32.1 %
  residency, and the highest code reuse of any candidate: `hc_mult 4`, `hc_sinkhorn_iters 20`,
  `sqrtsoftplus`, `noaux_tc`, `routed_scaling_factor 1.5`, `swiglu_limit 10.0`, `sliding_window 128`,
  `head_dim 512`, `o_lora_rank 1024`, `o_groups 8`, `index_topk 512`, yarn factor 16 and
  `compress_rope_theta 160000` are all identical to V4.1. Real differences:
  `compress_ratios [0,0,4,128,4,128,...]` introduces a ratio-128 layer class, `num_hash_layers: 3`
  replaces `engram_layer_ids [1,14]`, `index_n_heads 64` against 32, `q_lora_rank 1024` against 1280,
  `rms_norm_eps 1e-6` against 1e-20, one MTP layer instead of DSpark, and no vision tower.
  **Closed because it is V4.1's predecessor** — strictly less capable than what is running.
- **MiniMax-M3** (`minimax_m3_vl`). 57 MoE layers x 128 experts, top-4, `moe_intermediate_size 3072`
  on `hidden_size 6144` gives 56.6 M parameters and 30.38 MiB per expert. Top-4 routing means only
  228 reads per token, the lowest of any candidate, but the bank is 216.4 GiB at 4-bit — no better
  than V4.1's 3-bit bank — and it needs a new block-sparse attention (`sparse_topk_blocks 16`,
  `sparse_block_size 128`). No size win, new kernels: closed.
- **Kimi K3.** 2.8T total, 896 experts with 16 active. ~1.6 TB at 4-bit, ~3 % residency. Out of range
  for this machine by an order of magnitude.
- **Qwen3.8-Flash-Next** stays the first port for the reasons in section L, and **GLM-5.3-Flash**
  stays the acceptance test for the streaming core. Neither is a replacement for V4.1 Flash: Qwen is
  6B active and its bank is small enough not to stream at all, and GLM 5.3 scores 66.9 on DeepSWE
  against V4.1's 74.2 and 30.8-to-54.8 behind on AutomationBench.

**The conclusion this search supports.** V4.1 Flash is simultaneously the most capable open model
available and the least convenient workload on the shortlist — largest bank, worst residency, and a
compute floor that is 80 % hyper-connections, a cost most rivals do not pay at all. Those two facts
are not in tension; they are the reason this runtime exists. Nothing found here changes the ranking
of the levers in `HANDOFF.md` section 9, and nothing found here is a reason to move off V4.1 Flash.

*Benchmark figures above are third-party and are quoted only to rank capability coarsely. They are
not a substitute for this repository's own gates — `nll_expert_precision.py`, `repetition_quality.py`
and `code_validity.py` — which are the only quality evidence any adoption decision here has ever
been made on.*

---

## E. Compatibility matrix

`✓` works unchanged · `◐` works after a load-time change · `▲` needs new code of the same shape ·
`✗` needs genuinely new implementation · `?` unresolved, experiment named in section K.

| subsystem | file | DeepSeek | Qwen3.8-FN | GLM-5.3-Flash | GLM-5.3 |
|---|---|---|---|---|---|
| positional expert reader | `storage/reader.py` | ✓ | ✓ | ✓ | ✓ |
| contiguous-range merging | `storage/index.py:208` | ✓ | ✓ | ✓ | ✓ |
| stacked bank indexing | `storage/index.py:120` | ✓ | ◐ regex + names | ◐ | ◐ |
| bank detection | `storage/index.py:182` | ✓ | ◐ config key | ◐ | ◐ |
| slot pool | `cache/slots.py` | ✓ | ◐ name guard | ◐ | ◐ |
| resident store / LRU / SLRU | `cache/resident_store.py` | ✓ | ✓ | ✓ | ✓ |
| prefill quota planner | `resident_store.py:593` | ✓ | ◐ `num_layers` | ◐ | ◐ |
| decode prefetch + prediction | `resident_store.py:191`, `moe_layer_metal.py:94` | ✓ | ✓ mechanism, ? width | ✓ mechanism, ? width | ✓ mechanism, ? width |
| duplicate-load suppression | `resident_store.py` key locks | ✓ | ✓ | ✓ | ✓ |
| hotlist preload | `resident_store.py:216` | ✓ | ✓ (low value — bank nearly resident) | ✓ | ✓ high value |
| routing trace + offline sims | `metrics/routing_trace.py` | ✓ | ✓ | ✓ | ✓ |
| affine expert math | `model/expert_affine.py` | ✓ | ◐ names, ▲ grouped kernel | ◐ | ◐ |
| FP4/FP8 kernels | `fp4_*`, `fp8_*` | ✓ | n/a | n/a | n/a |
| **Q3g64 / Q2g128 bank format** | `build_affine_bank.py` + `quant_affine.py` | ✓ | ◐ geometry from config | ◐ | ◐ |
| router | `router_mlx.py`, `router_fused_metal.py` | ✓ | ✗ new scoring + shared gate | ✗ sigmoid + correction bias | ✗ same as GLM-F |
| MoE decode assembly | `moe_layer_metal.py` | ✓ | ✗ sibling function | ✗ sibling function | ✗ |
| hyper-connections | `hyper_connection_mlx.py`, `hc_sinkhorn_metal.py` | ✓ | ▲ `hc_lowrank 320` variant | ✓ likely direct (`hc_mult 4`, Sinkhorn 20, `mhc`) | n/a — absent |
| Engram / n-gram side table | `engram_*`, `io/engram_prefetch.py` | ✓ | ▲ PLE, same row-lookup shape | n/a | n/a |
| sliding-window attention | `attention_sliding_window.py` | ✓ | ✗ | ✗ | ✗ |
| compressed / DSA attention | `attention_compressed.py`, `indexer_mlx.py`, `compressor_mlx.py` | ✓ | ✗ (QSA differs) | ▲ **`deepseek_sparse_attention` layer type — largest single reuse opportunity** | ▲ |
| linear attention | — | n/a | ✗ Gated DeltaNet | ✗ Kimi-Delta | n/a |
| KV cache / state | `text_decode_runtime` caches | ✓ | ✗ recurrent state, new shape | ✗ recurrent + MLA | ▲ MLA only |
| prefix cache | `model/prefix_cache.py` | ✓ | ✗ must snapshot recurrent state | ✗ | ✓ likely |
| DSpark / MTP | `dspark_draft.py` | ✓ | ✗ | ✗ | ✗ |
| tokenizer, chat template | `generation.py` | ✓ | ✓ HF `AutoTokenizer` | ✓ | ✓ |
| CLI / HTTP server | `cli.py`, `server/` | ✓ | ◐ model id, discovery | ◐ | ◐ |
| benchmarks | `benchmarks/` | ✓ | ◐ see section J | ◐ | ◐ |
| memory guardian | `guarded_run.sh` | ✓ | ✓ | ✓ | ✓ |

Read down the columns: **everything below the storage/cache block is `✗` for every new model, and
everything above it is `✓` or `◐`.** That boundary is not a design proposal — it is where the code
already is.

---

## F. Proposed architecture

### F.1 Challenge to the boundary in the brief

The brief's diagram puts a "Model Architecture Adapter" under the runtime, with DeepSeek, Qwen and
GLM as peers beneath it. The repository evidence says the adapter should be **thinner and lower**
than that, for two reasons.

1. The only thing the runtime core asks of a model is *how to turn a bank directory into
   `ExpertEntry` objects and slot sizes*. Everything else it needs — routing decisions — arrives as
   a plain list of `ExpertEntry` at call time. There is nothing else for an adapter to adapt.
2. The execution side does not want an adapter at all; it wants a **sibling**. `TextDecodeRuntime`
   is 2,586 lines of DeepSeek topology with no polymorphism in it. A Qwen runtime would be a
   different 2,000 lines with the same *shape*. Making them share a base class would put a
   per-layer indirection into the only loop that matters, for no benefit: they share no block
   implementations.

So the boundary is:

```
                  shared, model-independent (unchanged hot-path code)
   ┌──────────────────────────────────────────────────────────────┐
   │  ExpertReader · ExpertSlotPool · ResidentExpertStore          │
   │  ResidentExpertPrefetcher · RoutingTracer · guarded_run.sh    │
   │  config budgets · benchmarks harness                          │
   └──────────────┬───────────────────────────────────────────────┘
                  │  consumes ONE contract:
                  │      ExpertBank = (ExpertFormat, {(layer,expert): ExpertEntry})
                  │      tensor_sizes: {slot_name: bytes}
   ┌──────────────┴───────────────────────────────────────────────┐
   │  bank adapters — pure load-time functions, ~40 lines each     │
   │  deepseek_fp4 · deepseek_stacked · mlx_switch_mlx             │
   └──────────────┬───────────────────────────────────────────────┘
                  │
   ┌──────────────┴───────────────────────────────────────────────┐
   │  runtimes — siblings, no shared base class                    │
   │  TextDecodeRuntime (DeepSeek)   QwenDecodeRuntime   GlmDecode │
   │  each owns its constants, its blocks, its router, its MoE fn  │
   └───────────────────────────────────────────────────────────────┘
```

The runtime layer *reuses leaf numerics by import*, not by inheritance: a GLM runtime imports
`hc_mixes_decode`, `hc_pre_norm_decode`, `hc_post_decode` and the DSA pieces from the existing
modules and calls them directly. That is how to get reuse without dispatch.

### F.2 Data structures

Only one new type, and it extends one that exists.

```python
# src/cachalot/storage/bank.py  (new, ~120 lines, load time only)

@dataclass(frozen=True)
class BankLayout:
    """How one architecture's routed experts are named and shaped on disk."""
    name: str                          # "deepseek_fp4" | "deepseek_stacked" | "mlx_switch_mlp"
    pattern: re.Pattern                # groups: layer, projection, field
    projections: tuple[str, str, str]  # source names in (gate, up, down) order
    slot_names: dict[str, str]         # source projection -> canonical slot prefix
    field_names: dict[str, str]        # "weight"/"scales"/"biases" spellings
    config_keys: tuple[str, ...]       # config.json keys that identify this layout
```

`ExpertFormat` gains two optional fields, both `None` for existing banks so nothing changes:

```python
    layout: str = "deepseek_stacked"   # which BankLayout produced it
    proj_order: tuple[str, str, str] = ("w1", "w3", "w2")   # gate, up, down
```

No descriptor object is introduced for the *model* (section 11 of the brief). The reason is in
section B.1: the only consumer of such a descriptor in the hot path would be the MoE forward, and
each architecture already has its own MoE forward, which can simply close over its own constants.
A model descriptor would be a data structure with exactly one reader per field, which is a constant
by another name. **Recommend against building it.** The one exception is
`num_layers` / `moe_layer_ids`, which `prepare_prefill_layer` genuinely needs — pass it as an
argument, as the signature already allows.

### F.3 Ownership

| concern | owner |
|---|---|
| byte ranges, reads, slots, residency, eviction, prefetch, accounting, budgets, guardian | shared core, unchanged |
| bank discovery and naming | `BankLayout` table, load time |
| quantization format of a bank | `ExpertFormat`, load time |
| routing semantics, attention, normalisation, positional encoding, side tables, speculation | per-architecture runtime module |
| expert *execution* (which kernel multiplies the slot bytes) | per-architecture, selecting from shared leaf kernels |

### F.4 Initialization flow (unchanged in shape)

```
CACHALOT_MODEL_PATH  →  config.json  →  architecture id
                                          ├─ "deepseek_v41"  → TextDecodeRuntime
                                          ├─ "qwen4_exp"     → QwenDecodeRuntime
                                          └─ "glm5_next"     → GlmDecodeRuntime
CACHALOT_EXPERT_BANK →  config.json  →  BankLayout  →  ExpertFormat + expert index
                                          →  tensor_sizes  →  ExpertSlotPool  →  ResidentExpertStore
```

One dictionary lookup, once per process. No factory, no registry object, no plugin loader.

### F.5 Hot-path implications

**Zero.** Every change in F.2 is read during `__init__` and never again. The one hot-path change
recommended anywhere in this document is an *optimization* that also helps DeepSeek: precompute the
per-projection `(dtype, shape)` views in `ExpertFormat` once rather than doing three dict lookups
per projection per expert per layer in `expert_affine._qmm` (240 experts x 3 projections x 3 lookups
= 2,160 dict lookups per token today). That is a candidate for the existing dispatch-reduction lever
(`HANDOFF.md` section 9.2), not part of multi-model work.

### F.6 Tensor storage classes (section 12 of the brief)

The repository already has four classes, implicitly, and they are correctly separated:

| class today | mechanism |
|---|---|
| always resident | `ResidentTrunk` + `ResidentLayer` views |
| cached, budgeted | `ResidentExpertStore` slots |
| transiently streamed | transient slots, released after eval |
| row-lookup from SSD | `EngramRowReader` + `EngramPrefetcher` |

Making these an explicit enum buys nothing today, because the trunk/expert split is decided by one
predicate (`resident_trunk.should_load_text_trunk_tensor`) and the Engram split by another. **The
one case that justifies a third class is Qwen's PLE n-gram table**, which is neither trunk nor
expert: ~32 GB, one layer, row-addressed. That is served by pointing it at the existing Engram
reader, not by inventing a taxonomy. Recommend deferring the enum until a model appears that needs a
fifth behaviour.

### F.7 Adaptive precision (section 13 of the brief)

The architecture already supports it and nothing needs to change to keep the option open. Evidence:
`ExpertFormat` carries `(bits, group_size)` per bank; `ExpertSlotPool` is sized from a
`{name: bytes}` map with no assumption of uniformity across pools; `mx.quantized_matmul` takes bits
and group as arguments. What blocks it today is deliberate: `build_stacked_expert_index` raises on
mixed quantization (`index.py:151`) because a single slot pool has a single slot size. A
tiered-precision design would need **two slot pools** (one per precision), not a new abstraction —
`ResidentExpertStore` already accepts an injected `slot_pool`. Note also the measured result that
argues against it: `expert_requant_error.py` found error spread evenly across layers (5 %) and
across projections (0.276/0.309/0.276), so a static mixed bank is not indicated
(`HANDOFF.md` section 8.3). A *dynamic* hot/cold split is a different question and remains open.
**Do not build it now; the architecture will not have to change to build it later.**

---

## G. Concrete file-level change plan

Nothing in this section should be executed before review. Ordering is section H.

### G.1 Modified files

| # | path | symbol | change | reason | depends on | perf risk | tests |
|---|---|---|---|---|---|---|---|
| 1 | `src/cachalot/storage/index.py` | `_STACKED_RE`, `build_stacked_expert_index` | take a `BankLayout` argument; default to the DeepSeek layout so every existing call is byte-identical | Qwen/GLM MLX banks use `mlp.switch_mlp.*` | new `storage/bank.py` | none — load time | `tests/test_storage_index.py`: add a fake `switch_mlp` bank; assert existing DeepSeek assertions unchanged |
| 2 | `src/cachalot/storage/index.py` | `detect_expert_bank` | try each `BankLayout.config_keys` in order, then MLX-LM's top-level `quantization`, then FP4 | MLX conversions do not write `omlx_deepseek_v41` | #1 | none | same file; assert DeepSeek detection still returns `FP4_FORMAT` / stacked identically |
| 3 | `src/cachalot/storage/index.py` | `ExpertFormat` | add `layout: str` and `proj_order` with DeepSeek defaults | lets the expert math know which slot is gate/up/down | — | none | `tests/test_expert_bank.py` |
| 4 | `src/cachalot/cache/slots.py` | `TENSOR_NAMES`, `ExpertSlotPool.__init__` guard | delete the module constant; replace the `w1/w2/w3.weight` check with "exactly three distinct weight tensors, all non-zero" | the names are architecture-specific; the *invariant* is not | — | none | `tests/test_resident_store.py` |
| 5 | `src/cachalot/cache/resident_store.py` | `tensor_sizes_from_entry` | same guard change as #4 | same | #4 | none | `tests/test_resident_store.py` |
| 6 | `src/cachalot/cache/resident_store.py` | `prepare_prefill_layer(num_layers=40)` | remove the default; update the two call sites in `text_decode_runtime.py` to pass `N_LAYERS` | a 40-layer default silently mis-plans quotas for a 45- or 48-layer model | — | none | existing prefill tests |
| 7 | `src/cachalot/model/expert_affine.py` | `affine_views`, `_qmm`, `affine_expert_forward` | read projection names from `fmt.proj_order`; make `swiglu_limit <= 0` skip the clamp (already true) | Qwen/GLM name projections differently; GLM-5.3 has no clamp | #3 | **watch** — see B.1; precompute view metadata rather than adding lookups | `tests/test_expert_bank.py` numerics parity on the DeepSeek bank, bit-exact |
| 8 | `src/cachalot/config.py` | `MODEL_DIR_NAME`, `TRUNK_BYTES`, `ALL_EXPERTS_BYTES`, `model_id` | move to a small per-architecture table keyed by `model_type`; keep the budget *formula* identical | auto-budget currently assumes DeepSeek's 12 GiB trunk and 275.7 GiB expert set | — | none | `tests/test_config.py` |
| 9 | `benchmarks/build_affine_bank.py` | `HIDDEN, INTER, N_EXPERTS, N_LAYERS`, `tensor_name()`, `write_config()` | read geometry from the *source* checkpoint's `config.json`; emit the source's own layout key | to build a 3-bit/2-bit Qwen or GLM bank at all | #1, #2 | none (offline) | `tests/test_bank_writer.py` extended with a fake Qwen-shaped source |
| 10 | `benchmarks/_common.py` | `MODEL_PATH` | allow a second model root via env | benchmarks must target two checkpoints | — | none | — |
| 11 | `src/cachalot/model/api.py` | `V41Model.from_pretrained` | dispatch on `config.json["model_type"]` to the right runtime class; keep `V41Model` as the DeepSeek name and add a neutral alias | one entry point for CLI and server | new runtime module | none | `tests/test_server.py` |

**Explicitly NOT changed:** `storage/reader.py`, the whole of `ResidentExpertStore` except items 5
and 6, `io/*`, `moe_layer_metal.py`, every `block_*.py`, every `attention_*.py`, every `fp4_*`/`fp8_*`
kernel, `dspark_draft.py`, `decode_fused_metal.py`. If a patch touches any of these while doing
multi-model work, that patch is wrong.

### G.2 New files

| path | contents | size |
|---|---|---|
| `src/cachalot/storage/bank.py` | `BankLayout` and the three known layouts | ~120 |
| `src/cachalot/model/qwen/__init__.py` | — | — |
| `src/cachalot/model/qwen/runtime.py` | `QwenDecodeRuntime`: constants, init, prefill, decode loop | ~900 |
| `src/cachalot/model/qwen/router.py` | top-10 of 512 routing + shared-expert gate | ~120 |
| `src/cachalot/model/qwen/moe.py` | `qwen_moe_layer_forward`, grouped `gather_qmm` expert path, prediction hook | ~200 |
| `src/cachalot/model/qwen/linear_attn.py` | Gated DeltaNet decode + prefill, recurrent state | ~400 |
| `src/cachalot/model/qwen/sparse_attn.py` | Qwen Sparse Attention + indexer | ~300 |
| `src/cachalot/model/qwen/ple.py` | n-gram PLE embedding over the Engram row reader | ~200 |
| `benchmarks/model_report.py` | the model-agnostic metric collector of section 14 | ~250 |
| `tests/test_bank_layouts.py` | fake `switch_mlp` bank; slicing parity against the DeepSeek path | ~120 |
| `tests/test_qwen_parity.py` | layer-level parity against a reference implementation | ~150 |

Per-architecture runtimes stay in their own package so that `grep -r deepseek src/cachalot/model/qwen`
returns nothing, which is the cheapest possible coupling test.

---

## H. Migration sequence

Every stage leaves `main` green, leaves DeepSeek inference byte-identical, and is independently
revertible. No long-lived branch.

**Stage 0 — freeze the baseline (half a day, no code).**
Record the three configurations with the exact methodology in section J. Store under
`benchmarks/results/baseline-2026-09-18/`. Nothing else starts until this exists.

**Stage 1 — parameterise the bank layout (1 day).** Changes G.1 #1–#3 plus `storage/bank.py` and
`tests/test_bank_layouts.py`. Acceptance: the full test suite passes and
`detect_expert_bank()` on `~/DeepSeek-V4.1-Flash-q3g64` returns an `ExpertFormat` equal field by
field to the one it returns today.

**Stage 2 — de-name the slot layer (half a day).** Changes #4–#6. Acceptance: `test_resident_store.py`
passes unchanged; a 64-token decode run reproduces the Stage 0 numbers within the 7 % spread.

**Stage 3 — generalise the bank builder and the budget table (1 day).** Changes #8–#10. Acceptance:
rebuilding two shards of the DeepSeek 3-bit bank produces byte-identical output to the shipped bank
(`build_affine_bank.py --verify`).

*Stages 1–3 are the entire "preparatory refactoring". They are ~250 lines and touch no hot-path code.
After Stage 3, DeepSeek optimization can resume without interference, and section 9's levers are
unaffected.*

**Stage 4 — Qwen loader, no inference (2 days).** Download `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit`
to X10Pro. Add the `mlx_switch_mlp` layout. Prove: the expert index has 48 x 512 entries; one expert
read lands 2,764,800 bytes in a slot; `mx.dequantize` of those bytes matches `mx.load` of the same
expert from the checkpoint, bit for bit. **No forward pass.** This alone answers most of the open
storage questions.

**Stage 5 — Qwen execution, correctness only (1 week).** Port Gated DeltaNet, QSA, PLE, the router
and the MoE assembly against the reference MLX implementation, layer by layer, asserting
layer-output parity. Run with the expert store configured so that everything is resident. Judge by
teacher-forced NLL and top-1 against the reference, not by comparing greedy text
(`HANDOFF.md` section 5 rule 8).

**Stage 6 — Qwen under the cache machinery (2 days).** Force streaming with a deliberately small
budget (`--expert-budget-gib 8`, ~13 % residency) so the SSD path is genuinely exercised, then sweep
up to 44 GiB. Record the section-14 metric set. This is the stage that actually tests the shared
core on a second architecture.

**Stage 7 — Qwen optimization (1 week).** Grouped-expert kernel decision, prediction width
re-measured at Qwen's expert size, hotlist value re-measured (likely near zero), 3-bit and 2-bit
Qwen banks screened with `expert_requant_error.py` and gated with `nll_expert_precision.py`.

**Stage 8 — GLM-5.3-Flash (2–3 weeks).** The genuinely valuable target for the streaming core, and
the one that can reuse hyper-connections and DSA. Same stage discipline.

**Stage 9 — GLM-5.3 feasibility (2 days, no implementation).** Replay a GLM-Flash routing trace
through `simulate_policies.py` with GLM-5.3's expert size and count to predict hit rate and bytes per
token before committing to anything.

---

## I. Qwen proof-of-concept plan

**The claim to prove:** Cachalot's existing reader, slot pool, resident store, prefetcher and
accounting serve a second architecture's routed experts correctly and at the drive's rated speed.

**Minimum that must work**
1. `detect_expert_bank()` returns `ExpertFormat(kind="affine", bits=4, group_size=64, ...)` and an
   index of 24,576 `ExpertEntry` objects from the Qwen MLX checkpoint.
2. `ExpertSlotPool` allocates from Qwen's `tensor_sizes` (2,764,800 B/expert) with no code change.
3. `read_expert_into()` fills a slot and `mx.dequantize` of the slot bytes equals the reference
   expert weights bit for bit for 12 random experts. *This is the whole correctness case for the
   storage layer, and it needs no forward pass.*
4. One MoE layer executes: router → `get_many` → grouped expert kernel → output matches the
   reference layer within bf16 tolerance.
5. A 128-token prompt and 32 decoded tokens produce text, with `stats()` reporting hit rate, bytes
   read and resident count.
6. `benchmarks/model_report.py` emits the section-14 table for Qwen and for DeepSeek from the same
   code path.

**Temporarily unsupported and explicitly fine**
vision tower; MTP/speculation; the prefix cache (recurrent state snapshotting is deferred — the
first sessions run `reset=True` per turn); typing-time prefill; long contexts beyond 8k; the
frequency-penalty tuning; the HTTP server. None of these are on the claim being proved.

**Honest framing of what it does and does not prove.** With the whole Qwen bank nearly resident at a
normal budget, steps 5–6 at 44 GiB will show a ~95 %+ hit rate and near-zero SSD traffic. That is a
correct result, not a failure, but it does **not** exercise eviction, the prefetcher or the quota
planner. Step 6 must therefore be run at `--expert-budget-gib 8` as well, and that arm is the one
quoted as evidence.

---

## J. Benchmark and regression plan

### J.1 The baseline to freeze, and exactly how

The numbers in the brief (3.0 / 4.7 / 7.6 tok/s) are close to but not identical with the ones
`docs/HANDOFF.md` records, because they were taken under different budgets and prompts. Record the
methodology, not the constant. Every command below goes through the memory guardian, one runtime at
a time, per `HANDOFF.md` section 5.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 2400 --tag base-q3g64 -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q3g64 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_throughput.py --prompt-tokens 512 --decode-tokens 64
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 3600 --tag base-anat -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q3g64 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 1800 --tag base-floor -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q3g64 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_resident.py --prompt-tokens 16 --decode-tokens 16 --passes 4
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 5400 --tag base-nll -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q3g64 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/nll_expert_precision.py --experts runtime --tokens 512
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q tests
```

Repeat the first command for `DeepSeek-V4.1-Flash-q2g128` and for the FP4 path
(`CACHALOT_EXPERT_BANK` unset, model on X10Pro) so all three rows of the brief's table are recorded
under one methodology. **Four runs per configuration, interleaved, with `benchmarks/settle.sh`
between them** — `HANDOFF.md` section 5 rule 3: run-to-run spread reaches 7 %.

Record for each: median and full range of ms/token; hit rate; misses/token; MiB read/token; cold
prefill seconds; `ready in` seconds; peak wired; the full `decode_anatomy` blocked-time split; the
NLL mean, paired median, sign test and top-1; git SHA; bank path and `MiB/expert` from the runtime's
own startup line.

### J.2 Regression gate for every stage

| dimension | gate |
|---|---|
| correctness | 79 tests pass; a fixed prompt decoded greedily at a fixed seed is **token-identical** to the frozen baseline |
| decode throughput | median ms/token within **2 %** of baseline, judged on 4 runs/side interleaved |
| prefill | cold 512-token prefill within 3 % |
| SSD traffic | MiB read/token within 1 % (this is deterministic given the same routing) |
| cache behaviour | hit rate, predicted/used, wasted bytes each within 1 point |
| memory | peak wired within 1 GiB; no new allocation before the first token |

**Threshold.** Zero measurable regression on the DeepSeek hot path for stages 1–3 — they are
load-time-only changes and anything else indicates a mistake, not a trade-off. From stage 4 onward,
DeepSeek code is not modified at all, so its gate is simply that the numbers do not move.

Stages 1–3 are also cheap to gate structurally: each is a pure-Python change with no forward pass in
it, so `pytest` plus one decode run per stage is sufficient, and the expensive NLL gate is not
required (no numerics change). *The single exception is G.1 #7, `expert_affine.py`, which does touch
numerics; that one needs the production NLL arm and must come out bit-identical.*

### J.3 The model-agnostic report (section 14 of the brief)

`benchmarks/model_report.py` collects one row per configuration, from sources that already exist —
`V41Model.stats()`, `ResidentStoreStats`, `decode_anatomy`'s blocked-time split, `mx.get_peak_memory`
and the guardian's CSV — so nothing new has to be instrumented:

```
model · bank · bits/group · budget GiB
load: ready_s · ttft_ms · prefill_tok_s · decode_tok_s · p50/p95/p99_ms
io:   ssd_MiB_per_token · reads_per_token · mean_read_KiB · GB_s_while_busy · io_wait_ms_per_token
cache: hit_% · misses_per_token · evictions · predicted · predicted_used · wasted_MiB · precision_%
mem:  resident_expert_GiB · trunk_GiB · kv_GiB · mlx_peak_GiB · peak_wired_GiB
comp: all_resident_floor_ms · attention_ms · moe_ms · dequant_ms · sync_ms   (where measurable)
```

Two metrics are missing from the repository today and are worth adding once: **evictions per token**
(one counter in `_drop_locked`) and **per-token latency percentiles** (a list in the decode loop).
Both are outside the hot path's critical section.

---

## K. Risk register

| # | risk | prob | impact | detection | mitigation |
|---|---|---|---|---|---|
| 1 | **Grouped expert kernel is slower than the per-expert loop at Qwen's shapes**, leaving 1,440 dispatches/token | med | high — Qwen decode could be dispatch-bound below 4 tok/s | `benchmarks/micro_gather_qmm.py` re-run at 10 x [640,2560] before any Qwen work; needs no checkpoint | decide before stage 5; if it loses, Qwen is still correct, just slower, and the dispatch lever (`HANDOFF.md` 9.2) applies to both models |
| 2 | **Hidden DeepSeek assumption in the store found only at runtime** (e.g. a place assuming 40 layers) | med | med | grep for `40`, `384`, `5120`, `2304`, `topk=6` in `cache/` and `storage/` before stage 4 — currently `num_layers=40` is the only hit in `cache/` | stage 6 at a small budget exercises eviction and the quota planner, where such an assumption would surface |
| 3 | **MLX conversions use mixed precision** (`pipenetwork/*-mixed-4_8bit` exists) and trip the uniform-quantization guard | high for those repos | low | `detect_expert_bank` raises with a clear message today | use the uniform 4-bit conversion; document mixed banks as unsupported |
| 4 | **Qwen PLE n-gram table does not fit the Engram row reader's geometry** (128 shards, g32, per-shard scales) | med | med — 32 GB would have to be resident instead | inspect the shard tensors at stage 4, before writing any PLE code | fall back to mmap; the table is 32 GB, not DeepSeek's 189 GiB, so a fallback is survivable |
| 5 | **Recurrent state breaks the prefix cache**, which snapshots arrays by shape | high | low — feature, not correctness | `test_prefix_cache.py` on the Qwen runtime | ship stage 5 with the prefix cache disabled; add state snapshotting in stage 7 |
| 6 | **MLX compilation / kernel cache churn** from a second architecture's kernels inflating startup or evicting the first model's | low | low | compare `ready in` across stages | one model per process, which the memory guardian already enforces |
| 7 | **Memory pressure / kernel panic** while two checkpoints are on disk and a second runtime is tried | low, but catastrophic | catastrophic | `guarded_run.sh` preflight | never bypass the guardian; never two runtimes; Qwen at 4-bit wires ~73 GiB, the same class as today, not more |
| 8 | **Abstraction creep**: the "adapter" grows into a per-token dispatch layer | med | high | code review rule: any diff touching `moe_layer_metal.py`, `block_*.py` or `resident_store.get_many` during multi-model work is rejected by default | the sibling-runtime design in F.1 removes the temptation |
| 9 | **Prediction width tuned for DeepSeek is wrong for Qwen/GLM** and wastes drive bandwidth | high | med | `predicted`, `predicted_used`, `wasted_MiB` are already reported | `PREDICT_TOPK` is an env var; re-run the width sweep per model, per `HANDOFF.md` section 8.2 |
| 10 | **Cache pollution between architectures** | n/a | n/a | — | not a risk: one model per process; the store is per-runtime |
| 11 | **Qwen quality at 3-bit/2-bit is much worse than DeepSeek's** because its experts are 7x smaller and have less redundancy per group | med | med | `expert_requant_error.py` at Qwen shapes, then the production NLL arm | keep 4-bit g64 as Qwen's default; the bank fits anyway |
| 12 | **Python overhead per layer rises** because Qwen has 48 layers and 10 experts, i.e. 1.6x the per-token Python of DeepSeek at 1/7 the arithmetic per expert | high | med | `profile_decode_components.py` at stage 6 | grouped expert kernel (risk 1) removes most of it: one `get_many` call and one dispatch triple per layer |
| 13 | **GGUF pursued by mistake**, burning days on a format MLX cannot execute | med, because the brief names GGUF artifacts | high | — | section D.5; start from the MLX affine conversions |
| 14 | **Losing I/O/compute overlap** in a new MoE forward by omitting `mx.async_eval(output)` | med | med — DeepSeek measured the GPU idling without it | `decode_anatomy`'s "drive busy" fraction | copy the pattern from `moe_layer_metal.py:225` deliberately, and assert it in review |
| 15 | **Disk exhaustion** — the internal SSD has 125 GiB free and X10Pro 871 GiB | med | med | `df -h` in the stage-4 preflight | all new checkpoints and banks go on X10Pro; the DeepSeek 3-bit bank stays on the internal drive where its speed matters |

---

## L. Final recommendation

**Option 2 — limited preparatory refactoring while DeepSeek optimization continues** — with a
sharper definition of "limited" than the brief assumes, and one correction to the target ordering.

**Why not option 1 (generalize now).** The repository does not contain an abstraction problem. It
contains ~1,600 lines of already-generic streaming machinery wearing DeepSeek names, and ~14,000
lines of DeepSeek execution that must not be generalised at all. Generalising now would mean
building a model-descriptor layer whose only consumers are constants, at the cost of touching the
files that produced the measured performance.

**Why not option 3 (finish DeepSeek first).** The preparatory work is ~250 lines across six files,
none of them in the hot path, and each independently gated by the existing test suite. Deferring it
has a real cost: `HANDOFF.md` section 9 has weeks of DeepSeek work queued (prefetch precision,
dispatch fusion, speculation), and every new expert-path patch written in the meantime will add
another `w1/w2/w3` literal or another `40` to unpick later. Doing stages 1–3 now is cheaper than
doing them in three months.

**The correction.** The brief's priority order — Qwen first, GLM-Flash second — is right for
*execution risk*, and the arithmetic in section D supports it: Qwen's expert bank is 63.3 GiB at
4-bit against GLM-Flash's 159.5 GiB, and it needs no MLA or DSA work. But it is wrong for *what gets
proved*. Qwen's bank is nearly resident at any sane budget, so it will not exercise eviction, the
quota planner or the prefetcher. **GLM-5.3-Flash is the model that validates the streaming core**,
and it also reuses more of this repository than Qwen does — hyper-connections nearly verbatim
(`hc_mult 4`, Sinkhorn 20, `mhc`, and `attn_hc.{base,fn,scale}` naming that maps onto
`hc_attn_{fn,scale,base}`) and a `deepseek_sparse_attention` layer type on 11 of its 45 layers.
Keep Qwen first because it is lower risk and genuinely useful on this hardware, but **write the
stage-6 evidence at a deliberately small budget**, and plan GLM-5.3-Flash as the acceptance test for
the architecture rather than as a follow-on.

**GLM-5.3 (full) needs no further redesign.** At 2-bit g128 its bank is 189.8 GiB — smaller than the
DeepSeek 3-bit bank running today — at 23.2 % residency, with 600 expert reads per token. It is a
bytes-and-layers problem, which is the problem this runtime is built for, and its architecture is
the *simplest* of the three: MLA plus a DSA indexer, no hyper-connections, no linear attention, no
n-gram table. The answer to the brief's question is yes, provided stages 1–3 land as described.

**No successor to DeepSeek V4.1 Flash exists today.** Section D.6 records the search: eight
candidates were costed from their own configurations, and the two that beat V4.1 Flash on size —
Step-3.5-Flash at a 99.6 GiB bank and DeepSeek V4-Flash at 137.1 GiB — both lose to it on
capability. Adopting either would repeat the trade the standing decision of 2026-09-18 rejected.
Multi-model work is therefore about portability and about not accreting new coupling, and it is
explicitly **not** about replacing the model this runtime was built for.

### The first engineering milestone

**Stage 0 followed by Stage 1: freeze the three-configuration baseline under one recorded
methodology, then land `src/cachalot/storage/bank.py` and the `BankLayout` parameterisation of
`build_stacked_expert_index()` / `detect_expert_bank()`, with `tests/test_bank_layouts.py` proving
that a fake `mlp.switch_mlp.*` bank slices to the same byte ranges the DeepSeek path produces, and
that DeepSeek bank detection is unchanged field for field.**

It is roughly 150 lines plus tests, it cannot move a single number on the DeepSeek hot path because
nothing it touches runs after `__init__`, it is the prerequisite for every later stage, and it makes
the storage layer's model-independence a property the test suite asserts rather than a property this
document claims.

**Do not start stage 4 (downloading a second checkpoint) until stages 0–3 are merged and the
baseline table is in the repository.**

---

## Open questions, and the experiment that settles each

| # | question | experiment | cost |
|---|---|---|---|
| 1 | Is `mx.gather_qmm` faster than a 10-expert loop at [640, 2560]? | extend `benchmarks/micro_gather_qmm.py`; synthetic weights, no checkpoint | 1 hour |
| 2 | Exact resident (non-expert) footprint of Qwen and GLM-Flash | sum the non-`switch_mlp`, non-`ngram` tensors in each MLX checkpoint's index | 30 min, after download |
| 3 | Does the Qwen PLE table's shard geometry fit `EngramRowReader`? | read the shard headers at stage 4 | 1 hour |
| 4 | Does the Qwen MLX conversion keep experts uniformly g64/4-bit? | `detect_expert_bank` at stage 4 — it raises if not | free |
| 5 | Qwen prediction-width saddle at 2.64 MiB/expert | the width sweep of `HANDOFF.md` 8.2, at a forced-small budget | 1 day, stage 7 |
| 6 | GLM-Flash hit rate at a 44 GiB budget | record a GLM routing trace, replay through `simulate_policies.py` | 1 day, stage 8 |
| 7 | Whether `expert_affine`'s per-call dict lookups are measurable | micro-benchmark against a precomputed-view variant | 2 hours — also a DeepSeek win |
| 8 | Whether the brief's 3.0 / 4.7 / 7.6 tok/s match this repository's methodology | stage 0 | half a day |
