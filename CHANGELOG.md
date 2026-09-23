# Changelog

## 0.12.0 (2026-09-23)

**Agent turns reuse the model's own reply, images in history are not re-encoded, and a long Hermes
session was run end to end.** HANDOFF section 15.5.

### Added
- **Reply splice** (`Engine._splice_own_replies`): after every reply the prefix cache holds a snapshot of
  prompt + reply, but Hermes sends each reply back re-serialized. Tool-call arguments come back with their
  keys in a different order than the model wrote them (`{"content":…,"path":…}` for a `path`-first call),
  and an empty thinking block comes back as `reasoning_content: " "`. Either way the re-rendered history
  diverges a few tokens into the reply and the whole reply is prefilled again. When the client's copy of a
  reply parses to the same message (same content and reasoning up to surrounding whitespace, same tool
  names, same argument values), the server now puts the model's own tokens back in its place, so the
  snapshot matches. Anything that does not parse to the same message is left as sent. On the replay of a
  10-turn Hermes session it removed 499 of 4,453 warm prefill tokens (11 %), up to 111 on one turn; live,
  every turn after a tool call reused prompt + reply (`spliced=N` on the `[request]` line). The saving grows
  with reply length: a `write_file` whose body is re-serialized re-prefilled all of it before.
- **Image span rows cached by content digest** (`VisionEncoder`): an agent resends every image of its history
  on every turn, and the ViT and aligner ran again each time (0.14-1.16 s per image). The rows are a pure
  function of the image bytes and are now kept (16 images). `/v1/stats` reports `vision_rows_reused`.
- **`miss/tok` and `hit` on the `[request]` line**: expert-cache misses per decoded token and the hit rate,
  which is what decode speed follows. They separate "this content routes to cold experts" from "the machine is
  slow right now" (section 15.5 has a case of the second).
- **`benchmarks/decode_vs_context.py`** (+ `.sh`): decode speed against context length with the task held
  fixed, one arm per process. 54 vs 16,054 tokens of context: 6.23/8.45 vs 7.38/7.45 tok/s at 23.7 vs 29.1
  misses/token. Context length does not slow decode.
- **`benchmarks/reply_splice_replay.py`**: replays a request dump offline (tokenizer and encoding only) and
  counts prefill tokens with and without the splice; its baseline reproduces the live `reused` numbers.
- **`docs/manual-tests/hermes-desktop.md`**: a step-by-step Hermes Agent Desktop test with what to expect
  at each step and what to report.
- `CACHALOT_SERVER_DUMP` also records each reply's token ids, so a prefix-cache miss can be traced to the
  exact token where a client's history left the model's output.

### Fixed
- **An agent's system-block snapshot was evicted by one long session.** The prefix cache keeps 16 snapshots
  in LRU order and every request adds two or more, so after ~8 requests the system block was gone from memory
  (it stayed on disk, which is read only at startup). The next new session paid the whole cold prefill
  again (185.9 s, measured). Boundary snapshots are now evicted only after every per-turn snapshot
  (`PrefixCache.max_pinned`, 8), and those loaded from disk at startup are pinned too. Verified live: a new
  session after a 15-request session with a compression reused 13,702 of 13,711 tokens, 1.07 s.

### Changed
- Disk snapshots: the eight newest are kept instead of four. Hermes writes its working directory into the
  system prompt (token 3,924 of 13,698), so each project has its own system block and its own snapshot.

### Measured, no change needed
- Hermes, 10-turn session through the CLI, context 13.7k → 26.3k tokens: every answer correct, `reused` never
  fell back to 0 after the first request, swap flat. Two images in one message, and follow-ups that resend
  them: correct answers, 511/532 and 548/564 tokens reused.
- Hermes (updated today) now sends `reasoning_effort: "medium"`, so its sessions run in thinking mode.
- Hermes compresses at ≥ 75 % of a window under 512k tokens (85 % when its 64k floor binds), ~49-56k on
  Cachalot's 65,536, whatever `compression.threshold` says; `compression.threshold_tokens` is the knob
  that lowers it. Triggered at 18k, twice: the summary request (2,145-token prompt) decoded 1,595 and 1,774
  tokens, 249 s and 343 s, the second past Hermes's 300 s budget (the turn still completed). Compression also
  adds a `skill_manage` tool mid-list, which changes the system block, so the next request prefills it cold
  once (254 s).

## 0.11.0 (2026-09-23)

**An agent's system prompt is reused whole, and survives a server restart.** Both levers of v38's Job 2,
measured live with the stock Hermes CLI against `./serve.sh`. HANDOFF section 15.4.

### Added
- **Snapshot where the system prompt ends** (`Engine.system_prefix_len`, `prefill_chunks(cuts=...)`): the
  server renders the leading system message (tools and reasoning-effort header included) on its own, and
  when its tokens are a prefix of the prompt's, prefill ends a call there and snapshots. A new Hermes
  session's first request now reuses 13,456 of 13,468 tokens and prefills in **1.09 s instead of 18.9 s**;
  the whole session took 15 s against 46 s. Checked across chat/thinking modes, reasoning efforts, with and
  without tools: the rendered system block is a token prefix in every case.
- **Prefix snapshots on disk** (`src/cachalot/model/snapshot_store.py`, `cachalot serve --snapshot-dir`,
  `CACHALOT_SNAPSHOT_DIR`; `serve.sh` sets `~/.cache/cachalot/prefix-snapshots`): boundary snapshots are
  written as safetensors (46 MB for Hermes's 13.5k-token system block, 10 ms) and loaded at startup (20 ms).
  Each file carries an identity of the runtime version, `max_seq_len`, the checkpoint's config and index,
  and the expert bank's files (size and mtime); a mismatched file is ignored. The four newest are kept. The
  first request after a restart prefilled in **3.31 s instead of 163 s**. Restore from disk is
  bit-identical to restore from memory on the real bank at 3,000 and 9,000 tokens
  (`benchmarks/prefix_snapshot_exactness.py`, new disk arm).
- `tests/test_system_boundary_snapshots.py`, 10 tests. 300 pass.

## 0.10.0 (2026-09-23)

**Hermes Agent drives Cachalot, images work end to end, and long prompts no longer run out of memory.** Job 1
was run from the session with the stock Hermes CLI and an isolated `HERMES_HOME`. Hamed's own Hermes config
was not touched. Parallel tool calls, a file write, a resumed session and an image all come back correct.
Four server breakages found on the way are fixed. Vision piece 4 wires image content through the server.
HANDOFF sections 15.1-15.3 and 16.4.

### Added
- **Image input through the server** (`src/cachalot/model/vision_prompt.py`): OpenAI `image_url` content
  parts (URL, path or base64 data URI) go through the official encoding, the MLX preprocessing, the ViT +
  aligner, and the piece 3 splice. The tower loads lazily on the first image (~1 s, 0.9 GiB). Checked on a
  real chart: every value read correctly.
- **Chunked prefill** (`generation.prefill_chunks`, `CACHALOT_PREFILL_CHUNK`, default 4096): long prompts
  are prefilled in chunks that never split an image span, with `cancel` checked between chunks. Quality was
  checked against token-by-token decode on 64 teacher-forced tokens: NLL 1.918 chunked vs 1.938 whole vs
  1.868 sequential at 1,536 tokens, within noise at 8,192.
- **Query-row chunking of the prefill indexer's scores** (`CACHALOT_INDEX_Q_CHUNK`, default 1024, balanced
  slices never below half a chunk). Bit-identical to the unchunked scores in the tests.
- **Prefix-cache snapshots at every prefill chunk boundary**, and least-recently-used eviction. A new Hermes
  session's first request reused 12,288 of 13,495 tokens: 18.9 s instead of 206.3 s.
- Server: one `[request]` line per request on stderr (prompt, reused, prefill seconds, decode tok/s,
  images), `CACHALOT_SERVER_DUMP=path` to append request bodies as JSON lines, SSE `: keep-alive` comments
  every 15 s of silence, and `images_served` / `vision_loaded` in `/v1/stats`.
- Benchmarks: `prefill_memory_sweep.py`, `prefill_chunk_check.py`, `prefill_chunk_quality.py`,
  `prefix_snapshot_exactness.py`.
- Tests: `test_vision_prompt.py`, `test_prefill_chunks.py`, `test_index_query_chunking.py`, and server tests
  for content parts, image errors and `reasoning_effort` aliases. 290 pass.

### Fixed
- **`serve.sh` served a 32,768-token context, and Hermes refuses anything below 64,000.** Now 65,536
  (+~84 MB of compressed-KV cache).
- **A 13.5k-token prompt ran the Metal heap out in prefill** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`).
  Fixed by the chunking above.
- **`reasoning_effort: "none"` returned a 500.** OpenAI-style values are now mapped (`none`/`minimal`
  switch thinking off, `medium` is 50, `xhigh`/`max` is `max`), and unknown ones are a 400.
- **A disconnected client still cost a full prefill.** Disconnects are polled every second, and queued
  cancelled requests are dropped.
- **Vision piece 3 against the reference**: image spans now get the learned `image_start`/`image_newline`/
  `image_end` rows, image positions are DEAD in the Engram hash with the gate shut, and the prefix cache
  keys image spans by content hash so two pictures of one size never share KV.
- `cachalot.__version__` was stuck at 0.9.8.

### Changed
- **Snapshots keep only the written rows of position-indexed caches** and restore pads zeros back:
  bit-identical on the real bank at 3k and 9k tokens, 14.7 / 33.0 MiB instead of ~215 MB. 12 live entries
  held 514 MB instead of 2.59 GB.

## 0.9.11 (2026-09-23)

**Vision phase 1 piece 3 is complete: per-token `bias_vl` routing and `image_mask` threading ship, wired
into `TextDecodeRuntime`'s prefill path and live-smoke-tested on the real bank.** Runtime change for prefill
only: every prefill call now threads two new optional keyword arguments end to end; omitting them (every
caller today) takes the exact old code path, confirmed bit-identical by the widened test suite and by a live
smoke test's unchanged first decode token. Separately, the FP8 GEMV family's post-fusion gap is re-measured,
after a discarded first attempt that mixed two scripts' incompatible measurement methods.

### Added
- **`src/cachalot/model/moe_prefill_batched.py`**: `route_topk_rows` gains `bias_vl`/`image_mask`, both
  optional and enforced both-or-neither. When given, each row selects `bias_vl` instead of the uniform
  `bias` where `image_mask` is set (`mx.where(image_mask[:, None], bias_vl[None, :], bias[None, :])`) before
  the top-k argsort — selection only, matching the official semantics' "correction bias affects selection
  only". Bit-identical to before when neither is given. This is the one router function that needed a real
  signature change (HANDOFF section 16.3): the decode router (`route_topk_fused`) and prefill's per-token
  fallback (`route_topk`) already take one bias per call, one token at a time, and a decode token is never
  an image position.
- **`src/cachalot/model/moe_prefill_grouped.py`**: matching `gate_bias_vl`/`image_mask` kwargs, forwarded to
  `route_topk_rows`; raises `NotImplementedError` if `image_mask` is given with `batched=False` (the
  per-token Python-loop fallback no production caller sets), before touching `expert_index`/`expert_store`.
- **`block_layer0_prefill.py`, `block_sliding_window_prefill.py`, `block_compressed_source_prefill.py`,
  `block_compressed_reuse_prefill.py`, `block_compressed_index_source_prefill.py`**: the same two
  keyword-only parameters, threaded straight through to each module's `moe_prefill_grouped` call.
- **`src/cachalot/model/text_decode_runtime.py`**: `prefill_tokens`/`_prefill_tokens_impl` gain
  `image_rows: mx.array | None = None` and `image_token_id: int = IMAGE_TOKEN_ID` (`129264`, now a module
  constant). The old `mx.stack([embed_token_decode(...) ...])` line is replaced unconditionally by piece 3
  step 1's `merge_image_embeddings` (0.9.10), bit-identical when `image_rows` is `None`. A local
  `image_mask` (`None` when there is no image) is built once and passed to all five layer-block call sites
  through a closure that returns `None` — not the tensor — when there is no image, so a plain text prefill
  never fetches `ffn.gate.bias_vl` at all.
- **`tests/test_route_topk_bias_vl.py`**: `route_topk_rows`'s new behavior checked against `router_mlx.
  route_topk` called once per row with the bias that row should have used — bit-identical with neither arg,
  both-or-neither and wrong-shape guards raise, text rows use `bias` and image rows use `bias_vl` exactly.
- **`tests/test_moe_prefill_grouped_image_mask.py`**: the `batched=False` guard.
- **`benchmarks/vision_piece3_prefill_smoke.py`**: live smoke test on the real 4-bit bank, same pattern as
  `qkv_fusion_live_smoke.py` — a plain text prefill (unchanged first decode token), then the same prompt with
  three interior token ids overwritten to `IMAGE_TOKEN_ID` and random `[3, 5120]` `image_rows` (no vision
  encoder is wired into the server yet, piece 4 is not started), through all 40 layers' `bias_vl` selection
  and eight follow-on decode tokens, finite output throughout, no crash.
- **HANDOFF sections 16.3, 9.36.**

### Fixed (documentation)
- HANDOFF/README's "43 MoE layers carry a `bias_vl`" corrected to 40: three of the 43 `bias_vl` tensors in
  the checkpoint index are unused `mtp.{0,1,2}.ffn.gate.bias_vl`, already excluded from this text-only
  runtime by `resident_trunk.py`'s existing filter. Confirmed by reading the shard header directly:
  `bias`/`bias_vl` are both `F32 [384]`, one pair per real layer, 40 pairs.

### Changed
- **The FP8 GEMV family's post-fusion gap, re-measured** (HANDOFF section 9.36). A first attempt summed
  numbers from `micro_fp8_gemv_kernel.py`'s isolated raw-kernel-launch measurement and
  `micro_shared_expert_roofline.py`'s real-function measurement for the same unfused shapes and found them
  ~20% apart — discarded before publishing, not reported. The self-consistent number, each shape group
  diffed only within the one script that measures its own before/after: **14.00 ms/token shipped today
  against 14.54 ms/token for the same shapes unfused, both measured the same way this session** — a 0.54
  ms/token combined win, the same order of magnitude as the two fusions' individually documented deltas.
  Gap against `mx.sum`: **2.36 ms/token today, against 2.90 ms/token unfused, measured the same way.** The
  original 3.48 ms figure (section 7.1.10) used a third methodology and is not directly comparable to
  either of these; if quoting a before/after, use 2.90 → 2.36.

### Verified
- 253/253 project tests pass (247 plus 5 new `test_route_topk_bias_vl.py`, plus 1 new
  `test_moe_prefill_grouped_image_mask.py`).

## 0.9.10 (2026-09-22)

**Two changes: a second FP8 GEMV fusion candidate is screened and correctly rejected, and vision phase 1
piece 3's first step ships.** No runtime behavior change on either path — the rejected fusion was never
wired in, and the vision splice is written and tested but not called from `TextDecodeRuntime` yet.

### Added
- **`src/cachalot/model/vision_mlx.py`**: `merge_image_embeddings`, piece 3 step 1 of the vision plan
  (HANDOFF section 16.2). Splices `vision_embed()`'s aligner rows into the embedded sequence at
  `image_token_id` positions, in reading order, for a text position falling back to the existing
  `embed_token_decode` unchanged. The official input boundary (`h = self.embed(input_ids);
  h = h.unsqueeze(2).repeat(..., hc_mult, ...)`) is a bare broadcast, not a learned expansion, so an image
  row needs the identical broadcast, not a new numerical path — the architectural risk flagged when piece 3
  was scoped did not materialize at this step. Raises `ValueError` if the prompt's `image_token_id` count
  and the supplied `image_rows` count disagree in either direction. Not wired into
  `TextDecodeRuntime._prefill_tokens_impl` — piece 3's other two steps (per-token `bias_vl` in the router
  kernel, threading `image_mask` through the prefill path) still need to exist first.
- **`tests/test_vision_prefill_splice.py`**: an all-text prompt is bit-identical to the unmodified
  `embed_token_decode` stack; an interleaved text/image prompt places each image row correctly and leaves
  text positions untouched; too few image rows, too many image rows, and image rows supplied for a prompt
  with no `image_token_id` positions all raise.
- **`benchmarks/micro_qb_indexer_fusion_roofline.py`**: screens whether attention's `wq_b` `[32768, 1280]`
  and the indexer's `wq_b` `[4096, 1280]` fuse the same way `wq_a`/`wkv` (0.9.9) and the shared expert's
  `w1`/`w3` (0.9.8) did — both read `qr`, the low-rank query, independently, a same-activation,
  independent-output pair the code's own comment names. Fused output is bit-identical to the shipped
  two-call path, but the indexer only runs on 8 of 40 layers and `wq_b` is not occupancy-bound (already
  456 GB/s, near its own `mx.sum` ceiling), so the recovered cost is 0.038 ms/token — two orders of
  magnitude below what a live session can confirm. **Rejected, not shipped**; the FP8 GEMV family's
  remaining ~3.1 ms gap (`wq_b`, `wo_b`, shared `w1`/`w3`/`w2`) now has no fusion candidate left unexamined.
- **HANDOFF sections 16.2, 9.35**.

### Verified
- 247/247 project tests pass (242 plus 5 new).

## 0.9.9 (2026-09-22)

**Two changes: the attention `wq_a`/`wkv` fusion ships, and vision phase 1 pieces 1-2 (the ViT+Aligner port
and the image preprocessing port) are numerically checked against the official reference.** Runtime change:
attention decode now issues one GEMV over `wq_a`/`wkv` instead of two, on all five decode call sites. Vision
change: two new modules, unwired, no runtime behavior change.

### Changed
- **`src/cachalot/model/attention_qkv_fusion.py`** (new): `wq_a` `[1280, 5120]` and `wkv` `[512, 5120]` read
  the same quantized activation and neither depends on the other's output, the same shape as 0.9.8's shared-
  expert `w1`/`w3` fusion, so they concatenate into one `[1792, 5120]` GEMV instead of two, cached per layer
  by weight-object identity (`fused_qr_kv_linear`, `_fused_wqkv`). These were the two worst-throughput
  shapes in the FP8 GEMV family (HANDOFF section 7.1.10: 171 and 87 GB/s against `wo_b`'s 383), occupancy-
  bound — 512 and 1280 output rows launch too few simdgroups to fill the GPU. Measured 2.01 → 1.66 ms/token
  across forty layers against a 1.55 ms `mx.sum` ceiling (`benchmarks/micro_qkv_fusion_roofline.py`),
  0.35 ms/token recovered, bit-identical by construction. No kill switch — attention is never called from
  inside an `mx.compile`d trace (only the MoE block is), so this fusion carries none of the shared expert's
  eval-inside-a-trace hazard.
- **`src/cachalot/model/attention_compressed.py`, `attention_layer0.py`, `attention_sliding_window.py`**:
  all five decode call sites (`compressed_attention_decode_source`, `_reuse`, `_index_source`,
  `layer0_attention_decode`, `sliding_window_attention_decode`) switched from two `fp8_linear` calls to
  `fused_qr_kv_linear`.

### Added
- **`src/cachalot/model/vision_mlx.py`**: MLX port of the official ViT (patch embed, 2D-RoPE bidirectional
  attention, SwiGLU MLP, RMSNorm) and `Aligner` (space-to-depth downsample reproduced by reshape/transpose,
  MLX has no `unfold`). Not wired into `TextDecodeRuntime`; `resident_trunk.py`'s filter is unchanged.
- **`src/cachalot/model/image_processor_mlx.py`**: near-verbatim port of the official resize-ratio solver
  and patchify, PIL replacing torch, `ml_dtypes` supplying the BF16 patches. Adds `pillow` as the new
  `vision` optional dependency extra.
- **`benchmarks/vision_parity_check.py`**, **`vision_parity_check_fp32.py`**: diff the MLX vision port
  against the official PyTorch reference on a real image, dynamically loaded from the checkpoint's own
  `inference/` directory. Preprocessing is bit-identical; the FP32 forward diff is at machine precision
  (`6.7e-6`), confirming the BF16 forward's `3.8e-2` max diff is accumulated rounding noise across 32
  layers, not a defect.
- **`benchmarks/micro_qkv_fusion_roofline.py`**, **`qkv_fusion_live_smoke.py`**: the `wq_a`/`wkv` fusion's
  roofline measurement and a live sanity check on the real 2-bit bank.
- **`tests/test_attention_qkv_fusion.py`**: bit-identical output between the fused and unfused paths on the
  checkpoint's real shapes; the concatenation is cached rather than rebuilt on every call and two layers'
  caches do not collide.
- **HANDOFF sections 16.1, 9.34**.

### Verified
- 242/242 project tests pass (238 plus 4 new).
- `wq_a`/`wkv` fusion: live smoke test on the real 2-bit bank (`benchmarks/qkv_fusion_live_smoke.py`),
  prefill plus twelve greedy decode tokens, no crash, correct completion (`'Hello'`, `'!'`, EOS for "Say
  hello in one short sentence.").
- Vision port: `benchmarks/vision_parity_check.py` against the official reference on a real image (a
  700x486 downsize of `assets/dsv41_kv_cache.png` — full resolution drives the ViT to ~9,200 patches and a
  CPU-only dense-bidirectional-attention reference forward over that many tokens does not finish in
  reasonable time). Grids match exactly, patches bit-identical, FP32 forward diff at machine precision.

## 0.9.8 (2026-09-22)

**The shared expert's `w1`/`w3` fusion is shipped, unconditionally, after one live crash and one fix.**
Runtime change: the shared FP8 expert now issues two GEMVs per layer instead of three on every decode
token. Bit-identical output, no quality change.

### Changed
- **`src/cachalot/model/shared_expert_metal.py`, `src/cachalot/model/moe_layer_metal.py`**: `w1` and `w3`
  read the same quantized activation and neither depends on the other, so they now concatenate into one
  `[2I, H]` GEMV instead of two `[I, H]` ones (`shared_expert_forward_fused`, `_fused_w13`), cached per layer
  by weight-object identity. Measured at 0.115 → 0.106 ms/layer, 4.60 → 4.23 ms per token across forty
  layers (`benchmarks/micro_shared_expert_roofline.py`), first identified and left unshipped in 0.9.2's
  ranking (HANDOFF section 9.27) because 0.4 ms is below what any live session can confirm.
- `shared_expert_forward` (the original two-GEMV form) stays for `moe_prefill_grouped.py`'s separate
  per-token prefill fallback, which was never benchmarked fused and is unchanged.

### Fixed
- The first version of the fusion called `mx.eval` inside `shared_expert_forward` itself, which
  `moe_layer_metal._compiled_moe_block`'s `mx.compile`d trace calls on the shipped 2-bit affine bank — MLX
  refuses `mx.eval` mid-trace (`ValueError: [eval] Attempting to eval an array during function
  transformations like compile or vmap is not allowed`). Every offline check before this was caught,
  including a bit-identical unit test and the full 238-test suite, called the fused path eagerly, never
  from inside `mx.compile`, so nothing caught it before a live `./chat.sh` session did, on the first decode
  step. Fixed by moving `w1`/`w3` concatenation into eager Python before the compiled call, and by giving
  `_compiled_moe_block` a stable cache key again once the code path was unconditional.

### Added
- **`tests/test_shared_expert_fused.py`**: bit-identical output between the fused and unfused paths on the
  checkpoint's real shapes; the concatenation is cached rather than rebuilt on every call and two layers'
  caches do not collide; a test that calls the fused path from inside an actual `mx.compile`, reproducing
  the crash; a test pinning that `mx.eval` inside a trace still raises if the bug is reintroduced.
- **HANDOFF section 9.33**: `kernel_consts.py:39`'s previously-unexplained store-blocked time (carried since
  0.9.5's next-session prompt) is explained — a profiler attribution artifact, not a real or recurring cost.

### Verified
- 238/238 project tests pass.
- `V41Model.from_pretrained` run directly against the real 2-bit affine bank (`/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128`)
  at a 24 GiB budget: warmup plus eight further `decode_token` calls, no traceback, the same eight greedy
  token ids before and after the fix, and again after the kill switch was removed.
- Tried live by Hamed: a 642-token story and a 1,744-token Objective-C json-to-CSV turn both read correct,
  9.91 and 8.27 tok/s, 92.35 % hit rate — in line with the shipped-52-GiB baseline (different prompts than
  the fixed four-prompt set, so not a controlled A/B, and 0.4 ms/token was never going to show up in a live
  read either way).

## 0.9.7 (2026-09-22)

**Server bug fix, budget investigation continued, vision scoped.** No change to the shipped runtime path.

### Fixed
- **`src/cachalot/server/app.py`**: `ServerConfig.default_frequency_penalty` was still 0.2, citing the
  62 %/12 % repetition-collapse claim that section 9.9 retracted (it was the transposed residual mix, not
  real repetition; the fixed runtime is 0 of 12 with the penalty off). An unmodified OpenAI client — the
  default case for a new Hermes connection — was the one remaining path still paying a penalty nothing
  causes any more. Now 0.0, matching `cachalot chat`. Two `tests/test_server.py` assertions that pinned the
  stale default updated to match; a third pins the knob is still configurable.

### Added
- **`serve.sh`**, mirroring `chat.sh`: the shipped 52 GiB / 80 GiB wired configuration as an HTTP server on
  port 8011. Smoke-tested this session against the real checkpoint: `/v1/models`, non-streaming and
  streaming chat completions, and a tool-calling round trip (valid JSON arguments, correct `finish_reason`).
- **HANDOFF section 7.2.10**: six budget arms (46-56 GiB) with `memwatch.sh` running beside them. No OS-level
  memory-pressure event at any budget; the one 54 GiB session's Objective-C-turn collapse (4.83 tok/s against
  8.4-8.8 at 48-52) has no memory signature and is turn-4-specific, not session-wide — refuting the prior
  physical-memory-cliff hypothesis for this run. 52 GiB stays shipped.
- **HANDOFF section 15**: the Hermes HTTP server audited, the frequency-penalty bug above, and what is not
  yet checked (Hermes's actual client behaviour against a single-flight engine).
- **HANDOFF section 16**: vision scoped. The checkpoint carries a real 263-tensor ViT + aligner (~480 M
  params, under 1 GiB) that `resident_trunk.py` currently filters out by name; a PyTorch reference
  implementation and image preprocessor ship in the checkpoint directory. Four-piece port plan, ordered by
  what can be checked before anything is wired into the text model.

## 0.9.6 (2026-09-22)

Measurement tooling and one live reading, no runtime change. **Hamed's first 54 GiB session decoded at 4.9-6.2
tok/s against 9.4-9.6 at 52, with 0.33 points more hit rate, and a 60 GiB attempt was abandoned as unusably
slow; a guarded replay of the same four prompts at 50 GiB on a clean machine then decoded at 9.58 and 8.52
tok/s with memory pressure normal.** The shipped configuration stays at 52 GiB. Why 54 is slow is not yet
established, and the tool that will establish it is new.

### Added
- **`benchmarks/memwatch.sh`**: samples free, available, wired, compressor, anonymous and file-backed memory,
  swap, pressure level, pageouts and decompressions once a second beside an interactive chat, never kills
  anything, and prints a summary on Ctrl-C. `guarded_run.sh` samples the same counters only for benchmarks it
  launches and kills on pressure, which a chat cannot be run under.
- `benchmarks/chat_turns.py --turns-file` and `--temperature`, so a live session's prompts can be replayed at
  the shipped temperature under `guarded_run.sh`. The 2026-09-22 session is in `benchmarks/sessions/`.

### Learned
- `CACHALOT_MLX_WIRED_LIMIT_GIB=90` does nothing: `resolve_wired_limit` takes the minimum of the request and
  the device's recommended working set, 77.8 GiB, and the banner printed 77.8 at 54 GiB exactly as at 52.
- The 50 GiB replay (`benchmarks/results/guarded/replay50_20260922-013337.*`): story 9.58 tok/s, Objective-C
  8.52, swap flat at 490 MB, compressor flat at 3.7 GiB, pressure 1, peak system wired 71.7 GiB. Section 7.2.9.

## 0.9.5 (2026-09-22)

Quality-gate change and no runtime change: **the coding gate now contains Objective-C, compiles it, and runs
it.** Two live Objective-C turns had failed a compiler and a third compiled only after a one-line repair and
still contradicted its own documented column order, while the 40-case gate contained no Objective-C and
executed nothing.

### Added
- **`benchmarks/code_validity.py`**: `check_objc` (clang `-fobjc-arc -fsyntax-only`, so a selector Foundation
  does not declare fails exactly as in a full build) and `run_objc`, which builds a block, runs it with a
  20 s timeout and no stdin, and diffs stdout against the text the task said it must print. `score()` runs a
  block only if it compiled; `summarise()` reports `executed`, `ran_clean` and `output_matches` next to the
  compile rate. A harness failure (no clang, no Foundation) is `valid: False` and never scored against the
  model. The C/C++ path was refactored into one shared `_check_clang` with no change in behaviour.
- **Six Objective-C tasks in `benchmarks/coding_tasks.json`** (26 tasks, up from 20), each a complete
  program with no input and an `expected_stdout`: CSV header in first-seen key order (the exact defect of the
  third live turn), word frequency, interval merge, LRU cache, Roman numerals, matrix transpose.
  `benchmarks/objc_reference/` holds a reference program per task; a test builds each one and requires its
  output to equal `expected_stdout` and to appear verbatim in the prompt.
- `coding_quality.py` recognises `objc` fences and writes `expected_stdout` into `rows.json`.
- Four tests (233 total): the live turn that calls `-map:` is rejected, the repaired turn compiles, running
  distinguishes right output from wrong output, every reference matches its expectation, and `score()` runs
  a block only when it compiled.

### Not done
- No model was run against the new cases; the corpus hash changed, so `--resume` will not continue a run
  begun on the 20-task corpus. Running the three banks on the 26 tasks (about 2 h) is the next quality job.

## 0.9.4 (2026-09-22)

Benchmark-instrument fix and one measurement; no runtime code changed. **Job 1 is answered: the miss is at
the drive's rated wall, and wired memory is a null.**

### Fixed
- **`benchmarks/expert_read_scaling.py`'s `--wire-gib` ballast-heartbeat thread crashed on its first tick
  on MLX 0.32.2** (`RuntimeError: There is no Stream(gpu, 0) in current thread`) because the pinged array
  was never `mx.eval`'d on the thread that created it, and 0.32.2 cannot resolve a default GPU stream for
  an array's first materialization from a different thread. The exception printed to stderr and the thread
  died; `main()` never saw it and the script's own exit code stayed 0, so three sweeps in a row silently
  measured an unwired machine after the first arm or two. Fixed with one `mx.eval()` call on the main
  thread before the heartbeat thread starts.

### Measurement
- **The miss is at the drive's rated wall, and wired memory pressure has no measurable effect.** At
  `io_workers=8`, 42.9 GiB wired (45 % of the machine): 6.81 GB/s, 1.46 ms/expert. Unwired: 6.76 GB/s,
  1.47 ms/expert. Both match section 3.1's "6.6-6.8 GB/s cold" rating, and the runtime's own 1.41 ms
  blocked-per-miss component sits inside that same band rather than above it. Tested at 45 % of the
  machine's RAM wired, not the shipped 52 GiB budget's 80-83 %, because available memory this session
  (60-64 GiB free) did not admit a bigger ballast. **The budget is the only lever left on the miss.**
  §7.1.11, §9.31.

## 0.9.3 (2026-09-21)

Documentation and one archived live turn; no code and no numerics changed. **The shipped 52 GiB
configuration was read a second time and replicates, and every MLX peak this project has ever quoted was
in GB against a limit in GiB.**

### Measurement
- **The second live session at 52 GiB, on a build with no runtime change against the first.** Prose
  **9.59 tok/s** against 9.42, Objective-C **8.15** on 1,484 tokens against 8.53 on 1,493, session hit
  rate **92.31 %** against 92.37 %, 5,276 residents against 5,314, and an MLX peak **identical to the
  byte**. The configuration replicates, so **52 GiB has two sessions and not one**. What is still unrun is
  section 9.24's A/B: this session had the Engram change on, so it is a second reading of the good arm.
  §7.2.7.
- **Section 7.1.9's price of a miss, checked against a conversation for the first time.** 502,435 expert
  requests over 240 per token is ~2,093 token-equivalents and 38,661 misses, so **18.5 misses per token**;
  at 1.7 ms each on the 79.6 ms floor that predicts **111 ms**, against **104.3 ms** observed on the prose
  turn and **122.7** on the coding one. The prediction lands between the session's own two long turns.
- **About 48 % of the session's drive traffic was speculative.** 755.07 GB read; the 38,661 demand misses
  account for 384.8 GB, the hotlist for 8.6 GB, and prefill was almost entirely prefix-cache reuse (13
  hits, 4,192 tokens, 619 of 623 on the coding turn). Section 9.10's 42.7 % was measured at 44 GiB on a
  different shape of session, and no instrument in the repository can say what fraction of the 360 GB
  earned its place at this budget.

### Corrections
- **`mlx_peak_bytes` is bytes, and four sections divided it by 10^9 while dividing the wired limit by
  2^30.** Every "peak against limit" pair before section 7.2.8 understates the headroom by 7.4 %: the
  52 GiB peak is **67.74 GiB, not 72.73**, against a 77.8 GiB wired limit, and the 44 GiB peaks are
  55.1-55.6 GiB rather than 59.2-59.7. The arithmetic settles which reading is right — 5,276 residents is
  48.91 GiB of experts, and the trunk, transient slots and MLX cache are about 14.4 GiB more, which 67.74
  clears and 72.73 would need 23.8 GiB of unaccounted memory to reach. **Nothing about any speed or
  quality conclusion moves**; the peaks were only ever used to decide whether a budget fits and the error
  was in the safe direction. What moves is the headroom, and therefore the next budget worth screening:
  **10.1 GiB free at 52, and 60 GiB projects to about 75.2 GiB** against a replay that says 52 → 60 is
  worth another 2.3 points of hit rate. §7.2.8.
- **`predicted_used` is not the numerator of a precision.** It increments in one place
  (`resident_store.py:629`), when a demand request catches a prediction **still in flight**; a prediction
  that lands before it is demanded is counted as an ordinary hit. Two sections called
  `predicted_used / predicted_loads` "prediction precision"; it is a lower bound by an unknown margin.
  §7.2.7.

### Quality
- **The third live coding turn put through a compiler, and the third to fail** — but the useful one.
  `[NSMutableArray map:]` is not declared by Foundation; one line repairs it and the program then compiles
  clean, runs, exits 0 and writes a valid CSV. **Its column order still contradicts both its own Notes and
  its own worked example**, because the header is built from `flat.allKeys`, which is unordered. This is
  the first case in the project where compiling is not the check either, and it is the argument for a gate
  that *runs* what it builds. Archived with the compiler output, the repair and the real output in
  `docs/live-turns/2026-09-21-json2csv-2/`.
- A bare "Hi" was answered in Chinese again, as in the first 52 GiB session, and did not recur once the
  session asked for English. A model behaviour on a multilingual checkpoint, reproducible across sessions,
  and nothing in the 40-case corpus looks for it.

229 tests pass. Version 0.9.3.

## 0.9.2 (2026-09-21)

Measurement and documentation; **nothing under `src/cachalot/` was touched and no numerics changed.**
**The last block in the ranking with no mechanism turned out not to be a block: it is the miss.** With
that settled, the three GPU blocks nobody had screened were screened, and all three came back at or near
the machine's own limit.

### Measurement
- **A streaming token at a 100 % hit rate is the all-resident floor.** `profile_decode_sync.py` gained
  `--stream-passes`, which replays the same continuation from the same snapshot; decoding is greedy, so
  pass 2 runs the identical tokens through the identical graph at the identical positions over the same
  240-experts-per-layer working set scattered across a 40 GiB slot pool, with everything already resident.
  It costs **79.6 ms against the all-resident arm's 79.6**, 58.5 ms inside `mx.eval` against 58.0, 19.3 ms
  of CPU against 19.3. So the "+20 ms inside `mx.eval`" that three prompts called unexplained is part of a
  miss, worth **0.31 ms of each one** on top of the 1.41 ms the store already charges. §7.1.9.
- **`io_workers` is a null from 2 to 16**, on eight interleaved arms with two reps of each width: every
  whole token inside 141.5–148.5 ms, and the two reps of the shipped width span 5.5 ms of that on their
  own. The in-eval / blocked split does not move. §9.26.
- **`CACHALOT_PAGE_CACHE=0` is a 16 ms loss**, in the blocking and the CPU, with the GPU getting nothing
  back. The shipped `=1` is confirmed from a direction nothing had tried. §9.26.
- **Attention is 86 % weight streaming.** A reuse layer's 0.441 ms is 0.270 ms of FP8 GEMV over `wq_a`,
  `wq_b`, `wkv` and `wo_b`, 0.107 ms of BF16 grouped matmul over `wo_a`, and **0.064 ms** of the sparse
  attention whose shapes three prompts wanted attacked. The weight inventory is read off the checkpoint
  headers: 126.6 MB per layer on disk, 160.2 MB resident. §7.1.10.
- **The FP8 GEMV family is the largest GPU path in the runtime** — four attention projections on forty
  layers plus all three shared-expert GEMVs, **5.14 GB per token against the routed experts' 2.3 GiB**,
  16.61 ms — and it runs at **79 % of `mx.sum` over the same bytes**. §7.1.10.
- **The shared expert, screened for the first time** after three prompts carried it unscreened: 307 GB/s
  against 385 for `mx.sum` over its own bytes, and the screen reproduces the profiler to 4 %. The one
  structural idea its shape allows — stacking `w1` and `w3` into one [4608, 5120] GEMV, which is
  **bit-identical by construction** because `K` and therefore the lane split are unchanged — is worth
  0.4 ms per token and is **not shipped**, because 0.4 ms is a tenth of what a live session resolves and
  this project does not ship on a screen. §9.27.
- **`wo_a` is held dequantized in BF16 and that is the right call.** It reads 67.11 MB per layer where the
  checkpoint ships 33.55, but MLX's BF16 grouped matmul runs it at **625 GB/s** and is faster than every
  FP8 form measured, including a one-launch upper bound that ignores the indexing a grouped FP8 kernel
  would need. §9.28.
- **The FP8 GEMV kernel's lanes-per-row policy, written for load balance with `N` not an input, picks the
  fastest split on five of six shapes.** Best split per shape is 16.56 ms against the shipped 16.61. §9.29.

### Instruments
- `benchmarks/profile_decode_sync.py`: `--stream-passes N` and `--io-workers N`, and the ready line now
  prints the reader width.
- `benchmarks/micro_fp8_gemv_kernel.py`: the largest GPU path priced per shape against `mx.sum`, every
  legal lanes-per-row split, each checked for bit-identical output against the shipped split.
- `benchmarks/micro_shared_expert_roofline.py`: the shared expert against the memory wall on forty
  distinct layers, with the `w1`/`w3` fusion and the activation-quantization launches priced separately.
- `benchmarks/micro_wo_a.py`: `wo_a` in BF16 against every FP8 form of the same projection.

### Pitfalls
- **Read which function an instrument calls before ranking a lever off it — fifth occurrence.** A
  four-variant kernel screen was written, run and found bit-identical against
  `fp8_gemv_metal.fp8_gemv_quantized`, which the runtime does not call: `fp8_linear_quantized` dispatches
  to `fp8_fused_metal.fp8_gemv_decoded` whenever `CACHALOT_FUSED_FP8` is set, and it is set by default.
  The shipped kernel was already 30 % faster than the best variant of the retired one.
- **`--mode both` and `--mode stream` do not produce the same streaming arm**: the all-resident arm leaves
  240 experts pinned and changes what the continuation evicts, which is 7 ms.
- **Interleave the arms of a sweep and run each twice.** The shipped `io_workers` width spanned 5.5 ms
  across its own two reps, the whole range of the sweep.

229 tests pass. Version 0.9.2.

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
