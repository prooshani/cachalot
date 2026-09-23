# Next-session prompt — **v37**, written 2026-09-22

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v37** | 2026-09-22 | Hamed asked for speed-then-vision, out of standing order, explicitly | **Vision piece 3 steps 2-3 are done: per-token `bias_vl` selection in `route_topk_rows` and `image_mask` threaded through `_prefill_tokens_impl` and all five prefill block variants, live-smoke-tested on the real bank. Piece 3 is fully wired end to end; piece 4 (server image-content parsing) is unblocked. Separately, the FP8 GEMV family's post-fusion gap was re-measured — first attempt mixed measurement methodologies and was discarded before publishing; the self-consistent number is a 2.90 → 2.36 ms/token gap reduction, not the "3.48 → X" a naive diff would have claimed.** §16.3, §9.36 |
| v36 | 2026-09-22 | Hamed asked for speed-then-vision, out of standing order, explicitly | `wq_b`/indexer `wq_b` fusion screened and correctly rejected (0.038 ms/token, §9.35). Vision piece 3 step 1 (`merge_image_embeddings`) written and tested, not wired. §16.2 |
| v35 | 2026-09-22 | Hamed approved "vision first, then speed" for this session, explicitly | Vision piece 1+2 numerically checked against the official PyTorch reference. `wq_a`/`wkv` fused, 0.35 ms/token, shipped. §16.1, §9.34 |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read this first.** `docs/HANDOFF.md` section **15** (the Hermes server, still not checked against Hermes's
own client — five sessions running now), section **16.3** (vision piece 3 steps 2-3, done this session, piece
4 next), and section **9.36** (the FP8 GEMV re-measurement and the cross-script methodology trap it found).

**Hamed's stated priority order for the next sessions: Hermes usage first, vision second, speed/performance
third.** This still holds. The 2026-09-22 speed-then-vision sessions (this one and the last) were explicit
one-off requests, asked for and confirmed in chat before starting, not a reordering of the standing priority —
**Job 1 (Hermes) is still first up and still nobody has run it, five sessions running now.**

**This session's changes are uncommitted.** `git status --short`: eight modified files
(`moe_prefill_batched.py`, `moe_prefill_grouped.py`, the five `block_*_prefill.py` modules,
`text_decode_runtime.py`) and three new files (two test files, one live-smoke benchmark). 253/253 tests pass.
Nobody asked for a commit this session — check with Hamed or commit with the usual version bump before
starting new work on top of it.

**The shipped chat configuration, unchanged:**

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

**The server:**

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

## Job 1 — get Hermes Agent Desktop talking to Cachalot. Needs Hamed's own session. Unchanged, still not done.

The server itself is implemented, its one known bug (a stale 0.2 frequency-penalty default, HANDOFF §15) is
fixed, and a curl smoke test passed: models, streaming, non-streaming, and a tool-calling round trip. **What
has never been checked is Hermes's own client against it — five sessions running now, this is still the top
of the list.** Start the server, point Hermes Agent Desktop at `http://127.0.0.1:8011/v1`, model id
`deepseek-v4.1-flash`, any placeholder API key, and have a real conversation, ideally one that uses tools.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

Watch for: a `content` shape `ChatCompletionRequest` does not model (a list of parts rather than a plain
string — this is also the shape an image would arrive in, and piece 4 below will need to parse it); a
`tool_choice` or `response_format` value the server ignores; Hermes retrying or opening a second conversation
while one is still generating, which queues rather than running concurrently (`engine.py`'s single-flight
lock, HANDOFF §15). Report back what broke, with the request body if possible.

## Job 2 — vision, phase 1 piece 4: the server's image-content parsing. Piece 3 is fully wired; this is now unblocked.

HANDOFF §16 has the four-piece plan, §16.1-§16.3 the checkpoints. **Piece 3 (the prefill splice:
`merge_image_embeddings`, per-token `bias_vl`, `image_mask` threading) is done and live-smoke-tested** on the
real 4-bit bank (`benchmarks/vision_piece3_prefill_smoke.py`) — a synthetic image-bearing prompt (three
positions overwritten to `IMAGE_TOKEN_ID`, random `image_rows` since no vision encoder is wired into the
server yet) prefills and decodes cleanly through all 40 layers, finite output, no crash.
`TextDecodeRuntime.prefill_tokens(token_ids, image_rows=..., image_token_id=...)` is the call piece 4 needs
to reach.

**What piece 4 actually is**, per §16 point 4: `ChatCompletionRequest`'s `content` field needs to accept
OpenAI's `content: [{"type": "image_url", ...}]` message shape (today it's typed as part of an untyped
`dict` — the same shape gap Job 1 above flags for Hermes), decode the URL or base64 payload, run it through
`image_processor_mlx.py` (piece 2, checked, unwired) and `vision_mlx.vision_embed()` (piece 1, checked) to
produce real `image_rows`, and call `prefill_tokens` with them. `image_processor_mlx.py`'s image-loading path
already handles a URL via `urlopen` and a data URI via `base64` (inherited from the reference's
`image_processor.py`), so the decode side is close to done — the message-shape typing and the call-site
wiring in the server are what's left.

**Once piece 4 exists, re-run `benchmarks/vision_parity_check.py`-style numerics on a real end-to-end
request** (real image through the real ViT+Aligner, not this session's random `image_rows`) — this session's
live smoke proves the plumbing is sound, not that a real image produces correct completions. That is the
first thing piece 4's own session should check once the server path exists.

## Job 3 — the 54 GiB collapse, re-tested with the right variable. Unchanged, still low priority.

Section 7.2.10 ruled out an OS-visible memory-pressure event at every budget from 46 to 56 GiB, including
the one 54 GiB session that collapsed on its Objective-C turn. The collapse was specific to that one long
turn, not the session, so the next check is duration and thermal state, not memory:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/memwatch.sh 54
```

then in the first terminal `CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 54` with the same
four prompts as `benchmarks/sessions/hamed_2026-09-22.txt`. If it reproduces, run `sudo powermetrics
--samplers cpu_power,gpu_power -i 1000` alongside a repeat to check for thermal throttling. Low priority —
do this after Jobs 1 and 2, or when Jobs 1-2 are blocked on something else.

## Job 4 — score the banks on the Objective-C cases the gate now has. Unchanged, still not run.

`benchmarks/coding_quality.py` on the 2-bit bank at the shipped configuration for the 26-task corpus (6 new
Objective-C tasks), a fresh run not `--resume`. About two hours, no live session needed.

## What is closed, so nobody spends a session there

- **The miss's own drive-wall arithmetic**, on both memory pressure and raw throughput. §7.1.11, §9.31.
- **The in-eval excess.** It is the miss. §7.1.9.
- **A physical-memory cliff between 50 and 56 GiB** — refuted, §7.2.10. What remains open is duration/thermal.
- **`kernel_consts.py:39`** — attribution artifact, not a cost. §9.33.
- `io_workers` 2-16, `CACHALOT_PAGE_CACHE=0`, attention's shapes, `mx.compile` on attention, quantizing
  `wo_a`, the FP8 GEMV lanes-per-row policy, `INDEX_TOPK`, the expert kernel, routing prediction, speculative
  decoding, the hyper-connection glue, the router, dispatch fusion, 1 bit per weight, mirror striping on
  this bank, the frequency penalty (as a *quality* lever), eviction policy. §7.1.9-7.1.10, §9.21-9.30, §11.
- **The shared expert's `w1`/`w3` fusion** and **the attention `wq_a`/`wkv` fusion** — both shipped as the
  only path, no kill switch. §9.27, §9.34.
- **The FP8 GEMV family's remaining gap, as a fusion-candidate hunt.** §9.35 found the one candidate left
  (`wq_b`/indexer `wq_b`), sized it at 0.038 ms/token, and rejected it. **Do not re-run this hunt without a
  new candidate in hand.** The gap itself was re-measured post-fusion this session (§9.36): 2.90 → 2.36
  ms/token, self-consistently. Don't quote the old 3.48 ms figure against the new 2.36 — they used different
  measurement methods; if quoting a before/after, use 2.90 → 2.36, both from §9.36's own scripts.
- **Vision piece 1 (ViT+Aligner) and piece 2 (image preprocessing)** — ported and numerically checked
  against the official reference. §16.1. Do not re-check piece 1/2's numerics again without a reason.
- **Vision piece 3, all three steps** — `merge_image_embeddings`, per-token `bias_vl` selection in
  `route_topk_rows`, and `image_mask` threading through `_prefill_tokens_impl` and all five prefill block
  variants. §16.2, §16.3. Live-smoke-tested clean on the real bank with a synthetic image span. Only piece 4
  (server plumbing) is left, and it can now build against a real, working `prefill_tokens(image_rows=...)`.

## Rules that still hold

- **A background thread's exception can die silently while the parent exits 0.** §7.1.11, §12.
- **Read which function an instrument calls before ranking a lever off it.**
- **`mlx_peak_bytes` is bytes; divide it by 2^30, and check the units on both sides of a comparison.**
- **Two arms must be launched the same way.** This session's §9.36 is the clearest example yet: a raw-kernel
  isolated launch and a call to the real production function measured the *same* unfused shape 20% apart.
  Never diff numbers across two different measurement scripts, even for a shape both claim to measure
  identically — diff a script's own before/after arms, then sum non-overlapping groups if you need a total.
- **The drive is shared, so measure I/O-touching code while it is busy.**
- **Check an offline simulation against a live run the first time one is possible.**
- **A live session resolves its hit rate, and a change of 20 ms or more; it cannot resolve 5 ms.** Extended
  this session: an offline microbenchmark's own cross-script diff cannot resolve a sub-millisecond win either,
  no matter how many decimal places it prints (§9.36).
- **Compile a live coding turn before calling it reference class.**
- **Never drop a case from a denominator.**
- **A duplicate file is not a second reading.**
- **`guarded_run.sh` needs `budget + 29` GiB available; a *chat* session is not bound by that guard.** Never
  pass `--force`.
- **Hand over `./chat.sh` or `./serve.sh`, never a one-liner.**
- **A same-activation, independent-output pair of GEMVs is a fusion candidate; check for one before ranking
  a kernel's per-shape throughput as fixed** — but bit-identical and correct is necessary, not sufficient;
  size it before shipping it.
- **A comment in the code naming a design intent is worth chasing down as a lever candidate, even when it
  turns out too small to ship, or too small to even measure cleanly (§9.36).**
- **An architectural worry stated in a planning doc is a hypothesis to check against the actual boundary
  code, not a fact to route around.** This session's second instance: §16/§16.2 flagged the router kernel as
  a real signature change, "not a parameter swap" — true for `route_topk_rows`, the one function that
  actually broadcasts a batch-uniform bias, but the *decode* router (`route_topk_fused`) already took one
  bias per token and needed no change at all, because a decode token is never an image position.
  `merge_image_embeddings`'s own "simpler than feared" finding (§16.2) is the same pattern, one step earlier.
- **A tensor-name count from a checkpoint index needs the layer range checked before it's quoted as "N MoE
  layers"** — §16/§16.2's "43" included three unused MTP layers this runtime already excludes; the real count
  for anything this runtime executes is 40. New this session (§16.3).
- **A CPU-only PyTorch reference forward over thousands of dense-bidirectional-attention tokens does not
  finish in reasonable time — downsize the test image, not the tolerance.**
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once — this includes `serve.sh` racing `chat.sh` or a benchmark.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/memwatch.sh N` | memory beside a live chat, without killing it | runs as long as the chat |
| `benchmarks/chat_turns.py --turns-file --temperature` | replays a live session's own prompts | ~1 min per turn set |
| `benchmarks/expert_read_scaling.py --wire-gib N` | what the drive gives under decode's memory conditions | ~3 min |
| `benchmarks/decode_anatomy.py` | where a live token's time goes by blocking cause | ~1 min |
| `benchmarks/profile_decode_gpu.py` | the GPU side, every layer | ~1 min |
| `benchmarks/decode_fingerprint.py` | whether a speed arm changed the numerics | ~1 min |
| `benchmarks/simulate_policies.py --expert-bytes 9953280` | hit rate against budget, offline | instant |
| `benchmarks/nll_expert_precision.py --experts runtime --tokens 512` | the production quality arm | ~2 min |
| `benchmarks/coding_quality.py --resume` | the 52-case corpus (26 tasks x 2 seeds), Job 4 | ~1.5 h |
| `benchmarks/server_smoke.sh` | curl-based server smoke test | ~2 min |
| `benchmarks/vision_parity_check.py [image]` | MLX vision port vs official PyTorch reference | ~2 min on a downsized image |
| `benchmarks/micro_qkv_fusion_roofline.py` | `wq_a`/`wkv` fused vs shipped vs `mx.sum` ceiling | ~10 s |
| `benchmarks/qkv_fusion_live_smoke.py` | live sanity check on the real bank | ~1 min |
| `benchmarks/micro_qb_indexer_fusion_roofline.py` | `wq_b`/indexer `wq_b` fused vs shipped vs `mx.sum` ceiling | ~10 s |
| `benchmarks/vision_piece3_prefill_smoke.py` | vision piece 3 (bias_vl + image_mask) live sanity check on the real bank, new 2026-09-22 | ~1 min |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | the official implementation |
| `src/cachalot/server/app.py`, `engine.py` | the OpenAI-compatible server, §15, piece 4's target |
| `src/cachalot/model/vision_mlx.py` | vision piece 1 + piece 3 step 1 (`merge_image_embeddings`). §16.1, §16.2 |
| `src/cachalot/model/image_processor_mlx.py` | vision piece 2, checked, unwired. §16.1 |
| `src/cachalot/model/moe_prefill_batched.py` | `route_topk_rows`, piece 3 step 2's per-token `bias_vl`. §16.3 |
| `src/cachalot/model/moe_prefill_grouped.py` | `gate_bias_vl`/`image_mask` passthrough, the `batched=False` guard. §16.3 |
| `src/cachalot/model/block_layer0_prefill.py`, `block_sliding_window_prefill.py`, `block_compressed_source_prefill.py`, `block_compressed_reuse_prefill.py`, `block_compressed_index_source_prefill.py` | the five prefill block variants, all threading `gate_bias_vl`/`image_mask` now. §16.3 |
| `src/cachalot/model/text_decode_runtime.py` | `prefill_tokens`/`_prefill_tokens_impl`, `image_rows`/`image_token_id`, `IMAGE_TOKEN_ID` constant. §16.3 |
| `tests/test_route_topk_bias_vl.py`, `tests/test_moe_prefill_grouped_image_mask.py` | piece 3 step 2's tests, new 2026-09-22 |
| HANDOFF section 16, 16.1, 16.2, 16.3 | vision: scoped, piece 1+2 checked, piece 3 all three steps done |
| HANDOFF section 15 | the Hermes server: what's fixed, what's not checked |
| HANDOFF section 9.34, 9.35, 9.36 | the two shipped fusions, the rejected one, and the post-fusion re-measurement |
| `./chat.sh` | the command to hand Hamed for chat — `--expert-budget-gib 52`, `CACHALOT_MLX_WIRED_LIMIT_GIB=80` |
| `./serve.sh` | the command to hand Hamed for the server — same config, port 8011 |
