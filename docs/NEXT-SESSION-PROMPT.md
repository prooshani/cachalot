# Next-session prompt — **v35**, written 2026-09-22

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v35** | 2026-09-22 | Hamed approved "vision first, then speed" for this session, explicitly, out of standing order | **Vision piece 1+2 (the ViT+Aligner port and the image_processor port) is numerically checked against the official PyTorch reference on a real image — bit-identical preprocessing, FP32 diff at machine precision, BF16 diff explained as rounding noise. Piece 3 (the prefill splice) is unblocked for whoever picks this up next. Separately, `wq_a`/`wkv` — the two worst-throughput shapes in the FP8 GEMV family (HANDOFF section 7.1.10) — fused into one GEMV the same way the shared expert's `w1`/`w3` did, 0.35 ms/token recovered, bit-identical, shipped with no kill switch, live-smoke-tested clean on the real bank.** §16.1, §9.34 |
| v34 | 2026-09-22 | Hamed asked for a speed-focused session out of order, explicitly | The shared expert's `w1`/`w3` fusion shipped as the only path (§9.27), and one mystery closed (`kernel_consts.py:39`, §9.33) |
| v33 | 2026-09-22 | Hamed reprioritized: Hermes, then vision, then back to speed | The server is more done than tracked and its one real bug is fixed and smoke-tested. Vision is scoped. The 54 GiB budget question moved (§7.2.10) |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read this first.** `docs/HANDOFF.md` section **15** (the Hermes server, still not checked against Hermes's
own client), section **16** and **16.1** (vision: scoped, piece 1+2 now numerically checked, piece 3 next),
and section **9.34** (the `wq_a`/`wkv` fusion, shipped this session).

**Hamed's stated priority order for the next sessions: Hermes usage first, vision second, speed/performance
third.** This still holds. Both the 2026-09-22 speed session (v34's Job 5) and this session's vision-then-
speed work were explicit one-off requests, asked for and confirmed in chat before starting, not a
reordering of the standing priority — **Job 1 (Hermes) is still first up and still nobody has run it.**

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
has never been checked is Hermes's own client against it — three sessions running now, this is still the
top of the list.** Start the server, point Hermes Agent Desktop at `http://127.0.0.1:8011/v1`, model id
`deepseek-v4.1-flash`, any placeholder API key, and have a real conversation, ideally one that uses tools.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

Watch for: a `content` shape `ChatCompletionRequest` does not model (a list of parts rather than a plain
string — this is also the shape an image would arrive in, now relevant since vision piece 1 is checked);
a `tool_choice` or `response_format` value the server ignores; Hermes retrying or opening a second
conversation while one is still generating, which queues rather than running concurrently (`engine.py`'s
single-flight lock, HANDOFF §15). Report back what broke, with the request body if possible.

## Job 2 — vision, phase 1 piece 3: the splice into prefill. Piece 1+2 are checked; this is now unblocked.

HANDOFF §16 has the four-piece plan, §16.1 has this session's checkpoint. **Piece 1 (ViT+Aligner,
`src/cachalot/model/vision_mlx.py`) and piece 2 (image preprocessing, `src/cachalot/model/image_processor_mlx.py`)
are ported and numerically checked** against the official PyTorch reference on a real image: bit-identical
preprocessing, FP32 forward diff at machine precision (`6.7e-6`), BF16 diff explained as rounding noise
(`3.8e-2` max, `6.9e-4` mean, over 32 layers — re-run in FP32 to confirm before treating any future BF16
diff on this port as a bug, not noise).

Piece 3 is real engineering, not a spike, and the part HANDOFF called "the trickiest piece
architecturally":

1. **`merge_image_embeddings`** — write the aligner's output rows into the embedded sequence at
   `image_token_id` positions before layer 0, in reading order. `vision_embed()` in `vision_mlx.py` already
   returns exactly those rows, in reading order, for one image.
2. **Per-token `bias_vl`** — the router kernel (`router_fused_metal.py`, `router_mlx.py`) currently takes
   one `bias` array applied uniformly to a whole batch. Selecting between `bias` and `bias_vl` per token by
   `image_mask`, for the 43 MoE layers that carry a `bias_vl`, is a real change to that kernel's signature,
   not a parameter swap.
3. Thread `image_mask` through `TextDecodeRuntime`'s prefill path to reach both (1) and (2).

**Do not start piece 4 (the server's image-content parsing in `ChatCompletionRequest`) until piece 3 has a
first working splice** — piece 4 can build in parallel once piece 3 exists, per HANDOFF §16, but there is no
image-bearing prompt to test piece 4 against before piece 3 can produce one.

Needs `torch` and `pillow` in the venv for the parity check to still run (`~/venvs/deepseek-v41/bin/pip
install torch pillow`, done this session); `pillow` is a real runtime dependency now, declared as the
`vision` extra in `pyproject.toml`.

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

## Job 5 — what is left on the CPU/GPU side, now smaller

- **`wq_a`/`wkv` fusion shipped, 2026-09-22** (§9.34). 0.35 ms/token, bit-identical, no kill switch. Of the
  FP8 GEMV family's 3.48 ms gap against its `mx.sum` ceiling (§7.1.10), 0.43 ms sat in exactly these two
  occupancy-bound shapes; the fusion recovers 0.35 of it (the fused kernel's own ceiling on the combined
  1792-row buffer is slightly different from the sum of the two separate ceilings, hence 0.35 not 0.43).
- **The remaining ~3.1 ms of the FP8 GEMV gap** (`wq_b`, `wo_b`, shared `w1`/`w3`/`w2` — the shapes that are
  NOT occupancy-bound): HANDOFF §7.1.10 still says "no obvious way to take it" and the lanes-per-row axis is
  separately closed (§9.29). Nobody has found a second structural fusion candidate here the way `wq_a`/`wkv`
  and `w1`/`w3` were found — worth one session of looking for another same-activation, independent-output
  pair before calling this fully closed, but do not re-run the lanes-per-row sweep again, it is answered.
- The 3.48 ms figure itself should be re-measured against the post-fusion baseline before being quoted
  again — it was measured before both the shared-expert and `wq_a`/`wkv` fusions shipped.

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
- **Vision piece 1 (ViT+Aligner) and piece 2 (image preprocessing)** — ported and numerically checked
  against the official reference. §16.1. Piece 3 (prefill splice) and piece 4 (server plumbing) are not
  started; do not re-check piece 1/2's numerics again without a reason, they are proven to machine precision
  in FP32.

## Rules that still hold

- **A background thread's exception can die silently while the parent exits 0.** §7.1.11, §12.
- **Read which function an instrument calls before ranking a lever off it.**
- **`mlx_peak_bytes` is bytes; divide it by 2^30, and check the units on both sides of a comparison.**
- **Two arms must be launched the same way.**
- **The drive is shared, so measure I/O-touching code while it is busy.**
- **Check an offline simulation against a live run the first time one is possible.**
- **A live session resolves its hit rate, and a change of 20 ms or more; it cannot resolve 5 ms.**
- **Compile a live coding turn before calling it reference class.**
- **Never drop a case from a denominator.**
- **A duplicate file is not a second reading.**
- **`guarded_run.sh` needs `budget + 29` GiB available; a *chat* session is not bound by that guard.** Never
  pass `--force`.
- **Hand over `./chat.sh` or `./serve.sh`, never a one-liner.**
- **A same-activation, independent-output pair of GEMVs is a fusion candidate; check for one before ranking
  a kernel's per-shape throughput as fixed.** New this session, from `wq_a`/`wkv` following the same shape
  as the shared expert's `w1`/`w3`.
- **A CPU-only PyTorch reference forward over thousands of dense-bidirectional-attention tokens does not
  finish in reasonable time — downsize the test image, not the tolerance.** New this session.
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
| `benchmarks/vision_parity_check.py [image]` | MLX vision port vs official PyTorch reference, new 2026-09-22 | ~2 min on a downsized image |
| `benchmarks/micro_qkv_fusion_roofline.py` | `wq_a`/`wkv` fused vs shipped vs `mx.sum` ceiling, new 2026-09-22 | ~10 s |
| `benchmarks/qkv_fusion_live_smoke.py` | live sanity check on the real bank, new 2026-09-22 | ~1 min |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | the official implementation |
| `src/cachalot/server/app.py`, `engine.py` | the OpenAI-compatible server, §15 |
| `src/cachalot/model/vision_mlx.py`, `image_processor_mlx.py` | vision piece 1+2, checked, unwired. §16.1 |
| `src/cachalot/model/resident_trunk.py:39-51` | where vision tensors are still filtered out today |
| `src/cachalot/model/attention_qkv_fusion.py` | the `wq_a`/`wkv` fusion, §9.34 |
| HANDOFF section 16, 16.1 | vision: scoped, piece 1+2 checked |
| HANDOFF section 15 | the Hermes server: what's fixed, what's not checked |
| HANDOFF section 9.34 | the `wq_a`/`wkv` fusion |
| HANDOFF section 7.1.10 | the FP8 GEMV family's roofline, before this session's fusion |
| `./chat.sh` | the command to hand Hamed for chat — `--expert-budget-gib 52`, `CACHALOT_MLX_WIRED_LIMIT_GIB=80` |
| `./serve.sh` | the command to hand Hamed for the server — same config, port 8011 |
