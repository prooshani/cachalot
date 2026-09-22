# Next-session prompt — **v33**, written 2026-09-22

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v33** | 2026-09-22 | Hamed reprioritized: Hermes, then vision, then back to speed | **The server is more done than tracked and its one real bug (a stale 0.2 frequency-penalty default) is fixed and smoke-tested against the shipped config, including a tool-calling round trip. Vision is scoped: the checkpoint carries the real 263-tensor ViT + aligner, `resident_trunk.py` filters it out by name today, and there is a four-piece port plan.** The 54 GiB budget question moved too: six budget arms sampled with `memwatch.sh` show no OS-level memory-pressure event anywhere, including the one slow session, which refutes the physical-memory-cliff hypothesis and points at decode duration or thermal state instead. §15, §16, §7.2.10 |
| v32 | 2026-09-22 | Hamed ran Job 1 and it failed | 54 GiB decoded at 4.9-6.2 tok/s against 9.4-9.6 at 52, 60 GiB was unusable, and a guarded replay of the same four prompts at 50 GiB ran at 9.58 and 8.52 with pressure normal. New tool `benchmarks/memwatch.sh`. §7.2.9 |
| v31 | 2026-09-22 | Job 3 was taken | the coding gate now contains Objective-C and runs it; no model scored on it yet |
| v30 | 2026-09-22 | Job 1 was answered | the miss is at the drive's rated wall; the budget is the only lever on it |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read this first.** `docs/HANDOFF.md` section **15** (the Hermes server, what is fixed and what is not
checked), section **16** (vision, scoped), and section **7.2.10** (the six-arm budget sweep and why the
memory-cliff hypothesis is refuted). Then section 7.1.11 and 9.31 for why the miss itself has no lever left.

**Hamed's stated priority order for the next sessions: Hermes usage first, vision second, speed/performance
third.** Do not reorder this without asking — it is an explicit instruction, not a measured ranking.

**The shipped chat configuration, unchanged:**

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

**The server, new this session:**

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

## Job 1 — get Hermes Agent Desktop talking to Cachalot. Needs Hamed's own session

The server itself is implemented, its one known bug (a stale 0.2 frequency-penalty default, HANDOFF §15) is
fixed, and a curl smoke test passed: models, streaming, non-streaming, and a tool-calling round trip. **What
has never been checked is Hermes's own client against it.** Start the server, point Hermes Agent Desktop at
`http://127.0.0.1:8011/v1`, model id `deepseek-v4.1-flash`, any placeholder API key (none is required unless
`--api-key` is passed), and have a real conversation, ideally one that uses tools.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

Watch for: a `content` shape `ChatCompletionRequest` does not model (a list of parts rather than a plain
string — this is also the shape an image would arrive in, relevant to Job 2); a `tool_choice` or
`response_format` value the server ignores rather than honours; Hermes retrying or opening a second
conversation while one is still generating, which queues rather than running concurrently
(`engine.py`'s single-flight lock, HANDOFF §15) and may read as a hang rather than a queue to a client that
does not expect it. Report back what broke, with the request body if possible — that is what turns this from
a scoping note into a fix.

## Job 2 — vision, phase 1: the feasibility spike, no model wiring yet

HANDOFF §16 has the full four-piece plan. **Do only the first piece next**, and do not start piece 3 (the
splice into the text model's prefill) until piece 1 is numerically checked — the residual-mix defect that
cost this project three sessions (§7.4.8) was exactly this kind of untested wiring.

1. Port the ViT and `Aligner` from `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/vision.py` (114
   lines) to MLX: patch embed, 2D-RoPE bidirectional attention, SwiGLU MLP, final RMSNorm, then the aligner's
   space-to-depth downsample and two-layer projection into the 5120-dim text embedding space. Load the real
   263 `vision.*`/`aligner.*` tensors from `model-00001-of-00048.safetensors` (no extra download).
2. Port `image_processor.py`'s resize-ratio solver and patchify (pure NumPy/PIL, no PyTorch dependency in
   the parts that matter).
3. **Check it, don't wire it yet.** Run one real image through both the ported MLX path and the reference
   PyTorch `vision.py` on the same patches, and diff the aligner's output. This is the checkpoint before any
   of it touches `TextDecodeRuntime`.

Do not start the router's per-token `bias_vl` selection (piece 3 of §16) or the server's image-content
parsing (piece 4) this session unless piece 1 and 2 are done and checked — they are real engineering, not a
spike, and should not be built on an unverified ViT port.

## Job 3 — the 54 GiB collapse, re-tested with the right variable this time

Section 7.2.10 ruled out an OS-visible memory-pressure event at every budget from 46 to 56 GiB, including
the one 54 GiB session that collapsed on its Objective-C turn (4.83 tok/s against 8.4-8.8 at 48-52). The
collapse was specific to that one long turn (1565 tokens, ~325 s of continuous decode), not the session, so
the next check is duration and thermal state, not memory:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/memwatch.sh 54
```

then in the first terminal `CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 54` with the same
four prompts as `benchmarks/sessions/hamed_2026-09-22.txt`. If it reproduces, run `sudo powermetrics
--samplers cpu_power,gpu_power -i 1000` alongside a repeat to check for thermal throttling. If it does not
reproduce, the one reading was a one-off interference and 54 GiB should be re-screened clean before it is
treated as anything but 52 GiB's equal. Low priority under Hamed's reordering — do this after Jobs 1 and 2,
or when a live session is free and Jobs 1-2 are blocked on something else.

## Job 4 — score the banks on the Objective-C cases the gate now has

Unchanged from v31/v32. `benchmarks/coding_quality.py` on the 2-bit bank at the shipped configuration for
the 26-task corpus (6 new Objective-C tasks), a fresh run not `--resume` (the corpus hash changed). About two
hours, no live session needed — can run in the background while Jobs 1-2 happen in the foreground.

## Job 5 — what is left on the CPU/GPU side, and it is small

- `kernel_consts.py:39` appears in the streaming arm at 0.5 calls per token with 1.7 ms of store-blocked
  time behind it, and nobody has explained it.
- The 3.48 ms between the FP8 GEMV kernel and `mx.sum` over its own bytes (§7.1.10).
- The `w1`/`w3` fusion in the shared expert, 0.4 ms per token and bit-identical (§9.27). Not shipped.

## What is closed, so nobody spends a session there

- **The miss's own drive-wall arithmetic**, on both memory pressure and raw throughput. §7.1.11, §9.31.
- **The in-eval excess.** It is the miss. §7.1.9.
- **A physical-memory cliff between 50 and 56 GiB** — refuted by direct sampling, §7.2.10. What remains open
  is duration/thermal, not memory.
- `io_workers` 2-16, `CACHALOT_PAGE_CACHE=0`, attention's shapes, `mx.compile` on attention, quantizing
  `wo_a`, the FP8 GEMV lanes-per-row policy, the shared expert, `INDEX_TOPK`, the expert kernel, routing
  prediction, speculative decoding, the hyper-connection glue, the router, dispatch fusion, 1 bit per weight,
  mirror striping on this bank, the frequency penalty (as a *quality* lever — it is still a live knob, just
  defaulted off everywhere now, §15), eviction policy. §7.1.9-7.1.10, §9.21-9.30, §11.

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
- **A duplicate file is not a second reading.** `Test-52-1.txt` and `test-54.txt` this session were the same
  54 GiB session saved twice; the turn-4 collapse is n=1, not confirmed reproducible. New this session.
- **`guarded_run.sh` needs `budget + 29` GiB available; a *chat* session is not bound by that guard.** Never
  pass `--force`.
- **Hand over `./chat.sh` or `./serve.sh`, never a one-liner.**
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once — this now includes `serve.sh` racing `chat.sh` or a benchmark.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/memwatch.sh N` | **memory beside a live chat, without killing it** — free/available/wired/compressor/swap/pressure/pageouts once a second, summarised on Ctrl-C. New 2026-09-22 | runs as long as the chat |
| `benchmarks/chat_turns.py --turns-file --temperature` | replays a live session's own prompts under `guarded_run.sh`, at the shipped temperature | ~1 min per turn set |
| `benchmarks/expert_read_scaling.py --wire-gib N` | what the drive gives under decode's memory conditions | ~3 min |
| `benchmarks/decode_anatomy.py` | where a live token's time goes by blocking cause | ~1 min |
| `benchmarks/profile_decode_gpu.py` | the GPU side, every layer | ~1 min |
| `benchmarks/decode_fingerprint.py` | whether a speed arm changed the numerics | ~1 min |
| `benchmarks/simulate_policies.py --expert-bytes 9953280` | hit rate against budget, offline | instant |
| `benchmarks/nll_expert_precision.py --experts runtime --tokens 512` | the production quality arm | ~2 min |
| `benchmarks/coding_quality.py --resume` | the 52-case corpus (26 tasks x 2 seeds), now includes Objective-C, Job 4 | ~1.5 h |
| `benchmarks/server_smoke.sh` | curl-based server smoke test (models, streaming, non-streaming, prefix cache) | ~2 min |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | the official implementation, including `vision.py` and `image_processor.py` |
| `src/cachalot/server/app.py`, `engine.py` | the OpenAI-compatible server, §15 |
| `src/cachalot/model/resident_trunk.py:39-51` | where vision tensors are filtered out today |
| `benchmarks/results/guarded/memwatch_*_20260922-1*.csv` | the six-arm budget sweep's memory traces, §7.2.10 |
| `benchmarks/sessions/hamed_2026-09-22.txt` | the four prompts used across the budget sweep |
| HANDOFF section 7.2.10 | **the six-arm budget sweep: no pressure cliff, one turn-4 collapse** |
| HANDOFF section 15 | **the Hermes server: what's fixed, what's not checked** |
| HANDOFF section 16 | **vision: scoped, four-piece plan** |
| HANDOFF section 7.1.11 | Job 1 (speed) answered: the miss is at the drive's rated wall |
| `./chat.sh` | the command to hand Hamed for chat — `--expert-budget-gib 52`, `CACHALOT_MLX_WIRED_LIMIT_GIB=80` |
| `./serve.sh` | the command to hand Hamed for the server — same config, port 8011 |
