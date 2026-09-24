# Next-session prompt — **v41**, written 2026-09-24

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v41** | 2026-09-24 | Hamed ran the Desktop manual test | **0.12.1-0.12.2: the Desktop test found three bugs, all fixed. The start guard matched the test's own `tee`. Retries after a cancelled request failed with MLX's "no Stream in current thread", so all generation now runs on one thread. Hermes cut a 917 s prefill at its 900 s local limit, so the server now sends empty-delta chunks while it prefills. Hermes's 22k-token system prompt flipped a tool description mid-session; declaring `supports_vision: true` fixed it. Retest: every check correct, a new chat reuses the system block in ~4 s, and images go to Cachalot natively.** §15.6 |
| v40 | 2026-09-23 | Hamed asked for integrations, features, a measured speed lever, and manual Desktop test steps | **0.12.0: the server reuses the model's own reply when Hermes sends it back re-serialized (tool arguments in another key order, an empty thinking block as a space): −11 % warm prefill tokens on a 10-turn session, all of a long `write_file` body. Images in history are not re-encoded. A 10-turn Hermes session to 26k context and a two-image vision conversation ran correctly. Decode speed does not depend on context length (54 vs 16k tokens, clean A/B). Decode alternates between ~8 and ~4 tok/s windows at equal misses/token, and the slow window is not the drive, the GPU clock or context: now Job 1.** §15.5 |
| v39 | 2026-09-23 | same | System-prompt snapshot at the message boundary (1.09 s), kept on disk across restarts (3.31 s). §15.4 |
| v38 | 2026-09-23 | same | Hermes CLI end to end, chunked prefill, 6.5x smaller snapshots, vision piece 4. §15.1-15.3, §16.4 |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read this first.** `docs/HANDOFF.md`'s "Start here (2026-09-24, 0.12.2)" block, sections **15.6** and
**15.5**, then 15.4 and 16.4.

**Hamed's standing priority order: Hermes usage first, vision second, speed/performance third.**

**Check the working tree before starting** (`git status --short`). 0.12.2 is pushed; if anything is
uncommitted, confirm with Hamed before committing, and bump the version with any push (release standards).

**The shipped configurations:**

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

## Job 1 — the slow-decode window. Measured-first, the biggest speed lever left.

Decode on the same server, at the same 20-34 misses/token, runs at ~8 tok/s in some stretches and 3.7-5.0
tok/s in others (§15.5: all of 17:22-18:00, 18:25-18:36 recovering without a restart; §15.6: Hamed's whole
Desktop retest ran at 3.8-5.0). The clean benchmark arms of §15.5 ran at 6.2-9.2. A cold 13.7k prefill took
598 s in a slow window and 163-190 s outside one. Hamed's real use, with the Desktop app and other apps open,
may simply *be* the slow window, which makes this the number that matters. Ruled out during a slow window: the
drive (5.0 GB/s single-stream raw reads), GPU compute (15.3 TFLOP/s bf16 beside the server), thermal warnings
(`pmset -g therm` empty), context length (§15.5 A/B), server overhead. Suspects, in order: the process's wired
working set going partly non-resident (measure it with `footprint -p PID` or `vm_stat` wired pages; process
RSS is meaningless for wired Metal buffers, it fell to 8.5 GB in a normal prefill), page-cache size
(file-backed pages 9.6-12.6 GB), CPU contention from other apps (the Codex service sat at ~60 % CPU for hours).
Steps:

1. Run `./serve.sh` and a Hermes session with a 10-second sampler beside it: `footprint -p PID`
   (phys_footprint), `vm_stat` wired / file-backed / compressor / pageins, swap, and the top CPU consumers.
   The `[request]` line's `miss/tok` and tok/s time-stamp each request.
2. A/B the obvious external factor back to back inside one window: the same `decode_vs_context.sh 0` arm
   with the Hermes Desktop app (and ChatGPT/Codex) open, then quit, then open again.
3. When a slow window appears, `sudo powermetrics --samplers cpu_power,gpu_power,disk -i 1000` for a minute
   (needs Hamed).
4. Only then pick a fix. If it is residency, the idle-heartbeat precedent (runtime facts: macOS un-wires an
   idle Metal working set within ~6 s) says where to look.

## Job 2 — the long Desktop session. Needs Hamed.

Hamed's retest (§15.6) covered short chats, tools and images; the long session was skipped. Still unexercised
through Desktop: 20+ turns, context growing past 30-50k, and a compression. `docs/manual-tests/hermes-desktop.md`
section 5 is the script; `threshold_tokens: 30000` under `compression:` brings compression within reach.

## Job 3 — Hermes compression: the one cold re-prefill after it. Measured, lever not built.

Done live via the CLI (§15.5): Hermes compresses at ≥ 75 % of any window under 512k tokens (~49-56k here);
`threshold_tokens` lowers that. The summary costs 4-6 minutes (1,595-1,774 tokens decoded) and once ran past
Hermes's 300 s budget. Compression also adds a `skill_manage` tool at position 13 of 25, so the system block
changes ~9k tokens in and the next request prefills it cold (254 s at 14.8k; ~7 minutes at Hamed's 22k). The
lever: also pin the chunk-boundary snapshots inside a system block (4,096, 8,192, …), since they survive the
insertion. That saves ~100 s (CLI) to ~5 minutes (Desktop) once per compressed session. Build it if Job 2 shows
compression happens in real use.

## Job 4 — vision: the ablation of §16.4's three fixes.

Add debug switches for the two numerical fixes of §16.4, the learned delimiter rows and the Engram image mask,
plus bias_vl (image identity in the prefix cache is a correctness guard, not an ablation candidate). Run a
handful of images with verifiable content with each switch off in turn and score the answers. The two-image
conversation of §15.5 and Hamed's Post-profile screenshot of §15.6 are ready-made test cases.

## Job 5 — batched prefill sits 0.014-0.02 mean KL from token-by-token decode. Unchanged.

`benchmarks/prefill_chunk_quality.py 1536:seq,whole --offset N` at several offsets, 8+ samples.

## Job 6 — the 54 GiB collapse, and Job 7 — the Objective-C corpus. Unchanged, low priority.

The 54 GiB collapse may be the same phenomenon as Job 1; run Job 1 first.

## What is closed, so nobody spends a session there

- **Hermes Agent Desktop, short chats, tools and images** (§15.6): works on 0.12.2 with the documented config.
  Known Hermes-side one-time costs, not server bugs: the first request per Hermes version, profile and working
  directory (~9 minutes cold at 22k tokens); the second turn of an image chat (Hermes rewrites the
  `[Image attached at: …]` line, ~1-3k tokens re-prefilled); compression (Job 3).

- **Decode speed vs context length** (§15.5): 54 and 16,054 tokens decode at the same speed. Not a lever.
- **Reply reuse across agent turns** (§15.5): the splice covers Hermes's re-serializations. If `reused`
  ever stops at the previous `prompt` on a turn after a reply, the request dump now holds the reply's token
  ids; find the first differing token before touching anything.
- Everything v39 listed as closed still is.

## Rules that still hold, and three new ones

- **MLX 0.32 arrays belong to the thread that built them until evaluated.** Evaluating an unevaluated array
  from another thread raises "There is no Stream(gpu, N) in current thread" (§15.6). All server generation runs
  on one thread (`generate_pool`); any new background thread that touches runtime arrays must evaluate what it
  builds before handing it over. New.

- **A live speed number is only comparable inside one quiet window.** §15.5's first session and the
  benchmarks run an hour later differ by 2x at the same misses/token. Take an A/B's arms back to back, and
  read `miss/tok` beside tok/s. New.
- **Hermes changes under you.** It auto-updates; the September 23 update switched it to thinking mode and
  reworded its system prompt. Diff the dumped system prompt against the snapshot's tokens before assuming a
  cache bug. New.
- `serve.sh`'s guard runs `pgrep -f "...|cachalot"`: launch it from a command line that says nothing but
  `./serve.sh`.
- `pgrep -f PATTERN` matches its own shell; wait on a PID.
- Quote heredoc delimiters (`<<'EOF'`) when the payload has backticks.
- Arms of an A/B in one process share a warmed cache: alternate order, separate processes.
- Two arms must be launched the same way; never diff numbers across two measurement scripts.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `./serve.sh` + `CACHALOT_SERVER_DUMP=path` | per-request `[request]` line with `reused`, `spliced`, `miss/tok`, `hit`; request bodies and reply token ids as JSON lines | — |
| `HERMES_HOME=<dir> hermes chat -Q -q ...` / `--resume SID` | Hermes against the server without touching Hamed's config | minutes per turn |
| `benchmarks/decode_vs_context.sh N` | decode tok/s and misses/token behind N filler tokens, shipped config, new | ~5 min per arm at 16k |
| `docs/manual-tests/hermes-desktop.md` | Hamed's Desktop test script, new | ~1 h |
| `benchmarks/prefix_snapshot_exactness.py N...` | snapshot → restore bit-identity, memory and disk | ~5 min |
| `benchmarks/prefill_chunk_quality.py` | NLL + KL of teacher-forced continuations | ~5 min |
| `benchmarks/memwatch.sh N` | memory beside a live chat | as long as the chat |
| `benchmarks/decode_anatomy.py` | where a live token's time goes | ~1 min |
