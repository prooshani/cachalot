# Next-session prompt — **v40**, written 2026-09-23

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v40** | 2026-09-23 | Hamed asked for integrations, features, a measured speed lever, and manual Desktop test steps | **0.12.0: the server reuses the model's own reply when Hermes sends it back re-serialized (tool arguments in another key order, an empty thinking block as a space): −11 % warm prefill tokens on a 10-turn session, all of a long `write_file` body. Images in history are not re-encoded. A 10-turn Hermes session to 26k context and a two-image vision conversation ran correctly. Decode speed does not depend on context length (54 vs 16k tokens, clean A/B). Decode alternates between ~8 and ~4 tok/s windows at equal misses/token, and the slow window is not the drive, the GPU clock or context: now Job 1.** §15.5 |
| v39 | 2026-09-23 | same | System-prompt snapshot at the message boundary (1.09 s), kept on disk across restarts (3.31 s). §15.4 |
| v38 | 2026-09-23 | same | Hermes CLI end to end, chunked prefill, 6.5x smaller snapshots, vision piece 4. §15.1-15.3, §16.4 |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read this first.** `docs/HANDOFF.md`'s "Start here (2026-09-23, 0.12.0)" block and section **15.5**, then
15.4 and 16.4.

**Hamed's standing priority order: Hermes usage first, vision second, speed/performance third.**

**Check the working tree before starting.** The 0.12.0 work may still be uncommitted (`git status --short`:
`engine.py`, `vision_prompt.py`, `snapshot_store.py`, the two new tests, `benchmarks/decode_vs_context.*`,
docs, version files). If so, confirm with Hamed and commit; the bump is already made.

**The shipped configurations:**

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

## Job 1 — the slow-decode window. Measured-first, the biggest speed lever left.

Decode on the same server, at the same 20-30 misses/token, runs at ~8 tok/s for long stretches and at
3.7-4.5 tok/s for others (§15.5: the whole of 17:22-18:00, then 18:25-18:36 recovering without a restart).
A cold 13.7k prefill took 598 s in a slow window and 163-190 s outside one. Already excluded during a slow
window: the drive (5.0 GB/s single-stream raw reads), GPU compute (15.3 TFLOP/s bf16 beside the server),
thermal warnings (`pmset -g therm` empty), context length (§15.5 A/B), server overhead. Suspects, in order:
the process's wired working set going partly non-resident (measure it with `footprint -p PID` or `vm_stat`
wired pages; process RSS is meaningless for wired Metal buffers, it fell to 8.5 GB in a normal prefill),
page-cache size (file-backed pages 9.6-12.6 GB), background
load from other apps. Steps:

1. Run `./serve.sh` and a Hermes long session (`docs/manual-tests/hermes-desktop.md` or the CLI loop of
   §15.5) with a 10-second sampler beside it: `footprint -p PID` (phys_footprint), `vm_stat`
   wired / file-backed / compressor / pageins, swap, and top CPU consumers. The `[request]` line's
   `miss/tok` and tok/s time-stamp each request.
2. When a slow window appears, `sudo powermetrics --samplers cpu_power,gpu_power,disk -i 1000` for a minute
   (needs Hamed), and `benchmarks/decode_anatomy.py`-style per-token split if the server can be paused.
3. Only then pick a fix. If it is residency, the idle-heartbeat precedent (runtime facts: macOS un-wires an
   idle Metal working set within ~6 s) says where to look.

## Job 2 — Hamed's own Hermes Agent Desktop session. Needs Hamed.

`docs/manual-tests/hermes-desktop.md` is the script: config, a short check, images, a long session with what
to watch, and what to send back. The CLI path is fully covered (§15.1, §15.5); what only Hamed can check is
the Desktop app's own UI flows, and his real `~/.hermes/config.yaml` (compression with
`provider: auto`, which may send the summary to Cachalot as a full prefill).

## Job 3 — Hermes compression: the one cold re-prefill after it. Measured, lever not built.

Done live (§15.5): Hermes compresses at ≥ 75 % of any window under 512k (~49-56k here), `threshold_tokens`
lowers it. The summary costs 4-6 minutes (1,595-1,774 tokens decoded) and once ran past Hermes's 300 s
budget. Compression adds a `skill_manage` tool at position 13 of 25, so the system block changes ~9k tokens
in and the next request prefills it cold (254 s). Lever: also pin the chunk-boundary snapshots inside a
system block (4,096, 8,192), which survive the insertion: ~100 s saved once per compressed session. Build it
only if Hamed's Desktop session shows compression is common.

## Job 4 — vision: the ablation of §16.4's three fixes.

Add three debug switches (delimiter rows, the Engram image mask, image identity in the prefix cache is not
an ablation candidate, bias_vl), run a handful of verifiable images with each off in turn, score the answers.
The two-image conversation in §15.5 is a ready-made test case.

## Job 5 — batched prefill sits 0.014-0.02 mean KL from token-by-token decode. Unchanged.

`benchmarks/prefill_chunk_quality.py 1536:seq,whole --offset N` at several offsets, 8+ samples.

## Job 6 — the 54 GiB collapse, and Job 7 — the Objective-C corpus. Unchanged, low priority.

The 54 GiB collapse may be the same phenomenon as Job 1; run Job 1 first.

## What is closed, so nobody spends a session there

- **Decode speed vs context length** (§15.5): 54 and 16,054 tokens decode at the same speed. Not a lever.
- **Reply reuse across agent turns** (§15.5): the splice covers Hermes's re-serializations. If `reused`
  ever stops at the previous `prompt` on a turn after a reply, the request dump now holds the reply's token
  ids; find the first differing token before touching anything.
- Everything v39 listed as closed still is.

## Rules that still hold, and two new ones

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
