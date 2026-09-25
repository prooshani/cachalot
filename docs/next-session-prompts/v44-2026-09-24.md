# Next-session prompt — **v44**, written 2026-09-24

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v44** | 2026-09-24 | Hamed ran the long Desktop session (v43 Job 1) with parallel subagents | **0.15.0: the prefix cache evicts by tier under a 1.5 GiB byte budget; 0.13.0's in-block pins had crowded out every parallel subagent's previous turn (replay: 479,899 to 256,597 prefilled tokens, ~46 min). `serve.sh` defaults to 8,192 new tokens (2,000 had truncated two summaries and a `delegate_task` call). The sampler showed macOS un-wiring the model inside every prefill chunk: the chunk waited on its Engram rows with the GPU idle; a keep-alive eval while waiting makes a 12k prefill 15 % faster, bit-identical.** §15.10, §15.11 |
| v43 | 2026-09-24 | same | 0.14.0: shared-memory Metal fences; snapshots keyed on numerics. §15.9 |
| v42 | 2026-09-24 | same | 0.13.0-0.13.1: slow window is compute; visible Hermes window is the main trigger. §15.7, §15.8 |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read this first.** `docs/HANDOFF.md`'s "Start here (2026-09-24, 0.15.0)" block and sections **15.10-15.11**,
then 15.9 and 15.7.

**Hamed's standing priority order: Hermes usage first, vision second, speed/performance third.**

**Check the working tree before starting** (`git status --short`). If 0.15.0 is uncommitted, confirm with
Hamed and commit; the bump is already made. `docs/translations/` is Hermes's working output from the
subagent session, not ours: ask Hamed before deleting or ignoring it.

**The shipped configurations:**

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

## Job 1 — the disk snapshot store under subagents. Autonomous.

§15.10 finding 4: every subagent's system block is persisted, and `snapshot_store.save(keep=8)` prunes by
mtime, so one batch of 7+ subagents evicts the main agent's block and the next restart pays a cold ~22k
prefill (~4.5 min). Subagent blocks are unique per task and never match again. Options: persist a block on
its second use (a `find` that returns a boundary snapshot), or keep a hit marker and prune never-hit files
first. Measure with `prefix_pin_replay.py` extended to simulate restarts (drop memory, reload the disk set)
on `/tmp/cachalot-requests.jsonl` (if gone, the session's requests are rows 130-204 of Hamed's dump; ask).

## Job 2 — the live read of 0.15.0. Needs Hamed.

Same shape as v43 Job 1 (`docs/manual-tests/hermes-desktop.md` §5, sampler beside it), and ask for a task
that dispatches `delegate_task` subagents again. Read back: every subagent turn after its first should show
`reused` near its previous `prompt + completion`, not 4,096; the main agent's first turn after a dispatch
should reuse its block or more; no `finish=length` at `completion=2000`; in the sampler CSV, no rows with
`wired_gib` under 60 during a request (§15.11). Keep Hermes's reasoning effort as it
was in the last session (§15.10 finding 3).

## Job 2b — hide the Engram wait itself. Autonomous, measured-first.

§15.11: the keep-alive stops the un-wire, but layer 1 of every 4,096-token chunk still waits ~10 s for ~100k
Engram rows (~20 keep-alive evals per chunk). Issue the *next* chunk's Engram reads while the current chunk
runs: `prepare_prompt` knows every chunk, but `engram_hash.push` is sequential, so compute the next chunk's hash
rows on a copy of the hash state. Measure with `benchmarks/prefill_unwire_timeline.py 12288`, ABBA, bit-identity
from its RESULT line. Expected: up to ~10 s per chunk, ~10 % of a long prefill.

## Job 3 — subagent first turns: ~15.6k tokens each, 190 s. Design only first.

The six subagent blocks share ~5.7k tokens with each other and then diverge at Hermes's per-task CONTEXT
section, after which ~14k tokens of tool schemas are identical again but unreachable to a prefix cache. Two
paths, both outside the server's control or large: ask Hermes to put per-task context after the tools (a
Hermes config or upstream change: check whether `delegate_task` has a template option), or position-independent
reuse of the tool-schema span (not possible with causal KV; do not start it). Write up what Hermes allows.

## Job 4 — compression inside Hermes's budget.

§15.10: one compression timed out queued behind subagents; two were truncated at 2,000 (fixed). With 8,192 a
summary may now run past Hermes's 300 s. Measure one summary on 0.15.0 (prompt size, decode time); if it
cannot fit, document pointing Hermes's auxiliary `compression` provider at a hosted model as the shipped
advice in the manual test and README.

## Job 5 — the rest of the slow-window trigger, and the MLX buffer knobs. Unchanged from v43 Jobs 2-3.

`sudo powermetrics --samplers gpu_power,cpu_power -i 1000 -n 30` once slow, once fast; wallpaper, iStatistica,
display scaling one at a time inside a slow window. Then `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER`
at 2x and 4x, ABBA, only in a slow window, `decode_fingerprint.py` for bit-identity. Low urgency: this
session's decode never entered a slow window with the Hermes window hidden.

## Job 6 — vision regression on Desktop-sized screenshots; Job 7 — prefill KL; Job 8 — 54 GiB collapse, Objective-C corpus. Unchanged from v43.

## What is closed, so nobody spends a session there

- **Prefix-cache eviction order and size** (§15.10): replayed eight policies; tiered + byte budget shipped;
  flat from 1.3 to 3 GB. Do not re-tune the budget without a new dump that disagrees.
- **Server default `max_tokens`** (§15.10): 8,192, clamped to `max_seq_len`.
- **The wired-memory collapses in the sampler** (§15.11): prefill's Engram wait; fixed by the keep-alive. A
  sampler row with wired under 60 GiB on 0.15.0 is new information.
- Everything v43 listed as closed still is (`MLX_METAL_FAST_SYNCH`, vision fix ablation, the slow window is
  not the drive / page cache / context / dispatch / clocks).

## Rules that still hold, and one new one

- **Replay a dump before touching the prefix cache.** `benchmarks/prefix_pin_replay.py DUMP` reproduced the
  live `reused` column exactly; every eviction change is judged on total prefilled tokens there. New.
- **`snapshot_store.NUMERICS_VERSION` moves with numerics, not with releases** (eviction changes do not bump it).
- **A slow-window A/B needs a slow window**; separate processes, interleaved; never two runtimes at once.
- **MLX 0.32 arrays belong to the thread that built them until evaluated** (§15.6).
- **Hermes changes under you.** Diff the dumped system prompt (and `reasoning_effort`) before assuming a cache bug.
- `serve.sh`'s guard runs `pgrep -f "deepseek-v41/bin/python|cachalot\.cli"`: launch it from a command line
  that contains neither.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `./serve.sh` + `CACHALOT_SERVER_DUMP=path` | per-request `[request]` line; request bodies and reply token ids | — |
| `benchmarks/prefill_unwire_timeline.py N` | per-chunk prefill seconds against wired memory; bit-identity line | ~4 min at 12k |
| `benchmarks/prefix_pin_replay.py DUMP [--gib G]` | prefilled tokens per request under the real `PrefixCache`, offline | ~1 min |
| `benchmarks/slow_window_sampler.py` | memory, swap, GPU %, display, server reads every 10 s | — |
| `benchmarks/decode_anatomy.py` | expert wait against `rest` per token; the slow-window discriminator | ~45 s |
| `benchmarks/decode_fingerprint.py --decode-tokens 24` | bit-identity of a scheduling change | ~1 min per arm |
| `benchmarks/vision_ablation.sh ARM [--cases ...]` | verifiable image cases per ablation arm | ~1-3 min per arm |
| `docs/manual-tests/hermes-desktop.md` | Hamed's Desktop test script | ~1-2 h |
