# Next-session prompt — **v45**, written 2026-09-24

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v45** | 2026-09-24 | Hamed asked for the v44 levers planned, built, measured and documented | **0.16.0: the disk snapshot store keeps 32 system blocks by last use, preloads 4 and loads the rest on demand (the main agent's 22k block had already been pushed off the disk by subagents). Next-chunk Engram lookahead: bit-identical but no faster on cold rows, shipped off. A local compression summary takes 850 s against Hermes's 120 s: hosted `auxiliary.compression` is the advice. Hermes subagents: moving their CONTEXT to the user turn would save ~15 min per six-subagent batch (a Hermes change).** §15.12 |
| v44 | 2026-09-24 | Hamed's long Desktop session with subagents | 0.15.0: tiered eviction under a byte budget; 8,192 default tokens; prefill keep-alive. §15.10, §15.11 |
| v43 | 2026-09-24 | same | 0.14.0: shared-memory Metal fences; snapshots keyed on numerics. §15.9 |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read this first.** `docs/HANDOFF.md`'s "Start here (2026-09-24, 0.16.0)" block and section **15.12**, then
15.10-15.11.

**Hamed's standing priority order: Hermes usage first, vision second, speed/performance third.**

**Check the working tree before starting** (`git status --short`). If 0.16.0 is uncommitted, confirm with
Hamed and commit; the bump is already made. `docs/translations/` is Hermes's working output from the
2026-09-24 subagent session, not ours: ask Hamed before deleting or ignoring it.

**Note for Hamed's next restart:** the snapshot directory has no main-agent block today (§15.12), so the first
main request after the upgrade pays one cold ~22k prefill; from then on 0.16.0 keeps it.

**The shipped configurations:**

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

## Job 1 — the live read of 0.15.0 + 0.16.0. Needs Hamed.

v44's Job 2, unchanged, plus the store: the Desktop session of `docs/manual-tests/hermes-desktop.md` §5 with the
sampler beside it and a task that dispatches `delegate_task` subagents. Read back: every subagent turn after its
first shows `reused` near its previous `prompt + completion`; the main agent's first turn after a dispatch reuses
its block or more; no `finish=length` at 2,000; no sampler rows with `wired_gib` under 60 during a request; after
a restart, startup prints `N loaded ..., M more on disk` and the main agent's first request reuses its block.
Before the session, ask Hamed to set a hosted `auxiliary.compression` provider (§15.12, Job 4), and keep Hermes's
reasoning effort as in the last session.

## Job 2 — the Hermes subagent change. Needs Hamed's decision.

`docs/hermes-subagent-context.md`: moving each subagent's CONTEXT (and workspace, context files) from the system
prompt to the first user turn makes all subagent blocks identical; replayed, ~76k fewer prefilled tokens per
six-subagent batch. Ask Hamed whether to (a) propose it upstream, (b) carry a local patch in his Hermes checkout,
or (c) leave it. Never edit `~/.hermes` without his yes. If he says yes, test it with `HERMES_HOME` isolation
(memory: Hermes isolated test) and replay the new dump.

## Job 3 — is the drive idle during prefill? Autonomous, measured-first.

§15.12's lookahead null says Engram reads compete with expert streaming during prefill. Confirm or refute with an
`iostat -d -w 1` (or `fs_usage`-free equivalent) trace beside `benchmarks/prefill_unwire_timeline.py 12288`
(fresh `FILLER_FILE`/`FILLER_OFFSET` text): per-layer drive MB/s against the chunk timeline. If the drive is idle
for a stretch of layers 15-39, start the lookahead there instead of at layer 14 and re-run the 8-arm A/B; if it is
saturated, prefill is read-bound and the lever is closed for good.

## Job 4 — the rest of the slow-window trigger and the MLX buffer knobs; Job 5 — vision on Desktop-sized screenshots; Job 6 — prefill KL; Job 7 — 54 GiB collapse, Objective-C corpus. Unchanged from v44 (Jobs 5-8 there).

## What is closed, so nobody spends a session there

- **Next-chunk Engram lookahead** (§15.12): bit-identical, removes 2/3 of the wait, no faster on cold rows.
  Off by default; reopen only on Job 3's evidence.
- **Local compression inside Hermes's timeout** (§15.12): 850 s against 120 s. Hosted provider is the advice.
- **Disk store "proven first" ranking** (§15.12): worse than last-use ranking on the replay (a session's main
  block is used by one conversation only).
- **Server-side reuse of the subagent tool-schema span**: needs position-independent KV; not possible.
- Everything v44 listed as closed still is (prefix-cache eviction and budget, `max_tokens` default, the
  wired-memory collapses, `MLX_METAL_FAST_SYNCH`, the vision fix ablation, the slow-window non-causes).

## Rules that still hold, and one new one

- **Replay a dump before touching a cache**, now including restarts: `benchmarks/snapshot_store_replay.py`.
- **A prefill A/B needs fresh text per arm** (`FILLER_FILE`/`FILLER_OFFSET`): the same filler twice warms the
  page cache for its Engram rows and experts and fakes a win (§15.12: -22 % warm, null cold). New.
- **`snapshot_store.NUMERICS_VERSION` moves with numerics, not with releases.**
- **A slow-window A/B needs a slow window**; separate processes, interleaved; never two runtimes at once.
- **MLX 0.32 arrays belong to the thread that built them until evaluated** (§15.6).
- **Hermes changes under you.** Diff the dumped system prompt (and `reasoning_effort`) before assuming a cache
  bug; the codebase-memory index of `~/.hermes/hermes-agent` can lag the checkout (§15.12).
- `serve.sh`'s guard runs `pgrep -f "deepseek-v41/bin/python|cachalot\.cli"`: launch it from a command line
  that contains neither.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `./serve.sh` + `CACHALOT_SERVER_DUMP=path` | per-request `[request]` line; request bodies and reply token ids | — |
| `benchmarks/snapshot_store_replay.py DUMP [--then-subagents N]` | prefilled tokens with restarts under the disk store; restart penalty | ~15 s |
| `benchmarks/prefix_pin_replay.py DUMP [--gib G]` | prefilled tokens per request under the real `PrefixCache`, offline | ~1 min |
| `benchmarks/prefill_unwire_timeline.py N` (+ `FILLER_FILE`, `FILLER_OFFSET`) | per-chunk prefill seconds, wired memory, keep-alives, lookahead hits, bit-identity | ~4 min at 12k |
| `benchmarks/slow_window_sampler.py` | memory, swap, GPU %, display, server reads every 10 s | — |
| `benchmarks/decode_anatomy.py` | expert wait against `rest` per token | ~45 s |
| `benchmarks/decode_fingerprint.py --decode-tokens 24` | bit-identity of a scheduling change | ~1 min per arm |
| `benchmarks/vision_ablation.sh ARM [--cases ...]` | verifiable image cases per ablation arm | ~1-3 min per arm |
| `docs/manual-tests/hermes-desktop.md` | Hamed's Desktop test script | ~1-2 h |
