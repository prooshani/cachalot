# Next-session prompt — **v42**, written 2026-09-24

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v42** | 2026-09-24 | Hamed asked for the remaining levers planned, built, measured and documented | **0.13.0: the slow-decode window is compute, not reads: `rest` doubles (97-121 to 203-208 ms/token) at identical misses, read latency and page-cache share, and it only ever appeared with the display on; display-on alone is not sufficient and the trigger is not named. `serve`/`chat` now run with the focused-app Darwin role (+6 to +40 % in every slow-window pair, null elsewhere). Every expert read is timed (`read=`/`fast=` on `[request]`), and `slow_window_sampler.py` watches the machine. In-block chunk snapshots are pinned (v41 Job 3). Vision ablation done (v41 Job 4): only the Engram image mask is load-bearing.** §15.7 |
| v41 | 2026-09-24 | Hamed ran the Desktop manual test | 0.12.1-0.12.2: three Desktop bugs fixed; retest correct. §15.6 |
| v40 | 2026-09-23 | same | 0.12.0: reply splice, long session, decode vs context. §15.5 |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read this first.** `docs/HANDOFF.md`'s "Start here (2026-09-24, 0.13.0)" block and section **15.7**, then
15.6 and 15.5.

**Hamed's standing priority order: Hermes usage first, vision second, speed/performance third.**

**Check the working tree before starting** (`git status --short`). If 0.13.0 is still uncommitted, confirm
with Hamed and commit; the bump is already made.

**The shipped configurations:**

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

## Job 1 — the long Desktop session, with the sampler beside it. Needs Hamed.

This is v41's Job 2 and now also the best chance to catch the slow window in real use. `docs/manual-tests/
hermes-desktop.md` section 5 is the script (20+ turns, past 30-50k context, `threshold_tokens: 30000` under
`compression:` to reach a compression). In a second terminal, before the first request:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && /usr/bin/python3 benchmarks/slow_window_sampler.py
```

What to read back: every `[request]` line (tok/s, `miss/tok`, `read=`, `fast=`), the sampler CSV in
`benchmarks/results/slow_window/`, and after the compression the `reused` of the next main request. With
0.13.0 it should be a 4,096-multiple near where the block changed, not 0 (§15.7, in-block pins).

## Job 2 — the rest of the slow-window trigger. Needs Hamed at the desk.

**Main trigger found (§15.8): the visible Hermes Desktop window.** Hidden, decode went from 4.2-5.2 to
5.5-6.7 tok/s at identical reads, but not to the 7.2-8.6 of a fast window. What follows is for the rest.

Known (§15.7): the window doubles `rest` only; reads, misses and page cache are unchanged; it never appeared
with the display off; it came back on a display wake; a synthetic GPU client does not reproduce it; a GPU
keep-alive makes it worse. Suspects on this desk: the Aerial video wallpaper (`WallpaperAerialsExtension`
21-35 % CPU, `VTDecoderXPCService`), two 4K panels at the scaled "looks like 3360x1890" mode (rendered at
6720x3780 and downsampled every frame), iStatistica Pro and MenuBarAgent redrawing the menu bar on three
screens. Steps, one change at a time, each inside a slow window (the `[request]` tok/s or a
`decode_anatomy.py` run tells you which window you are in):

1. `sudo powermetrics --samplers gpu_power,cpu_power -i 1000 -n 30` once in a slow window and once in a fast
   one: GPU frequency and residency are what nothing without root can see.
2. Still wallpaper instead of the Aerial; then quit iStatistica Pro; then native "looks like 1920x1080" on
   the two 4K panels. Take one anatomy arm (no server running) before and after each change; `rest` near
   200 ms/token is the slow window, near 90-120 the fast one:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=80 CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json CACHALOT_HOTLIST_GIB=8 CACHALOT_EXPERT_CACHE_BUDGET_GIB=52 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py 2>&1 | grep -E "^decode|expert wait|^  rest"
```

3. If one of them is the trigger, document it in the README's Performance section and the manual test. If
   none is, the next lever is structural: fewer GPU round trips per token (the 40 routing syncs, §7.1.6),
   since the window only hits a token whose GPU work is broken up by read waits.

## Job 3 — decide what else the in-block pins need. Only after Job 1.

The pins are in memory only. If Hamed's sessions show a changed system block right after a server restart
(the replay found one such case), write the in-block chunk snapshots to disk too (`snapshot_store`, eight
files kept today; the pins would need a larger cap or their own). Measure with
`benchmarks/prefix_pin_replay.py` on the new dump first.

## Job 4 — vision: harder cases for the delimiters and `bias_vl`.

§15.7's ablation found only the Engram mask load-bearing on five easy cases. Add dense-layout cases (a
6x6 grid, a table with many rows, small text) to `benchmarks/vision_ablation.py` and rerun
`benchmarks/vision_ablation.sh` for `none`, `delims` and `bias_vl`. The code stays as the reference has it
either way; this decides whether they are worth a regression test.

## Job 5 — batched prefill sits 0.014-0.02 mean KL from token-by-token decode. Unchanged.

`benchmarks/prefill_chunk_quality.py 1536:seq,whole --offset N` at several offsets, 8+ samples.

## Job 6 — the 54 GiB collapse, and Job 7 — the Objective-C corpus. Unchanged, low priority.

The 54 GiB turn-4 collapse (§7.2.9) looks like the slow window (a long decode, one turn, no memory signal);
re-read it after Job 2.

## What is closed, so nobody spends a session there

- **The slow window is not the drive, the page cache, context length, `max_seq_len`, GPU dispatch latency,
  the all-resident compute path, or GPU clock starvation** (§15.5, §15.7). Do not re-measure those.
- **The page cache as a fast-window explanation** (§15.7): 15-19 % of reads are page-cache speed in slow and
  fast arms alike.
- **Vision fix ablation on easy cases** (§15.7): done; only Job 4's harder cases remain.
- Everything v41 listed as closed still is.

## Rules that still hold, and two new ones

- **A slow-window A/B needs a slow window.** Check that the default arm is slow (`rest` near 200 ms) before
  reading a lever's arm; a pair taken in a fast window says nothing either way. New.
- **Separate processes, interleaved, for every speed arm**; the window changes within minutes. New.
- **MLX 0.32 arrays belong to the thread that built them until evaluated** (§15.6).
- **A live speed number is only comparable inside one window.** Read `miss/tok` and `read=` beside tok/s.
- **Hermes changes under you.** Diff the dumped system prompt before assuming a cache bug.
- `serve.sh`'s guard runs `pgrep -f "deepseek-v41/bin/python|cachalot\.cli"`: launch it from a command line
  that contains neither.
- Quote heredoc delimiters (`<<'EOF'`) when the payload has backticks.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `./serve.sh` + `CACHALOT_SERVER_DUMP=path` | per-request `[request]` line with `reused`, `spliced`, `miss/tok`, `read=`, `fast=`; request bodies and reply token ids | — |
| `benchmarks/slow_window_sampler.py` | page cache, memory, GPU %, display, top processes, server reads every 10 s, new | — |
| `benchmarks/decode_anatomy.py` | expert wait against `rest` per token, reads by pool; the slow-window discriminator | ~1 min |
| `benchmarks/prefix_pin_replay.py DUMP` | prefill tokens with and without in-block pins, offline, new | ~2 min |
| `benchmarks/vision_ablation.sh ARM` | five verifiable image cases scored per ablation arm, new | ~3 min per arm |
| `HERMES_HOME=<dir> hermes chat -Q -q ...` / `--resume SID` | Hermes against the server without touching Hamed's config | minutes per turn |
| `benchmarks/decode_vs_context.sh N` | decode tok/s, misses and read times behind N filler tokens | ~1-5 min per arm |
| `docs/manual-tests/hermes-desktop.md` | Hamed's Desktop test script | ~1 h |
| `benchmarks/prefill_chunk_quality.py` | NLL + KL of teacher-forced continuations | ~5 min |
