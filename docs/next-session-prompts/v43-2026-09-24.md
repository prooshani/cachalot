# Next-session prompt — **v43**, written 2026-09-24

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v43** | 2026-09-24 | Hamed asked for the remaining levers planned, built, measured and documented | **0.14.0: `serve`/`chat` run with `MLX_METAL_FAST_SYNCH=1` (shared-memory Metal fences): bit-identical, +8 to +25 % decode in slow and mid windows over the Darwin role, null in a fast window, prefill unchanged. Snapshots on disk are keyed on `NUMERICS_VERSION`, so a release no longer costs Hermes one cold 22k re-prefill. Vision Job 4 done: on a 6x6 grid the delimiters and `bias_vl` are load-bearing too.** §15.9 |
| v42 | 2026-09-24 | same | 0.13.0-0.13.1: slow window is compute; focused-app role; visible Hermes window is the main trigger. §15.7, §15.8 |
| v41 | 2026-09-24 | Hamed ran the Desktop manual test | 0.12.1-0.12.2: three Desktop bugs fixed. §15.6 |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read this first.** `docs/HANDOFF.md`'s "Start here (2026-09-24, 0.14.0)" block and section **15.9**, then
15.8 and 15.7.

**Hamed's standing priority order: Hermes usage first, vision second, speed/performance third.**

**Check the working tree before starting** (`git status --short`). If 0.14.0 is uncommitted, confirm with
Hamed and commit; the bump is already made.

**The shipped configurations** (both now export `MLX_METAL_FAST_SYNCH=1`; startup prints it):

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

## Job 1 — the long Desktop session, with the sampler beside it. Needs Hamed.

Unchanged from v42, and now also the first live read of 0.14.0's fences. `docs/manual-tests/hermes-desktop.md`
section 5 (20+ turns, past 30-50k context, `threshold_tokens: 30000` under `compression:`). Before the first
request, in a second terminal:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && /usr/bin/python3 benchmarks/slow_window_sampler.py
```

Read back: every `[request]` line (tok/s, `miss/tok`, `read=`, `fast=`), the sampler CSV, the `reused` of the
first main request after the upgrade (the 22k snapshot should load: startup says `prefix snapshots: 4 loaded`),
and after the compression the `reused` of the next main request (a 4,096 multiple, not 0).

## Job 2 — the rest of the slow-window trigger. Needs Hamed at the desk.

v42's Job 2 unchanged: `sudo powermetrics --samplers gpu_power,cpu_power -i 1000 -n 30` once slow, once fast;
then one change at a time inside a slow window (still wallpaper instead of Aerial; quit iStatistica Pro;
native "looks like 1920x1080" on the two 4K panels), one anatomy arm before and after each. Anatomy arms do not
set the fences or the role; to measure a served token add both:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && env MLX_METAL_FAST_SYNCH=1 CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=80 CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json CACHALOT_HOTLIST_GIB=8 CACHALOT_EXPERT_CACHE_BUDGET_GIB=52 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py 2>&1 | grep -E "^decode|expert wait|^  rest"
```

(That line has no Darwin role; §15.9 used a scratch `runpy` wrapper that calls `apply_darwin_role()` first.
Adding a `--darwin-role` flag to `decode_anatomy.py` is a five-line job worth doing first.)

## Job 3 — the other MLX scheduling knobs. Autonomous, measured-first.

Listing `libmlx.dylib`'s `MLX_*` strings found the fence switch; two more were never tried here:
`MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` (how many ops/MB go into one Metal command buffer before
MLX commits it). Fewer commits per token may matter for the same reason fences did: every commit is a point
where the compositor's GPU work interleaves with ours. Screen 2x and 4x the defaults with `decode_anatomy.py`
under fences + role, ABBA, **only inside a slow window** (`rest` near 150-200 ms), then `decode_fingerprint.py`
for bit-identity. Also still open: why fences help (instrument the time from GPU completion to the next
launch, §15.9 has only a reading).

## Job 4 — in-block pins on disk. Only after Job 1.

v42's Job 3 unchanged: `benchmarks/prefix_pin_replay.py` on the new dump first.

## Job 5 — vision: make grid6 a live regression, then larger images.

`benchmarks/vision_ablation.sh none --cases grid6,count,chart` is the three-case check that catches all three
§16.4 fixes (grid6: delimiters and `bias_vl`; count and chart: the Engram mask). Run it after any change to
`vision_prompt.py`, `engram*`, or the prefill path. Next case worth adding: a screenshot at Desktop's typical
size (the §15.6 Post-profile screenshot), since every case so far is under 1,000 image rows.

## Job 6 — batched prefill sits 0.014-0.02 mean KL from token-by-token decode. Unchanged.

`benchmarks/prefill_chunk_quality.py 1536:seq,whole --offset N` at several offsets, 8+ samples.

## Job 7 — the 54 GiB collapse, and the Objective-C corpus. Unchanged, low priority.

## What is closed, so nobody spends a session there

- **`MLX_METAL_FAST_SYNCH`** (§15.9): shipped; bit-identical; null in a fast window. Do not re-screen it.
- **Vision fix ablation** (§15.7, §15.9): all three fixes load-bearing (Engram mask on count/chart, delimiters
  and `bias_vl` on grid6).
- **The slow window is not the drive, the page cache, context length, `max_seq_len`, GPU dispatch latency,
  the all-resident compute path, or GPU clock starvation** (§15.5, §15.7).
- Everything v42 listed as closed still is.

## Rules that still hold, and one new one

- **`snapshot_store.NUMERICS_VERSION` moves with numerics, not with releases.** Bump it with any change that can
  move a prefill's KV bits (kernels, quantization, attention, Engram, tokenizer path, chunk sizes); never for a
  scheduling or documentation change. New.
- **A slow-window A/B needs a slow window**; check `rest` before reading a lever's arm.
- **Separate processes, interleaved, for every speed arm. Never start an arm while another runtime is alive**:
  §15.9 lost two arms and one fingerprint run to exactly that.
- **MLX 0.32 arrays belong to the thread that built them until evaluated** (§15.6).
- **Hermes changes under you.** Diff the dumped system prompt before assuming a cache bug.
- `serve.sh`'s guard runs `pgrep -f "deepseek-v41/bin/python|cachalot\.cli"`: launch it from a command line
  that contains neither.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `./serve.sh` + `CACHALOT_SERVER_DUMP=path` | per-request `[request]` line with `reused`, `spliced`, `miss/tok`, `read=`, `fast=` | — |
| `benchmarks/slow_window_sampler.py` | page cache, memory, GPU %, display, top processes, server reads every 10 s | — |
| `benchmarks/decode_anatomy.py` | expert wait against `rest` per token; the slow-window discriminator | ~45 s |
| `benchmarks/decode_fingerprint.py --decode-tokens 24` | bit-identity of a scheduling change | ~1 min per arm |
| `benchmarks/vision_ablation.sh ARM [--cases ...]` | eight verifiable image cases per ablation arm | ~1-3 min per arm |
| `benchmarks/decode_vs_context.sh N` | prefill s, decode tok/s, misses and read times behind N filler tokens | ~1.5 min at 4k |
| `benchmarks/prefix_pin_replay.py DUMP` | prefill tokens with and without in-block pins, offline | ~2 min |
| `docs/manual-tests/hermes-desktop.md` | Hamed's Desktop test script | ~1 h |
