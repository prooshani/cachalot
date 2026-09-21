# Next-session prompt — **v25**, written 2026-09-21

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v25** | 2026-09-21 | the ranking was rebuilt on measured sizes because the rate had not moved in four sessions | **two of `profile_decode_gpu.py`'s rows were timing retired paths**: routed experts are **6.7 ms** of GPU per token and not 19.8, the router **0.9** and not 7.5; the expert kernel is **closed** (the traced block costs what its three matmuls cost); the largest named GPU piece is **attention, 14.3 ms**; and the two biggest blocks in the token are **~20 ms of GPU nothing accounts for** and **~33 ms of CPU above the floor**, together 50 ms of a 128 ms token |
| v24 | 2026-09-21 | two live sessions on 0.7.0, coding turns compiled | the live A/B is a null a third time; 46.9 % of bytes are unused predictions; one of two coding turns does not compile |
| v23 | 2026-09-21 | prediction levers closed, shipped budget profiled | admission 4.3 % ceiling, blocklist a loss, submission 0.040 ms; 36 and 44 GiB run again |
| v22 | 2026-09-21 | the CPU third attacked again | memoised kernel constants ship at 1.2 ms; prediction is 11 ms of the floor |
| v21 | 2026-09-21 | the traced runtime run interactively | hand over `./chat.sh` |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**This session is about speed, and the previous four did not deliver any.** 7.6-7.9 tok/s on prose and
5.6-6.9 on code, unchanged across three runtime versions. Read `docs/HANDOFF.md`'s opening block, then
**9.20**, then **7.1.4**, then 7.1.3. The second is the ranking rebuilt on measured sizes and the reason
nothing moved; the third is the GPU side decomposed on the code that actually runs; the fourth is the
shipped configuration's profile.

## Where the time is

A live prose token is **128.5 ms**; a benchmark token at the same budget is 170 ms with a colder working
set. The all-resident floor is 76.4 ms — **54.8 inside `mx.eval`, 21.6 outside it.**

| block | size | state |
|---|---:|---|
| **GPU time no measured piece accounts for** | **~20 ms** | head, Engram, ten unprofiled attention layers, 44 eval drains. **No mechanism.** |
| **`rest` above the all-resident floor while streaming** | **~33 ms** | not the GIL switch interval, not reader scheduling. **No mechanism.** |
| blocking on misses no prediction covered | 46.4 ms at 83.5 % hit, ~25 at a session's 90 % | drive idle 45 %; width, lead and precision closed |
| attention, 30 of 40 layers | 14.3 ms | largest named GPU piece; `shapeless=True` never tried |
| routing prediction | 11 ms | every named way of making it cheaper is closed |
| routed experts, traced | 6.7 ms | **closed**: equals its three matmuls |
| shared expert | 6.3 ms | never screened |
| hyper-connection glue | 6.1 ms | **closed**: tracing is 0.2 ms |
| the layer's own router | 0.9 ms | **closed**: nothing there |

**Everything worked on since 2026-09-19 lives in the bottom half of that table.** The top two lines are
about 50 ms of a 128 ms token, and nobody has a mechanism for either.

**Two numbers in the old ranking were measured on code the runtime had stopped running**, which is how
19.8 ms of "routed experts" survived three sessions as the largest GPU item: `profile_decode_gpu.py` timed
`affine_expert_forward` in a loop (retired by the traced MoE block) and `route_topk` (retired by
`route_topk_fused`). Both rows are fixed and both old paths are kept beside the shipped ones so the gap
stays visible. **Before ranking anything off an instrument in this repository, read which function it
calls.** That is now four occurrences of the same mistake.

## Job 1 — the ~20 ms of GPU nothing accounts for

The measured pieces sum to **34.3 ms**; a token spends **54.8 ms** inside `mx.eval`. Close that gap, because
it is the largest single unexplained block in the project and it is pure measurement.

Four candidates, all cheap, in order:

1. **The ten attention layers the profile does not cover.** It times the two compressed-reuse classes on 30
   layers. Layer 0, the sliding-window layers and the source and index-source layers have their own modules
   (`attention_layer0.py`, `attention_sliding_window.py`, `block_compressed_source.py`,
   `block_compressed_index_source.py`) and none of them has ever been timed. If attention is 14.3 ms on 30
   layers it is plausibly 19 on 40.
2. **The head and the Engram rows.** `engram_rows.py:59` already shows up in `profile_decode_sync.py`'s
   per-site table at 0.7 ms inside eval and 1.8 ms of gap; the head is one large matmul against a 130k
   vocabulary and has never been priced.
3. **What 44 evals cost to drain.** Every layer ends in an `mx.eval` to bring routing to the CPU. If the
   drain itself is worth milliseconds, that is an argument for a different synchronisation shape, and it is
   measurable by chaining N launches with and without intervening evals.
4. **Whether the sum is allowed to close.** Chained launches exclude the per-eval floor by construction, so
   the pieces may simply be cheaper measured this way than paid in sequence. Test it directly: time the
   forty layers of one token with the pieces chained, then the same work with an eval per layer.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh && benchmarks/guarded_run.sh --budget-gib 40 --max-seconds 1800 --tag gpu40 -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/profile_decode_gpu.py --prompt-tokens 512
```

## Job 2 — the ~33 ms a streaming token spends above the floor

Carried from v23 and still first among the CPU-side questions. A 44 GiB token spends 115.5 ms outside the
store's blocking calls against an all-resident 76-82. `sys.setswitchinterval` at 0.001, 0.005 and 0.020 s
changes nothing, so reader-thread scheduling is ruled out.

The measurement nobody has run is `profile_decode_sync.py`'s eval/gap split **on a streaming token** rather
than one restored from a snapshot. If the gap before `moe_layer_metal.py:180` grows with the miss rate, the
cost is in the store's admission path — slot acquisition, eviction, the LRU, `slot_views` — and it is
addressable. If it grows inside eval, it is the GPU waiting on memory the SSD DMA is also using, and it is
not.

## Job 3 — attention, now that it is the largest named GPU piece

14.3 ms across 30 layers, more across 40, and the only untried instrument in the repository is
`mx.compile(shapeless=True)`. Section 6.4's cap of 4.7 ms is about how attention **grows** with the context,
not what it costs at a fixed one, so that cap does not bound this.

**Screen it before writing anything.** `benchmarks/micro_compile_hc_fused.py` is the template: time
construction with no eval in the loop, chain 40 launches for the GPU side, assert bit-identical output on
all arms. Two screens this project ran in twenty minutes each closed two ranked jobs, and one screen this
session closed the expert kernel.

## Job 4 — the hit rate, which is the only lever a live session can see

Every interactive session peaks at **59.2-59.7 GiB against a 72 GiB wired limit**, holding 4,480-4,495
experts at a 44 GiB budget. Section 9.4 closed a larger budget by simulation at a different expert size.

1. **Re-simulate at 9.49 MiB**: `benchmarks/simulate_policies.py --expert-bytes`, offline and free.
2. **If it pays, raise it in one interactive session, not a benchmark**:
   `CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52`, and read the session hit rate against
   90.0 %, not the tok/s of any one turn.

## What is closed, so nobody spends a session there

- **The expert kernel.** The traced block is 0.169 ms per layer, which is what its three
  `mx.quantized_matmul` calls cost alone (0.163). `mx.compile` fuses every cast, clamp and accumulate away.
  `gather_qmm` is 20-25 % better on pre-stacked weights — **≤1.5 ms per token** — and needs six LRU slots
  made contiguous. §7.1.4, §11.
- **Routing prediction**, every named way of making it cheaper: admission (4.3 % ceiling), a re-read
  blocklist (8.1 wasted reads saved for 3.4 demand misses added), the submission bookkeeping (0.040 ms),
  width (top-6 again), two layers ahead (a null at 55 % drive busy), the GIL switch interval. §9.19, §11.
- **Speculative decoding**, re-checked this session on current constants rather than FP4 ones. Bytes per
  accepted token rise 515 → 600 → 876 MiB from width 1 to 2 to 5, and a K-position forward measures
  **242.6 ms + 26.9 per extra position**, because the only multi-position path here is the prefill path.
  Width 5 is 123 ms per accepted token against a live 128.5, for twice the bytes. It needs a decode-shaped
  batched forward before the economics are worth recomputing. §9.20.
- The hyper-connection glue, the router, dispatch fusion, 1 bit per weight, mirror striping on this bank,
  the frequency penalty, eviction policy. §11.

## Rules that still hold

- **Read which function an instrument calls before ranking a lever off it.** Four occurrences now, and this
  session's two were the largest GPU item and the largest router number in the document.
- **Never quote a screen as a gate** — and price a loop with one before ranking it.
- **Check that a profile's parts add up to its whole before ranking anything off it.** This is what found
  the 20 ms.
- **Print what an arm read before attributing a difference to CPU work.**
- **A thread does not hide Python work.**
- **Compile a live coding turn before calling it reference class.**
- **A live session resolves its hit rate and nothing else.** Four sessions repeat their steady state to a
  tenth of a point and cannot see a 1 % change in rate. Speed is measured with `profile_decode_sync.py`,
  `profile_decode_gpu.py` or `decode_anatomy.py`.
- **An A/B only finds defects that differ between its arms.** Compare against
  `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/`.
- **Write a test when a comparison finds a match, not only when it finds a bug.**
- **Never drop a case from a denominator.**
- **A curve drawn through the turns of a live session is not a curve.**
- **Read a run's footprint line before its numbers.**
- **Check a confound before reporting an effect.** A screen that replays the same six experts forty times
  reads 57 MiB from cache; `micro_expert_roofline.py --distinct` builds 240 experts for that reason, and the
  answer moved by 6 %.
- **Memory.** `guarded_run.sh` needs `budget + 29` GiB available and sets no wired limit of its own. 44 GiB
  ran four times on 2026-09-21 and was refused once at 72.0 GiB available; 40 always runs. Never pass
  `--force`; lower the budget.
- **Hand over `./chat.sh`, never the section 4 one-liner.** It does not survive line wrapping and the
  failure is silent — FP4 over USB at 0.6 tok/s. Section 12.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once. Launch each run as its own command.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/profile_decode_gpu.py` | **the GPU side, piece by piece, on the paths that ship** | ~1 min |
| `benchmarks/profile_decode_sync.py` | the CPU/GPU split per eval site, and what the arm read | ~1 min |
| `benchmarks/decode_anatomy.py` | where a live token's time goes, by blocking cause; ±0.3 % at 44 GiB | ~1 min |
| `benchmarks/micro_topk_core.py` | `topk_core` built back one op at a time, ending at the compiled block | instant, no model |
| `benchmarks/micro_expert_roofline.py` | the expert matmuls against `gather_qmm`, a dense matvec, 240 distinct experts | instant, no model |
| `benchmarks/micro_compile_hc_fused.py` | **the template for a CPU screen** | ~1 min, no model |
| `benchmarks/predict_ghost.py` | what happens to a mispredicted expert after it is dropped | ~1 min |
| `benchmarks/verify_forward_cost.py` | what a K-position forward costs on the path that exists | ~1 min |
| `benchmarks/speculation_bytes.py` | misses and bytes per accepted token by verification width | instant, no model |
| `benchmarks/decode_rate_by_block.py` | the rate over a long generation, in blocks | ~5 min |
| `benchmarks/nll_expert_precision.py --experts runtime --tokens 512` | the production quality arm | ~2 min |
| `benchmarks/coding_quality.py --resume` | the 40-case corpus, restartable | ~1.5 h |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | **the official implementation** |
| `benchmarks/results/coding/q2g128-v17/` | **the 2-bit gate**, 40/40 |
| `benchmarks/results/guarded/gpu40c_*` | **the corrected GPU decomposition** |
| `benchmarks/results/guarded/anat44_*` | the shipped configuration's profile |
| `benchmarks/results/guarded/vfc40_*` | what a K-position forward costs |
| `benchmarks/results/speculation_bytes_q2_44.json` | bytes per accepted token by width, 2-bit |
| HANDOFF section 9.20 | **the ranking, by measured size** |
| HANDOFF section 7.1.4 | the GPU side, and the two withdrawn numbers |
| HANDOFF section 7.2.5 | two live sessions on 0.7.0 and their compiled coding turns |
| `./chat.sh` | **the command to hand Hamed** |
