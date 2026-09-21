# Next-session prompt — **v26**, written 2026-09-21

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v26** | 2026-09-21 | the GPU side was closed and the first lever with a mechanism since 2026-09-19 was built and measured | **the ~20 ms of unattributed GPU time is gone**: 12 ms of it was the ten attention layers, the head and Engram, which nobody had ever timed, and the rest is the **0.20 ms round trip an `mx.eval` costs**, paid 44 times a token; **attention is 22.5 ms**, three times the routed experts; `mx.compile` on attention is **closed three ways**; and giving the GPU the shared expert to run during the routing sync takes **10 ms out of `mx.eval`** and puts **13 ms back on the CPU**, while putting it in the routing's own eval costs 5 ms inside it |
| v25 | 2026-09-21 | the ranking was rebuilt on measured sizes | two profiler rows were timing retired paths; the expert kernel is closed; 20 ms of GPU and 33 ms of CPU had no mechanism |
| v24 | 2026-09-21 | two live sessions on 0.7.0 | the live A/B is a null a third time; 46.9 % of bytes are unused predictions; one of two coding turns does not compile |
| v23 | 2026-09-21 | prediction levers closed, shipped budget profiled | admission 4.3 % ceiling, blocklist a loss, submission 0.040 ms |
| v22 | 2026-09-21 | the CPU third attacked again | memoised kernel constants ship at 1.2 ms; prediction is 11 ms of the floor |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Nothing shipped this session and the decode rate is unchanged** — 7.6-7.9 tok/s on prose, 5.6-6.9 on
code — but the GPU side of the token is now fully accounted for, and one lever came out of it with a
measured mechanism rather than a guess. Read `docs/HANDOFF.md`'s opening block, then **7.1.5**, then
**7.1.6**, then **9.22**, then 9.23. The first is where the GPU time is, on every layer; the second is what
a synchronisation costs; the third is the arm that moved 10 ms and lost anyway; the fourth is the ranking
as it now stands.

## Where the time is, after this session

A live prose token is **128.5 ms**; a benchmark token at the same budget is 170 ms with a colder working
set. The all-resident floor is **76.4-79.6 ms**, about 56 of it inside `mx.eval` and 21.5 outside.

| block | size | state |
|---|---:|---|
| blocking on misses no prediction covered | 46.4 ms at 83.5 % hit, ~25 at a session's 90 % | every prediction lever closed; **a larger budget is the only untried one and needs no code** |
| **`rest` above the all-resident floor while streaming** | **~33 ms** | **the only block left with no mechanism** |
| **attention, all forty layers** | **22.5 ms** | 12.8 reuse + 8.9 source + 0.8 sliding; `mx.compile` closed three ways |
| routing prediction | 11 ms | closed |
| **the 44 `mx.eval` round trips** | **~9-12 ms** | 0.20 ms each, fixed; the overlap arm moves 10 ms and costs 13 |
| routed experts, traced | 6.9 ms | closed |
| **compressor and indexer, eight source layers** | **5.5 ms** | **never examined** |
| shared expert | 4.8 ms | never screened |
| hyper-connection glue | 4.2 ms | closed |
| head | 1.6 ms | never attacked |
| Engram, two layers | 0.8 ms | — |
| the layer's own router | 0.8 ms | closed |

## Job 1 — the 33 ms a streaming token spends above the floor

Carried from v23, v25 and now the **only** unattributed block in the token. A 44 GiB token spends 115.5 ms
outside the store's blocking calls against an all-resident 76-82. `sys.setswitchinterval` is ruled out and
so is reader scheduling.

The measurement still nobody has run is `profile_decode_sync.py`'s eval/gap split **on a streaming token**.
The instrument restores a snapshot and decodes the same token twelve times, which is all-resident by
construction; it needs an arm that decodes a real continuation at a budget low enough to miss, reporting
the same per-site table. If the gap before `moe_layer_metal.py:208` grows with the miss rate, the cost is
in the store's admission path — slot acquisition, eviction, the LRU, `slot_views` — and it is addressable.
If it grows inside eval, it is the GPU waiting on memory the SSD DMA is also using, and it is not.

## Job 2 — the compressor and the indexer, 5.5 ms nobody has looked at

Section 7.1.5 measured a source layer's attention at 1.483 ms against a reuse layer's 0.441 on the same
context, and an index-only source at 0.839. The difference is the compressed-KV write and the index
selection, **5.5 ms per token across the eight source layers**, which is larger than the routed experts and
has never been decomposed. Start by splitting `compressed_attention_decode_source` into attention,
compressor and indexer rows in `profile_decode_gpu.py` — the same edit that closed the 20 ms this session —
and read `INDEX_TOPK` (512) against what the indexer costs at that width.

## Job 3 — the syncs themselves

Section 7.1.6 says a token pays 44 round trips of about 0.20 ms and cannot avoid them while the expert
store is addressed on the CPU. Section 9.22 tried to hide one behind the shared expert and paid more in
submission than it saved in waiting. Two things remain untried, in order:

1. **Fewer syncs, not cheaper ones.** Three arrays in one `mx.eval` cost 0.28 ms against 1.03 for three
   separate ones. Every layer already fuses its routing with the predictor's look at L+1. What is left is
   whether two *layers* can share one sync — they cannot as the graph stands, because layer L+1's router
   needs layer L's output, but the predictor already computes an approximation of exactly that, and its
   recall is 73 %.
2. **The head and the tail.** Four of the 44 evals are outside the forty layers. `text_decode_runtime.py`
   evaluates hidden and logits together at line 2551 and the Engram rows twice; whether any of those can
   join a neighbour is a five-minute read.

## Job 4 — the budget, which is the only lever a live session can see

Re-simulated at 9.49 MiB this session (section 9.4): **44 to 52 GiB buys 2.6 points of decode hit rate**,
52 to 60 another 2.3. A chat session peaks at 59.2-59.7 GiB against a 72 GiB wired limit and leaves 12 GiB
unused. This is Hamed's run, not a benchmark:

```bash
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52
```

Read the session hit rate against 90.0 %, not the tok/s of any one turn.

## Job 5 — the shared expert, 4.8 ms and never screened

Three FP8 gemvs per layer, 0.121 ms per launch, and the only piece of the MoE block that has never had a
screen of its own. `benchmarks/micro_expert_roofline.py` is the template for asking whether it is at the
memory wall.

## What is closed, so nobody spends a session there

- **`mx.compile` on attention**, in all three of its shapes. §9.21.
- **The expert kernel.** The traced block costs what its three `mx.quantized_matmul` calls cost;
  `gather_qmm` is ≤1.5 ms and needs six LRU slots made contiguous. §7.1.4, §11.
- **Routing prediction**, every named way of making it cheaper: admission, a re-read blocklist, the
  submission bookkeeping, width, lead, two layers ahead, the GIL switch interval. §9.19, §11.
- **Speculative decoding**, re-checked on current constants: a K-position forward is 242.6 ms + 26.9 per
  extra position, because the only multi-position path is the prefill path. §9.20.
- The hyper-connection glue, the router, dispatch fusion, 1 bit per weight, mirror striping on this bank,
  the frequency penalty, eviction policy. §11.

## Rules that still hold

- **A chained profile excludes the per-eval floor by construction.** Every row of
  `profile_decode_gpu.py` is many launches in one `mx.eval`; a token pays 44 evals at about 0.20 ms each.
  Add that back before comparing a sum of pieces to a token. New this session, and it is what closed the
  20 ms. §7.1.6.
- **Read which function an instrument calls before ranking a lever off it.** Four occurrences.
- **Check that a profile's parts add up to its whole before ranking anything off it.**
- **Never quote a screen as a gate** — and price a loop with one before ranking it.
- **Prove a speed arm's numerics, do not argue them.** `benchmarks/decode_fingerprint.py` is 16 greedy
  tokens with ids and fp32 logit checksums; diff the two arms. New this session. §9.22.
- **Write a multi-arm A/B to a file and run `bash the-file`.** `settle.sh` matches whole command lines for
  a live runtime, so a script passed as text waits for itself forever, at 73 GiB available and pressure 1.
  New this session. §12.
- **Print what an arm read before attributing a difference to CPU work.**
- **A thread does not hide Python work** — and neither does `mx.async_eval`: submitting a graph is Python
  on the decode thread, 0.3 ms of it per layer. New this session. §9.22.
- **Compile a live coding turn before calling it reference class.**
- **A live session resolves its hit rate and nothing else.** Speed is measured with
  `profile_decode_sync.py`, `profile_decode_gpu.py` or `decode_anatomy.py`.
- **An A/B only finds defects that differ between its arms.** Compare against
  `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/`.
- **Write a test when a comparison finds a match, not only when it finds a bug.**
- **Never drop a case from a denominator.**
- **A curve drawn through the turns of a live session is not a curve.**
- **Read a run's footprint line before its numbers.**
- **Check a confound before reporting an effect.**
- **Memory.** `guarded_run.sh` needs `budget + 29` GiB available. 40 GiB ran nine times on 2026-09-21;
  44 needs 73 and was not available late in the day. Never pass `--force`; lower the budget.
- **Hand over `./chat.sh`, never the section 4 one-liner.** §12.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once. Launch each run as its own command.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/profile_decode_gpu.py` | **the GPU side, every layer, plus what an eval costs to drain** | ~3 min |
| `benchmarks/profile_decode_sync.py` | the CPU/GPU split per eval site, and what the arm read | ~3 min |
| `benchmarks/micro_eval_floor.py` | **what one `mx.eval` costs, and whether anything makes it cheaper** | instant, no model |
| `benchmarks/decode_fingerprint.py` | **whether a speed arm changed the numerics** | ~3 min |
| `benchmarks/micro_compile_attention.py` | the `mx.compile` screen for attention, retraces included | ~3 min |
| `benchmarks/decode_anatomy.py` | where a live token's time goes, by blocking cause; ±0.3 % at 44 GiB | ~3 min |
| `benchmarks/simulate_policies.py --expert-bytes 9953280` | hit rate against budget, offline | instant, no model |
| `benchmarks/micro_expert_roofline.py` | the expert matmuls against `gather_qmm` and 240 distinct experts | instant, no model |
| `benchmarks/micro_compile_hc_fused.py` | **the template for a CPU screen** | ~1 min, no model |
| `benchmarks/predict_ghost.py` | what happens to a mispredicted expert after it is dropped | ~3 min |
| `benchmarks/nll_expert_precision.py --experts runtime --tokens 512` | the production quality arm | ~2 min |
| `benchmarks/coding_quality.py --resume` | the 40-case corpus, restartable | ~1.5 h |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | **the official implementation** |
| `benchmarks/results/coding/q2g128-v17/` | **the 2-bit gate**, 40/40 |
| `benchmarks/results/guarded/gpu40full3_*` | **the GPU side, all forty layers, head and Engram** |
| `benchmarks/results/guarded/attncomp2_*` | the attention `mx.compile` screen |
| `benchmarks/results/guarded/sync0_*`, `sync1*_*`, `sync2*_*` | the three prelaunch arms |
| `benchmarks/results/guarded/fp0_*`, `fp1_*`, `fp2_*` | their identical fingerprints |
| `benchmarks/results/guarded/anat44_*` | the shipped configuration's profile |
| HANDOFF section 7.1.5 | **where the GPU time is, layer by layer** |
| HANDOFF section 7.1.6 | what a synchronisation costs |
| HANDOFF section 9.22 | the prelaunch arms |
| HANDOFF section 9.23 | **the ranking** |
| `./chat.sh` | **the command to hand Hamed** |
