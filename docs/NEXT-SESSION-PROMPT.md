# Next-session prompt — **v23**, written 2026-09-21

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v23** | 2026-09-21 | every named way of making routing prediction cheaper was measured and closed, and the shipped 44 GiB configuration was profiled for the first time | admitting mispredicted bytes has a **4.3 %** ceiling, a re-read blocklist trades 8.1 wasted reads for 3.4 demand misses, the submission bookkeeping is **0.040 ms** and not 3.6; `PREDICT_AHEAD=2`, top-4, top-8 and the GIL switch interval are nulls at 44 GiB; **36 and 44 GiB run again** and the wired-plan hypothesis is refuted; the token at 44 is **170 ms, 83.5 % hit rate, 627 MiB**, and the largest unexplained block is now **33 ms** between a streaming token's `rest` and the all-resident floor |
| v22 | 2026-09-21 | the CPU third was attacked again and the reply-length effect refuted | memoised kernel constants ship at 1.2 ms; tracing the glue and the router are worth 0.2 and 0.3 ms; routing prediction is 11 ms of the 77 ms floor; the live spread is working set, not reply length |
| v21 | 2026-09-21 | the traced runtime was run interactively, both arms | two full sessions at 7.76 and 7.82 tok/s; the live A/B is a null by construction; hand over `./chat.sh` |
| v20 | 2026-09-21 | the CPU third was attacked with `mx.compile` | the decode MoE block traces once instead of forty times a token: 82-85 ms to **77 ms** |
| v19 | 2026-09-21 | the compute floor was measured properly | hyper-connections are 4.6 ms, not 68.7 |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read `docs/HANDOFF.md`'s opening block, then 7.1.3, then 9.19, then section 11's routing-prediction
block.** The first tells you what ships; the second is the shipped configuration's first profile and the
four A/Bs it made readable; the third is three ranked jobs closed in one session by two new instruments; the
fourth is the list of things not to try again.

## Where the project stands

The shipped configuration is unchanged and **nothing shipped this session**: the 2-bit g128 bank at a 44 GiB
budget with the hotlist, no mirror, no frequency penalty. Quality is untouched — no numerics were modified,
219 tests pass, the tree is clean. What changed is what is known.

**The machine came back.** 36 and 44 GiB had been refused or killed for three days; on 2026-09-21 four
benchmarks ran at 44 and two 310-second ones at 36, every one under 3.2 GiB of compressor. **v22's
wired-plan hypothesis is refuted**: the same 1,792-token run finished in 310 s at a forced 72 GiB wired
limit and 313 s at the runtime's auto limit. It was the rest of the machine. Check `settle.sh`'s available
figure before believing a budget is dead.

**The shipped configuration finally has a profile**, and it says what the 40 and 32 GiB ones said:

    decode 64 tokens: 10.85 s = 5.90 tok/s (170 ms/token)   [44 GiB, 512-token context]
      hit rate 83.5% | 39.7 misses/token | 627 MiB read/token
      blocked  54.0 ms/token (31.9%)  ->  46.4 coverage, 6.5 timing, 1.2 store
      rest    115.5 ms/token (68.1%)
      drive busy 55.3% of decode, 2.12 reads in flight while busy
      prediction: 45% precision, 26.4 wasted loads/token

The instrument repeats to ±0.3 % here — four shipped runs at 10.85, 10.85, 10.89, 10.91 s — which is what
made the session's A/Bs readable.

**Every candidate section 9.18 named for making prediction cheaper is closed** (section 9.19):

| candidate | v22 said | **measured 2026-09-21** |
|---|---|---|
| admit the mispredicted bytes | "the best idea on the list" | a dropped expert is demanded again within 8 tokens on **4.3 %** of demand reads; holding drops that long costs 2.7 GiB of a 24 GiB cache |
| predict fewer, better experts | "the width that pays is a function of what a wasted read costs" | **top-6 again**, on a 170 ms token reading 627 MiB: top-8 is 1.0 % slower, top-4 is 2.3 % slower |
| make the submission cheaper | "3.6 ms per token across 240 dict lookups and forty `.tolist()`s" | that bookkeeping is **0.040 ms** per token; the best rewrite saves 0.015; fusing the `.tolist()`s with `mx.concatenate` costs 8.4 |

**And a fourth thing fell out that corrects the floor.** `profile_decode_sync.py`'s "all-resident" token is
all-resident on the demand path only: it issues **40 speculative reads per token, expires all 40 and reads
380 MiB off the SSD**, because a mispredicted expert was never demanded and the probe repeats the same token
twelve times. That I/O is what `CACHALOT_PREDICT_SUBMIT=0` removed, so the 3.6 ms attributed to
"submitting" the prediction is the cost of issuing forty wasted reads, not of building the list.

**A new instrument, `benchmarks/predict_ghost.py`,** answered the first two rows above and one question
nobody had asked: **29.2 % of all speculative reads re-read an expert a prediction had already read and
dropped**, because the router barely moves between adjacent tokens. Blocking those re-reads was simulated at
every lifetime from 1 to 64 passes and is a loss — about 30 % of them turn out right, so a TTL of 8 saves
8.1 wasted reads per token and adds 3.4 demand misses.

## The lessons this session added

The eighth: **price a bookkeeping loop before ranking it.** `micro_predict_submit.py` cost ten minutes and
showed that a job two prompts had carried was worth 0.015 ms. The screen rule cuts both ways — a screen is
not a gate, but it is cheap enough to kill a job before a benchmark is booked for it.

The ninth: **a profiler's arm name is a claim about I/O too.** "All-resident" meant all-resident for the
demand path and 380 MiB per token of speculative reading. Before attributing a difference between two arms
to CPU work, print what each arm read.

The tenth: **a budget that was refused yesterday is not a property of the runtime.** Three prompts in a row
lowered the budget and one of them proposed a mechanism for it. Both budgets ran unchanged the next day.

## Job 1 — the 33 ms between a streaming token's `rest` and the all-resident floor

A 44 GiB token spends 115.5 ms outside the store's blocking calls. The all-resident floor is 77 ms at its
minimum and 82 median (section 9.18, and read its correction in 9.19 — that floor contains 40 speculative
reads of its own). **Nothing accounts for the 33 ms difference**, and it is now larger than any other
unattributed block in the token, larger than the 11 ms prediction costs and seven times what attention's
growing shapes can be worth (section 6.4).

The one mechanism anybody proposed is reader threads stealing the interpreter, and **it is ruled out**:
`sys.setswitchinterval` at 0.001, 0.005 and 0.020 s decodes 64 tokens in 10.85-10.89 s (section 7.1.3).

What has never been done is running `profile_decode_sync.py`'s eval/gap split on a token that is *streaming*
rather than restored from a snapshot. That is the measurement: the same CPU-outside-eval statistic, per call
site, at a budget low enough to miss. If the gap before `moe_layer_metal.py:180` grows with the miss rate,
the cost is in the store's admission path — slot acquisition, eviction, the LRU, `slot_views` — and it is
addressable; if it grows inside eval instead, it is the GPU waiting on memory the SSD DMA is also using, and
it is not.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh && benchmarks/guarded_run.sh --budget-gib 44 --max-seconds 1800 --tag anat44 -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64
```

## Job 2 — coverage, which is 86 % of the blocked time and has a free offline instrument

46.4 ms per token of blocking is misses no prediction covered, and the drive is idle 45 % of the time, so
the bandwidth to cover them exists. Width and lead time are both exhausted (section 11), which leaves the
predictor's *signal*: the L+1 router applied to layer L's input.

**The decomposition nobody has run, and it needs no runtime change**: of the misses a token takes, how many
were used at that same layer within the last K tokens, how many the router prediction named, how many both,
how many neither. `benchmarks/predict_ghost.py` already logs every demand read and every speculative read
with its decode pass; extending it to answer this is an hour and no new run. A large "neither" closes the
predictor question for good; a large "recent but unpredicted" is a second signal to union in, and it costs
one dict lookup per layer.

## Job 3 — attention, still the only unscreened `mx.compile` candidate

Unchanged from v22 and still capped at the 4.7 ms of `rest` that a fourfold context change moves
(section 6.4). Its shapes grow with the context, so a plain trace retraces every token;
`mx.compile(shapeless=True)` has never been tried here. **Screen it before writing anything** —
`benchmarks/micro_compile_hc_fused.py` is the template: time construction with no eval in the loop, chain 40
launches for the GPU side, assert bit-identical output on all arms.

## Job 4 — the loose threads, still open, still cheap

1. **A stray character.** Turn 2 of an earlier session returned `wHi! How can I help you today?`. It has now
   failed to reproduce in four clean sessions. Four greedy repeats plus `token_rank_probe.py` settles it in
   ten minutes, or write it off. Sections 7.2.2, 7.2.3, 7.2.4.
2. **Typing-time prefill**: 169-288 ms per token across four sessions against a batched turn's 90-116 ms.
   Measured five times, never explained.
3. **The gate has never been run at 44 GiB.** It can be now. `nll_expert_precision.py --experts runtime
   --tokens 512` was killed at a 24 GiB budget on 2026-09-21 and passed at 20; at 44, with the machine in
   the state it was in today, it should simply run.

## Rules that still hold

- **Never quote a screen as a gate** — and price a loop with one before ranking it. New this session.
- **Check that a profile's parts add up to its whole before ranking anything off it.**
- **Print what an arm read before attributing a difference to CPU work.** New this session.
- **Check which code path a number was measured on before ranking a lever off it.**
- **A thread does not hide Python work.**
- **An A/B only finds defects that differ between its arms.** Compare against
  `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/`.
- **Write a test when a comparison finds a match, not only when it finds a bug.**
- **Never drop a case from a denominator.**
- **A curve drawn through the turns of a live session is not a curve.**
- **Read a run's footprint line before its numbers.** A `predict_ghost` run at 13.3 GiB of compressor
  decoded at 392 ms/token against 190 clean; its *counts* were identical to five reads in 6,800, so the
  correlations survived and the timing was discarded.
- **Check a confound before reporting an effect.**
- **Memory.** `guarded_run.sh` needs `budget + 29` GiB available and sets no wired limit of its own. 36 and
  44 both ran on 2026-09-21; if one is refused, it is the rest of the machine, not the runtime. Never pass
  `--force`; lower the budget or close something.
- **Hand over `./chat.sh`, never the section 4 one-liner.** It does not survive line wrapping and the
  failure is silent — FP4 over USB at 0.6 tok/s. Section 12.
- One change at a time, measured. Terse in chat, complete prose in files. Full copy-paste commands.
- Do not run two runtimes at once. Launch each run as its own command.

## The instruments

| tool | what it answers | cost |
|---|---|---|
| `benchmarks/decode_anatomy.py` | **where a live token's time goes, split by blocking cause**; ±0.3 % at 44 GiB | ~1 min |
| `benchmarks/predict_ghost.py` | **what happens to a mispredicted expert after it is dropped**, and what a blocklist of any lifetime would have done | ~1 min |
| `benchmarks/profile_decode_sync.py` | the CPU/GPU split of a token per eval site, and what the arm read | ~1 min |
| `benchmarks/profile_decode_layers.py` | per-layer and per-class time with no barrier added | ~1 min |
| `benchmarks/profile_decode_gpu.py` | per-piece GPU time the way a token pays it, chained | ~2 min |
| `benchmarks/profile_decode_cpu.py` | the same token under cProfile | ~1 min |
| `benchmarks/micro_predict_submit.py` | the prediction submission bookkeeping, four arms | instant, no model |
| `benchmarks/micro_compile_hc_fused.py` | **the template for a CPU screen** | ~1 min, no model |
| `benchmarks/micro_compile_moe.py`, `micro_compile_router.py` | what `mx.compile` is worth on the MoE block, on the router | ~1 min, no model |
| `benchmarks/decode_rate_by_block.py` | the rate over a long generation, in blocks | ~5 min |
| `benchmarks/nll_expert_precision.py --experts runtime --tokens 512` | the production quality arm | ~2 min |
| `benchmarks/chat_turns.py --max-new-tokens 160` | six chat turns exactly as the CLI runs them | ~3 min |
| `benchmarks/coding_quality.py --resume` | the 40-case corpus, restartable | ~1.5 h |

## Reference points

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | **the official implementation** |
| `~/cachalot-runA-relace-fp4-scored/` | the hosted reference arm, 40/40 |
| `benchmarks/results/coding/q2g128-v17/` | **the 2-bit gate**, 40/40 |
| `benchmarks/results/guarded/anat44_*` | **the shipped configuration's profile** |
| `benchmarks/results/guarded/anat44ahead2_*`, `anat44k4_*`, `anat44k8_*`, `anat44si*` | the four nulls beside it |
| `benchmarks/results/guarded/ghost*_*` | the prediction-waste logs |
| `benchmarks/results/guarded/rateblk36w72_*`, `rateblk36auto_*` | the wired-limit control pair, 310 s and 313 s |
| `benchmarks/results/guarded/syncpred_*` | the arm that reads 380 MiB per "all-resident" token |
| HANDOFF section 7.1.3 | the 44 GiB profile and its A/Bs |
| HANDOFF section 9.19 | the three closed ways to reclaim the prediction's waste |
| `./chat.sh` | **the command to hand Hamed**: the shipped environment, exported, one runtime at a time |
