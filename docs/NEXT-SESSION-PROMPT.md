# Next-session prompt — **v14**, written 2026-09-20

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v14** | 2026-09-20 | the probes v13 asked for, run | the defect is **in-context copying**, and six suspects are cleared including the whole expert path and Engram. v13's Job 2, a 10-to-20-hour slow-path corpus run, is cancelled: it was built to bisect optimizations and the optimizations are not the fault. **The checkpoint ships the official implementation and nobody had opened it.** |
| v13 | 2026-09-20 | v12's knob table re-read out of `src/` | corrected knob table; cheap probes ahead of the overnight run; `--resume` added |
| v12 | 2026-09-20 | the independent reference arm | a hosted endpoint is clean where Cachalot is corrupt. Drafted and superseded the same day; never committed |
| v11 | 2026-09-19 | an outside review, acted on | the coding gate was broken and every C++ syntax-error figure withdrawn |
| v10 | 2026-09-19 | the FP4 bank finally profiled | mirror striping shipped; four levers closed |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read `docs/HANDOFF.md` section 7.4.7 first, then 7.4.6.** The raw tables are in
`docs/HANDOFF-2026-09-20-quality.md`.

## The goal, unchanged

Make Cachalot reproduce the original model's quality: every C++ block compiles, zero malformed `#include`
lines, matching the hosted reference arm. Throughput is not a constraint and not a tiebreaker. A slow
correct path is the deliverable.

## Read this before planning anything

**The official implementation ships with the weights.**
`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` contains `model.py`, `engram.py`, `kernel.py`,
`convert.py`, `generate.py`, and there is an `encoding/` directory beside it. No session before 2026-09-20
had opened it. Three bank-building sessions chased quality through quantization while a copy of the answer
sat in the model directory. Before reasoning about what the runtime *should* do, read what the reference
*does*.

It cannot be executed here — torch is not installed and the model is 552B — but it can be read, and reading
it cleared Engram in under an hour.

## What the defect is

Not blur, not FP4, not the sampler, not a kernel. **In-context copying.**

| class | n | rank 1 | mean logprob | worst rank |
|---|---:|---:|---:|---:|
| continuation, first occurrence | 28 | 86 % | -0.699 | 19 |
| continuation, **repeat of an earlier word** | 101 | **51 %** | -3.578 | **23,989** |
| everything else | 1,371 | 88 % | -0.595 | 3,703 |

Finishing a word for the first time is healthy. Copying one the text already spelled out collapses, and the
model prefers words from *neighbouring sentences* — it retrieves the wrong thing rather than nothing. That
covers every artefact on record: `#include <stdex>`, `LRUCache` written `LRcache`, `map_._.end()`,
`std std::string`, `utfutf-8`.

This is a within-run control. It needs no reference arm, because no correct model fails to copy a word it
emitted ten tokens ago.

## What is already cleared — do not re-test these

| suspect | evidence |
|---|---|
| detokenization | 34/34 include lines exact: batch, per line, one token at a time |
| the sampler | greedy manifest records `temperature 0.0`, `frequency_penalty 0.0`; that path is plain `argmax` |
| the prefill path | 540 tokens batched vs one at a time: 60 % vs 65 %, within noise |
| depth and position | identical block at token 568 scores 100 %; no decay over 640 positions |
| routed-expert kernel, streaming, cache | dense fp32 vs production, same weights: median NLL difference **+0.0000**, 96 % greedy agreement |
| Engram | nine checks against the shipped reference; prime sums match `engram_num_embeddings` exactly |
| fused decode, fused FP8, bf16 head | all three off together: 51 % against 51 %, unchanged |
| copy distance | controlled probe at 8/32/96/288 tokens: 73 %, 79 %, 62 %, 75 % — flat |

**v13's Job 2 is cancelled.** The 10-to-20-hour slow-path corpus run was designed to bisect optimizations,
and the expert path, the prefetch family and all three custom kernels are now cleared without it. Do not
start it. If a fix lands, the corpus is how it gets *confirmed*, not how it gets *found*.

## Job 1 — read `attention_compressed.py` beside `inference/model.py`

No machine time. This is the job.

`src/cachalot/model/attention_compressed.py` is 34 KB implementing V4.1's sparse attention and has never
been compared against the reference. The scheme, from `config.json`: `index_source_layer_ids = [2, 8, 14,
20, 24, 28, 32, 36]`, `kv_source_layer_ids = [2, 8, 14, 20]`, `candidate_source_layer_id = 20`,
`candidate_topk_blocks = 2048`, `candidate_block_size = 8`, `index_topk = 512`, `index_n_heads = 32`,
`index_head_dim = 128`, `sliding_window = 128`, `compress_ratios` 2 for layers 2-19 and 1 for 20-39.

Already compared and **matching** — do not redo:

- `get_window_topk_idxs`, prefill and decode, against `model.py:410`
- the whole Engram path, against `engram.py` and `model.py:296-380`

**Uncompared, in the order they are most likely to be wrong:**

1. **The RoPE positions the compressed latents are rotated with.** The reference (`model.py:527`, the
   `Indexer.forward` docstring) says a latent stands for the first token of its group, so group *j* takes
   position `j * ratio`, and at decode it indexes `self.freqs_cis[start_pos + 1 - ratio]`. That expression
   is exactly the kind an independent implementation gets subtly wrong, and a wrong position on a
   compressed key scrambles *which* earlier position a query matches — the measured symptom.
2. **The indexer's scoring and top-k.** Reference: `weights = weights_proj(x) * (softmax_scale *
   n_heads**-0.5)`, `index_score = einsum("bshd,btd->bsht", q, index_k)`, then
   `(index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)`, then `topk(...).indices.sort(dim=-1).values`,
   then `where(idxs < compress_lens, idxs + offset, -1)`. Check the relu, the weight broadcast, the sum
   axis, the re-sort into position order, and the `offset`.
3. **The compressor's partial-group state.** `latent` is `None` while a group is still filling, so during
   decode it only yields every `compress_ratio` steps and holds the partial group in `kv_state` /
   `score_state`. Check what the runtime attends to for tokens in an incomplete group.
4. **The candidate pre-filter.** Layer 20 publishes `shared_attn.candidates` via `select_candidate_blocks`;
   layers after it mask their own `index_score` to those blocks. Check that the publish/consume split
   matches `is_candidate_source` and `uses_candidates`.
5. **`compress_lens`** — how many compressed positions a query may see. Reference masks with
   `arange(seqlen // ratio) >= compress_lens` in prefill and uses `end_pos // ratio` in decode.

Write what each comparison found into the handoff as you go, matching or not. A component compared and
matching is a real result and stops the next session redoing it.

## Job 2 — when a candidate defect is found, prove it with a probe, not the corpus

The instruments are built and take minutes:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 5400 --tag copyprobe -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/token_rank_probe.py --probe <probe.txt> --prefill 16 --tokens 1500 --top-k 5 --out /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/<name>.json
```
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/continuation_rank.py benchmarks/results/<name>.json --worst 10
```

The repeat-continuation number is the gate for a fix: it must go from 51 % to near the run's own baseline.
Probe texts used on 2026-09-20 are in the job scratch directory and are cheap to rebuild — each long
identifier appears three times in its own sentence, which makes the second and third occurrences pure copies.

`benchmarks/nll_expert_precision.py --experts runtime|fp4` is the dense-reference comparison, about six and
eleven minutes respectively for 620 tokens.

## Job 3 — only once a probe is clean, confirm on the corpus

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 28800 --tag coding-fix -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/coding_quality.py --resume --out-dir /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/fix
```
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/fix/ /Users/hamedprooshani/cachalot-runA-relace-fp4-scored /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/
```

`--resume` is new and re-issuing the same command continues an interrupted run. It refuses to merge rows
written under a different configuration and names the field that differs.

Match the corrupt arm exactly so the fix is the only variable: 24 GiB budget, `max_seq_len` 8192, seeds
20260919 and 20260920, temperature 0.6, frequency penalty 0.2, 2000 max new tokens. Clean means
reference-equal compile rate and **0 malformed includes**.

## Rules that still hold

- **Never quote a screen as a gate.** This document has been wrong about screens five times.
- **Never drop a case from a denominator.** Three times now, most recently two missing Hermes sessions.
- **Check a confound before reporting an effect.** This session found a 40 %-against-77 % result that looked
  like a clean mechanism and was entirely rarity masquerading as distance. The controlled rerun killed it.
- **One change at a time, measured.**
- **Read the reference before theorizing.** It is on disk.
- Terse in chat, complete prose in files. Full copy-paste commands, no relative paths.
- Do not run two runtimes at once; `guarded_run.sh` refuses when one is alive. Note that its `pgrep`
  preflight also trips on a wrapping shell whose command line contains the pattern, so launch each run as
  its own command rather than from a loop or a heredoc that mentions the script name.

## What exists to compare against

| path | what it is |
|---|---|
| `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` | **the official implementation** |
| `~/cachalot-runA-relace-fp4-scored/` | the target: FP4 reference, 40 of 40, no harness, provider pinned |
| `~/cachalot-runA-deepinfra-fp8-scored/` | FP8 reference, the control |
| `benchmarks/results/coding/20260919-085931_*/` | Cachalot's corrupt 40-case arm, 24 GiB, `max_seq_len` 8192 |
| `benchmarks/results/rankprobe-*.json` | this session's probes: prod, decode16, prefill540, deep, cont, cont-nofused, dist |
| `benchmarks/results/nll_experts_*probe-long.json` | the runtime-vs-dense expert comparison |

## What this session is not

Not a speed session. Section 9 is frozen. Do not sweep a knob for throughput, do not build a bank, do not
re-open a closed lever, and do not start the slow-path corpus run — it was cancelled for cause.
