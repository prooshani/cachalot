# Next-session prompt — **v13**, written 2026-09-20

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v13** | 2026-09-20 | v12's knob table re-read out of `src/`, line by line | the plan is unchanged and the knob table is rebuilt. v12 asked for a cache budget this bank cannot have (eviction cannot be disabled on FP4), dropped `CACHALOT_EXPERT_BANK` from its own command, disabled mirror striping with a variable that does not disable it, and missed six knobs that are on by default — one of them `CACHALOT_BF16_HEAD`, a custom Metal kernel on the final logit path. Cheap probes moved ahead of the overnight run; `--resume` added as a prerequisite. |
| v12 | 2026-09-20 | the independent reference arm, finally run | a hosted endpoint serving the same model is **clean** where Cachalot is corrupt: the defect is in this runtime, not FP4, not the model, not the sampler. Speed work is frozen. The job is a bisection. **Drafted and superseded the same day; never committed, so there is no `v12-2026-09-20.md` in the archive.** Its findings are carried above in full; only its knob table was wrong. |
| v11 | 2026-09-19 | an outside review, acted on | the coding gate was broken and every C++ syntax-error figure withdrawn; predicted loads had no lifetime and now do; non-cumulative lookahead closed at ~24 ms/token |
| v10 | 2026-09-19 | the FP4 bank finally profiled | mirror striping shipped; speculation, eviction, prefetch lead time and prefetch precision all closed; dispatch count demoted |
| v9 | 2026-09-18 | the searched, weighted 3-bit bank built and gated | lever 0 closed; FP4's 0.6 corrected to 8.9 (**both withdrawn by v11**) |
| v8 | 2026-09-18 | FP4 vs 3-bit vs 2-bit, matched | 3-bit retired; FP4 on the internal SSD |
| v7 | 2026-09-18 | 3-bit bank built and gated | q3g64 adopted (later retired) |
| v6 | 2026-09-18 | chat-collapse investigation | quality over speed (**replaced by v12**) |

---

You are continuing work on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B parameters,
40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming routed
experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

**Read `docs/HANDOFF.md` section 7.4.6 and section 2 before anything else.** They are short and they
invalidate the framing of every prompt before this one.

## The goal of this session, in one sentence

**Make Cachalot reproduce the original model's quality.** Not most of it, not enough of it to be useful —
the same quality the hosted endpoint gives on the same corpus: every C++ block compiles, zero malformed
`#include` lines. Nothing else is being measured. Throughput is not a constraint, not a tiebreaker and not
worth a sentence in the write-up. Hamed's instruction, verbatim in substance: *it does not matter if the
runtime produces 0.1 tok/s — the goal is to reproduce the model's original quality.*

## What changed on 2026-09-20, and why it changes the whole project

The 40-case coding corpus was run against a hosted endpoint serving the same model, with the provider pinned
and **no harness at all** — bare API calls, no system prompt, no tools, no content retries, reasoning off,
the pack's own sampling settings.

| | Cachalot, FP4 | run A, `relace/fp4` | run A, `deepinfra/fp8` |
|---|---:|---:|---:|
| C++ blocks that compile | 0 / 42 | **20 / 20** | **20 / 20** |
| Python blocks that parse | 5 / 26 | **18 / 18** | **18 / 18** |
| `#include` lines malformed | 49 / 154 (32 %) | **0 / 102 (0 %)** | **0 / 103 (0 %)** |
| `import` lines malformed | 4 / 57 (7 %) | **0 / 32** | **0 / 31** |
| C++ errors per 100 lines | 42.2 | 0.0 | 0.0 |

Greedy, on the six tasks at temperature 0: Cachalot 0 of 6 compile with **1980 of 2004** include lines
malformed; the reference 6 of 6 with **0 of 34**.

All arms are 40 of 40 cases, each served by exactly one provider. **Three explanations are dead:**

- **Not FP4.** The clean arm *is* FP4, the same `expert_dtype` the checkpoint declares.
- **Not a harness effect.** Run A has no harness. A Hermes arm was also scored and it proves nothing about
  harnesses: every assistant turn of all 40 sessions was scanned, first drafts included, and **no malformed
  include appears anywhere**. Nine cases answered in one shot with no tool call. There was nothing to repair.
- **Not sampling.** Greedy keeps the gap at 99 % against 0 %.

**The defect is in Cachalot's own path.** Its shape is a single-token drop at a high-confidence position:
38 of 49 malformed lines in the sampled run and 1977 of 1980 in the greedy run are "missing opening
delimiter". That is a bug, not blur. A bug has a line number, and the work of this session is to find it.

## The method

Strip the runtime down to the slowest path that could possibly be correct, confirm it against the reference,
then reinstate one optimization at a time until the corpus breaks. The optimization that breaks it is the
culprit. That is a bisection, and it only works if two things hold: the gate is trusted — it is, since
section 7.4.1 — and each step's configuration is the one actually running, which is why Job 0 exists.

Two facts shape the ordering below, and they were both checked in `src/` on 2026-09-20:

1. **The cheap probes are very cheap and the corpus run is very expensive.** A tokenizer round-trip is
   seconds on the CPU. `token_rank_probe.py` is one forward pass. The slow-path corpus is a night or two.
   Run the cheap ones first; they cost nothing and they can reorder the suspect list before the expensive
   run starts.
2. **`benchmarks/coding_quality.py` has no resume, and `guarded_run.sh` SIGKILLs on `--max-seconds`.**
   `rows.json` is rewritten after every case, so a killed run leaves partial data and a manifest that
   correctly says `"complete": false` — but the next attempt starts from case 1. At the speeds this session
   will see that is a lost night. Fix it in Job 0 before starting anything long.

---

## Job 0 — the cheap checks and the tooling, before any long run

None of this needs the GPU for more than a moment, and the first two need no model load at all, so they can
run while something else occupies the machine.

**0a. Round-trip the tokenizer on the exact construction that fails.** A dropped `<` is as consistent with a
byte-level BPE decode fault as with a numerics fault, and it has never been checked. Encode and decode a
corpus of `#include <x>` lines — every header in `benchmarks/debug_fused_parity.py`'s own list is a ready
made set — plus the incremental-decode path the runtime actually uses during generation, one token at a
time, because a streaming detokenizer can drop a byte that a batch decode keeps. Compare byte for byte. This
is minutes of work and it would be embarrassing to find it in month two.

**0b. Verify the expert bytes are the checkpoint's bytes.** `~/DeepSeek-V4.1-Flash-fp4-experts` is 40 files
named `model-000NN-of-00048.safetensors`, 275 GB, and the claim is that they are copies of the
expert-carrying shards of `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash`. That is a layout consistent with a
plain copy rather than a re-encode, which is reassuring but is not the same as checked. Hash both copies and
compare. It reads about 550 GB, it takes minutes, and it retires an assumption that sits under every other
result.

```bash
cd /Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts && for f in model-*.safetensors; do printf '%s ' "$f"; shasum -a 256 "$f" | cut -d' ' -f1; done > /tmp/bank.sha256
cd /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash && while read -r f h; do printf '%s ' "$f"; shasum -a 256 "$f" | cut -d' ' -f1; done < /tmp/bank.sha256 > /tmp/checkpoint.sha256
diff /tmp/bank.sha256 /tmp/checkpoint.sha256 && echo "IDENTICAL"
```

**0c. Add `--resume` to `benchmarks/coding_quality.py`.** A case whose reply file already exists under
`--out-dir` is skipped and its row is carried forward from the existing `rows.json`. Keep the manifest
honest: `planned_cases` stays 40, `completed_cases` counts what is on disk, and `complete` stays false until
all 40 are there. This is not an optimization and it is not scope creep — without it a killed run cannot be
finished, and the run in Job 2 is the long pole of the session.

**0d. Write the test that proves a disabled knob is actually disabled.** This project has been burned three
times by a measurement that measured nothing (sections 7.4.1, 9.13, 7.4.6). Every constant in the table
below except six is read **at module import time**, so the env has to be set before the Python process
starts — an `env VAR=... python ...` prefix works, assigning into `os.environ` after the import does not,
and neither does exporting it from inside a test that already imported the module. Write
`tests/test_slow_path_knobs.py` that launches a subprocess with the slow-path env, imports the modules, and
asserts the values that are actually in force:

- `cachalot.model.moe_layer_metal.ASYNC_MOE is False` and `PREDICT_TOPK == 0`
- `cachalot.model.decode_fused_metal.FUSED_DECODE is False`
- `cachalot.model.fp8_linear_metal.FUSED_FP8 is False`
- `cachalot.model.moe_prefill_grouped.PREFILL_SGMM is False` and `SPECULATIVE_PREFILL is False`
- `cachalot.model.attention_prefill_batched.WOA_GEMM is False`
- `cachalot.model.moe_prefill_batched.PREFILL_BF16_GEMM is False`
- `cachalot.cache.resident_store.EVICT_POLICY == "lru"`

and then asserts the two that are decided per call rather than per import, by exercising them: that
`moe_layer_forward` selects the unfused branch, and that the head path does **not** reach
`bf16_gemv_f32` (`model_boundary_mlx.py:181`). No test in `tests/` covers any of these today — checked on
2026-09-20.

**0e. Record what is in force, from the run itself.** `coding_quality.py` already writes every `CACHALOT_*`
variable into `manifest.json` (`coding_quality.py:182`). Read that manifest after the run and paste it into
the handoff with the result. It is the receipt that the run was the run you think it was.

---

## Job 1 — `token_rank_probe.py`, which is written and has never been run

One forward pass, no sampling, no compiler, and it splits the remaining suspects in half before the
expensive run starts. Teacher-force a text dense in correct `#include <...>` lines and read where the model
ranks the token that should follow `#include`, with Python `import` lines as a within-run control — the
corpus says includes fail at 32 % and imports at 7 %, so a real defect should show that asymmetry here too.

Run it on two arms: the production path, then with `--experts fp4` dense substitution, which bypasses the
expert bank and its reader.

| rank on production path | rank on dense path | what it means |
|---|---|---|
| wrong | right | the bank, its reader, or the streaming and cache machinery around it |
| wrong | wrong | the trunk, the attention path, the head, or detokenization — and Job 2's knob list is aimed at the wrong half of the runtime |
| right | right | the model knows; the token is being lost after the logits, at sampling or detokenization |

The third row is the one worth pausing on. If `<` sits at rank 1 with high probability wherever it belongs
and the runtime still drops it, the defect is downstream of the forward pass entirely, and the whole knob
table below is irrelevant. Check that before spending a night on Job 2.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 3600 --tag rankprobe-prod -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/token_rank_probe.py --tokens 600 --top-k 5 --out /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/rankprobe-prod.json
```

Read `benchmarks/token_rank_probe.py`'s own header before running it; it documents the two arms and what
each one substitutes.

---

## Job 2 — build the slowest obviously-correct path and score it

### The knobs, verified against `src/` on 2026-09-20

Every row below was read out of the file named. The **default** column is the value that was in force in the
corrupt run, so every row is a live suspect.

| knob | default | set to | read at | note |
|---|---|---|---|---|
| `CACHALOT_ASYNC_MOE` | `1` | `0` | `model/moe_layer_metal.py:15` | import time. Drops the `mx.async_eval` dispatch |
| `CACHALOT_FUSED_MOE` | `1` | `0` | `model/moe_layer_metal.py:96` | **per call**, not import time |
| `CACHALOT_FUSED_DECODE` | `1` | `0` | `model/decode_fused_metal.py:179` | import time. Also switches the router from `route_topk_fused` to `route_topk` |
| `CACHALOT_FUSED_FP8` | `1` | `0` | `model/fp8_linear_metal.py:19` | import time |
| `CACHALOT_WOA_GEMM` | `1` | `0` | `model/attention_prefill_batched.py:53` | import time |
| `CACHALOT_BF16_HEAD` | `1` | `0` | `model/model_boundary_mlx.py:181` | **per call.** See below — this one is not in any earlier prompt |
| `CACHALOT_PREDICT_TOPK` | `6` | `0` | `model/moe_layer_metal.py:33` | import time. `0` disables prediction outright: `moe_layer_metal.py:119` gates the whole predictor on `PREDICT_TOPK > 0` |
| `CACHALOT_PREDICT_AHEAD` | `1` | leave | `model/moe_layer_metal.py:43` | dead once `PREDICT_TOPK=0` |
| `CACHALOT_PREDICT_LEAD` | `1` | leave | `model/moe_layer_metal.py:59` | dead once `PREDICT_TOPK=0` |
| `CACHALOT_PREDICT_WORKERS` | `2` | `1` | `cache/resident_store.py:175` | thread pool width |
| `CACHALOT_PREFETCH_DEPTH` | = workers | `1` | `io/resident_prefetch.py:31` | `0` is accepted and sets depth 0, whose queue behaviour is unverified — use `1` |
| `CACHALOT_SPECULATIVE_PREFILL` | `1` | `0` | `model/moe_prefill_grouped.py:38` | import time |
| `CACHALOT_PREFILL_SGMM` | `1` | `0` | `model/moe_prefill_grouped.py:35` | import time |
| `CACHALOT_PREFILL_BF16_GEMM` | `1` | `0` | `model/moe_prefill_batched.py:28` | import time |
| `CACHALOT_PREFILL_BATCHED_ATTN` | `1` | `0` | five prefill blocks, e.g. `model/block_layer0_prefill.py:178` | per call |
| `CACHALOT_PREFILL_BATCHED_HC` | `1` | `0` | `model/hc_prefill_exact.py:54` | per call |
| `CACHALOT_PREFILL_BATCHED_ENGRAM` | `1` | `0` | `model/text_decode_runtime.py:1564` | per call |
| `CACHALOT_ENGRAM_PREFETCH` | `1` | `0` | `model/text_decode_runtime.py:1566` | per call |
| `CACHALOT_RDAHEAD` | on | `0` | `storage/reader.py:53` | keeps the page cache, drops the readahead hint |
| `CACHALOT_EVICT` | `lru` | leave `lru` | `cache/resident_store.py:41` | import time. See the correction below |
| `CACHALOT_MIRROR_PATH` | unset | **leave unset** | `storage/reader.py:62` | striping is off only when the *path* is unset. `CACHALOT_MIRROR_FRACTION` alone does nothing |
| `CACHALOT_HOTLIST` | unset | leave unset | `model/text_decode_runtime.py:900` | |
| `CACHALOT_EXPERT_BANK` | — | **keep set** | `model/text_decode_runtime.py:369` | see below |
| `CACHALOT_MODEL_PATH`, `CACHALOT_PAGE_CACHE=1` | — | keep | | |

**Three corrections to v12, because acting on its knob table would have wasted a night.**

- **Eviction cannot be turned off on FP4, and the earlier instruction to "raise the budget high enough that
  nothing is evicted" is impossible.** An FP4 expert is 17.93 MiB, 57.1 experts per GiB of budget; the bank
  is 15,360 experts and 275.4 GiB on a 96 GiB machine. At the largest budget this machine safely runs, the
  resident set holds about 2,500 of 15,360. Eviction runs in every configuration, so it is not part of the
  slow path — it is a separate suspect, reinstated in Job 3 by *changing the policy*, not by disabling it.
- **`CACHALOT_EXPERT_BANK` must stay set.** Dropping it does not neutralize anything: it silently serves the
  same FP4 experts off the X10Pro at roughly a quarter of the speed (handoff section 4), which costs a night
  and changes no variable under test.
- **`CACHALOT_BF16_HEAD` is missing from v12 and from every prompt before it, and it belongs near the top of the list.** It
  is on by default and it routes the final logit computation — the 129,280 x 5,120 head GEMV, the last
  arithmetic before the token is chosen — through a hand-written Metal kernel, `bf16_gemv_metal.py`. A wrong
  row bound or a bad reduction there corrupts *specific* logits and leaves the rest intact, which is exactly
  the observed shape. Reading the source found no defect: 32 lanes per row, `K % 64 == 0` enforced, the
  simdgroup boundary aligns with the row boundary, and `simd_sum` is uniform across the row. But it has
  never been parity-tested against `F.linear`-equivalent dense arithmetic, and "I read it and it looked
  fine" is not a measurement. Disable it in the slow path and parity-test it in Job 3.

### Budget, and why it is 24 GiB and not 36

Match the corrupt arm exactly. Its manifest says `expert_budget_bytes: 25769803776` — 24 GiB — with
`max_seq_len: 8192`, seeds 20260919 and 20260920, temperature 0.6, frequency penalty 0.2, 2000 max new
tokens, at git `a103b845`. The whole value of this run is that the **only** difference from the corrupt run
is the knobs. Do not also change the budget, the sequence length, the seeds or the sampling.

### The run

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 86400 --tag coding-slowpath -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 CACHALOT_ASYNC_MOE=0 CACHALOT_FUSED_MOE=0 CACHALOT_FUSED_DECODE=0 CACHALOT_FUSED_FP8=0 CACHALOT_WOA_GEMM=0 CACHALOT_BF16_HEAD=0 CACHALOT_PREDICT_TOPK=0 CACHALOT_PREDICT_WORKERS=1 CACHALOT_PREFETCH_DEPTH=1 CACHALOT_SPECULATIVE_PREFILL=0 CACHALOT_PREFILL_SGMM=0 CACHALOT_PREFILL_BF16_GEMM=0 CACHALOT_PREFILL_BATCHED_ATTN=0 CACHALOT_PREFILL_BATCHED_HC=0 CACHALOT_PREFILL_BATCHED_ENGRAM=0 CACHALOT_ENGRAM_PREFETCH=0 CACHALOT_RDAHEAD=0 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/coding_quality.py --out-dir /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/slowpath
```

Note the explicit `--out-dir`: with `--resume` from Job 0c, a killed run is restarted by re-issuing the same
command against the same directory.

Then score it against the reference:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/slowpath/ /Users/hamedprooshani/cachalot-runA-relace-fp4-scored /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/
```
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/include_integrity.py /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/slowpath/ /Users/hamedprooshani/cachalot-runA-relace-fp4-scored /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/
```

Three arms in one call, so the slow path, the reference and the corrupt baseline appear in the same table.

### How long this takes, and what to do about it

The corrupt arm took 2 h 03 m for 40 cases at about 3 tok/s. The slow path removes the prefetcher, so every
expert read is a miss on the critical path, and it removes the async dispatch, so nothing overlaps. Decode
on FP4 already moves 1,858 MiB per token. Expect something in the region of 10 to 20 hours, possibly more,
and treat any estimate as a guess until the first few cases have run. `--max-seconds 86400` is a day;
raise it rather than shrink anything.

**Raise the wall clock, not the sample count.** Do not shrink the corpus. The denominator is the one thing
this project has repeatedly got wrong, three times now.

**A subset is allowed as a screen, and never as a gate.** `coding_quality.py --only cpp-csv-to-json,cpp-lru-cache,cpp-thread-pool`
plus `include_integrity.py` scores in seconds without a compiler and will show a 32 % include-malform rate
collapsing to 0 % long before the full run finishes. Use it to find out early whether the night is worth
starting. Never quote it as a result — handoff section 8.3 is about exactly what screens are worth, and this
document has been wrong about screens five times.

### Two outcomes, and they fork the session

- **Clean** — reference-equal compile rate and **0 malformed includes**. The defect is in an optimization.
  Go to Job 3.
- **Still corrupt** — the defect is below all of it. Go to Job 4. **Nothing in section 9 matters in that
  case.**

---

## Job 3 — bisect the optimizations, one at a time

Reinstate exactly one knob to its default. Re-run the corpus. Score it. Repeat. Order, highest-prior first,
but reorder on whatever Job 1 and Job 2's diagnostics suggest:

1. **`CACHALOT_BF16_HEAD`** — a custom kernel on the final logit path, never parity-tested. Before spending
   a corpus run on it, test it directly: compare `bf16_gemv_f32(x, W)` against `mx.matmul(x, W.T)` in fp32
   over real hidden states, and look at the rank of the argmax, not only the norm of the difference.
2. **`CACHALOT_FUSED_FP8`, then `CACHALOT_FUSED_DECODE`, then `CACHALOT_FUSED_MOE`** — custom Metal kernels
   are the highest-prior remaining suspects, because a wrong tile or row bound corrupts *some* values and
   leaves the rest intact, which is the observed shape. Note that `FUSED_DECODE` also switches the router
   between `route_topk_fused` and `route_topk`, so it is really two changes in one knob; if it is implicated,
   separate them.
3. **`CACHALOT_ASYNC_MOE`** — asynchrony over a shared expert cache.
4. **The prediction and prefetch family** — `PREDICT_TOPK` back to 6, then `PREDICT_WORKERS` to 2, then
   `PREFETCH_DEPTH`. Section 9.13 already found one lifetime bug here; a stale or released buffer read as an
   expert is a very plausible single-token corruptor.
5. **Eviction policy** — `CACHALOT_EVICT=lfu` and `slru`, as a policy comparison. Eviction itself is always
   on at this bank size and was on in the slow path too, so if the slow path was clean, eviction is already
   exonerated as a mechanism and only the policy code is left.
6. **`CACHALOT_MIRROR_PATH` plus `CACHALOT_MIRROR_FRACTION=0.10`** — the one that splits every expert read
   across two drives and reassembles it. Then `CACHALOT_HOTLIST`.
7. **The prefill family** — `PREFILL_SGMM`, `SPECULATIVE_PREFILL`, `PREFILL_BF16_GEMM`, the three
   `PREFILL_BATCHED_*`, `ENGRAM_PREFETCH`, `WOA_GEMM`, `RDAHEAD`.

**A step is "clean" only when it matches the reference**: reference-equal compile rate and 0 malformed
includes. Not "better than the last step".

**Screen each step before gating it.** `include_integrity.py` on a `--only` subset costs minutes; a step
that malforms includes is caught there. Use the full compile gate to confirm a clean screen, not to discover
a dirty one.

**Write each step into `docs/HANDOFF.md` as it finishes**, with its run directory and its manifest `env`
block, so the bisection survives a context reset. Do not batch the write-ups to the end.

---

## Job 4 — if the slow path is still corrupt

Then it is below the optimizations, and these are the probes, in order:

1. **Whatever Job 1 said.** If the rank probe already localized it, start there rather than here.
2. **Detokenization, if Job 0a has not already settled it.** Widen it: the incremental path during
   generation, not only a batch round-trip, and the sampler's interaction with `--frequency-penalty 0.2
   --penalty-window 128`. That penalty is applied over a 128-token window, and `<` after `#include` is a
   token that recurs constantly in C++ and rarely in Python — which is an asymmetry with the same shape as
   the defect. It was ruled out for *sampling* by the greedy run, but check that the penalty is not applied
   on the greedy path too before treating it as closed.
3. **The FP4 expert kernel against a dense reference**, per layer, on real activations.
   `benchmarks/debug_fused_parity.py` exists; extend it if it does not already cover the routed path end to
   end. Compare argmax rank, not only tensor norms — a blur metric will not see a single flipped logit.
4. **The expert-bank reader**, if Job 0b came back identical and the kernel is clean. Bytes on disk being
   right does not mean the right bytes reach the kernel: check the index, the offsets and the mirror
   reassembly path in `storage/reader.py`.

---

## Rules that still hold

- **Never quote a screen as a gate.** This document has been wrong about screens five times.
- **Never drop a case from a denominator.** A short run that looks complete is this project's most expensive
  recurring mistake — three times now, most recently two missing Hermes sessions on 2026-09-20.
- **`code_validity.py` reads the compiler's exit status** and `include_integrity.py` tests a pre-registered
  prediction. Both are trustworthy now. Section 7.4.1.
- **One change at a time, measured.** That is the whole method this session.
- **Verify a knob is honoured before trusting a run that depends on it.** Job 0d.
- **Do not run two runtimes at once.** Two will exhaust memory; the `pgrep` guard in section 4 is not
  decoration, and `guarded_run.sh` refuses to start when another runtime is alive. Job 0a and 0b are the
  only work here that is safe to run alongside a long benchmark, because neither loads the model.
- Terse in chat, complete prose in files. Full copy-paste commands, no relative paths.

## What exists to compare against

| path | what it is |
|---|---|
| `~/cachalot-corpus-pack/` | the 20 prompts, `settings.json`, `run_reference.py`, `score.sh`, `RUN.md` |
| `~/cachalot-runA-relace-fp4-scored/` | **the target.** FP4 reference, 40 of 40, no harness, provider pinned |
| `~/cachalot-runA-deepinfra-fp8-scored/` | FP8 reference, 40 of 40 — the control |
| `~/cachalot-runA-relace-fp4-greedy-scored/` | six tasks at temperature 0, paired with Cachalot's greedy run |
| `~/cachalot-runB-hermes-scored/` | the Hermes harness arm, and an `ANALYSIS.md` on why it answers a different question |
| `benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/` | Cachalot's 40-case arm, the one being fixed — 24 GiB budget, `max_seq_len` 8192, git `a103b845` |
| `benchmarks/results/coding/20260919-140553_DeepSeek-V4.1-Flash-fp4-experts/` | Cachalot's greedy six |

The corpus is 20 tasks x 2 seeds: ten `cpp-*`, ten `py-*`, of which `cpp-string-split-snippet` has
`expect_compiles: false` because the prompt forbids includes and `main`. It belongs in the snippets column,
not the compile column; every `-scored` directory carries the `manifest.json` and `rows.json` that make
`code_validity.py` honour that.

To re-run the reference arm at any time:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && set -a && . ~/.hermes/.env && set +a && python3 benchmarks/run_reference.py --pack ~/cachalot-corpus-pack --out ~/cachalot-runA-relace-fp4 --model deepseek/deepseek-v4.1-flash --provider relace/fp4 --attempts 5 --sleep 1.5
```

Pinning is fiddly: with fallbacks off, OpenRouter returns `404 No endpoints found` if the pinned provider
does not support every parameter sent. First-party `deepseek` lacks `seed` and 404s; `sail-research/fp4`
returns 429 on every attempt; `relace/fp4` serves. Section 7.4.6.

## What this session is not

Not a speed session. Section 9 of the handoff is frozen and says so at the top. Do not sweep a knob for
throughput, do not build a bank, do not re-open a closed lever, do not report a tokens-per-second figure as
though it were a result. If Job 2 comes back clean and Job 3 names the culprit in an afternoon, the correct
next move is to write it up and stop, not to start re-optimizing.
