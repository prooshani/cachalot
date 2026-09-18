# Next-session prompt — **v9**, written 2026-09-18

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v9** | 2026-09-18 | the searched, weighted 3-bit bank built and gated | lever 0 closed; FP4's 0.6 corrected to 8.9; dispatch count is now the top lever |
| v8 | 2026-09-18 | FP4 vs 3-bit vs 2-bit, matched | 3-bit retired; FP4 on the internal SSD; searched fit above 2 bits named the top lever |
| v7 | 2026-09-18 | 3-bit bank built and gated | q3g64 adopted (later retired) |
| v6 | 2026-09-18 | chat-collapse investigation | quality over speed; repetition and code-validity gates |
| v5 | 2026-09-17 | DSpark / compute floor | compute floor 93 ms; DSpark 1.20x; hotlist shipped |

---

You are continuing performance engineering on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B
parameters, 40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming
routed experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

## Read `docs/HANDOFF.md` in full first

Section 2 carries a standing decision that governs everything. Section 9 opens with the ranking, which
changed on 2026-09-18. Section 12.1 is how to read a live session's counters without misreading them. The
dated logs (`docs/HANDOFF-2026-09-16.md`, `-09-17.md`) are derivations only; section 10 lists which of their
conclusions have expired — and **four entries were added to it on 2026-09-18**, one of which corrects a
number this document itself published.

## The standing decision, already made

**Quality over speed.** Hamed will trade decode speed for code that compiles. Do not re-open it.

FP4 is the bank in use and nothing has come close to it. On matched three-seed arms, C++ blocks only:

| bank | C++ syntax errors per 100 lines | collapses, long turns | decode |
|---|---:|---:|---:|
| **FP4** (in use) | **8.9** | 0 of 6 | 3.3 tok/s @36 GiB |
| 3-bit g64, searched + activation-weighted | 31.6 | 2 of 6 | ~4.7 tok/s @44 GiB |
| 3-bit g64, `mx.quantize` | 19.6 | — | ~4.7 tok/s @44 GiB |
| 2-bit g128 | 25.3 | — | 5.5 tok/s @36 GiB |

**Two quantized expert banks above 2 bits have now been built, gated and rejected.** Do not build a third
expecting quality: section 9.0's conclusion is that **bits, not fit, are binding**, and MLX offers nothing
between 3 bits and FP4's 4.25 (4-bit affine at group 128 is byte-identical to FP4 and strictly worse).

## What the last session established, because it constrains what you trust

1. **The searched fit was improved as far as it usefully goes, and it did not matter.** `pack_bits` writes
   MLX's layout at any width; the searched fit plus least-squares refinement plus activation weighting cuts
   3-bit routed-expert output error **28 %**, measured on the model's own recorded activations and shown to
   transfer across texts. The bank built from it is worse than the one it replaced on the only metric that
   decides. **A screen metric improving is not evidence. This is the third time.**
2. **`FP4 = 0.6 errors per 100 lines` was wrong** — 163 lines from an arm the memory guardian truncated at
   two of three seeds, carried by one lucky clean block. It is **8.9** on 461 lines. Check that every arm ran
   to completion before comparing arms; the result file cannot tell you it is short, but the guarded run's
   log can.
3. **Python does not discriminate between banks; long C++ does.** Python blocks score 2.2 errors per 100
   lines on FP4 and on the rejected 3-bit bank alike. Gate on the hardest thing the model is asked to write.
4. **`capture_activations.py` exists now** and is the first tool here that looks at what the model actually
   multiplies. Inside an average group of 64 columns the mean square of the input spans a factor of 7.9.
   Quantization is not obviously its only use.

## Job 1 — dispatch count (section 9.2), now the top lever

93 ms of compute per token over roughly 400 GPU dispatches, none dominating; the model spends longer in
hyper-connections (74 ms across 80 sublayers) than in its routed experts (24.6 ms). `mx.compile` over a
layer, or one kernel for the hyper-connection triple. **Bank-independent, so it pays whatever is mounted, and
it is the largest lever that depends on nothing else.**

Deciding measurement: count and time dispatches directly with `profile_decode_gpu.py`, fuse the
cheapest-to-fuse group, and re-run `decode_resident.py` — the only clean read of the floor. Profiling is
hours; fusion is days.

## Job 2 — DSpark speculative decoding (section 9.1), parked pending arithmetic

It accepts 2.85 tokens per main forward and projected 1.20x, but its cost is the bytes a K-position
verification reads, which scale with expert size — on FP4 that is 1.9x worse than the 2-bit bank the
projection was computed on. **Redo the arithmetic before building anything**, which is free:
`speculation_bytes.py` and `speculation_policy.py` need no model. If it still projects above about 1.15x on
FP4's 17.93 MiB experts it is worth a session; if not, it closes.

## Ground rules

1. **Memory safety is not optional.** Never an automatic budget; every benchmark through
   `benchmarks/guarded_run.sh`; one runtime at a time. **A budget safe on one bank is not safe on another** —
   FP4 experts are 17.93 MiB against the 2-bit bank's 9.49. FP4 arms want 24 GiB unless measured otherwise.
2. **Measure before changing behaviour**, one change at a time, interleaved both ways, `settle.sh` between.
3. **Two arms per side is not an A/B**, and a *stochastic* failure needs a rate, not an A/B. Four seeds
   minimum. And **check that each arm finished**: a guarded arm killed at its timeout looks like a complete
   result in the JSON, which is how an order-of-magnitude error reached this document.
4. **A teacher-forced gate cannot see a free-running failure.** Numerics and sampling changes need
   `repetition_quality.py` and `code_validity.py` as well as `nll_expert_precision.py`.
5. **Compare arms on matched generations.** A code-validity ratio is meaningless unless the arms produced
   comparable block lengths — check `avg block` before believing it — and meaningless across languages,
   because short Python does not separate banks that long C++ separates 3.5x.
6. **Judge a quality arm by the paired median and the sign test, never the mean NLL** (paired SE ≈ 0.04 nats).
7. **A screen is for choosing what to gate, never for predicting what a gate will say.** Section 8.3. This
   has now been established three times, most recently with a screen deliberately made more faithful.
8. **Nothing timing-sensitive is valid while anything else is on the GPU.** `kill -STOP` a background build.
9. **Watch your own helper processes.** `pgrep -f foo` matches the shell running it (use `[f]oo`); an inline
   `bash -c` monitor puts the whole benchmark command in its own command line, so `guarded_run.sh`'s
   preflight sees a phantom runtime; a log can contain the sentinel a waiter greps for. **Put anything
   long-running in a script file and kill by recorded PID, never by pattern.**
10. **Every repository edit goes through shell commands**; every command copy-paste ready with absolute `cd`,
    `PYTHONPATH=src` and the full interpreter path.
11. **After each production patch:** byte-compile, focused test, `git diff --check`, inspect the diff.

## How to start

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && git log --oneline -3 && git status --short && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q tests && /bin/df -g /System/Volumes/Data | tail -1 && ls -d /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash /Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128
```

The command Hamed runs. Keep it working; hand it back verbatim when asked:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && pgrep -fl "deepseek-v41/bin/python|cachalot" || CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json CACHALOT_HOTLIST_GIB=8 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat --expert-budget-gib 44 --max-seq-len 32768 --max-new-tokens 4096 --temperature 0.6 --frequency-penalty 0.2 --penalty-window 128
```

Swap `CACHALOT_EXPERT_BANK` to `.../DeepSeek-V4.1-Flash-q2g128` for speed over quality, and check the
`expert bank:` line at startup — a wrapped copy-paste that loses the variable silently serves FP4 from the USB
drive at a quarter of the speed. The frequency penalty is not decoration: without it 62 % of long code replies
collapse into a repeating loop.
