# Next-session prompt — **v8**, written 2026-09-18

**This is the file to paste.** `docs/NEXT-SESSION-PROMPT.md` is always current; superseded ones live in
`docs/next-session-prompts/`.

| version | written | produced by | what changed |
|---|---|---|---|
| **v8** | 2026-09-18 | FP4 vs 3-bit vs 2-bit, matched | 3-bit retired; FP4 on the internal SSD; searched fit above 2 bits is the new top lever |
| v7 | 2026-09-18 | 3-bit bank built and gated | q3g64 adopted (later retired) |
| v6 | 2026-09-18 | chat-collapse investigation | quality over speed; repetition and code-validity gates |
| v5 | 2026-09-17 | DSpark / compute floor | compute floor 93 ms; DSpark 1.20x; hotlist shipped |

---

You are continuing performance engineering on **Cachalot**, an MLX runtime that runs DeepSeek V4.1 Flash (552B
parameters, 40 layers, 384 routed experts per layer, top-6) on a single 96 GiB Mac Studio M3 Ultra by streaming
routed experts from SSD. The user is Hamed; he runs the interactive model himself in a separate terminal and
expects terse replies in chat, complete prose in files.

## Read `docs/HANDOFF.md` in full first

Section 2 carries a standing decision that governs everything. Section 9.0 is your first job. Section 12.1 is
how to read a live session's counters without misreading them. The dated logs
(`docs/HANDOFF-2026-09-16.md`, `-09-17.md`) are derivations only; section 10 lists which of their conclusions
have expired.

## The standing decision, already made

**Quality over speed.** Hamed will trade decode speed for code that compiles. Do not re-open it.

What that cost to learn, because it constrains what you try next:

| bank | syntax errors per 100 generated lines | decode |
|---|---:|---:|
| **FP4** (in use) | **0.6** | 3.3 tok/s @36 GiB |
| 3-bit g64 | 19.6 | ~4.7 tok/s @44 GiB |
| 2-bit g128 | 25.3 | 5.5 tok/s @36 GiB |

A 3-bit bank was built, gated, adopted and **retired the same day**. It bought +5.3 points of top-1 and
nothing you can compile. **Do not assume a small quality metric predicts usable output** — top-1 and NLL both
said 3-bit was a clear win, and a compiler said it was not.

## State

Cachalot 0.5.0, `main` clean and pushed, 100 tests passing. FP4 experts are on the internal SSD
(`DeepSeek-V4.1-Flash-fp4-experts`, 275.4 GiB, 40 expert-bearing shards copied from the USB checkpoint, which
is untouched and still serves trunk, Engram, head and tokenizer). 70 GiB free. `q2g128` is kept as the fast
option; `q2g64` and `q3g64` are deleted.

## Job 1 — the searched fit above 2 bits (section 9.0)

The only remaining route to a bank that is both smaller than FP4 and good enough to use.

`mx.quantize`'s affine fit is max-abs symmetric and wastes one level at every width; the 3-bit bank that
failed was built with it. The searched fit, generalised beyond 2 bits on 2026-09-18, cuts 3-bit output error
by **26 %** — 0.3521 to **0.2600**, past the calibrated oQ3e download's 0.2996 — at 14.77 MiB per expert,
**18 % smaller than FP4**.

1. **Write `pack_3bit`**, pinned against `mx.quantize`'s own layout exactly as `tests/test_quant_affine.py`
   pins the 2-bit packer. MLX packs 3-bit across word boundaries, so the 2-bit packer does not generalise.
   This is the blocker; `quant_affine.dequantized()` can already screen any width without it.
2. Let `build_affine_bank.py` take a searched fit above 2 bits, build, verify.
3. **Gate with `code_validity.py` on the matched long-C++ conversation** (`--conversation`), not with NLL.
   The bar is FP4's 0.6 per 100 lines. The screen cannot answer this: the map from output error to syntax
   errors is steeply non-linear.

If it lands near 19.6, bits rather than fit are binding and this lever closes for good.

## Job 2 — prefetch width on FP4

A sweep was **running when the last session ended**; its results are in
`benchmarks/results/guarded/pf-*.out` and `$CLAUDE_JOB_DIR/tmp/prefetch_sweep.log`. **Read them before
re-running.**

Why it matters: FP4 decode wasted **34.2 predicted loads per token — 613 MiB read and thrown away**, at 55 %
precision, on top of 1,248 MiB of real misses. The top-6 width was tuned on 9.49 MiB experts; at 17.93 MiB
every wasted read costs 1.9x as much. Widths 0, 2, 3, 4 and 6 were swept three times, interleaved both ways.

## Job 3 — dispatch count (section 9.2)

93 ms of compute over roughly 400 GPU dispatches, none dominating; the model spends longer in
hyper-connections (74 ms across 80 sublayers) than in its routed experts. `mx.compile` over a layer, or one
kernel for the hyper-connection triple. Bank-independent, so it pays whatever is mounted.

**Parked:** DSpark speculative decoding. It accepts 2.85 tokens per main forward and projected 1.20x on the
2-bit bank, but its cost is the bytes a K-position verification reads, which scale with expert size — on FP4
that is 1.9x worse. Redo the arithmetic before building anything.

## Ground rules

1. **Memory safety is not optional.** Never an automatic budget; every benchmark through
   `benchmarks/guarded_run.sh`; one runtime at a time. **A budget safe on one bank is not safe on another** —
   FP4 experts are 17.93 MiB against the 2-bit bank's 9.49, and an FP4 arm at 36 GiB drove wired to 58.5 GiB
   and was killed correctly. FP4 arms want 24 GiB unless measured otherwise.
2. **Measure before changing behaviour**, one change at a time, interleaved both ways, `settle.sh` between.
3. **Two arms per side is not an A/B**, and a *stochastic* failure needs a rate, not an A/B: two conclusions
   in the 2026-09-18 session survived five replies and died on the sixth. Four seeds minimum.
4. **A teacher-forced gate cannot see a free-running failure.** Numerics and sampling changes need
   `repetition_quality.py` and `code_validity.py` as well as `nll_expert_precision.py`.
5. **Compare arms on matched generations.** A code-validity ratio is meaningless unless the arms produced
   comparable block lengths — check `avg block` before believing it. A confounded "4.6x" was published and
   withdrawn on 2026-09-18 for exactly this.
6. **Judge a quality arm by the paired median and the sign test, never the mean NLL** (paired SE ≈ 0.04 nats).
7. **Nothing timing-sensitive is valid while anything else is on the GPU.** `kill -STOP` a background build.
8. **Watch your own helper processes.** Three distinct self-reference bugs cost real experiments in one
   session: `pgrep -f foo` matches the shell running it (use `[f]oo`); an inline `bash -c` monitor puts the
   whole benchmark command in its own command line, so `guarded_run.sh`'s preflight sees a phantom runtime and
   refuses the next arm; and a log can contain the sentinel string a waiter greps for. **Put anything
   long-running in a script file and kill by recorded PID, never by pattern.**
9. **Every repository edit goes through shell commands**; every command copy-paste ready with absolute `cd`,
   `PYTHONPATH=src` and the full interpreter path.
10. **After each production patch:** byte-compile, focused test, `git diff --check`, inspect the diff.

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
