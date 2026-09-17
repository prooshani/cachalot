# Cachalot — Engineering Handoff

**Authoritative state as of 2026-09-17, end of the third session of that day.** This document supersedes
`HANDOFF-2026-09-16.md` and `HANDOFF-2026-09-17.md` wherever they differ. Those two remain as the session
logs: they carry the derivations, the discarded attempts and the raw tables behind the numbers quoted here,
and section 14 indexes them. Read this document in full before running anything or proposing any change.

**Version:** Cachalot 0.4.0, tag `v0.4.0`, `main` clean and pushed to `github.com/prooshani/cachalot`, 60
tests passing.

---

## 1. What this project is

Cachalot is an MLX runtime that runs **DeepSeek V4.1 Flash** — 552B parameters, 40 layers, 384 routed experts
per layer, top-6 routing — on a single **96 GiB Mac Studio M3 Ultra**, by holding the trunk wired in memory and
streaming routed experts from SSD on demand. The model does not fit in memory; the design question is therefore
always the same one: how few expert bytes can a token be made to cost, and how much of the reading can be
hidden under computation.

## 2. Headline state

| | 2026-09-16 | now |
|---|---|---|
| decode, 36 GiB budget, 512-token prompt | 275 ms/token, 3.64 tok/s | **182.5 ms/token, 5.48 tok/s** |
| interactive chat, 44 GiB budget | 4.3–5.6 tok/s | **6.0–7.5 tok/s** |
| expert bank | 3-bit affine, 14.77 MiB/expert | 2-bit affine, 9.49 MiB/expert |
| bytes read per decoded token | 1030 MiB | 486 MiB |
| expert hit rate, benchmark / chat session | 73.4 % / — | 80.9 % / 87.3 % |
| quality, production path, 512 tokens | 2.4997 nats, 50.8 % top-1 | 2.5187 nats, 44.5 % top-1 |

Two changes produced that, both from 2026-09-17: a 2-bit expert bank built here from the FP4 checkpoint, and
`CACHALOT_PREDICT_TOPK` raised from 3 to 6, which only became worth doing once experts got smaller.

**The most important structural fact in this document: decode is now compute-bound.** Per decoded token at a
36 GiB budget, compute is 120 ms, the drive contributes 73 ms of transfer that hides underneath it, the drive
sits idle 55 % of the decode, and 62 ms is exposed wait. Every ranking this project carried before today
assumed the opposite, and several conclusions were reached under that assumption and must not be reused
blindly; section 10 lists them.

## 3. The machine

```
repository   /Users/hamedprooshani/Projects/deepseek-v41-mac      (github.com/prooshani/cachalot)
interpreter  ~/venvs/deepseek-v41/bin/python                      (3.14.7)
always       PYTHONPATH=src
MLX          0.32.2
hardware     Mac Studio M3 Ultra, 96 GiB unified memory, Darwin 27.2.0
Metal recommended working set  77.8 GiB
```

### 3.1 Storage — read this before running anything

| what | where | size | speed |
|---|---|---|---|
| FP4 checkpoint — **only complete copy** | `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash` | 475.2 GiB | 1.0 GB/s, USB 3.2 Gen 2 |
| 3-bit oQ3e bank — **only copy** | `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-oQ3e-mtp` | 331 GB | 1.0 GB/s |
| 2-bit g128 bank, **in use** | `/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128` | 142.4 GiB | 6.6–6.8 GB/s cold |
| 2-bit g64 bank, alternative | `/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g64` | 158.2 GiB | same |
| free space, internal | | 190 GiB | |

The X10Pro must stay connected. It holds the only copy of the FP4 checkpoint, the only copy of the 3-bit bank,
and the Engram tables the runtime reads on every prefill. The internal copy of the 3-bit bank was deleted on
2026-09-17 to make room for the 2-bit banks, after verifying that the X10Pro copy can restore it; restoring is
a 331 GB copy at 1 GB/s.

The runtime reads its trunk, Engram tables, head and tokenizer from `CACHALOT_MODEL_PATH` and its routed
experts from `CACHALOT_EXPERT_BANK`. Those are independent, which is why switching banks is one environment
variable and no code.

### 3.2 Checkpoint composition, measured from the shard headers

The FP4 checkpoint is 48 shards totalling 475.2 GiB: trunk 10.4 GiB smeared across 46 shards that also hold
experts, routed experts 275.7 GiB across 43 shards, and Engram 189.1 GiB in exactly two pure shards, 47 and 48.
Engram is therefore trivially separable and the trunk is not. Engram layers are 1 and 14; each holds
`embed.weight` as F8_E4M3 of shape [384006168, 256] plus `embed.scale` as F8_E8M0 of shape [384006168, 8].

It also contains **three complete multi-token-prediction layers**, `mtp.0`, `mtp.1` and `mtp.2`: 2,401 tensors,
7.39 GiB in total, of which 6.72 GiB is routed experts. Each layer has **128 routed experts** of its own
(ids 0 to 127, the same six-tensor FP4 layout as a trunk expert at 18,800,640 B each) plus its own attention,
norms and hyper-connections. Three layers means a draft depth of up to three tokens. Nothing in the runtime
uses any of it today. Section 9.1 is about that.

### 3.3 Expert formats

| | FP4, shipped | 3-bit oQ3e | 2-bit g64 | 2-bit g128, in use |
|---|---|---|---|---|
| bytes per expert | 18,800,640 | 15,482,880 | 11,059,200 | 9,953,280 |
| MiB per expert | 17.93 | 14.77 | 10.55 | 9.49 |
| experts per GiB of budget | 57.1 | 69.3 | 97.1 | 107.9 |
| bank total | 275.7 GiB | 221.5 GiB | 158.2 GiB | 142.4 GiB |
| layout | six tensors, two contiguous reads | nine stacked tensors, nine reads | same | same |
| encoding | E2M1 nibbles, UE8M0 scales, group 32 | MLX affine, bf16 scales and biases | affine, searched fit | affine, searched fit |
| built by | Meta | `Jundot/DeepSeek-V4.1-Flash-oQ3e-mtp` | `benchmarks/build_affine_bank.py` | same |

## 4. The configuration to use

**Interactive chat, machine otherwise idle.** This is the command Hamed runs himself, in his own terminal.
Keep it working and hand it back verbatim whenever he asks to try the model.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && pgrep -fl "deepseek-v41/bin/python|cachalot" || CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat --expert-budget-gib 44 --max-seq-len 8192 --max-new-tokens 1024 --temperature 0.6
```

**Interactive chat, other applications open.** Identical but `CACHALOT_MLX_WIRED_LIMIT_GIB=64` and
`--expert-budget-gib 36`. At a 44 GiB budget the runtime wires 72 GiB of a 96 GiB machine, which is the
configuration class that panicked this machine twice on 2026-09-15.

The `pgrep` prefix is not decoration: two runtimes at once will exhaust memory. If it prints a process, do not
start another one.

Why each variable is there: `CACHALOT_MODEL_PATH` points at the FP4 checkpoint for trunk, Engram, head and
tokenizer; `CACHALOT_EXPERT_BANK` selects the routed-expert bank; `CACHALOT_PAGE_CACHE=1` leaves the OS page
cache enabled, which is free here and slightly ahead on genuine repeat hits; `CACHALOT_MLX_WIRED_LIMIT_GIB`
sets the Metal residency limit, without which macOS compresses cold expert buffers and decode collapses to
seconds per token; `--expert-budget-gib` is always explicit, never automatic.

## 5. Operating rules

1. **Memory safety is not optional.** This machine kernel-panicked twice on 2026-09-15 because the runtime ran
   with an automatic expert budget and wired about 67 GiB while other applications were open: swap grew to 29
   swapfiles, the compressor hit its segment limit, and `kernel_task` spun until the watchdog fired. Never run
   with an automatic budget. Every benchmark goes through `benchmarks/guarded_run.sh`, one runtime process at a
   time, never two in parallel. If its preflight refuses a budget, **lower the budget**; do not force it. A
   36 GiB budget needs 65 GiB available and other applications can hold 8 GiB wired on their own.
2. **Measure before changing behaviour.** One architectural change at a time. A/B every optimization with the
   arms run sequentially, interleaved in both orders, with `benchmarks/settle.sh` between them. Two arms
   started back to back share memory and produce outliers.
3. **Two arms per side is not an A/B.** Decode throughput's run-to-run spread reaches 7 %, so a 5 % effect
   needs at least four runs per side before it is real. The prediction-width decision in section 8.2 was made
   on six per side for this reason, after two per side had produced a misleading table.
4. **Quality is gated, not assumed, and gated on the production path.** Any change touching expert or Engram
   numerics must pass `benchmarks/nll_expert_precision.py` before adoption. The dense reference arms
   (`--experts fp4`, `--experts oq3e`, `--experts requant`) rank *weights*; the production arm
   (`--experts runtime`) ranks what the model actually computes. **The two disagree in sign** between the
   3-bit and 2-bit banks (section 7.3). Run the production arm. Use 512 tokens, not 160, for any top-1
   comparison: the binomial standard deviation at 160 tokens is 3.9 points, which is wider than the effects
   being judged.
5. **Judge numerics by teacher-forced NLL and top-1, never by comparing greedy text.** This model's greedy
   decoding flips tokens on changes as small as one floating-point unit.
6. **Every repository edit goes through shell commands**, never prose asking Hamed to edit a file by hand.
7. **Every command given to Hamed is complete and copy-paste ready**: absolute `cd`, `PYTHONPATH=src`, the full
   interpreter path `~/venvs/deepseek-v41/bin/python`. Never a bare `python`, never an ellipsis. Repeat the
   full command in every message that asks for something to be run.
8. **After each production patch**: byte-compile, run the focused test, `git diff --check`, inspect the diff.
   Keep benchmark scripts out of runtime code.
9. **Chat replies terse. Prose in files, commits and documents stays normal and complete.**

## 6. Where the time goes

`benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64`, 2-bit g128 bank, 36 GiB budget:

    decode 64 tokens: 12.22 s = 5.24 tok/s (191 ms/token)          [top-3 prediction]
      expert hit rate 80.9% | 45.8 misses/token | 486 MiB read/token
      expert wait 4.53 s =  70.8 ms/token (37.1% of decode)
      rest        7.69 s = 120.1 ms/token (62.9% of decode)
      drive busy 5.49 s of 12.22 s decode (45.0%); 2.03 reads in flight while busy
      blocked total           4.53 s =  70.8 ms/token
      a demand read in flight 4.36 s =  68.1 ms/token (96.2% of blocked) -- never predicted
      only a predicted read   0.10 s =   1.6 ms/token ( 2.2%) -- predicted, issued too late
      no read outstanding     0.07 s =   1.0 ms/token ( 1.5%) -- slots, eviction, locks

The arithmetic that should drive every decision below, per token at a 36 GiB budget with top-6 prediction:

    bytes      45.8 misses x 9.49 MiB   = 486 MiB
      at 6.7 GB/s, what the drive gives =  73 ms
    compute                             = 120 ms
    measured                            = 182 ms

So the ceiling on anything that only improves overlap is about **120 ms per token, 8.3 tok/s**, and the ceiling
on anything that only reduces bytes is nothing at all, because bytes already hide under compute. The 62 ms
between 120 and 182 is exposed wait, 96 % of which is misses that were never predicted.

There is still no fixed per-read latency tax: 3.28 ms for a 9.49 MiB expert is 2.9 GB/s per stream, the same
per-stream rate the 15.48 MiB expert gave at 5.28 ms. Reads got smaller, not slower. While the drive is busy the
runtime moves about 6 GB/s of the 6.7 GB/s available, so there is no bandwidth left to recover at a given
concurrency; what changed is that the drive is busy far less often.

## 7. Measured baselines

### 7.1 Decode, 2-bit g128 bank

| budget | prediction | ms/token | tok/s | hit rate | misses/token |
|---|---|---|---|---|---|
| 36 GiB | top-3 | 190–193 (median 192.5) | 5.19–5.26 | 80.9 % | 45.8 |
| 36 GiB | **top-6** | **180–184 (median 182.5)** | **5.43–5.55** | 81.0 % | 45.7 |
| 36 GiB | top-8 | 184–190 | 5.26–5.43 | — | — |
| 36 GiB | top-12 | 202–219 | 4.56–4.95 | 80.8 % | 46.0 |
| 28 GiB | top-3 | 202 | 4.94 | 76.9 % | 55.4 |

2-bit g64 at 36 GiB and top-6: 189–191 ms/token, 81.2 % hit, 45.2 misses. Cold 512-token prefill is 16.4 s with
g128 and 17.9 s with g64; it was not measured on the 3-bit bank at this budget.

For the 3-bit bank, which is no longer on the internal SSD, the recorded curve is in
`HANDOFF-2026-09-16.md` section 4.1: 3.22–3.36 tok/s at 28 GiB rising to 3.89 at 44 GiB, 275 ms/token at
36 GiB, prefill flat at 24.2 s across budgets.

### 7.2 Interactive chat, 44 GiB budget, 72 GiB wired

    ready in 16.4 s
    turn 1   prefill   5 tokens (2 reused)   0.8 s    51 tokens   6.03 tok/s
    turn 2   prefill  69 tokens (65 reused)  0.6 s    68 tokens   6.73 tok/s
    turn 3   prefill 157 tokens (152 reused) 0.7 s   117 tokens   7.52 tok/s
    session  87.3 % hit rate, 4,456 resident experts, 108.4 GiB read
             6,495 predicted loads, 2,991 used, 16 prefix-cache entries, 14 hits

A chat session beats the benchmark's hit rate — 87.3 % against 80.9 % — because it reuses experts across turns,
and 44 GiB now holds 4,456 experts where the 3-bit bank held 2,984.

### 7.3 Quality

All arms are teacher-forced NLL on the same text at a 24 GiB budget. The FP4 reference of 2.3004 nats at 160
tokens is stored in `benchmarks/results/nll_experts_fp4.json`.

**Dense reference math** — ranks the weights, 160 tokens:

| arm | MiB/expert | mean NLL | vs FP4 |
|---|---|---|---|
| FP4 | 17.93 | 2.3004 | — |
| 3-bit g64, oQ3e calibrated | 14.77 | 2.3398 | +0.0394 |
| 3-bit g64, `mx.quantize` from FP4 | 14.77 | 2.3097 | +0.0093 |
| 2-bit g64, searched fit | 10.55 | 2.3207 | +0.0203 |
| 2-bit g128, searched fit | 9.49 | 2.3234 | +0.0230 |
| 2-bit g64, `mx.quantize` | 10.55 | 2.3395 | +0.0391 |
| 2-bit g128, `mx.quantize` | 9.49 | 17.5460 | model destroyed |

**Production path** — ranks what the model computes:

| bank | 160 tokens | top-1 | 512 tokens | ppl | top-1 |
|---|---|---|---|---|---|
| 3-bit g64, oQ3e | 2.3080 | 53.1 % | 2.4997 | 12.178 | 50.8 % |
| 2-bit g128 | 2.3337 | 45.6 % | 2.5187 | 12.413 | 44.5 % |
| 2-bit g64 | 2.3279 | 46.9 % | 2.5327 | 12.587 | 46.5 % |

The absolute level rises with length because the text continues into harder material; only comparisons at equal
length mean anything. **The cost of the 2-bit bank against the 3-bit one is +0.019 nats, 1.9 % of perplexity
and 6.3 points of top-1.** At 512 tokens the standard deviation is 2.2 points, so the top-1 cost is real at
2.9 sigma. The two metrics disagree about which 2-bit bank is better — g128 by 0.014 nats, g64 by 2.0 points of
top-1 — and nothing measured so far separates them confidently.

The cost is visible in output, which matters more than the nats: a 100-word story came back with "whitewas
crumbling houses" and renamed its own character from Nikos to "Niks" in the final sentence. Dropped and mangled
tokens are what a 6-point top-1 loss looks like. Cosmetic in prose; not cosmetic in code or arithmetic.

## 8. What was learned about quantization

### 8.1 MLX's affine fit wastes a level at 2 bits

`mx.quantize` does not fit the affine grid to a group's range. Measured by quantizing known groups and reading
back what it produces, its fit is max-abs symmetric:

    scale = +-max|w| / 2**(bits-1)        bias = -2**(bits-1) * scale

so the representable values are `scale * (q - 2**(bits-1))`. At 4 bits this costs little. At 2 bits the four
levels are `{-2, -1, 0, +1} * scale`: one level sits where the data never goes, and one side of the
distribution is clipped at half its range, which is why the sign of `scale` flips from group to group — the fit
puts the longer tail on the side that has three levels.

The stored format is only `w ~= scale * q + bias` per group, and `mx.dequantize` and `mx.quantized_matmul` take
the scales and biases as ordinary arrays, so **a better fit needs no runtime change at all** — only a packer.
`benchmarks/quant_affine.py` has one: `pack_2bit` (sixteen values per `uint32`, least-significant field first,
pinned against `mx.quantize` by `tests/test_quant_affine.py`), `fit_search` (a 5x5 grid over shrunk range ends,
keeping the lowest-squared-error pair per group) and `fit_minmax` (the textbook fit, which is **worse than
MLX's** at 2 bits — with four levels, clipping outliers beats spanning them).

At 2 bits and group 128 the searched fit is the difference between a working model and a destroyed one:
2.3234 nats against 17.5460.

### 8.2 Prediction width is a function of expert size

Every predicted expert that is not resident is a read, so recall and bytes rise together and the two settle at
a saddle. On the 15.48 MiB 3-bit bank the saddle was at top-3. On the 9.49 MiB 2-bit bank it is at **top-6**,
because the floor the extra reads raise stays under compute: six runs per side, interleaved both ways, top-3
median 192.5 ms/token and top-6 median 182.5, with every top-6 run faster than every top-3 run. `PREDICT_TOPK`
in `src/cachalot/model/moe_layer_metal.py` now defaults to 6 and both saddles are recorded in the comment
there. **Re-measure this whenever the bank changes.**

### 8.3 The cheap screen, and its limit

`benchmarks/expert_requant_error.py` re-quantizes sampled FP4 experts and reports the relative error of the
routed-expert output on random unit inputs. It costs seconds where the gate costs an hour, and it correctly
ranked the formats within one quantizer. **Across quantizers it is wrong**: it puts the calibrated oQ3e
download ahead of our naive 3-bit on both of its metrics (0.296 against 0.352), and the gate puts the naive one
ahead by 0.030 nats. Use it to choose which formats deserve an NLL run; never as a substitute for one.

It also settled two design questions cheaply. Error is spread evenly across layers (5 % spread, no gradient
from layer 0 to 39) and across the three projections (0.276 for w1, 0.309 for w2, 0.276 for w3), so no
per-layer or per-projection mixed bank is indicated; raising w2 alone to 3 bits buys 12 % of error for 13 %
more bytes. That is convenient, because `storage/index.py` raises on mixed expert quantization by design.

### 8.4 Cost of a gate run

Each gate arm re-quantizes about 20,000 experts, not 15,360, because the 8 GiB packed cache evicts. At
`mx.quantize` speed (19 ms per expert) an arm is 540 s; with a 13x13 search grid (720 ms per expert) the first
attempt was still running when the guard's 5400 s timeout killed it. The 5x5 grid costs 125 ms per expert and
about 2200 s per arm, and buys 0.0005 of relative weight error against the finer grid. If a future arm needs a
better fit, raise the timeout or build the bank and gate it with `--experts runtime`; do not raise the grid.

---

## 9. Open levers, ranked

Ranked by expected value per unit of work, with the evidence, the cost and — most importantly — the
measurement that decides each one before any code is written.

### 9.1 Lever 1 — MTP speculative decoding

**What.** The FP4 checkpoint carries three complete next-token-prediction layers, `mtp.0` to `mtp.2`, 7.39 GiB
in total, each with 128 routed experts of its own and its own attention and hyper-connections (section 3.2).
Drafting with them and verifying several positions in one forward pass is the only lever on this list whose
ceiling is another 1.5x rather than another 10 %. Three layers means the draft can be up to three tokens deep,
so the acceptance measurement below should report per-position acceptance, not one number.

**Why it is worth more than the 2026-09-16 handoff thought.** That document lists speculative decoding as a
null result, on the grounds that "verification of K positions multiplies bytes per token, which is precisely
the resource we are short of". That was correct when a token cost 1030 MiB and the drive was the floor. It is
no longer the situation: a token costs 486 MiB, the drive is idle 55 % of the decode, and compute is the floor.
The premise expired; the conclusion has to be re-derived, not inherited.

**Why it also attacks the right resource.** Sections 10 and 11 of `HANDOFF-2026-09-17.md` establish that 96 %
of blocked time is misses that were never predicted, that prediction timing is already fine, and that neither
more width nor more lead time fixes coverage — the runtime needs to know a *future* token's routing, which its
own router cannot tell it. A draft head is exactly that mechanism. And at batch 1 a verification forward over
two positions is close to the cost of one, because decode is latency-bound rather than FLOP-bound.

**Rough arithmetic, to be replaced by measurement.** At 85 % acceptance, ~1.85 tokens per forward; expert reads
per forward rise by about 1.7x from the measured 29 % adjacent-token expert overlap, so ~0.9x reads per token;
and all three MTP layers' experts together are 384 slots, 3.6 GiB at 2 bits, so they can simply stay resident
and make every draft step pure compute. That lands near 100 ms per accepted token if it works at all, and it
will not fully work.

**Deciding measurement, before any runtime work.** Acceptance rate. Load the MTP layers, run them over real
decoded prefixes from a handful of prompts, and measure how often draft position k matches what the main model
actually produced, for k = 1, 2, 3. That single number sets the entire ceiling, it needs no changes to the
decode path, and it is an afternoon of work rather than a week. If acceptance is below about 60 %, stop and
write it down as a null.

**Cost if it passes.** The largest on this list: a draft forward, verification of K positions in one pass, KV
rollback on rejection, prefix-cache interaction, and quantizing the MTP experts (`build_affine_bank.py` matches
`layers.N` only, so it needs a small extension for the `mtp.N` naming — or they can be served from FP4 on the
USB drive once and held resident, since 7.39 GiB read once per session is 7 seconds). Expect this to be a whole
session's subject, and note that the draft layers' own quality matters less than the main model's: a wrong
draft is rejected, not emitted.

### 9.2 Lever 2 — Compute, 120 ms per token

**What.** Compute is now 66 % of decode and has never been attacked, because it was never the binding
constraint.

**The known unknown.** The component split on record — attention, MoE, head, Python — is from 2026-09-16, when
experts were 15.48 MiB and the MoE used the 3-bit `mx.quantized_matmul`. The 2-bit kernel micro-benchmarks at
1.7 us per expert row against 3-bit's 4.6 us, so the MoE's 18.6 ms is probably nearer 7 ms now and something
else owns the remaining ~113 ms. Attacking it before profiling it would be guessing.

**An inconsistency worth resolving first.** `HANDOFF-2026-09-16.md` section 8 states, in passing, that this
runtime's all-resident autoregressive decode is 68 ms per token. If that is still true, then 120 ms of "rest"
in section 6 contains roughly 50 ms that is neither expert wait nor arithmetic, and finding it would be worth
more than any kernel work. If it is not true — different bank, different configuration, or a stale number —
then 120 ms is simply the cost of the model and the lever is ordinary kernel optimization. **Settle this
first**: run decode at a budget and prompt where the hit rate reaches ~100 % on a second pass and compare the
per-token time against the anatomy's `rest`.

**Deciding measurement.** `benchmarks/profile_decode_components.py` and `benchmarks/profile_decode_gpu.py`
against the 2-bit g128 bank, plus the all-resident comparison above. My prior, unverified, is Python and MLX
dispatch overhead across 40 layers, which `mx.compile` or per-layer graph fusion would attack; the repo already
has fused Metal kernels for several pieces, so the remaining overhead may be dispatch rather than math.

**Cost.** Profiling is hours. Acting on it depends entirely on what the profile says.

### 9.3 Lever 3 — Recover the quality the 2-bit bank cost

**What.** +0.019 nats and 6.3 points of top-1 against the 3-bit bank (section 7.3), visible as dropped and
mangled tokens in output. This lever costs **build time only** — no runtime change, no risk to the decode path,
one 33-minute rebuild plus one 512-token production arm per attempt.

**Three things to try, in order of expected value.**
1. **A better search grid.** `fit_search`'s 5x5 grid is deliberately coarse because the *gate* quantizes
   20,000 experts per run. A bank is built once, so it can afford 13x13 or a two-stage refinement, and the
   search can be asymmetric in the two range ends rather than gridded.
2. **Per-channel scaling folded into the stored scales**, AWQ-style. The format stores one scale per group; a
   per-output-channel rescale can be absorbed into those scales without changing what the kernel does, which
   keeps the bank a drop-in.
3. **Real activation weighting.** Capture activations from a prefill and weight the per-group fit by what the
   model actually multiplies, which is what imatrix calibration approximates. Note that calibration is worth
   less here than the 2026-09-16 handoff assumed: naive `mx.quantize` from FP4 beat the calibrated download by
   0.030 nats at identical bits.

**Deciding measurement.** `expert_requant_error.py` to screen candidate fits in seconds, then
`nll_expert_precision.py --experts requant` on the winner, then a rebuild and `--experts runtime` at 512
tokens. The bar is the 3-bit bank's 2.4997 nats and 50.8 % top-1.

### 9.4 Lever 4 — A larger expert budget

**What.** 44 GiB already gives 87.3 % in a chat session. Experts are 39 % smaller than the budget curve on
record was measured with, so that curve has moved and nobody has re-measured it.

**The constraint is wired memory, not bytes.** A 44 GiB budget wires 72 GiB of a 96 GiB machine, and that is
the configuration class that panicked this machine twice. A 52 GiB budget would wire ~80 GiB. **Do not simply
try it.**

**Deciding measurement, free and safe.** `benchmarks/simulate_policies.py` replays a routing trace against any
budget without loading the model. If 52 GiB buys less than a point of hit rate, the question closes without
ever putting the machine at risk. Only if the simulation promises something substantial is a guarded run at a
raised budget worth discussing with Hamed, and it needs the machine genuinely idle.

### 9.5 Lever 5 — Startup hotlist preload

Unchanged from 2026-09-16, and now slightly more attractive because the bank is smaller: 16.4 s to ready, and
the first turn of a session pays full miss cost while later turns run at 87 % hit. Preloading a recorded hot
set at startup helps exactly one turn per session. Small, self-contained, low risk.

### 9.6 Lever 6 — Long-prompt prefill

512-token cold prefill is 16.4 s and follow-up prefills in chat are under a second thanks to typing-time
prefill and the prefix cache. A 2048-token prefill has not been measured on any recent bank. Low expected value
for interactive use, real value if long documents become a use case.

### 9.7 Lever 7 — Engram onto the internal SSD

Measured as a speed null on 2026-09-17 (section 2 of that log: a prefill is barely blocked on Engram, and the
faster drive changes nothing), so this is **robustness only** — it would remove one of the three reasons the
X10Pro must stay connected. It is newly affordable: 189.1 GiB of FP4 Engram against 190 GiB free, or 91.9 GiB
if taken from the oQ3e conversion. Do it if disk pressure ever eases further, not for throughput.

### 9.8 Lever 8 — Below 9.49 MiB per expert

Effectively closed inside MLX. `mx.quantized_matmul` accepts group sizes 32, 64 and 128 only, and 2 bits at
group 128 is already in use, so 9.49 MiB is the floor for any format the existing kernels can read. Going lower
means custom Metal kernels for a custom encoding — a much larger piece of work than the 2-bit bank was, and it
would be attacking bytes, which are no longer the constraint. Mentioned for completeness; do not start here.

---

## 10. Retired premises — conclusions whose reasons expired

These were correct when written and are now misleading. Anyone reading the older logs will meet them.

| claim | where | why it no longer holds |
|---|---|---|
| A smaller bank is blocked on GGUF k-quants MLX cannot read | 09-16 §7.5 | A better bank was built here from the FP4 checkpoint in 33 minutes. No GGUF, no k-quants. |
| Their mixed-precision recipe is the quality trick to copy | 09-16 §7.5 | Naive `mx.quantize` from FP4 beats the calibrated download by 0.030 nats at equal bits. Calibration is worth ~16 % of weight error, not a recipe worth copying. |
| Speculative decoding and MTP are a null result | 09-16 §8 | Rejected because verification multiplies bytes and bytes were the constraint. Bytes are no longer the constraint. See lever 1. |
| Do not re-run the prediction-width sweep | 09-17 §13 | True for the 15.48 MiB bank only. On a 9.49 MiB expert the saddle moved from top-3 to top-6 and paid 5.2 %. |
| Reading fewer bytes per expert is the only lever with real room | 09-17 §15.3 | It was, and it was taken. Bytes now hide under compute; the byte lever is spent. |
| Decode's floor is set by bytes | 09-17 §15 | The floor is compute, 120 ms per token. |

## 11. Null results — do not repeat these

Each was measured and rejected, and the reasoning still holds. Re-running them costs hours and returns nothing.

**Storage and I/O**
- **Closing the achieved-bandwidth gap.** While the drive is busy the runtime already moves about 6 GB/s of the
  6.6–6.8 GB/s the drive delivers on cold experts.
- **The OS page cache under a large wired set.** A full 2x2 at 2 and 8 concurrent loaders, every arm on disjoint
  cold experts: all eight readings fall between 6.62 and 6.81 GB/s. Both variables null. `CACHALOT_PAGE_CACHE=1`
  stays on because it costs nothing and helps genuine repeat hits.
- **`F_RDAHEAD 0`** alongside `F_NOCACHE`: null.
- **Chunked expert reads**: a single expert read already saturates a stream.
- **Mirror striping with a stacked bank**: harmful; leave it off.
- **Engram on the internal SSD** for speed: null (it survives as lever 7 for robustness only).

**Caching and scheduling**
- **Eviction policy work.** Segmented LRU and decayed frequency are worth 1 to 2 points against 13 % fewer
  bytes from a larger budget, and total store overhead is 1.0 ms per token, so there is nothing to win.
- **LRU prefill admission** as an alternative to the quota planner: identical decode misses, worse prefill.
- **Prediction from an earlier activation** (more lead time): free, and useless — timing is not the problem,
  coverage is.
- **A GPU heartbeat to prevent clock drop**: the clock does not drop during decode.

**Numerics and kernels**
- **Expert pruning or pinning a hot subset**: zeroing experts outside a calibrated pinned half costs about 91 %
  of gsm8k accuracy. Fetch-on-miss keeps the computation exact.
- **The miss-budget approximation**: 9.7 tok/s at budget 0, but KL 0.64 and 7 of 40 tokens identical.
- **A fused multi-expert kernel and `gather_qmm` chunking**: both slower than per-expert quantized matmul.
- **A simdgroup FP4 GEMM**: 2.5x less GPU time, zero wall-clock change.
- **`fit_minmax` at 2 bits**: worse than MLX's own max-abs fit.

## 12. Pitfalls worth knowing before touching the code

- **`F_NOCACHE` does not reliably keep expert reads out of the page cache.** Repeat reads of the same experts
  through a reader with `bypass_page_cache=True` went 5.55 GB/s, then 9.07, then 9.07 — the second and third
  passes were partly served from memory. `CACHALOT_PAGE_CACHE=0` therefore does not mean what its name
  suggests, and several bandwidths above 7 GB/s recorded in the older logs are partly memory hits.
- **A storage arm is only valid on experts nothing has read yet**, including in a previous process. Warmth
  survives across runs; one discarded matrix reported 17 to 64 GB/s, which is physically impossible for this
  drive and was the tell. `expert_read_scaling.py --expert-offset` slices one seeded shuffle disjointly so each
  arm gets cold experts by construction. Interleave wired and unwired arms, and never compare an arm that ran
  first against one that ran after it on the same experts.
- **MLX streams are thread-local.** Evaluating a runtime graph from a worker thread raises
  `There is no Stream(gpu, N) in current thread`. Typing-time prefill runs on the chat thread for this reason;
  the heartbeat and forward passes share `runtime._gpu_lock`.
- **Free memory is always near zero on macOS** because the file cache fills it. Judge pressure by
  `kern.memorystatus_vm_pressure_level` and compressor growth, not by free pages.
- **A wrapped copy-paste silently drops the expert bank.** If a shell-wrapped chat command loses
  `CACHALOT_EXPERT_BANK`, the runtime serves FP4 experts from the USB drive and everything is four times slower
  with no error. Check the `expert bank:` line the runtime prints at startup; it reports path, format, bits and
  MiB per expert.
- **`tty.setcbreak` flushes pending input** unless called with `TCSANOW`. Without it, anything typed while the
  model was generating is silently discarded.
- **A pseudo-terminal harness must drain the child's output while typing**, or the chat blocks writing to a
  full buffer and the test appears to hang.
- **The occasional stutter in generated text** — "con concurrency", "nib nibbled" — appears with every numerics
  path including the fully unfused baseline, and predates the 2-bit bank. It is model behaviour. The *new*
  artefacts the 2-bit bank adds are different: dropped characters inside words and renamed entities.
- **Nothing under `benchmarks/results` is tracked by git**, and result files are named by arm. The
  `--experts runtime` arm used to write one filename whatever bank it served, and overwrote the 3-bit bank's
  per-token data on 2026-09-17; it now writes `nll_experts_runtime_<kind><bits>g<group>.json`. Check for
  collisions before adding an arm.

## 13. Reference commands

All are copy-paste ready and assume nothing about the current directory.

**Full test suite**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m pytest -q tests
```

**Decode benchmark, current best configuration**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 2400 --tag decode -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_throughput.py --prompt-tokens 512 --decode-tokens 64
```

**Where decode's time goes**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 3600 --tag anatomy -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64
```

**Quality gate, production path — required for any bank or numerics change**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 5400 --tag nll512 -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/nll_expert_precision.py --experts runtime --tokens 512
```

**Quality gate, dense reference math for a format that has no bank yet**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 7200 --tag nll_requant -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/nll_expert_precision.py --experts requant --requant-bits 2 --requant-group 128 --requant-fit search --tokens 160
```

**Screen candidate quantization formats in seconds**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/expert_requant_error.py --experts-per-layer 2 --probes 8
```

**Build a bank** (resumable; skips shards a previous run finished)
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/build_affine_bank.py --out /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 --bits 2 --group 128 --fit search --verify 12
```

**Storage probe on genuinely cold experts**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/expert_read_scaling.py --experts 512 --expert-offset 0 --wire-gib 0
```

**Settle memory between arms**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh --budget-gib 36
```

## 14. Files that matter

| path | role |
|---|---|
| `src/cachalot/storage/index.py` | expert bank detection, stacked and FP4 indexing, per-expert byte ranges; raises on mixed quantization |
| `src/cachalot/storage/reader.py` | positional reads into slot buffers, concurrent pieces, mirror striping |
| `src/cachalot/cache/resident_store.py` | residency, LRU, prefill admission planner, decode prefetch |
| `src/cachalot/cache/slots.py` | pre-allocated wired slot pool, any tensor set |
| `src/cachalot/model/expert_affine.py` | routed-expert math for affine banks; takes bits and group from the format |
| `src/cachalot/model/moe_layer_metal.py` | decode MoE path, routing prediction, `PREDICT_TOPK` and its two measured saddles |
| `src/cachalot/model/moe_prefill_grouped.py` | expert-major prefill, format branch |
| `src/cachalot/model/text_decode_runtime.py` | runtime assembly, bank selection, budget and wired limit |
| `benchmarks/quant_affine.py` | the 2-bit packer and the two fits MLX does not provide |
| `benchmarks/build_affine_bank.py` | bank builder: one shard per layer, one expert in memory, resumable, self-verifying |
| `benchmarks/expert_requant_error.py` | the seconds-long screen for candidate formats |
| `benchmarks/nll_expert_precision.py` | the quality gate; dense arms and the production arm |
| `benchmarks/decode_anatomy.py` | blocked time split by cause, reads split by worker pool |
| `benchmarks/guarded_run.sh`, `benchmarks/settle.sh` | the memory guardian and the between-arms gate |
| `tests/test_quant_affine.py` | pins the 2-bit packing against `mx.quantize`'s own layout |

### Session logs, for history

| document | what it holds |
|---|---|
| `docs/HANDOFF-2026-09-16.md` | the five changes that reached 4.3–5.6 tok/s; budget sweeps on the 3-bit bank; the panic postmortem |
| `docs/HANDOFF-2026-09-17.md` §1–9 | concurrency diagnosis, Engram null, prediction-knob nulls, segmented LRU |
| `docs/HANDOFF-2026-09-17.md` §10–15 | what decode blocks on; coverage versus timing; the three measurement traps; the real drive speed |
| `docs/HANDOFF-2026-09-17.md` §16–19 | the 2-bit bank, the fit, the gate, the width sweep, and the ranking this document replaces |
