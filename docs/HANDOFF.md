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

**The most important structural fact in this document: a decode token is roughly half arithmetic and half
expert streaming, and the arithmetic half is dispatch-bound.** Per decoded token at a 36 GiB budget, 93 ms is
the all-resident floor — measured, not inferred, by decoding the same tokens twice and reading the second pass
at a 100 % hit rate — 62 ms is exposed expert wait, and the remaining 27 ms appears only when experts are
being fetched and is not accounted for anywhere. So 49 % of a token is the cost of streaming, and the ceiling
on a perfect cache is 10.8 tok/s.

The 93 ms is not arithmetic that a better kernel would shrink. Profiled piece by piece it is about 400 GPU
dispatches at roughly 0.2 ms each, no one of which dominates: the model spends more time in hyper-connections
than in its experts. Section 9.2 is about that.

Earlier versions of this document put the floor at 120 ms and called decode compute-bound. That number was
`decode_anatomy`'s "rest", which contains the streaming overhead as well as the arithmetic; section 10 lists
the conclusions that moved with it.

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

It also contains **three complete DSpark draft stages**, `mtp.0`, `mtp.1` and `mtp.2`: 2,401 tensors,
7.39 GiB in total, of which 6.72 GiB is routed experts. Each stage has **128 routed experts** of its own
(ids 0 to 127, the same six-tensor FP4 layout as a trunk expert at 18,800,640 B each) plus its own attention,
norms and hyper-connections; stage 0 also carries `main_proj` and `main_norm`, and stage 2 carries the final
norm, a rank-256 Markov head and a confidence head. Together they draft **five** tokens per main forward, not
one per stage — section 21 of `HANDOFF-2026-09-17.md` describes the mechanism, and
`src/cachalot/model/dspark_draft.py` implements it. Nothing in the decode path uses them today.

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
4. **Judge a quality arm by the paired median and the sign test, never by the mean NLL.** The 512-token mean
   has a paired standard error of about 0.04 nats and five tokens out of 512 routinely move it further than
   the effect being measured; two different texts have disagreed in its sign while both medians sat at zero.
   The gate prints all of it now. See section 9.3.1.
5. **Quality is gated, not assumed, and gated on the production path.** Any change touching expert or Engram
   numerics must pass `benchmarks/nll_expert_precision.py` before adoption. The dense reference arms
   (`--experts fp4`, `--experts oq3e`, `--experts requant`) rank *weights*; the production arm
   (`--experts runtime`) ranks what the model actually computes. **The two disagree in sign** between the
   3-bit and 2-bit banks (section 7.3). Run the production arm. Use 512 tokens, not 160, for any top-1
   comparison: the binomial standard deviation at 160 tokens is 3.9 points, which is wider than the effects
   being judged.
6. **Judge numerics by teacher-forced NLL and top-1, never by comparing greedy text.** This model's greedy
   decoding flips tokens on changes as small as one floating-point unit.
7. **Every repository edit goes through shell commands**, never prose asking Hamed to edit a file by hand.
8. **Every command given to Hamed is complete and copy-paste ready**: absolute `cd`, `PYTHONPATH=src`, the full
   interpreter path `~/venvs/deepseek-v41/bin/python`. Never a bare `python`, never an ellipsis. Repeat the
   full command in every message that asks for something to be run.
9. **After each production patch**: byte-compile, run the focused test, `git diff --check`, inspect the diff.
   Keep benchmark scripts out of runtime code.
10. **Nothing timing-sensitive is valid while anything else is on the GPU.** Suspend a background build with
    `kill -STOP` and resume it with `kill -CONT` rather than measuring through it.
11. **Chat replies terse. Prose in files, commits and documents stays normal and complete.**

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
    all-resident decode, measured       =  93 ms   <- benchmarks/decode_resident.py, 512-token context
    exposed expert wait                 =  62 ms
    unaccounted, fetch-only overhead    =  27 ms
    measured                            = 182 ms

So the ceiling on a perfect cache is **93 ms per token, 10.8 tok/s**, and 89 ms per token — 49 % — is
attributable to expert streaming. Note that 73 ms of transfer and 62 ms of exposed wait means the drive
essentially does not hide under compute: overlap is not the lever, coverage is.

The 27 ms line is the difference between the all-resident floor and `decode_anatomy`'s 120 ms of "rest". It is
whatever only happens while experts are being fetched — slot acquisition, eviction, promotion, GPU stalls
against concurrent DMA. Eviction-policy work measured total store bookkeeping at 1.0 ms per token, so it is
probably not CPU time. Nobody has measured it.

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
length mean anything.

**The cost of the 2-bit bank against the 3-bit one is real, but read it from the median and not the mean.**
Paired per token, the 2-bit g128 bank is worse than the 3-bit one on 62.5 % of the 512 tokens by a median of
+0.035 nats — a sign test beyond 5 sigma — and 6.3 points of top-1, which is 2.9 sigma. The mean difference of
+0.019 nats is only 0.43 sigma and was over-quoted in earlier versions of this document; section 9.3.1
explains why, and the gate now prints the paired statistics. The two metrics still disagree about which 2-bit
bank is better and nothing measured so far separates them confidently.

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

### 9.1 Lever 1 — DSpark speculative decoding

**What it actually is.** Not three multi-token-prediction layers, as this document said before 2026-09-17's
fourth session: the `mtp.*` namespace holds **DSpark**, which the model card describes as "semi-autoregressive
draft generation with confidence-scheduled verification". One main forward drafts **five** tokens, not one to
three. `src/cachalot/model/dspark_draft.py` implements it and section 21 of `HANDOFF-2026-09-17.md` describes
the mechanism.

**Acceptance, measured.** Four prompts, 384 draft blocks, greedy on both sides: 72.7 % at depth 1 falling to
16.8 % at depth 5, mean accepted prefix 1.85, so **2.85 tokens per main forward**. The confidence head
separates accepted from rejected positions cleanly, 1.763 against 0.309. This passed the bar comfortably.

**The draft is cheap: 22.9 ms per block** on a quiet machine with 2-bit draft experts, 25.3 ms with the FP4
experts as shipped, and the spread across 40 blocks is under a millisecond. (Every draft figure taken while a
bank build was running — 75 to 95 ms — was inflated three to four times over. Operating rule 10 exists because
of this.)

**It is worth 1.20x, and the limit is bytes.** Widening a forward is cheap in compute — a fixed cost plus
19.8 ms per extra position — but not in reads: adjacent tokens share only about 29 % of their experts, so
verifying W positions to accept T tokens reads W/T times the bytes, and bytes are already half of a token.
Verifying all five drafted positions is a **loss** at every budget measured. Extending the block while
confidence >= 1.0 gives width 2.30 for 1.94 tokens per forward and projects to **1.20x** at a 36 GiB budget —
153 ms per accepted token against 182.5. Sections 22 to 24 of `HANDOFF-2026-09-17.md` have the tables and the
model's assumptions, the main one being that reads and compute do not overlap, which the anatomy says is close
to true.

**The lever is worth more in interactive use than this says**, because the miss curve behind the projection
comes from a trace running at a 74 to 77 % hit rate while a chat session at 44 GiB runs at 87.3 %.

**One thing already settled, and it is about memory rather than speed.** Quantizing the draft's own experts to
2-bit g128 costs 2.8 % of accepted tokens and saves 2.4 ms per block, which is a wash — both formats project
to 1.20x. What it buys is residency: 1.71 GiB instead of 3.24 GiB of a budget that is the binding constraint
on everything else. Take it for that reason.

**Cost if it proceeds.** The largest on this list: a draft forward, verification of K positions in one pass,
KV rollback on rejection, prefix-cache interaction. A whole session, for a projected 1.1x. Levers 2 and 3
below are cheaper and two of them make this one worth more.

### 9.2 Lever 2 — Dispatch count, 93 ms per token

**What.** The all-resident floor is 93 ms per token at a 512-token context, and it is roughly 400 GPU
dispatches at about 0.2 ms each. The inconsistency the previous version of this section asked to settle is
settled (section 20 of `HANDOFF-2026-09-17.md`): 68 ms was a stale number, 120 ms was "rest" with streaming
overhead inside it, and the real floor is 93 ms.

**Where it goes**, profiled against the 2-bit bank with the fused decode entry points, as isolated pieces each
paying its own evaluation barrier — which is why they sum to 162 ms against a 93 ms token:

    hyper-connections (mixes, pre+norm, post)   74.2 ms across 80 sublayers
    compressed reuse attention                  30.7 ms across 30 layers
    routed experts, 2-bit affine g128           24.6 ms across 40 layers
    shared expert (fp8)                         16.7 ms across 40 layers
    router                                      12.7 ms across 40 layers
    final head + norm                            2.1 ms once

**The finding is the shape, not any one row.** No piece dominates, every piece is a fraction of a millisecond,
and the model spends three times longer in hyper-connections than in its routed experts. A faster expert
kernel cannot move this; fewer, larger dispatches can. `mx.compile` over a whole layer, or one kernel for the
hyper-connection triple, is where to start.

**Deciding measurement.** Count and time dispatches directly — `profile_decode_gpu.py` — then fuse the
cheapest-to-fuse group and re-run `decode_resident.py`, which is the only clean read of the floor.

**Cost.** Profiling is hours. Fusion work is days, and it is the largest lever that depends on nothing else.

### 9.3 Lever 3 — Recover the quality the 2-bit bank cost

**What.** +0.019 nats and 6.3 points of top-1 against the 3-bit bank (section 7.3), visible as dropped and
mangled tokens in output. This lever costs **build time only** — no runtime change, no risk to the decode
path.

**A better fit was built and gated, and it is a null.** A grid search picks the best of 25 shrunk ranges per
group; once the level assignment is fixed, the best (scale, bias) for that assignment is the closed-form
least-squares fit of the weights against the levels, which is not a grid point. Alternating the two converges
in four iterations. On the cheap screen this looked decisive — 24 real FP4 experts at 2-bit group 128:

| fit | mean w-err | mean y-err | vs the bank in use |
|---|---:|---:|---:|
| `search`, the bank in use | 0.3702 | 0.5912 | — |
| `search` + least squares | 0.3518 | 0.5552 | −6.1 % |
| 9x9 wide grid + least squares | 0.3331 | 0.5289 | **−10.6 %** |

A whole bank was built with the last of those (`build_affine_bank.py --fit wide-lsq`, 115 minutes,
`/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128-lsq`, verified byte for byte) and gated on the production
path at 512 tokens on two different texts. Paired against the bank in use, per token:

| text | mean | paired SE | median | refined better on |
|---|---:|---:|---:|---:|
| model README | +0.0216 | 0.0386 | +0.0017 | 46.9 % |
| `encoding.py` | −0.1392 | 0.0494 | +0.0001 | 47.9 % |

**The typical token does not move.** Both medians are within 0.002 nats of zero and neither sign test is
significant; the two means disagree in direction and each is driven by about five tokens out of 512. A 10.6 %
reduction in the screen's output error bought nothing the model can be shown to notice.

So section 8.3's warning — the screen ranks correctly within one quantizer and wrongly across quantizers — is
**too generous**. It is also wrong within one quantizer when the fit's *character* changes, and a searched fit
that clips outliers is a different character from one refined onto its own level assignment.

**Still untried.** Real activation weighting — capture activations from a prefill and weight the per-group fit
by what the model actually multiplies. Note that calibration is worth less here than the 2026-09-16 handoff
assumed: naive `mx.quantize` from FP4 beat the calibrated download by 0.030 nats at identical bits. Given the
above, screen any candidate against the *production* path or not at all.

### 9.3.1 The gate's own statistic was the bigger finding

Every quality claim this project has made rests on a 512-token mean NLL, and that mean has a **paired standard
error of 0.039 to 0.049 nats** — wider than every difference it has been asked to rank. Re-tested paired, on
the same recorded per-token data, against the 3-bit bank:

| bank | mean | z on the mean | median | worse on | top-1 |
|---|---:|---:|---:|---:|---:|
| 3-bit g64 oQ3e | — | — | — | — | 50.8 % |
| 2-bit g128 `search` | +0.0191 | +0.43 | +0.0348 | 62.5 % | 44.5 % |
| 2-bit g64 `search` | +0.0330 | +0.81 | +0.0353 | 62.9 % | 46.5 % |
| 2-bit g128 `wide-lsq` | +0.0407 | +0.94 | +0.0387 | 61.7 % | 46.5 % |

**The conclusion survives but the evidence for it was the wrong evidence.** "+0.019 nats" is 0.43 sigma on the
mean and should never have been quoted as established. What is established, and strongly, is the median and
the sign test: every 2-bit bank is worse than the 3-bit one on about 62 % of tokens by about 0.035 nats — a
sign test at more than 5 sigma — which agrees with the top-1 result and with the visible artefacts.
`nll_expert_precision.py` now prints the paired median, the sign test and the five tokens that move the mean
most, so no future arm is judged on the mean alone.

**Still untried.** Real activation weighting — capture activations from a prefill and weight the per-group fit
by what the model actually multiplies. Note that calibration is worth less here than the 2026-09-16 handoff
assumed: naive `mx.quantize` from FP4 beat the calibrated download by 0.030 nats at identical bits.

### 9.4 Lever 4 — A larger expert budget — **closed**

Re-simulated at the current expert size, which nobody had done since experts got 39 % smaller. Decode hit rate
against budget, first-come admission, LRU, within a point of the measured 80.9 % at 36 GiB:

| budget GiB | 36 | 40 | 44 | 48 | 52 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode hit | 81.9 | 83.8 | 85.1 | 86.5 | 87.7 | 88.9 | 91.0 |

Going from the 44 GiB in use to 52 GiB buys 2.6 points and would wire about 80 GiB of a 96 GiB machine — the
configuration class that kernel-panicked this machine twice. **Not worth it.** Reopen only if speculation
lands, which changes the arithmetic (section 9.1).

### 9.5 Lever 5 — Startup hotlist preload — **measured, and better than it looked**

A session is 16.4 s to ready and its first turn pays full miss cost; later turns run at 87.3 % because they
reuse what the first turn dragged in. The open question was never the cost, it was whether a hot set
generalizes: do the experts a *new* prompt wants resemble the ones old prompts wanted?

`benchmarks/hotlist_coverage.py` answers it leave-one-prompt-out — rank on the other prompts in the trace,
score on the held-out one, so nothing is credited for memorizing its own prompt:

| hot set | experts | of the bank | load time | unseen prefill | unseen decode |
|---:|---:|---:|---:|---:|---:|
| 2 GiB | 215 | 1.4 % | 0.3 s | 15.9 % | 14.5 % |
| 4 GiB | 431 | 2.8 % | 0.6 s | 23.1 % | 20.9 % |
| 8 GiB | 863 | 5.6 % | 1.3 s | 32.4 % | 29.6 % |
| 16 GiB | 1,726 | 11.2 % | 2.6 s | 45.3 % | 41.9 % |

**5.6 % of the bank covers 30 % of an unseen prompt's requests**, for 1.3 s added to a startup that already
takes 16.4 s, and LRU evicts whatever the session turns out not to want — so the budget cost is transient
rather than permanent. Routing is far more concentrated than a top-6-of-384 router suggests.

Two honest limits. Coverage is of requests, not of misses: in a real first turn the cache also fills as it
goes, so this is an upper bound on what a preload buys. And the trace holds five prompts, so leave-one-out
ranks on four — indicative, not tight. Recording a hot set over a wider spread of real sessions is the next
step, and it is cheap.

This is now the best ratio of value to risk on the list: no numerics change, no decode-path change, one read
at startup.

### 9.6 Lever 6 — Long-prompt prefill

512-token cold prefill is 16.4 s and follow-up prefills in chat are under a second thanks to typing-time
prefill and the prefix cache. A 2048-token prefill has not been measured on any recent bank. Low expected value
for interactive use, real value if long documents become a use case.

### 9.7 Lever 7 — Engram onto the internal SSD

Measured as a speed null on 2026-09-17 (section 2 of that log: a prefill is barely blocked on Engram, and the
faster drive changes nothing), so this is **robustness only** — it would remove one of the three reasons the
X10Pro must stay connected. It is newly affordable: 189.1 GiB of FP4 Engram against 190 GiB free, or 91.9 GiB
if taken from the oQ3e conversion. Do it if disk pressure ever eases further, not for throughput.

### 9.8 Lever 8 — Below 9.49 MiB per expert — **reopened, conditionally**

Closed in the previous version on the grounds that bytes were no longer the constraint. They are: 49 % of a
token is expert streaming (section 6), and under speculation bytes per accepted token rise further (section
9.1), so a smaller expert is worth more now than when this was written.

What has not changed is the cost. Effectively closed inside MLX. `mx.quantized_matmul` accepts group sizes 32, 64 and 128 only, and 2 bits at
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
| Decode's floor is set by bytes | 09-17 §15 | The floor is compute, and it is 93 ms per token. |
| The compute floor is 120 ms per token, 8.3 tok/s | HANDOFF §2, §6, before 09-17 session 4 | 120 ms was `decode_anatomy`'s "rest", which contains streaming overhead as well as arithmetic. The all-resident floor, measured directly, is 93 ms and 10.8 tok/s. |
| Bytes now hide under compute, so the byte lever is spent | HANDOFF §6, before 09-17 session 4 | 73 ms of transfer against 62 ms of exposed wait: the drive essentially does not hide. 49 % of a token is streaming cost. Lever 8 is reopened. |
| The `mtp.*` layers are three MTP layers giving a draft depth of up to three | HANDOFF §3.2, §9.1 | They are DSpark: one block of five drafted tokens, a bidirectional draft block, a rank-256 Markov correction and a confidence head. |
| Speculative decoding's ceiling is another 1.5x | HANDOFF §9.1 | Acceptance is excellent — 2.85 tokens per main forward — but verification reads W/T times the bytes. The projection is 1.07x to 1.17x. |
| A larger expert budget is an open lever | HANDOFF §9.4 | Re-simulated at the current expert size: 44 to 52 GiB buys 2.6 points of hit rate and wires 80 GiB of 96. Closed. |

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

**Speculation and drafting**
- **Verifying all five drafted DSpark positions.** A loss at every budget measured, because bytes per accepted
  token rise by the ratio of positions verified to tokens accepted. Confidence-gated widths near 2.3 are the
  optimum.
- **Serving the draft's experts from FP4.** The FP4 path repacks each projection into MLX's 8-bit affine
  layout on every matmul, 0.391 ms on top of a 0.547 ms product. Quantize the draft's experts once at load.

**Numerics and kernels**
- **Expert pruning or pinning a hot subset**: zeroing experts outside a calibrated pinned half costs about 91 %
  of gsm8k accuracy. Fetch-on-miss keeps the computation exact.
- **The miss-budget approximation**: 9.7 tok/s at budget 0, but KL 0.64 and 7 of 40 tokens identical.
- **A fused multi-expert kernel and `gather_qmm` chunking**: both slower than per-expert quantized matmul.
- **A simdgroup FP4 GEMM**: 2.5x less GPU time, zero wall-clock change.
- **`fit_minmax` at 2 bits**: worse than MLX's own max-abs fit.
- **Least-squares refinement of the affine fit** (`--fit wide-lsq`): 10.6 % less routed-expert output error on
  the screen, and nothing the model notices — paired median within 0.002 nats of zero on two texts, neither
  sign test significant. A whole bank was built and gated to find this out. Section 9.3.
- **Storing affine scales and biases in fp32 instead of bf16**: 0.5767 against 0.5789 of routed-expert output
  error, inside the noise. Scale precision is not where the 2-bit error lives.
- **A grid finer than 9x9 for the affine fit**: a 17x17 grid plus least-squares refinement buys 0.05 %.
- **Segmented LRU, decayed frequency and popularity-ordered prefill admission, re-run at the 9.49 MiB
  expert**: 0.9 and 0.5 points respectively over plain LRU, and popularity ordering is worse than first-come.
  The 2026-09-17 null survives the smaller bank.

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

**The all-resident compute floor** — the only clean read of what a token costs with no reads in it
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 1800 --tag resident -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_resident.py --prompt-tokens 16 --decode-tokens 16 --passes 4
```

**Where the 93 ms of compute goes**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 1800 --tag components -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/profile_decode_components.py --prompt-tokens 512
```

**DSpark draft acceptance** — the measurement that decides lever 1
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 16 --max-seconds 3600 --tag dspark -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/dspark_acceptance.py --prompt-tokens 128 --decode-tokens 96 --prompts 4 --draft-expert-bits 2
```

**What the draft itself costs** — run this on a quiet machine, nothing else on the GPU
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/dspark_draft_cost.py --blocks 40 --draft-expert-bits 2
```

**Speculation's bytes and the confidence-gated projection** — free, no model
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/speculation_bytes.py benchmarks/results/trace_routing_v7.trace.npz --budgets-gib 36,44 && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/speculation_policy.py --acceptance benchmarks/results/dspark_acceptance_q2.json --budget-gib 36 --draft-ms 30
```

**Budget and eviction-policy sweep** — free, no model
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/simulate_policies.py benchmarks/results/trace_routing_v7.trace.npz --budgets-gib 36,44,52 --expert-bytes 9953280 --orders first-come,popularity --policies lru,slru,lfu
```

**Screen candidate 2-bit fits at a fixed format**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/quant_fit_screen.py --experts 24 --probes 8 --groups 128,64
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
| `src/cachalot/model/dspark_draft.py` | the three DSpark stages, their window caches, the Markov and confidence heads; optional one-time quantization of the draft's own experts |
| `benchmarks/dspark_acceptance.py` | per-depth draft acceptance and per-block confidence |
| `benchmarks/dspark_draft_cost.py` | what one draft block costs, split by stage, head and Markov correction |
| `benchmarks/verify_forward_cost.py` | the cost of a K-position forward, measured on the prefill path |
| `benchmarks/speculation_bytes.py` | misses per forward against verification width, replayed from a trace |
| `benchmarks/speculation_policy.py` | the confidence-gated projection, and every assumption behind it |
| `benchmarks/decode_resident.py` | the all-resident compute floor |
| `benchmarks/quant_fit_screen.py` | candidate affine fits at one format, in minutes |
| `tests/test_dspark_draft.py` | pins the draft's attention index set, whose failure mode is a false null |

### Session logs, for history

| document | what it holds |
|---|---|
| `docs/HANDOFF-2026-09-16.md` | the five changes that reached 4.3–5.6 tok/s; budget sweeps on the 3-bit bank; the panic postmortem |
| `docs/HANDOFF-2026-09-17.md` §1–9 | concurrency diagnosis, Engram null, prediction-knob nulls, segmented LRU |
| `docs/HANDOFF-2026-09-17.md` §10–15 | what decode blocks on; coverage versus timing; the three measurement traps; the real drive speed |
| `docs/HANDOFF-2026-09-17.md` §16–19 | the 2-bit bank, the fit, the gate, the width sweep, and the ranking this document replaces |
| `docs/HANDOFF-2026-09-17.md` §20–27 | the compute floor, DSpark and its acceptance, speculation's economics, the refined 2-bit fit, and lever 4 closed |
