# Cachalot — Engineering Handoff

**Authoritative state as of 2026-09-19, end of the storage-latency session.** This document supersedes
`HANDOFF-2026-09-16.md` and `HANDOFF-2026-09-17.md` wherever they differ. Those two remain as the session
logs: they carry the derivations, the discarded attempts and the raw tables behind the numbers quoted here,
and section 14 indexes them. Read this document in full before running anything or proposing any change.

**Version:** Cachalot 0.5.0, tag `v0.5.0`, `main` clean and pushed to `github.com/prooshani/cachalot`, 143
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

**That table describes the 2-bit bank, which is not the bank in use.** FP4 is, and until 2026-09-19 no
timing measurement in this document had been taken on it. Measured on FP4 at a 36 GiB budget with a
512-token prompt, `decode_anatomy.py`:

| | 2-bit g128 | **FP4, the bank in use** |
|---|---:|---:|
| decode | 182.5 ms/token, 5.48 tok/s | **341.5 ms/token, 2.93 tok/s** |
| with mirror striping (section 9.11) | not measured | **325 ms/token, 3.08 tok/s** |
| cold prefill, 512 tokens | 16.4 s | **29.1 s**, 27.1 s with mirror striping |
| expert hit rate | 80.9 % | 71.1 % |
| bytes read per token | 486 MiB | 1,858 MiB |
| all-resident compute floor | 93.0 ms | **84.6 ms** |

The shape of the problem is different on the two banks and the ranking of levers follows the shape, not the
other way round. On the 2-bit bank streaming is 49 % of a token and the drive is busy 45 % of decode. On FP4
streaming is 73 % of a token, **the drive is busy 80.5 % of decode**, and 1,858 MiB at the achieved
5.8-6.0 GB/s is about 320 ms of drive time inside a 341 ms token. **Decode on FP4 is drive-bound almost
end to end**, and the 84.6 ms of compute is very nearly free underneath it. Section 6.1.

> ### The standing decision: quality over speed — and what it cost to learn
>
> Hamed chose to give back speed for quality. Implementing that took two wrong turns, both measured and both
> recorded here, because the reasoning behind them is the useful part.
>
> **A 3-bit bank was built, gated, adopted — and is not the answer.** It buys +5.3 points of top-1 over the
> 2-bit bank (49.8 % against 44.5 %, sign test 3.8 sigma), and that is real. It buys **nothing you can
> compile**: on matched full-length C++ programs it scores 19.6 syntax errors per 100 lines against the 2-bit
> bank's 25.3, while **FP4 scores 8.9** (section 7.5 — the 0.6 this document quoted until 2026-09-18's fourth
> session came from 163 lines of a truncated arm and is superseded). The bank was deleted on 2026-09-18.
>
> **FP4 is the quality bank, and it is now on the internal SSD.** 275.4 GiB of expert-bearing shards copied
> from the USB checkpoint, which is untouched and still serves trunk, Engram, head and tokenizer. FP4 runs at
> **3.3 tok/s** on long generations at a 36 GiB budget, against the 2-bit bank's 5.5 — **1.7x the time for
> roughly 3.5x fewer syntax errors** in C++, and no free-running collapses where a quantized bank has them.
>
> **The better *fit* was built and gated, and it is not the answer either. Lever 0 is closed.** MLX's own
> affine fit does waste one level at every width, and fixing that is worth a great deal of measured error:
> a searched fit with least-squares refinement and activation weighting cuts 3-bit routed-expert output error
> by **28 %**, from 0.3335 to 0.2384, screened on the activations the model really produces and shown to
> transfer across texts. A whole 221.5 GiB bank was built with it in 47 minutes and gated on matched
> generations. It writes **31.6** C++ syntax errors per 100 lines against FP4's 8.9, and collapses on 2 of 6
> long turns where FP4 collapses on none. **Bits, not fit, are binding.** Section 9.0.
>
> That is now three times a screen metric has improved while the compiler has not, and this was the strongest
> test of the idea available: the screen was made *more* faithful — real activations rather than random unit
> vectors — and still failed to predict the only thing that matters.

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
| FP4 experts, **in use for quality** | `/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts` | 275.4 GiB | 6.6–6.8 GB/s cold |
| 2-bit g128 bank, the fast alternative | `/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128` | 142.4 GiB | same |
| free space, internal | | 68 GiB | |
| free space, X10Pro | | 871 GiB | |

`fp4-experts` holds **only** the 40 expert-bearing shards (numbers 3 to 42), copied from the USB checkpoint
and never modified. The other 8 shards carry embeddings, the MTP stages and Engram, and are still read from
`CACHALOT_MODEL_PATH` on the USB drive. `build_expert_index` globs `model-*.safetensors` and needs no config,
tokenizer or Engram shard, which is what makes the split legal — no code change was required. Verified: all
40 shards match the source byte count exactly, 15,360 experts index, and the runtime produces text identical
to the USB copy on the same seeds.

The X10Pro must stay connected. It holds the only copy of the FP4 checkpoint, the oQ3e download, and the
Engram tables the runtime reads on every prefill. The 2-bit g64 bank was deleted on 2026-09-18 to make room
for the 3-bit bank; it was strictly dominated and unused, and `build_affine_bank.py` rebuilds it in about half
an hour if it is ever wanted. Restoring the oQ3e download is *not* something to plan for: our own 3-bit bank
ties it on the production path (section 7.5).

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

A fourth format was built and rejected on 2026-09-18: **3-bit g64 with the searched, activation-weighted
fit**, 15,482,880 B per expert, the same bytes as the columns below but the best affine fit this project can
produce. It writes 31.6 C++ syntax errors per 100 lines against FP4's 8.9 (section 9.0), which is why the
table below still has four columns and not five.

| | **FP4, in use** | 2-bit g128, the fast option | 3-bit g64 (retired) | 3-bit oQ3e |
|---|---|---|---|---|
| bytes per expert | **18,800,640** | 9,953,280 | 15,482,880 | 15,482,880 |
| MiB per expert | **17.93** | 9.49 | 14.77 | 14.77 |
| experts per GiB of budget | **57.1** | 107.9 | 69.3 | 69.3 |
| bank total | **275.4 GiB** (experts only) | 142.4 GiB | 221.5 GiB | 331 GB |
| encoding | E2M1 nibbles, UE8M0 scales, group 32 | affine, searched fit | `mx.quantize`, bf16 scales | MLX affine, bf16 scales |
| C++ syntax errors / 100 lines | **8.9** | 25.3 | 19.6 | not measured |
| where | `~/DeepSeek-V4.1-Flash-fp4-experts` | `~/...-q2g128` | **deleted** | `/Volumes/X10Pro/...` |

Deleted on 2026-09-18: the 2-bit g64 bank (strictly dominated, unused), the 3-bit g64 bank (built, gated,
adopted and retired the same day — see section 7.5) and the searched, activation-weighted 3-bit bank (built,
gated and rejected the same day — see section 9.0). All three rebuild from the FP4 checkpoint in under an
hour, and each one's evidence outlives it.

The oQ3e download is **not worth restoring**: our own 3-bit bank tied it on the production path (paired median
+0.0017, sign z −1.68), and a searched 3-bit fit now beats it on the screen (0.2600 against 0.2996, section
9.0).

## 4. The configuration to use

**Interactive chat, machine otherwise idle.** This is the command Hamed runs himself, in his own terminal.
Keep it working and hand it back verbatim whenever he asks to try the model.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && pgrep -fl "deepseek-v41/bin/python|cachalot" || CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json CACHALOT_HOTLIST_GIB=8 CACHALOT_MIRROR_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_MIRROR_FRACTION=0.10 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat --expert-budget-gib 44 --max-seq-len 32768 --max-new-tokens 1024 --temperature 0.6 --frequency-penalty 0.2 --penalty-window 128
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
seconds per token; `CACHALOT_HOTLIST` and `CACHALOT_HOTLIST_GIB` preload a recorded hot set in the background
while the runtime finishes starting, worth 5.7 % of a cold prefill and 4.0 points of first-turn hit rate
(section 9.5); `--expert-budget-gib` is always explicit, never automatic.

`CACHALOT_MIRROR_PATH` and `CACHALOT_MIRROR_FRACTION` are new on 2026-09-19 and are the one setting on this
list that pays on *both* phases: the tail 10 % of every expert read is issued to the X10Pro concurrently with
the head on the internal SSD, which cuts the critical-path read from 9.37 ms to 8.10 ms. Measured at a 36 GiB
budget on a 512-token prompt: cold prefill −6.9 %, decode −5 %. The X10Pro is opened read-only and holds the
byte-identical shards the internal copy was made from, so there is no quality question and nothing to keep in
sync. **Expect less than the measured 5 % in Hamed's own configuration** — a 44 GiB budget with a hotlist runs
at an 87 to 90 % hit rate, where there are far fewer misses for the second drive to help with. Section 9.11.
The fraction matters: 0.08 to 0.12 are indistinguishable, and **0.15 is worse than off**.

Dropping the two hotlist variables changes nothing but the first turn, so a copy-paste that loses them is not
a correctness problem — unlike one that loses `CACHALOT_EXPERT_BANK`, which silently serves FP4 from the USB
drive at a quarter of the speed.

`--frequency-penalty 0.2 --penalty-window 128` is not optional decoration: without it, 62 % of long code
replies collapse into a repeating loop (section 9.9). `--max-seq-len 32768` rather than something enormous:
the caches and the 16-entry prefix cache scale with it, and 262000 costs up to about 13 GB against 0.4 GB at
8192, on a machine already wiring 72 of 96 GiB (section 12).

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
4. **A teacher-forced gate cannot see a free-running failure.** Any change that touches numerics or sampling
   also needs `benchmarks/repetition_quality.py`, which measures whether generation falls into a repetition
   loop. The NLL gate feeds the correct prefix at every step and therefore cannot produce one; it scored the
   2-bit bank at 44.5 % top-1 while that bank collapsed on 62 % of long code replies. Judge by `max_run`, not
   by the trigram rate: healthy code repeats trigrams 34 % of the time.
5. **A stochastic failure needs a rate, and a rate needs samples.** Two conclusions in the 2026-09-18 session
   survived five replies and died on the sixth. Four seeds minimum before believing a collapse rate.
6. **Judge a quality arm by the paired median and the sign test, never by the mean NLL.** The 512-token mean
   has a paired standard error of about 0.04 nats and five tokens out of 512 routinely move it further than
   the effect being measured; two different texts have disagreed in its sign while both medians sat at zero.
   The gate prints all of it now. See section 9.3.1.
7. **Quality is gated, not assumed, and gated on the production path.** Any change touching expert or Engram
   numerics must pass `benchmarks/nll_expert_precision.py` before adoption. The dense reference arms
   (`--experts fp4`, `--experts oq3e`, `--experts requant`) rank *weights*; the production arm
   (`--experts runtime`) ranks what the model actually computes. **The two disagree in sign** between the
   3-bit and 2-bit banks (section 7.3). Run the production arm. Use 512 tokens, not 160, for any top-1
   comparison: the binomial standard deviation at 160 tokens is 3.9 points, which is wider than the effects
   being judged.
8. **Judge numerics by teacher-forced NLL and top-1, never by comparing greedy text.** This model's greedy
   decoding flips tokens on changes as small as one floating-point unit.
9. **Every repository edit goes through shell commands**, never prose asking Hamed to edit a file by hand.
10. **Every command given to Hamed is complete and copy-paste ready**: absolute `cd`, `PYTHONPATH=src`, the full
   interpreter path `~/venvs/deepseek-v41/bin/python`. Never a bare `python`, never an ellipsis. Repeat the
   full command in every message that asks for something to be run.
11. **After each production patch**: byte-compile, run the focused test, `git diff --check`, inspect the diff.
   Keep benchmark scripts out of runtime code.
12. **Nothing timing-sensitive is valid while anything else is on the GPU.** Suspend a background build with
    `kill -STOP` and resume it with `kill -CONT` rather than measuring through it.
13. **Chat replies terse. Prose in files, commits and documents stays normal and complete.**

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

### 6.1 Where the time goes on FP4, the bank actually in use

`benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64`, FP4 bank, 36 GiB budget, top-6,
mirror striping off:

    decode 64 tokens: 22.43 s = 2.85 tok/s (350 ms/token)
      expert hit rate 71.1% | 69.5 misses/token | 1858 MiB read/token
      expert wait 12.93 s = 202.1 ms/token (57.7% of decode)
      rest         9.50 s = 148.4 ms/token (42.3% of decode)
      reader threads 10.86 ms/miss | aggregate 2.58 GB/s | wall-clock 5.56 GB/s
      demand   27.7 reads/token | mean  9.49 ms, p50  8.87, p90 15.22
      predict  76.0 reads/token | mean  6.47 ms, p50  6.07, p90 10.87
      drive busy 18.05 s of 22.43 s decode (80.5%); 2.67 reads in flight while busy
      blocked total            12.93 s = 202.1 ms/token
      a demand read in flight  10.22 s = 159.8 ms/token (79.1% of blocked) -- coverage
      only a predicted read     2.65 s =  41.4 ms/token (20.5% of blocked) -- timing
      no read outstanding       0.06 s =   0.9 ms/token ( 0.5% of blocked) -- store overhead

Three things in that block are different in kind from the 2-bit numbers above, not merely in size.

**The drive is busy 80.5 % of decode, against 45 % on the 2-bit bank.** Several conclusions in this document
rest on the drive having spare capacity during decode. On FP4 it does not. Section 9.10 keeps its conclusion
and loses its stated reason.

**Timing is 20.5 % of blocked time, against 2.2 % on the 2-bit bank.** The 2026-09-17 null "prediction from
an earlier activation: timing is not the problem, coverage is" was measured where timing was worth 1.6 ms per
token. On FP4 it is worth 41.4 ms. The null's premise has expired even though nothing has yet beaten it.

**Store overhead is 0.9 ms per token**, so the 27 ms of unattributed fetch-only overhead the 2-bit anatomy
carried does not appear here. On FP4 the accounting closes: 202 ms of expert wait plus 148 ms of rest, of
which 84.6 ms is the measured compute floor.

**The all-resident floor on FP4 is 84.6 ms, slightly *below* the 2-bit bank's 93.0.**
`profile_decode_components.py --prompt-tokens 512` on FP4:

    hc_mixes (attention HC, sinkhorn kernel)    0.347 ms x 80 = 27.8 ms
    hc_pre + rms_norm                           0.281 ms x 80 = 22.5 ms
    hc_post                                     0.230 ms x 80 = 18.4 ms
    compressed reuse attention                  0.940 ms x 30 = 28.2 ms
    fused routed experts (6, FP4 kernel)        0.589 ms x 40 = 23.5 ms
    shared expert (fp8, 3 gemv)                 0.428 ms x 40 = 17.1 ms
    router route_topk                           0.329 ms x 40 = 13.2 ms
    sliding-window attention                    0.838 ms x  2 =  1.7 ms
    final head + norm                           2.134 ms x  1 =  2.1 ms
    ----
    sum of isolated pieces 154.5 ms; whole token 84.6 ms

The FP4 fused expert kernel costs 23.5 ms per token against the 2-bit affine path's 24.6. **Compute is
bank-independent in fact and not only in principle**, which is worth knowing before spending days on fusion:
the lever is the same size whatever is mounted, and on FP4 it is 84.6 ms of a 341 ms token that is already
hidden under the drive.

**`decode_resident.py` cannot measure the floor on FP4 at a 36 GiB budget.** 36 GiB holds 2,056 FP4 experts,
which is below the probe's own working set, so its repeat pass plateaus at 61 % hit and 141 misses per token
instead of reaching 100 %. The floor above comes from `profile_decode_components.py`, which times an
all-resident token directly. Do not read `decode_resident.py`'s FP4 output as a floor.

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

### 7.4 Quality as a user sees it: three metrics, one conclusion

Teacher-forced NLL and top-1 are the only quality numbers this project carried until 2026-09-18, and both are
blind to how the model behaves when it is generating freely. Two more exist now, and all three agree.

**Free-running repetition** (`benchmarks/repetition_quality.py`). Replays a real multi-turn conversation
through the chat path and reports the longest span covered by a k-gram repeating back to back. Judge by
`max_run`, not by the trigram rate -- healthy code repeats trigrams 34 % of the time. On the 2-bit bank,
generating a long reply from a short prompt collapses into a loop on **62 % of replies**; with
`--frequency-penalty 0.2 --penalty-window 128` that falls to **12 %**. Section 9.9.

**Code that compiles** (`benchmarks/code_validity.py`). Extracts fenced code blocks from saved replies and
syntax-checks them with `ast.parse` or `clang++ -fsyntax-only`.

> **Correction, 2026-09-18.** An earlier version of this section reported "6.6x fewer syntax errors" for the
> 3-bit bank and a "4.6x" figure against the 2-bit one. **Both were confounded and are withdrawn.** The arms
> were not matched on what they generated: one produced three ~107-line C++ programs, the other ten ~19-line
> snippets, and long blocks accumulate errors while short ones do not. The metric was measuring block
> composition as much as correctness. Always compare arms on the same conversation *and* check the average
> block length before believing a ratio.

Matched properly — same three-turn conversation, same sampling, C++ blocks only:

| bank | blocks | avg block | lines | errors | errors per 100 lines |
|---|---:|---:|---:|---:|---:|
| **FP4**, 3 seeds, 2026-09-18 session 4 | 4 | 115 | 461 | 41 | **8.9** |
| FP4, the 2-seed arm the guard truncated | 3 | 54 | 163 | 1 | 0.6 |
| 3-bit g64, searched + activation-weighted | 3 | 138 | 414 | 131 | 31.6 |
| 3-bit g64, `mx.quantize` | 5 | 98 | 491 | 96 | 19.6 |
| 2-bit g128 | 13 | 18 | 233 | 59 | 25.3 |

> **Correction, 2026-09-18 session 4.** FP4's **0.6** was real but thin: 163 lines from an arm the memory
> guardian killed after two of three seeds, carried by one lucky 153-line block that compiled clean. Re-run to
> three full seeds on the internal SSD — 461 lines, 4 blocks — FP4 scores **8.9**, and only one of its four
> blocks is clean. Every ratio this document quoted against 0.6 is therefore too large by an order of
> magnitude. **FP4 is still in a different regime, but the regime is about 3.5x, not 30 to 40x.**
>
> The lesson is the one section 7.4 already carried and did not apply to itself: a code-validity figure is
> only as good as the volume behind it, and a truncated arm is a small sample dressed as a measurement. Check
> that every arm ran to completion before comparing them, and prefer the guarded run's own log over the
> result file, which cannot tell you it is short.

**The quantized banks are not meaningfully different from each other**, which is the finding that matters:
the +5.3 points of top-1 the `mx.quantize` 3-bit bank genuinely bought translated into no usable improvement
in code, and neither did the 28 % of output error the searched, activation-weighted fit bought after it.

**Python is not the discriminator; long C++ is.** On the same 2026-09-18 session-4 arms, Python blocks score
**2.2** errors per 100 lines on *both* FP4 and the 3-bit bank, at 37 and 45 lines per block. The entire
difference lives in the long C++ programs, where blocks run past 100 lines and an artefact every few hundred
tokens is certain to land inside one. Screen on the hardest thing the model is asked to write, not on the
average of what it writes.

**The three together.** Top-1 said 50.8 % against 44.5 %. The paired median said the 2-bit bank is worse on
62 % of tokens. The compiler says 15.2 errors against 2.3. That is why the standing decision in section 2 is
what it is.

The repetition loops are a *separate* failure with a *separate* fix: they are sampling dynamics, they happen
on FP4 too, and the frequency penalty handles them. A better bank will not stop loops and the penalty will not
stop artefacts. Both are needed.

### 7.5 The 3-bit bank's gate, 2026-09-18

Three metrics, all against the 2-bit g128 bank it replaces, each on the protocol that metric was defined for.

| gate | 2-bit g128 | **3-bit g64** | how it was read |
|---|---:|---:|---|
| top-1, production path, 512 tokens | 44.5 % | **49.8 %** | +5.3 points |
| paired NLL against the 2-bit bank | — | median **−0.0208**, better on 58.4 % | **sign z +3.80** — the mean, at −0.0443 +- 0.0457, is z −0.97 and says nothing |
| free-running collapse rate, penalty on | 1/8 (12 %) | 1/8 (12 %) | **unchanged, and expected** |
| syntax errors per 100 generated lines | 22.7 | **4.9** | 596 lines / 135 errors against 512 / 25 |
| decode, 44 GiB budget, interactive | 6.3–7.1 tok/s | **4.8–5.0 tok/s** | the price |
| expert hit rate, chat session | 90.7 % | 83.3 % | 1.56x the bytes per expert |

**The collapse rate being unchanged is a result, not a null.** It confirms the two failures are independent:
the bank causes the token-level artefacts, sampling dynamics cause the loops, and neither fix substitutes for
the other. Ship both.

**The error counts differ in shape, not only in size.** The 2-bit arm's worst two blocks carried 81 and 50
errors -- files that are write-offs rather than files with typos -- while the 3-bit arm's worst block had 2.

**And our bank ties the download.** Against oQ3e on the same 512 tokens: paired median +0.0017, better on
46.3 %, sign z −1.68. Eight minutes of building matches a 331 GB restore, so section 3.1's "restoring is a
331 GB copy at 1 GB/s" is now a note of historical interest.

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

**Generalised to every width on 2026-09-18.** `pack_bits` writes MLX's layout at any width — one contiguous
little-endian bit stream per row, least-significant bit first, no padding, so a level straddles a word
boundary whenever its width does not divide 32 — and `tests/test_quant_affine.py` pins it against
`mx.quantize` at 2, 3, 4, 5, 6 and 8 bits. Every fit takes `bits` and an optional per-column `weights`, and
`quantize_affine` is the general entry point. The searched fit helps *more* at higher widths, not less: 26 %
of output error at 3 bits against 17 % at 2. **None of which was enough to make a 3-bit bank usable** —
section 9.0 is the whole story and it closes the lever.

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

**And the limit is worse than "across quantizers", 2026-09-18 session 4.** The screen was made *more*
faithful — `quant_fit_screen.py --activations` scores on the vectors the model really hands its experts
instead of random unit ones — and it still failed. It ranked a searched, activation-weighted 3-bit fit 28 %
better than the fit the retired bank used, correctly as far as output error goes, and the bank built from it
wrote 31.6 C++ syntax errors per 100 lines against the retired bank's 19.6. **Output error is not a proxy for
usable output at this resolution, however it is measured.** Use the screen to decide which formats deserve a
bank; never to predict what a bank will do.

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

**The ranking changed again on 2026-09-19, and this time because the bank in use was finally measured.**
Every timing number behind the 2026-09-18 ranking came from the 2-bit bank. On FP4 the token is 341 ms, of
which about 320 ms is drive time and 84.6 ms is compute that hides underneath it (section 6.1). That
demotes dispatch count and closes speculation:

| lever | state on FP4 |
|---|---|
| 11, mirror striping across both drives | **shipped 2026-09-19**: −5 % decode, −7 % cold prefill, no quality change |
| 12, prefetch precision | **bounded and mostly closed**: the missing 28.5 % is the router's selection boundary, and no cheap re-use of the stale scores beats plain top-k |
| 2, dispatch count | open, but worth close to nothing until bytes come down: 84.6 ms is already hidden |
| 1, DSpark speculative decoding | **closed**: 1.03x on measured constants, against its own 1.15x bar |
| 0, a searched fit above 2 bits | closed 2026-09-18, built and gated |
| 3, 4, 10 | closed |
| 5, 9, 11 | shipped |
| 6 | low value for interactive use |
| 7 | robustness only |
| 8 | do not start there |

**The general lesson of the day: a lever's rank is a property of the bank, not of the runtime.** Three
conclusions in this document were correct on 9.49 MiB experts and wrong on 17.93 MiB ones — the drive's spare
capacity, the irrelevance of prefetch timing, and speculation's economics. Re-measure the anatomy whenever the
bank changes, before re-ranking anything. Read sections 6.1 and 9.11 first.

**And the state after 2026-09-19 is that nothing cheap is left.** Mirror striping is taken. Speculation,
eviction policy, prefetch width, prefetch lead time and prefetch precision are all measured and closed, four
of them in a single session for a few hours of machine time and no code. What remains is expensive and
honest: **fewer bytes per expert at FP4 quality**, which MLX cannot express (section 9.8) and which no affine
bank above 2 bits has survived (section 9.0); **a genuinely better routing predictor**, which needs a signal
the stale vector does not carry (section 9.12); and **dispatch fusion**, which is real work on a term that is
already hidden under the drive (section 9.2) and pays only after one of the other two lands. A session that
starts by looking for another environment variable will not find one.

### 9.0 Lever 0 — a searched fit above 2 bits — **built, gated and closed, 2026-09-18**

**Where this came from.** FP4 writes C++ that mostly compiles; the 3-bit and 2-bit banks do not. That looked
like a cliff rather than a slope, and a cliff between 4 and 3 bits was suspicious enough to investigate:
section 8.1 had already shown that `mx.quantize`'s fit is max-abs symmetric and **wastes one level at every
width**, and the 3-bit bank that failed was built with it. The question was whether the cliff was the bits or
the fit.

**It is the bits.** The fit was improved as far as it can usefully be improved, the bank was built, and the
compiler said no.

#### What was built

`pack_bits` in `benchmarks/quant_affine.py` writes MLX's layout at any width, which was the blocker. MLX's
layout turned out simpler than the 2-bit special case suggested: the levels of a row are one contiguous
little-endian bit stream, filled least-significant bit first with no padding, so a level straddles a word
boundary whenever its width does not divide 32 and 32 three-bit levels occupy exactly three words.
`tests/test_quant_affine.py` pins it against `mx.quantize`'s own output at 2, 3, 4, 5, 6 and 8 bits and at
every group size the kernels accept — worth doing, because a bank written with the wrong layout fills the
shard header exactly and reads back as noise with no error anywhere.

Everything downstream is width-agnostic now, and that part is worth keeping whatever happens to this lever:
every fit takes `bits` as a keyword, `quantize_affine` is the general entry point, `build_affine_bank.py`
takes a searched fit at any width, and `quant_fit_screen.py --bits` and `nll_expert_precision.py
--experts requant` screen any width.

**Activation weighting was added and it works.** Section 9.0.1 called it the part of the "dynamic
quantization" idea this project can use, and it was untried. The premise had never been checked either, so it
was checked first: `benchmarks/capture_activations.py` wraps `moe_layer_forward` in every block module and
records the vector the decode path actually hands the routed experts. **Inside an average group of 64 columns
the most and least excited column differ by a factor of 7.9 in mean square**; across a whole layer, 73. An
unweighted fit spends the same effort on both ends of that.

Every fit now takes a `weights` array and minimizes the weighted squared error, with `_lsq_step` solving the
weighted normal equations. The weighting is per column, because that is the axis the input is contracted over
and the axis the groups run along: w1 and w3 are weighted by the recorded mean square of the MoE input, a
property of the layer; w2 by the mean square of the SwiGLU hidden that input produces *through this expert's
own FP4 weights*, which makes w2's weighting per expert. A weighting of all ones reproduces the unweighted fit
exactly, and that is pinned.

#### What the screen said, and it was not a small effect

At 3-bit group 64 on 16 real FP4 experts, scored on **recorded MoE inputs** rather than random unit vectors —
random inputs make every column equally important by construction, which is the assumption being tested:

| fit | w-err | y-err | against `search`+lsq |
|---|---:|---:|---:|
| `mlx`, the fit the retired bank used | 0.2108 | 0.3335 | +32.7 % |
| `search` | 0.1614 | 0.2560 | +1.9 % |
| `search`+lsq | 0.1587 | 0.2513 | — |
| **`search`+lsq+act** | 0.1630 | **0.2384** | **−5.1 %** |

Weight error *rises* while output error falls, which is the signature of the trade working rather than of a
generically better fit. Against `mx.quantize` the total is **−28.5 %**. The 9x9 wide grid is a null at 3 bits
(0.2588 against `search`+lsq's 0.2587) for 378 ms per expert against 143, so it was not used.

**And it transfers, which is the standing objection to anything calibrated.** Fitted on an English-prose
recording and scored on activations from a code text it keeps 82 % of what a self-calibrated weighting gets
(0.2347 against 0.2310, where the unweighted fit is 0.2465). A weighting merged from both recordings —
`capture_activations.py --merge`, which adds the second moments and pools the probes without loading the model
— comes within 0.3 % of the self-calibrated number on *both* texts, so the bank was built with that.

#### What the compiler said

`DeepSeek-V4.1-Flash-q3g64-act` — 3-bit group 64, `search-lsq`, activation-weighted from the merged
recording, 15,482,880 B per expert, 221.5 GiB — was built on the X10Pro in 47 minutes at 166 ms per expert,
and 12 sampled experts read back byte for byte through the real index. The internal SSD has 70 GiB free, so
there was nowhere else to put it; the source was read from the internal FP4 copy so the reads stayed off the
drive being written.

Gated against FP4 on the same conversation, the same three seeds and the same sampling
(`--frequency-penalty 0.2 --penalty-window 128`, 1400 tokens, 24 GiB budget), C++ blocks only:

| arm | blocks | avg block | lines | errors | clean | errors per 100 lines |
|---|---:|---:|---:|---:|---:|---:|
| **FP4** | 4 | 115 | 461 | 41 | 1/4 | **8.9** |
| **3-bit searched + weighted** | 3 | 138 | 414 | 131 | 0/3 | **31.6** |

Free-running collapse, penalty on: **2 of 9 replies (22 %)** against FP4's **0 of 9**, which is **2 of 6**
long turns against none — the short first turn of each seed never collapses on either bank. Three seeds is
below the four this document asks for a rate (rule 5), so read the long-turn count rather than the
percentage; what is not in doubt is that FP4 produced six full 1400-token replies and the 3-bit bank produced
four, having run two of them into a loop at 279 and 129 tokens.

Block lengths are comparable — 138 against 115 — so this is not the block-composition artefact that withdrew
the "4.6x" earlier the same day. And Python blocks score **2.2** errors per 100 lines on *both* arms, so the
banks are indistinguishable on short code and separated only by long C++, which is exactly where a
once-every-few-hundred-tokens artefact is certain to land.

Section 9.0 set its own closing condition before the bank was built: *if it lands near 19.6, bits rather than
fit are binding and this lever closes for good.* It landed at 31.6. **Closed.**

#### The three things to carry forward

1. **Bits are binding above the fit.** 28 % less routed-expert output error, measured on real activations and
   shown to generalise, changed nothing a compiler can see. There is no reason left to expect a cleverer
   affine fit at 3 bits to reach FP4, and `mx.quantized_matmul` offers nothing between 3 bits and FP4's 4.25
   (section 9.0's own finding that 4-bit affine at group 128 is byte-identical to FP4 and strictly worse).
2. **A more faithful screen is still not the gate.** Section 8.3 said the screen ranks correctly within one
   quantizer and wrongly across them; section 9.3 said it is also wrong within one quantizer when the fit's
   character changes. This adds the strongest version: making the screen *more* faithful, by scoring on the
   model's own activations instead of random vectors, did not make it predictive. The map from output error
   to syntax errors is not merely non-linear, it is not a function of output error at all at this resolution.
3. **The machinery is worth keeping even though the lever is closed.** `pack_bits` unblocks any future width;
   activation weighting is a general capability that costs no bytes and no runtime; `capture_activations.py`
   is the first tool this project has for looking at what the model actually multiplies, and nothing says its
   only use is quantization.

**The bank was deleted on 2026-09-18** — 221.5 GiB for a measured negative, and it rebuilds in 47 minutes
from the command in section 13 if anyone ever wants to re-open the question. Its gated output is kept under
`benchmarks/results/replies/lc_q3g64act`, alongside FP4's matched arm, so the comparison can be re-scored
without generating anything.

### 9.0.1 Two questions asked and answered on 2026-09-18

**Can Unsloth-style dynamic quantization help?** Its premise is that some tensors matter far more than others,
found by calibration, then bits spent unevenly. **That premise is measured absent here** (section 8.3): error
spread across layers is 5 % with no gradient from layer 0 to 39, and across projections it is 0.276 / 0.309 /
0.276 for w1 / w2 / w3. Raising w2 alone to 3 bits buys 12 % of error for 13 % more bytes — break-even. The
one vendor-calibrated bank available, oQ3e, *lost* to naive `mx.quantize` by 0.030 nats at identical bits.
There is also a hard blocker: `mx.quantized_matmul` takes one width per tensor and `storage/index.py` raises
on mixed expert quantization by design. The part worth keeping is **activation-weighted fitting** — the
imatrix idea without the mixed precision — which needs no format change.

**Update, 2026-09-18 session 4: it was tried.** The premise the *mixed-precision* half of the idea rests on
is absent here, as measured above, but the weighting half is not: inside an average group of 64 columns the
mean square of the input spans a factor of 7.9. Weighting the fit by it is worth 5.1 % of routed-expert
output error on top of the searched fit, transfers across texts, and costs no bytes and no runtime — and it
made no difference the compiler could see (section 9.0). So the answer to "can Unsloth-style dynamic
quantization help?" is now fully settled: the mixed-precision premise is absent, the weighting premise is
real and the weighting does not rescue 3 bits.

**Would FP8 experts be more accurate?** No, and the checkpoint settles it: `quantization_config` reads
`{quant_method: fp8, expert_dtype: fp4}`. The trunk ships FP8; the routed experts ship **FP4**. FP4 is the
source of truth, which is why `nll_experts_fp4.json` is the reference arm every other bank is scored against.
An FP8 expert bank would dequantize FP4 and re-store identical values at twice the bytes — 551 GiB, no
quality gain, and it would not fit. The untouched FP8 question is *activations*, not weights.

### 9.1 Lever 1 — DSpark speculative decoding — **closed on FP4, 2026-09-19**

> **Closed before any code was written, which is what the arithmetic is for.** v9 of the next-session prompt
> set the bar: redo the projection on FP4's expert size, and if it still clears about 1.15x it is worth a
> session. Done with `speculation_bytes.py --expert-bytes 18800640` and `speculation_policy.py`, no model
> loaded, in minutes.
>
> On the constants this document carried it projects **1.17x** and would have opened. On the constants this
> session *measured* it projects **1.03x** and closes. The three that moved it:
>
> | constant | carried | measured 2026-09-19 |
> |---|---:|---:|
> | drive rate | 6.7 GB/s | **5.97 GB/s achieved** (section 6.1) |
> | all-resident compute | 93.0 ms (2-bit) | **84.6 ms** (FP4) |
> | baseline to beat | 342 ms | **325 ms** with mirror striping |
>
> Best confidence-gated policy: threshold 2.0, mean width 1.75, 1.62 tokens per forward, 314 ms per accepted
> token against 325. **The entire projected value of speculation lived in the gap between the drive's assumed
> and achieved bandwidth, and mirror striping has taken part of that gap directly, for hours of work instead
> of a session.** Verifying all five drafted positions is 0.67x.
>
> Reopen only if bytes per expert fall, which changes the byte term that dominates every row. The acceptance
> measurements (2.85 tokens per main forward, the confidence head's separation) are unaffected and stand; it
> is the economics that fail. Results in `benchmarks/results/speculation_policy_fp4_measured.json`.

The measurements behind it, which remain valid:


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

### 9.2 Lever 2 — Dispatch count — **demoted on FP4, 2026-09-19**

> **It is real work on a term that is already free.** The floor is **84.6 ms on FP4** (section 6.1), inside a
> 341 ms token of which about 320 ms is drive time. Fusing compute to nothing would not be visible until
> bytes come down. The lever is unchanged in size and bank-independent — the FP4 expert kernel costs 23.5 ms
> per token against the affine path's 24.6, and hyper-connections cost 68.7 ms across 80 sublayers on both —
> but it is not the place to spend days while decode is drive-bound. **Prefill is the exception**: it is
> compute-bound in a way decode is not and has never been profiled at the layer level on FP4.

The figures below were taken on the 2-bit bank and the shape holds on FP4.

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

A whole bank was built with the last of those (`build_affine_bank.py --fit wide-lsq`, 115 minutes, verified
byte for byte) and gated on the production path at 512 tokens on two different texts. Paired against the bank
in use, per token:

| text | mean | paired SE | median | refined better on |
|---|---:|---:|---:|---:|
| model README | +0.0216 | 0.0386 | +0.0017 | 46.9 % |
| `encoding.py` | −0.1392 | 0.0494 | +0.0001 | 47.9 % |

**The typical token does not move.** Both medians are within 0.002 nats of zero and neither sign test is
significant; the two means disagree in direction and each is driven by about five tokens out of 512. A 10.6 %
reduction in the screen's output error bought nothing the model can be shown to notice.

The bank was deleted after the gate: 142.4 GiB for a measured null, and `--fit wide-lsq` rebuilds it in
115 minutes from the FP4 checkpoint if anyone ever wants to re-open the question. Its per-token results are
kept in `benchmarks/results/nll_experts_runtime_affine2g128_DeepSeek-V4-1-Flash-q2g128-lsq*.json`, which is
what section 9.3.1's paired comparisons are computed from.

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

**5.6 % of the bank covers 30 % of an unseen prompt's requests.** Routing is far more concentrated than a
top-6-of-384 router suggests.

**Implemented, and measured end to end.** `CACHALOT_HOTLIST` names a file from
`benchmarks/build_hotlist.py`, `CACHALOT_HOTLIST_GIB` caps what is read (default 8 GiB), and unset the runtime
behaves exactly as before. Four runs a side, interleaved both ways, 512-token cold prompt, 36 GiB budget:

| | off | on | change |
|---|---:|---:|---:|
| ready | 15.3 s | 15.5 s | +0.2 s |
| cold prefill | 16.6 s | 15.7 s | **−5.7 %** |
| first-turn hit rate | 34.2 % | 38.2 % | **+4.0 pts** |
| read during the turn | 117.4 GiB | 111.0 GiB | **−5.5 %** |
| decode | 201.1 ms | 206.1 ms | +2.5 % |
| cold start to end of first turn | 38.3 s | 37.8 s | −1.1 % |

The prefill gain is the reliable part: the four arms do not overlap, 16.5–16.7 s against 15.6–16.2 s. The
decode difference and the whole-session figure are inside the 7 % run-to-run spread.

**The first attempt was a net loss, and the reason is worth keeping.** Reading the hot set in the foreground
cost 1.3 s of startup to save 0.7 s of prefill. The same bytes read *during* prefill hide under prefill's own
compute; read at startup they hide under nothing. Moving the read to a thread that is joined before the first
prompt — so it overlaps RoPE precompute, the Engram reader and the prefetcher — cut the startup cost to 0.2 s
and turned the lever positive.

**Two honest limits remain.** The trace holds five prompts, so leave-one-out ranks on four: indicative, not
tight, and recording a hot set over a wider spread of real sessions is cheap. And 8 GiB of a 36 GiB budget is
a large static reservation; the sweep in `hotlist_coverage.py` suggests 4 GiB gives two thirds of the coverage
for half the reservation and has not been A/B'd.

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

### 9.9 Repetition collapse, and the sampler that now stops it

Free generation on code prompts falls into a repeating loop. It is **not** a quantization failure -- FP4 does
it too -- and it is not the prefix cache, which was tested and cleared. It tracks **generating a long reply
from a short prompt**: 62 % of replies collapse in that shape, against 0 of 7 when the same turn is generated
from about 1,700 tokens of context. A collapsed turn then poisons the next one.

`frequency_penalty` is the fix, because it grows with the count; a classic repetition penalty fires once per
unique token and a confident loop rides straight over it. At 0.2 with a 128-token window the rate falls from
62 % to 12 %. The survivor had period 14, which only puts each of its tokens in the window about nine times --
long-period loops want a wider window or `no_repeat_ngram_size`, and neither is tuned.

Defaults are on in the HTTP server (0.2 / 128) and off in the library, because an unmodified OpenAI client
cannot ask for something it does not know exists. The CLI exposes all four knobs.

**Untuned and worth an hour:** the window and the penalty against a code-validity score, so the setting is
chosen on whether the code still compiles rather than only on whether it loops.

### 9.10 Wasted prefetch: 42.7 % of every byte read

From a real chat session's `/stats`, the byte accounting closes exactly:

    demand misses 30,184 + wasted predicted loads 22,465 = 52,649
    52,649 x 9,953,280 B = 524,030,238,720 B = ssd_bytes_read, to the byte

So **42.7 % of all SSD traffic was read and never used**, at 39.6 % prediction precision, while 48.8 % of real
misses did get a head start. The top-6 width was tuned at an 80.9 % hit rate on the benchmark; a chat session
runs at 90.7 %, where there are far fewer misses to predict and the same width wastes proportionally more --
and those reads occupy the drive and the loader threads during exactly the windows the demand misses need.

**Swept on FP4 and closed, 2026-09-18.** Widths 0/2/3/4/6 at a 36 GiB budget, three passes interleaved:
2.78, 2.83, 2.86, 2.89 and **2.93 tok/s**. Top-6 is optimal on 17.93 MiB experts as well, monotonically, with
non-overlapping ranges; prediction off is the worst setting.

> **Correction, 2026-09-19.** This section explained that result by saying the drive is not saturated during
> decode, so a speculative read is nearly free. **That is false on FP4**: the drive is busy 80.5 % of decode
> (section 6.1), and the predict-worker sweep in section 11 shows it is at its knee — adding in-flight reads
> lowers achieved bandwidth rather than raising it. The width result stands, on a different mechanism: a
> predicted read costs 6.47 ms against a demand read's 9.49 because it is off the critical path, and
> prediction serves 41.8 of the 69.5 misses per token early. Width is settled; **precision is not, and it is
> now the largest open lever** (section 9.12).

**Keep the general lesson, and note it cuts both ways:** a large waste figure is not a lever unless the
wasted resource is the binding one — and when the bank changes, check whether it has become binding before
reusing the conclusion's reasoning for anything else.

### 9.11 Lever 11 — Mirror striping across both drives — **shipped, 2026-09-19**

**What.** The FP4 experts live on the internal SSD; the X10Pro holds the checkpoint they were copied from,
byte for byte, under the same shard names. `reader.py` has carried a mirror path since 2026-09-16: the tail
`CACHALOT_MIRROR_FRACTION` of every expert read is issued to a second drive concurrently, so one expert lands
sooner than either drive alone could deliver it. It was recorded as harmful and left off.

**It is not harmful; it was measured past its optimum.** Decode is drive-bound on FP4 (section 6.1) and the
gain is a cliff rather than a plateau. `decode_anatomy.py`, 36 GiB budget, 512-token prompt, four runs per
arm interleaved in both directions with `settle.sh` between:

| mirror fraction | ms/token | median | demand read mean | wall-clock GB/s | coverage block |
|---:|---|---:|---:|---:|---:|
| off | 342, 341, 342, 337 | 341.5 | 9.37 ms | 5.80 | 156.6 ms |
| **0.10** | 326, 328, 323, 328 | **327** | **8.10 ms** | **5.97** | **145.8 ms** |
| 0.15 | 352, 352, 352, 352 | 352 | 9.75 ms | 5.54 | 167.5 ms |

The ranges do not overlap. **At 0.15 it is worse than off**, which is what a mirror looks like past its
optimum: the USB carries 280 MiB per token at 1.0 GB/s, about 293 ms inside a 352 ms token, and the slow tail
becomes the long pole. At 0.10 it carries 186 MiB, about 195 ms inside a 327 ms token, and stays off the
critical path.

**Between 0.08 and 0.12 the effect is flat** — medians 328, 324 and 322.5, with a within-arm spread of 7 to
12 ms that swamps the 3 to 5 ms between them. The shipped default of 0.10 sits in the middle of that plateau
and does not need tuning. Do not read a winner out of those three numbers; four runs a side cannot separate
them.

**It is a pure latency win and the diagnostics say so.** Bytes read and hit rate are identical across arms
(1,861 against 1,864 MiB, 71.0 % both). Only the arrival time moves: the demand read, which is the critical
path, drops from 9.37 ms to 8.10 ms.

**And it pays on prefill, which was the thing to check before recommending it.** The X10Pro also serves
trunk, Engram, head and tokenizer, so decode's win might have been taken out of prefill. `decode_throughput.py`,
four runs per arm, interleaved:

| | cold prefill, 512 tokens | median | decode ms/token | median |
|---|---|---:|---|---:|
| off | 29.1, 29.0, 29.1, 29.1 | 29.1 s | 332, 333, 333, 340 | 333 |
| **0.10** | 27.1, 27.6, 27.1, 27.1 | **27.1 s** | 323, 326, 324, 323 | **323.5** |

**Prefill −6.9 %, decode −2.9 %, both with non-overlapping ranges, and no quality question to answer at all**
— the bytes are the same bytes, read from a byte-identical copy, and the X10Pro is opened `O_RDONLY`
(`storage/reader.py:103`).

**Two honest limits.** Everything above was measured at a 36 GiB budget on a 512-token prompt with the
machine idle, where the hit rate is 71 %. Hamed runs 44 GiB with a hotlist and a live conversation at 87 to
90 %, where there are far fewer misses for the second drive to help with, so **expect less than 5 % there**;
it has not been measured. And the whole lever is contingent on the X10Pro staying connected, which section
3.1 already requires for three other reasons.

### 9.12 Lever 12 — Prefetch precision — **bounded and mostly closed, 2026-09-19**

**Where it came from.** A token reads 1,858 MiB, of which **613 MiB is predicted and never used** — 34.2
wasted loads at 55 % precision, about 105 ms of drive time on the binding resource. Width is settled at top-6
and swept on FP4; accuracy had never been touched. It looked like the largest open lever.

**It was bounded before anything was written, offline, with no GPU and no experts loaded.**
`benchmarks/predictor_recall.py` reads `capture_activations.py`'s per-layer MoE inputs — token-aligned across
layers, because every layer sees every token — and the gate tensors straight out of the shard headers, then
scores the predictor the runtime actually uses against the truth it is trying to guess.

| width | one layer early, as shipped | two layers early | previous token |
|---:|---:|---:|---:|
| 6 | **71.5 %** | 65.0 % | 34.2 % |
| 7 | 75.8 % | 69.0 % | 36.5 % |
| 8 | 78.9 % | 72.1 % | 38.4 % |
| 12 | 85.7 % | 79.8 % | 43.9 % |
| 16 | 88.8 % | 83.9 % | 47.4 % |
| 24 | 92.0 % | 88.4 % | 52.2 % |

71.5 % at top-6 reproduces the "~73 %" in `moe_layer_metal.py`'s own comment, which is the check that the
script measures the right thing.

**The missing 28.5 % is the router's selection boundary, not lost information.** Two extra layers of
staleness cost only 6.5 points, so the residual stream drifts slowly and the stale vector is nearly as
informative as the true one. Recall by the router's own ranking says where the loss is:

    rank 1 (of 6)  95.5%      rank 4 (of 6)  69.7%
    rank 2 (of 6)  89.9%      rank 5 (of 6)  54.0%
    rank 3 (of 6)  79.7%      rank 6 (of 6)  40.5%

The predictor nails the expert the router wants most and coin-flips the one it wants least. What it loses are
experts whose scores sit within drift of the cut.

**So the obvious idea was to spend extra predictions only where the stale router is unsure, and it is a
null.** Take top-6 plus every expert within `margin` of the 6th score, `margin` in units of the spread from
the 1st to the 6th so it does not depend on a layer's score scale:

| margin | mean predicted | recall | fixed width at the same cost |
|---:|---:|---:|---|
| 0.00 | 6.00 | 71.5 % | top-6 at 71.5 % |
| 0.05 | 8.82 | 78.7 % | **top-8 at 78.9 %** |
| 0.10 | 12.44 | 83.1 % | **top-12 at 85.7 %** |
| 0.20 | 17.22 | 87.6 % | top-12 at 85.7 % |
| 0.35 | 20.73 | 90.0 % | top-12 at 85.7 % |

**Adaptive width is not better than fixed width per byte, and at 12 predictions it is 2.6 points worse.** The
score gap does not predict which way drift will flip the ordering; the ambiguous layers simply absorb extra
predictions without proportionally more hits.

**What that leaves.** No cheap re-use of the stale scores beats plain top-k, and top-k's own optimum on FP4
is settled at 6 (section 9.10). Recovering the remaining 28.5 % needs a *different signal* — a learned
correction, or an input that includes part of layer L's own update — which is a research project rather than
a session, and it would have to pay for whatever it costs to compute inside the layer it is trying to run
ahead of.

**Untested, and cheap for whoever wants it.** The knob cannot express "predict L+2 *instead of* L+1":
`PREDICT_AHEAD` is cumulative, so 2 predicts both and doubles the bytes (section 11). The offline table says
predicting two layers early costs only 6.5 points of recall, so a non-cumulative version would double the lead
time at constant bytes — worth a code change and a 20-minute A/B if the timing term (40 ms per token) is ever
worth attacking on its own. Nobody has measured whether it wins; the recall table is a screen, and section 8.3
is about what screens are worth.


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
| FP4 scores 0.6 syntax errors per 100 lines, 30 to 40x better than any quantized bank | HANDOFF §2, §3.3, §7.4 | 163 lines from an arm the guardian truncated after two of three seeds, carried by one clean 153-line block. Re-run to three full seeds: **8.9** on 461 lines, one clean block in four. The gap is real and is about **3.5x**. |
| A better affine fit above 2 bits is the only route left to a smaller quality bank, and the next session's first job | HANDOFF §2, §9.0 | Taken. The fit was improved 28 % on a screen made faithful with real activations, a 221.5 GiB bank was built and gated, and it wrote 31.6 C++ errors per 100 lines against FP4's 8.9 and collapsed on 2 of 6 long turns. Bits, not fit, are binding. **Lever 0 closed.** |
| Activation-weighted fitting is the untried part of the calibration idea worth keeping | HANDOFF §9.0.1, §9.3 | Tried. Worth 5.1 % of routed-expert output error at 3 bits, transfers across texts, costs no bytes — and worth nothing a compiler can see. The *capability* is kept; the lever it was meant to open is closed. |
| The drive is not saturated during decode, so a speculative read is nearly free | HANDOFF §9.10 | True on 9.49 MiB experts, where the drive was busy 45 % of decode. On FP4 it is busy **80.5 %** and at its knee: raising `CACHALOT_PREDICT_WORKERS` from 2 to 8 *lowers* achieved bandwidth from 5.70 to 5.34 GB/s. The width conclusion survives on a different mechanism. §6.1, §9.10. |
| Prediction from an earlier activation is free and useless: timing is not the problem, coverage is | 09-17 §13, HANDOFF §11 | Measured where timing was 2.2 % of blocked time and worth 1.6 ms per token. On FP4 timing is **20.5 % and 41.4 ms per token**. Still unbeaten, but the premise has expired and `CACHALOT_PREDICT_AHEAD` has never been swept on FP4. §6.1. |
| Mirror striping is harmful; leave it off | HANDOFF §11 | Measured past its optimum. At fraction 0.15 it is indeed worse than off; at **0.10 it is −5 % decode and −7 % cold prefill** with non-overlapping ranges and no quality change. Shipped. §9.11. |
| DSpark is parked pending arithmetic on FP4's expert size | HANDOFF §9.1, prompt v9 | Arithmetic done. **1.03x** on measured constants against its own 1.15x bar. Closed. §9.1. |
| The all-resident compute floor is 93 ms | HANDOFF §6, §9.2 | 93.0 ms is the 2-bit bank's. On FP4 it is **84.6 ms**, and the FP4 expert kernel costs 23.5 ms per token against the affine path's 24.6. Compute is bank-independent in fact. §6.1. |
| Dispatch count is the largest open lever and the only large one depending on nothing else | HANDOFF §9.2, prompt v9 | True by size, misleading by value on the bank in use: 84.6 ms of a 341 ms token that is already hidden under ~320 ms of drive time. It pays after bytes come down, not before. §9 ranking. |
| There is no `pack_3bit`, and MLX's packing above 2 bits is the blocker | HANDOFF §9.0, §8.1 | `pack_bits` writes MLX's layout at any width and is pinned against `mx.quantize` at 2, 3, 4, 5, 6 and 8 bits. The layout is one contiguous little-endian bit stream with no padding. |

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
- **Mirror striping** was listed here as harmful until 2026-09-19. It is not: it was measured past its optimum. See section 9.11 — shipped at fraction 0.10.
- **Engram on the internal SSD** for speed: null (it survives as lever 7 for robustness only).

**Caching and scheduling**
- **Eviction policy work.** Segmented LRU and decayed frequency are worth 1 to 2 points against 13 % fewer
  bytes from a larger budget, and total store overhead is 1.0 ms per token, so there is nothing to win.
- **LRU prefill admission** as an alternative to the quota planner: identical decode misses, worse prefill.
- **Prediction from an earlier activation** (more lead time): free, and useless — timing is not the problem,
  coverage is.
- **A GPU heartbeat to prevent clock drop**: the clock does not drop during decode.
- **More predict-pool workers on FP4.** `CACHALOT_PREDICT_WORKERS` 2 / 4 / 8 at a 36 GiB budget, four runs
  each interleaved both ways: **341 / 363 / 369 ms per token**, ranges non-overlapping, monotonically worse.
  The drive is already at its knee at 2.67 reads in flight; more concurrency lowers achieved bandwidth
  (5.70 / 5.35 / 5.34 GB/s) and lengthens the demand reads that are on the critical path (9.59 / 11.56 /
  11.78 ms). The timing term does fall as intended, 41.2 to 26.4 ms, and the coverage term rises further than
  that gain, 160.3 to 199.8 ms. **`decode_anatomy.py`'s own "at 7.3 GB/s with 8 reads in flight" line is a
  model, not a measurement, and the model is wrong on this drive.** The default of 2 is correct.
- **`CACHALOT_PREDICT_AHEAD=2` on FP4.** 330.5 against 359.5 ms per token, four runs a side interleaved,
  8.8 % worse. The extra lead time did what it was meant to -- coverage blocking fell 22 ms, from 147.1 to
  125.1 -- and it lost anyway, because the knob is cumulative: bytes rise from 1,868 to 2,514 MiB per token,
  precision falls from 55 % to 39 %, and on a saturated drive the queue pushes the timing term up 40 ms. The
  2026-09-17 null survives for a third distinct reason. **Lead time is worth having; this knob cannot buy it
  without bytes.**
- **Adaptive prefetch width by the stale router's own score margin.** Extra predictions only where the 6th
  and 7th scores are close: 78.7 % recall at a mean of 8.82 predicted against fixed top-8's 78.9 % at 8.00,
  and 83.1 % at 12.44 against fixed top-12's 85.7 %. **Not better than fixed width per byte, and worse when
  wide.** Measured offline in minutes with `predictor_recall.py`, no GPU. Section 9.12.
- **Segmented LRU at FP4 expert size.** Re-run because the 2026-09-17 null was measured with a resident set
  1.8x larger, and `simulate_policies.py --expert-bytes fp4` predicted +0.9 points of decode hit at 36 GiB
  and +1.7 at 44. The runtime delivers **−0.25 points** (70.6-70.7 % against LRU's 70.8-70.9 %) and 326.5
  against 327 ms per token, fully overlapping. **The simulator over-predicts SLRU and should not be used as a
  runtime prediction for it**; the 44 GiB figure was never measured and should not be quoted.

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
- **A searched, activation-weighted 3-bit bank** (`--bits 3 --group 64 --fit search-lsq --importance`):
  28 % less routed-expert output error than the fit the retired 3-bit bank used, measured on the model's own
  recorded activations and shown to transfer across texts, and **31.6 C++ syntax errors per 100 lines against
  FP4's 8.9**, with 2 of 6 long turns collapsing against FP4's none. A 221.5 GiB bank was built in 47
  minutes and gated on matched generations to find this out. Bits, not fit, are binding above 2 bits.
  Section 9.0. Do not build another affine expert bank above 2 bits expecting quality.
- **A 9x9 wide grid for the affine fit at 3 bits**: 0.2588 against the 5x5 grid's 0.2587, for 378 ms per
  expert against 143. The 2-bit finding that a finer grid does not pay holds at 3 bits too.
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

## 12.1 Reading a live session's numbers

A healthy chat session at a 44 GiB budget with the hotlist on looks like this, from `/stats` on 2026-09-18:

    expert_hit_rate 90.7 %       better than the 87.3 % on record; the hotlist is part of it
    resident 4,431 experts       93.4 % of the budget, and 4,431 x 9,953,280 B exactly
    decode 6.3 to 7.1 tok/s      the upper half of the recorded 6.0-7.5 range
    prefix cache 19 hits, 1 miss
    mlx peak 59.6 GiB            under the 72 GiB wired limit

Two ways to misread the machine while it runs:

**"RAM is at 81 %, so there is headroom."** There is not. MLX alone peaked at 59.6 GiB and 81 % of 96 GB is
about 77.8 GB. The wired limit is already 72 GiB, and the measured budget curve says 44 to 52 GiB buys
2.6 points of hit rate while wiring about 80 GiB — the configuration class that panicked this machine twice.
Section 9.4.

**"The GPU is only at 56 %, so there is compute headroom."** That idle *is* the expert stall. At 7.08 tok/s a
token is 141 ms and the measured all-resident floor is 93 ms, so about a third of every token is the GPU
waiting on the drive. You cannot convert it by giving the GPU more of the same work. Only two things use it:
removing stalls (sections 9.10 and 9.4) or filling them with speculative work (section 9.1).

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

**Screen candidate affine fits at a fixed format** — any width, and `--activations` scores on the model's
own recorded inputs rather than random unit vectors
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/quant_fit_screen.py --experts 24 --probes 8 --groups 128,64
```
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/quant_fit_screen.py --bits 3 --experts 16 --groups 64,128 --fits mlx,search,search+lsq --activations benchmarks/results/activations_moe_input_both.npz
```

**Record what the routed experts are multiplied by** — needed for any activation-weighted fit, and the only
tool here that looks at the model's own activations
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 2400 --tag acts -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/capture_activations.py --tokens 128 --prefill 128 --samples 64
```

**Add recordings together**, so a weighting is about the model and not about one document (no model loaded)
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/capture_activations.py --merge benchmarks/results/activations_moe_input.npz,benchmarks/results/activations_moe_input_code.npz --out benchmarks/results/activations_moe_input_both.npz
```

**Does free generation fall into a repetition loop?** — required for any numerics or sampling change
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 5400 --tag rep -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/repetition_quality.py --seeds 4 --max-new-tokens 900 --save-text /tmp/replies
```

**Does the generated code compile?** — the paired bank comparison, no GPU needed
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py /tmp/replies_bank_a /tmp/replies_bank_b
```

**The whole deciding gate for a bank, both arms matched** — this is what closed lever 0. Three seeds, 1400
tokens, the production sampling, `settle.sh` between arms, and the paired comparison at the end. It takes
about four hours with one arm served from the USB drive, and **both arms must run to completion**: FP4's
truncated two-seed arm is what put a 0.6 in this document for a day (section 7.4).
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && for arm in "lcgate-fp4 /Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts /tmp/lc_fp4" "lcgate-cand /path/to/candidate-bank /tmp/lc_cand"; do set -- $arm; benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 10800 --tag "$1" -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK="$2" CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/repetition_quality.py --seeds 3 --max-new-tokens 1400 --frequency-penalty 0.2 --penalty-window 128 --save-text "$3"; benchmarks/settle.sh --budget-gib 24; done && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py /tmp/lc_fp4 /tmp/lc_cand
```

**Build a 3-bit bank** (`mlx` fit: about 19 ms per expert, so I/O bound rather than fit bound)
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/build_affine_bank.py --out /Users/hamedprooshani/DeepSeek-V4.1-Flash-q3g64 --bits 3 --group 64 --fit mlx --verify 12
```

**Build the searched, activation-weighted 3-bit bank** — 166 ms per expert, 47 minutes, 221.5 GiB. **This
bank was built, gated, rejected and deleted on 2026-09-18 (section 9.0);** the command is kept because the
recipe is the evidence, and because it is the template for any future width. The output goes on the X10Pro
because the internal SSD has 68 GiB free, and the source is read from the internal FP4 copy so the reads stay
off the drive being written.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/build_affine_bank.py --model-path /Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts --out /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q3g64-act --bits 3 --group 64 --fit search-lsq --importance benchmarks/results/activations_moe_input_both.npz --verify 12
```

**Mirror striping A/B** — the 2026-09-19 lever. Four runs a side, interleaved both ways, `settle.sh` between;
drop `CACHALOT_MIRROR_*` for the off arm. The X10Pro is read only.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 1800 --tag mir-0.10 -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 CACHALOT_MIRROR_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_MIRROR_FRACTION=0.10 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64
```

**Where the time goes on the FP4 bank** — run this first whenever the mounted bank changes; every timing
conclusion in this document is a property of the bank, not of the runtime
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 3600 --tag fp4-anatomy -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 64
```

**The FP4 compute floor** — `decode_resident.py` cannot measure it at 36 GiB (2,056 slots is under its own
working set); `profile_decode_components.py` times an all-resident token directly and reports 84.6 ms
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 2400 --tag fp4-components -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/profile_decode_components.py --prompt-tokens 512
```

**Speculation's economics on any expert size** — free, no model; this is what closed lever 1 on FP4
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/speculation_bytes.py benchmarks/results/trace_routing_v7.trace.npz --budgets-gib 36,44 --expert-bytes 18800640 --out benchmarks/results/speculation_bytes_fp4.json && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/speculation_policy.py --acceptance benchmarks/results/dspark_acceptance_conf.json --bytes benchmarks/results/speculation_bytes_fp4.json --budget-gib 36 --baseline-ms 325 --compute-fixed-ms 84.6 --draft-ms 25.3 --miss-scale 0.8654 --expert-bytes 18800640 --drive-gbps 5.97
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
| `benchmarks/quant_affine.py` | the packers (`pack_2bit`, `pack_bits` at any width) and the fits MLX does not provide, all of them width-agnostic and optionally activation-weighted |
| `benchmarks/build_affine_bank.py` | bank builder: one shard per layer, one expert in memory, resumable, self-verifying |
| `benchmarks/expert_requant_error.py` | the seconds-long screen for candidate formats |
| `benchmarks/nll_expert_precision.py` | the quality gate; dense arms and the production arm |
| `benchmarks/decode_anatomy.py` | blocked time split by cause, reads split by worker pool |
| `benchmarks/guarded_run.sh`, `benchmarks/settle.sh` | the memory guardian and the between-arms gate |
| `tests/test_quant_affine.py` | pins the packing against `mx.quantize`'s own layout at 2, 3, 4, 5, 6 and 8 bits, including the word-straddling case |
| `benchmarks/predictor_recall.py` | bounds the routing predictor offline: recall against width, against staleness, and by the router's own ranking; no GPU, no experts |
| `benchmarks/capture_activations.py` | records what the routed experts are multiplied by, per layer; `--merge` adds recordings without loading the model |
| `benchmarks/activation_importance.py` | recorded activations to the per-column weighting a fit takes; w2's is derived per expert from the SwiGLU hidden |
| `tests/test_activation_importance.py` | pins the weighting's orientation, w2's derivation, and that a weighting of all ones is the unweighted fit |
| `src/cachalot/model/dspark_draft.py` | the three DSpark stages, their window caches, the Markov and confidence heads; optional one-time quantization of the draft's own experts |
| `benchmarks/dspark_acceptance.py` | per-depth draft acceptance and per-block confidence |
| `benchmarks/dspark_draft_cost.py` | what one draft block costs, split by stage, head and Markov correction |
| `benchmarks/verify_forward_cost.py` | the cost of a K-position forward, measured on the prefill path |
| `benchmarks/speculation_bytes.py` | misses per forward against verification width, replayed from a trace |
| `benchmarks/speculation_policy.py` | the confidence-gated projection, and every assumption behind it |
| `benchmarks/decode_resident.py` | the all-resident compute floor |
| `benchmarks/quant_fit_screen.py` | candidate affine fits at one format, in minutes |
| `tests/test_dspark_draft.py` | pins the draft's attention index set, whose failure mode is a false null |
| `benchmarks/repetition_quality.py` | free-running collapse rate; canned-context and no-prefix-cache arms |
| `benchmarks/code_validity.py` | syntax-checks generated code blocks; the paired bank comparison |
| `tests/test_sampling_penalties.py` | pins that the frequency penalty grows with the count and survives greedy |
| `benchmarks/quant_affine.py` | the fits; `fit_search`/`refine_lsq` work at any width, `dequantized()` screens without packing |
| `tests/test_bank_writer.py` | pins that the quantizer's output fills exactly what the shard header reserved |

### Session logs, for history

| document | what it holds |
|---|---|
| `docs/HANDOFF-2026-09-16.md` | the five changes that reached 4.3–5.6 tok/s; budget sweeps on the 3-bit bank; the panic postmortem |
| `docs/HANDOFF-2026-09-17.md` §1–9 | concurrency diagnosis, Engram null, prediction-knob nulls, segmented LRU |
| `docs/HANDOFF-2026-09-17.md` §10–15 | what decode blocks on; coverage versus timing; the three measurement traps; the real drive speed |
| `docs/HANDOFF-2026-09-17.md` §16–19 | the 2-bit bank, the fit, the gate, the width sweep, and the ranking this document replaces |
| `docs/HANDOFF-2026-09-17.md` §20–27 | the compute floor, DSpark and its acceptance, speculation's economics, the refined 2-bit fit, and lever 4 closed |
