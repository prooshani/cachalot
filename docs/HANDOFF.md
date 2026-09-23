# Cachalot — Engineering Handoff

**Authoritative state as of 2026-09-24, after Hamed's first Hermes Agent Desktop session (section 15.6)
and the session before it that made agent turns reuse the model's own replies (section 15.5).** The first
block below is new; the blocks after it still hold.

> ## Start here (2026-09-24, 0.12.2): Hamed's first Desktop session found two server bugs, both fixed
>
> - **Retries after a cancelled request failed with "There is no Stream(gpu, 12) in current thread"**: MLX
>   thread affinity. All generation now runs on one thread (section 15.6).
> - **Hermes cut a 917 s cold prefill at its 900 s local stale limit**: the server now sends empty-delta
>   chunks while it prefills.
> - **Hermes's 22k-token system prompt changes mid-session** (a vision-dependent tool description, the
>   provider label), each change a 7-minute cold prefill. `supports_vision: true` for the Cachalot model
>   stabilizes it; see `docs/manual-tests/hermes-desktop.md`.

**Previous block, 0.12.0:**

> ## Start here (2026-09-23, 0.12.0): long agent sessions work; one speed mystery is now the top job
>
> - **Agent turns reuse the model's own reply** (section 15.5). Hermes sends every reply back re-serialized
>   (tool arguments in another key order, an empty thinking block as a space), which broke the prefix match
>   a few tokens into the reply. The server now recognizes its own reply and uses its own tokens: 11 % fewer
>   warm prefill tokens over a 10-turn session, and all of a long `write_file` body.
> - **Fixed: one long session evicted the system-block snapshot**, so the next new session paid the full
>   cold prefill (185.9 s). Boundary snapshots are now pinned; verified live at 1.07 s. Hermes compression
>   works but costs 4-6 minutes of summary decoding plus one cold re-prefill, because it adds a tool.
> - **A 10-turn Hermes session up to 26k context and a two-image vision conversation ran correctly.** Images
>   in the history are no longer re-encoded every turn.
> - **Decode speed does not depend on context length** (54 vs 16k tokens, clean A/B). But decode alternates
>   between ~8 and ~4 tok/s windows at the same misses/token, and the slow window is not the drive, GPU
>   compute or context. That is the next session's Job 1.
> - **Version 0.12.0.** 309 tests pass. Hamed's manual Desktop test: `docs/manual-tests/hermes-desktop.md`.

**Previous block, 0.11.0:**

> ## Start here (2026-09-23, 0.11.0): a new agent session costs ~1 s of prefill, a restart ~3 s
>
> - **The server snapshots where the system prompt ends** (section 15.4). A new Hermes session reused
>   13,456 of 13,468 tokens and prefilled in 1.09 s, against 18.9 s in 0.10.0 and 206 s before it.
> - **That snapshot is kept on disk** (`--snapshot-dir`, set by `serve.sh`). The first Hermes request after a
>   server restart prefilled in 3.31 s instead of 163 s. Restores from disk are bit-identical.
> - **Version 0.11.0.** 300 tests pass.

**Previous block, 0.10.0:**

> ## Start here (2026-09-23): Hermes works, images work, long prompts no longer run out of memory
>
> - **Hermes Agent drives Cachalot** (section 15.1): parallel tool calls, a write task, a resumed session and
>   an image, all correct, through the stock Hermes CLI and an isolated `HERMES_HOME`. Four server-side
>   breakages were found and fixed on the way: a 32k context Hermes refuses, a Metal OOM on its 13.5k-token
>   prompt, a 500 on `reasoning_effort: "none"`, and abandoned requests that still cost a full prefill.
> - **Prefill is chunked** (section 15.2), 4,096 tokens per call. Quality was checked against token-by-token
>   decode (NLL and KL over 64 teacher-forced tokens), and it is as close to ground truth as whole-prompt prefill.
> - **Snapshots are 6.5x smaller and a new agent session reuses the system prompt** (section 15.3): 18.9 s
>   instead of 206 s for a new Hermes session's first request. Restores are bit-identical.
> - **Vision works end to end through the server** (section 16.4): the model read every number off the
>   checkpoint's KV-cache chart. Three things piece 3 had missed against the reference were fixed first:
>   delimiter embeddings, Engram masking on image positions, and image identity in the prefix cache.
> - **Version 0.10.0.** 290 tests pass.

**Previous state, 2026-09-22, end of the session that answered Job 1 and closed the miss's own
arithmetic.** This document supersedes `HANDOFF-2026-09-16.md` and `HANDOFF-2026-09-17.md`
wherever they differ. Those two remain as the session logs: they carry the derivations, the discarded
attempts and the raw tables behind the numbers quoted here, and section 14 indexes them. Read this document
in full before running anything or proposing any change.

**Version:** Cachalot 0.9.6 (a live reading and a memory sampler, section 7.2.9), not yet tagged or pushed; 0.9.5 (a quality-gate change, section 9.32) is likewise unpushed; 0.9.4 below is unchanged and also not pushed. 0.9.0 is the runtime change this document's section
9.24 is about; 0.9.1 is the live reading in section 7.2.6; 0.9.2 is the measurement session behind sections
7.1.9, 7.1.10 and 9.26-9.30; 0.9.3 is the second live reading of the shipped configuration in section 7.2.7
and the unit correction in section 7.2.8; 0.9.4 is Job 1 answered (section 7.1.11) and a benchmark-instrument
bug fixed with it (`benchmarks/expert_read_scaling.py`, one line). **Only 0.9.0 carries a runtime change under
`src/cachalot/`; 0.9.4 touches only a benchmark script.**
**229 tests pass**, including `tests/test_engram_reader_parallel.py`, which pins the parallel Engram row
path against the serial one that section 9.24 replaced, `tests/test_hyper_connection.py`, which pins the contraction that section 7.4.8 is about,
the eleven prefill-parity tests added on 2026-09-20, the slot-view aliasing test added on 2026-09-21, the
traced-MoE-block parity test added the same day and the three memoised-kernel-constant tests added after
it.

> ## Start here: Job 1 is answered, and there is no memory-pressure lever
>
> **The miss is at the drive's rated wall, and wired memory pressure is a null.** v29's Job 1 asked whether
> the 1.7 ms per-miss cost is the drive slowed by decode's own 43-63 GiB of wired memory — nobody had run
> `expert_read_scaling.py --wire-gib` near a real budget. Two bugs in the instrument had to be fixed first:
> the ballast's keep-alive heartbeat thread was crashing on its first tick on MLX 0.32.2
> (`RuntimeError: There is no Stream(gpu, 0) in current thread`, from evaluating a never-materialized array
> on a thread that did not create it) and dying silently, so three sweeps in a row measured an unwired
> machine after the first arm or two without the script saying so. Fixed with one `mx.eval()` call. Measured
> clean at `io_workers=8`, 42.9 GiB wired (45 % of the machine, the most this session's free memory would
> admit — not the shipped budget's 80-83 %): **6.81 GB/s wired against 6.76 unwired**, both matching section
> 3.1's cold rating, and the runtime's own 1.41 ms blocked component sits inside that same band rather than
> above it. There is no store-side overhead and no memory-pressure tax hiding in the miss. **The budget is
> the only lever on it, now measured rather than assumed.** Sections 7.1.11, 9.31, 12.
>
> ## The shipped configuration changed on 2026-09-21, and it is nearly 3x faster
>
> **Cachalot now runs the 2-bit g128 bank at 7.6 tok/s with reference-quality output.** The configuration in
> section 4 is the 2-bit bank at a 44 GiB budget with the hotlist and **nothing else** — no mirror striping,
> no frequency penalty. Hamed's own session on 2026-09-21: 314 tokens at **7.62 tok/s**, 234 at 7.55, a
> **90.2 %** session hit rate, correct C++-free TypeScript, a coherent story and a scanning German poem
> (section 7.2.2).
>
> **Everything that argued against that configuration was the transposed residual mix.** Three sessions
> ranked quantization formats and concluded quality lived in the weights; re-gated through the fixed runtime
> the 2-bit bank compiles 20 of 20 C++ blocks and malforms 0 of 94 include lines, **equal to FP4 and to the
> hosted reference on every column** (section 7.6). The repetition collapse that justified
> `--frequency-penalty 0.2` was the same defect: 0 of 8 against the original 5 of 8, same bank and protocol,
> p = 0.026 (section 9.9). The artefacts blamed on 2-bit quantization — "whitewas crumbling houses", Nikos
> renamed "Niks" — do not reproduce.
>
> **Four levers turned out to be properties of the bank rather than of the runtime**: dispatch count,
> prefetch timing, mirror striping (section 9.11.1, an 18 % *loss* at 9.49 MiB and a −5 % gain at 17.93) and
> the whole quantization ranking. **Re-measure section 9's ordering before trusting it** — it was computed on
> a 341 ms token that read 1,858 MiB, and the token is now 131 ms reading 804.
>
> **The drive has stopped being the wall.** 57.1 % busy against FP4's 84.5 %, with 69 % of the token now
> compute (section 6.2).
>
> **And the compute is not shaped the way this document said it was.** The hyper-connection kernels are
> **4.6 ms of a token, not 68.7** — the 68.7 was an artifact of the profiler that produced it, which evaluates
> each piece behind its own barrier (section 6.3). Measured the way a token actually pays, an all-resident
> token is **65 % waiting inside `mx.eval` and 35 % Python building the next graph while the GPU has nothing
> queued**: 29 ms per token, at 40 layers of 0.73 ms each, on the critical path and invisible to every profile
> taken before 2026-09-21. That is the largest addressable block, and one tenth of it is now taken
> (section 9.15).
>
> **A quarter of it is taken as of the second session that day, and the instrument is `mx.compile`.** The
> decode MoE block — six routed experts, the shared expert, their sum — traces once and is replayed instead
> of being rebuilt on all forty layers of every token. The all-resident token falls from 82-85 ms to
> **77 ms** at its minimum and the CPU side from 27 to 21.5 ms, ranges non-overlapping on every statistic,
> with mean NLL identical to four decimals (section 9.16). Custom Metal kernels trace too, so the same
> treatment is available for the hyper-connection glue and, with more care about shapes, for attention.
> Two full interactive sessions on it, one per arm, decode at **7.76 and 7.82 tok/s** on long replies with
> reference-class output and no artefacts; the live A/B is a null because a 5-8 ms change on a 128 ms token
> is below what a chat session resolves (section 7.2.3).
>
> **The CPU third is now 20.7 ms, and the largest thing in it is not graph building.** The scalar parameter
> arrays the fused Metal kernels are handed — about 1,600 constructions per token across rope, the norms,
> the hyper-connection glue, attention, FP8 and the router — were rebuilt on every call and are now built
> once: 1.2-1.3 ms of CPU, non-overlapping on three statistics, NLL identical to four decimals
> (section 9.17). With that in, **routing prediction is 11 ms of the 77 ms floor**, 7 of CPU and 4 of GPU,
> split evenly between computing a prediction and submitting it (section 9.18) — three times the old
> estimate, and the biggest named item left. **The two levers the last two prompts ranked first are both
> worth under a millisecond**: tracing the fused hyper-connection glue is 0.2 ms once the constants are
> memoised, and tracing the router pass is 0.3 ms. Both nulls are in section 11, together with a third:
> moving the prediction submission to its own thread is a null and worse on the median, because the GIL
> means Python moved to another thread is not Python taken off the decode thread.
>
> **A live read of the shipped runtime found seven long turns spread from 7.95 tok/s to 5.61, and the
> obvious explanation is wrong.** Section 7.2.4 read the ordering as an effect of reply length. Measured,
> **context length is worth 4.7 ms of `rest` across a fourfold change and is not monotone** (section 6.4),
> and a 1,792-token continuation decoded in one pass gets **faster** as it runs, not slower — +8.6 % at a
> 24 GiB budget and +18 % at 36, because the hit rate rises as the reply converges on a working set the
> cache holds. Two different prompts at the same length and budget decode **15 % apart**, which is the
> whole effect. The live spread is between-turn working-set variation; section 7.2.4's table stands and
> its explanation is withdrawn. Attention's growing shapes are worth at most those 4.7 ms.
>
> **The shipped 44 GiB configuration has now been profiled, and 36 GiB runs again.** Both budgets had been
> refused or killed for three days; on 2026-09-21 four benchmarks ran at 44 and two 310-second ones at 36,
> all under 3.2 GiB of compressor. The wired-plan hypothesis prompt v22 carried is refuted — the same long
> run finished in 310 s at a forced 72 GiB wired limit and 313 s at the auto limit — so it was the rest of
> the machine, and `settle.sh`'s available figure is what to check. At 44 the token is **170 ms at an
> 83.5 % hit rate reading 627 MiB**, and every structural ratio matches the 40 GiB profile within a point:
> **46.4 ms per token of blocking on misses no prediction covered, with the drive idle 45 % of the time**
> (section 7.1.3).
>
> **Everything section 9.18 named as the way to make prediction cheaper is now closed.** Admitting the
> mispredicted bytes has a ceiling of 4.3 % of demand reads; a blocklist on re-reading a dropped expert
> trades 8.1 wasted reads for 3.4 demand misses per token; the submission bookkeeping is 0.040 ms per
> token, not 3.6 (section 9.19). Two layers ahead, top-4, top-8 and every GIL switch interval are nulls at
> 44 GiB (section 7.1.3). **And the floor itself is misnamed**: `profile_decode_sync.py`'s "all-resident"
> token issues 40 speculative reads and reads 380 MiB per token, which is what the 3.6 ms of "submission"
> was paying for.
>
> **Both arms were then run interactively on 0.7.0 and the runtime's steady state is the same to a tenth of
> a point**: 90.00 % and 89.92 % session hit rate, 33.58 % and 33.38 % prediction precision, 4,495 and 4,490
> residents, 55.1 and 55.5 GiB of MLX peak against a 72 GiB limit (§7.2.8), on turns at **7.78-7.92 tok/s** for prose
> and **6.90-6.93** for 1,300-1,500 tokens of Objective-C. The live A/B of the memoised constants is a null
> for the third time. **46.9 % of every byte a live session reads is a prediction nobody used**, with the
> accounting closing to the byte. And the first live coding turn ever put through `clang` — one of the two —
> **does not compile**: a helper declared `NSString *` handed `id` values, which is a model-level type error
> and not this runtime's defect class, but it means reading a program by eye is not the check it was being
> used as (section 7.2.5).
>
> **The rate has not moved in four sessions, and section 9.20 says why.** Laid out by measured size, a
> 128.5 ms live token has two blocks nobody has ever attacked — **about 20 ms of GPU time no measured piece
> accounts for**, and **about 33 ms by which a streaming token's `rest` exceeds the all-resident floor** —
> against levers of 1-5 ms each, which is all that has been worked on. Getting there needed the GPU side
> decomposed on the code that actually runs, and **two of `profile_decode_gpu.py`'s rows were timing retired
> paths**: routed experts are **6.7 ms** per token and not 19.8, the layer's own router **0.9 ms** and not
> 7.5 (section 7.1.4). Both are fixed in the instrument. The expert kernel is closed with them — the traced
> block costs exactly what its three matmuls cost — and the largest *named* GPU piece is now attention at
> 14.3 ms across 30 of 40 layers.
>
> **And the 30 ms of "never looked at" in section 6.3 is not concentrated anywhere.** Per-layer timing with
> no added barrier puts the eight source and index-source layers at 4.5 ms of excess over eight plain
> layers, Engram at 1.6 ms, and everything outside the layers at 7.0 ms; the rest is spread evenly across
> forty layers at 1.4 ms of GPU and 0.55 ms of CPU each (section 6.3.1).
>
> **The 20 ms of GPU time nothing accounted for is gone, and half of it was never timed.** The profiler
> covered thirty of the forty layers' attention and none of the head or Engram. Measured: the eight
> compressed-source layers are **8.9 ms**, the two sliding-window ones 0.8, the head 1.6 and Engram 0.8, so
> the shipped pieces now sum to **41.6 ms** against 54.8 inside `mx.eval`. **Attention is the largest GPU
> block in the token at 22.5 ms** — three times the routed experts — and a source layer costs about a
> millisecond more than a reuse layer, which is **5.5 ms per token of compressor and indexer nobody has
> looked at**. The rest of the old gap is arithmetic nobody had done: **one `mx.eval` costs about 0.20 ms
> of round trip whatever it evaluates**, a token pays 44 of them, and no chained measurement can see it
> (sections 7.1.5, 7.1.6).
>
> **`mx.compile` on attention is closed three ways**, which retires the last untried instrument in the
> repository: `shapeless=True` will not run against custom Metal kernels, a plain trace is not
> bit-identical, and it retraces every token anyway for a net loss (section 9.21).
>
> **And the first lever with a mechanism since 2026-09-19 was built, measured in two shapes and loses in
> both.**
> The shared expert does not depend on the routing, so it can be issued before the layer blocks for it;
> that takes **10 ms per token out of `mx.eval`**, almost exactly what the model-free screen predicted, and
> puts **13 ms back on the CPU**, because `mx.async_eval` walks and enqueues the graph on the decode thread
> forty times a token; adding it to the routing's own eval instead costs nothing on the CPU and 5 ms inside
> eval, because then the sync waits for it with nothing else queued. 79.1-79.6 ms shipped, 81.8-82.6 and
> 83.5-85.7 for the two arms. It ships off (`CACHALOT_PRELAUNCH_SHARED=0`), and every arm's 16-token
> greedy fingerprint is identical to the shipped one (section 9.22).
>
> **And the 33 ms nobody had a mechanism for is the Engram row reads. It is fixed, and it is the first
> thing to move the decode rate in five sessions.** `profile_decode_sync.py` now has a streaming arm and a
> column for the Engram `pread`s, and against an all-resident token a streaming one at a 36 GiB budget pays
> 63 ms more blocked in the expert store, 20 more inside `mx.eval`, **27 more inside `_apply_engram`** and
> only 4 more of real CPU. The mechanism is arithmetic: a decode token asks for 24 Engram rows twice, each
> row is two `pread`s, and the reader only used its sixteen-worker pool at 64 rows or more — so 96 reads
> went out one at a time from the decode thread, behind a queue the expert stream was filling. Reading them
> through the pool (`CACHALOT_ENGRAM_PARALLEL_MIN`, default 8) and issuing them at the top of the token
> instead of at the layer that needs them (`CACHALOT_DECODE_ENGRAM_PREFETCH`, default 1) takes the column
> from 28.4 ms to 2.4, the token from 188.3 ms to 162.4, and `decode_anatomy.py`'s rate from **5.65 to 6.51
> tok/s (+15.2 %)** with the hit rate, the byte count, the miss count and the prediction precision all
> unchanged and a 16-token greedy fingerprint identical. Sections 7.1.7, 9.24.
>
> **Live, at a 52 GiB budget, the session runs at 9.42 tok/s on prose and 8.53 on 1,493 tokens of
> Objective-C, at a 92.37 % hit rate.** Against 0.7.0's four sessions at 44 GiB — 7.78-7.92 and 6.90-6.93 at
> 90.00 % — that is 22-27 ms off a token, of which the miss arithmetic gives the budget about 6 and the
> Engram change the remaining 16-21. **The budget lever also landed exactly where
> `simulate_policies.py` said it would**, +2.37 points of hit rate against a predicted +2.6, at an MLX peak
> of 67.74 GiB against a 77.8 GiB wired limit, so 52 GiB fits with 10.1 GiB to spare (§7.2.8 corrects a
> unit error that had this as 72.7). The two causes were not separated in that
> session. Sections 7.2.6, 9.4. **And the coding turn does not compile** — one wrong method name made
> twice, `-stringValue` sent to an `NSString` — which is the second live coding turn ever compiled and the
> second model-level type error, not this runtime's defect class.
>
> **The source layers' 5.5 ms is also decomposed**: about two thirds of it is the indexer, a tenth the
> compressor and the rest the compressed-KV write — and **`INDEX_TOPK` is not a lever**, because an
> eightfold change in the width is worth 0.066 ms per layer (section 7.1.8).

**The defect is found, fixed and gated: `hc_post` applied the hyper-connection mixing matrix transposed.**
`comb @ residual` where DeepSeek's `Block.hc_post` does `comb.T @ residual`, in both the MLX path and the
fused Metal kernel. On the 40-case corpus this runtime now compiles **20 of 20** C++ blocks, parses **18 of
18** Python blocks, and emits **0 of 101** malformed `#include` lines — every column equal to the hosted
reference arm, against 0/42, 5/26 and 49/154 before. Section 7.4.8.

**The checkpoint ships the official implementation and no session before 2026-09-20 had opened it.**
`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` holds `model.py`, `engram.py`, `kernel.py`,
`convert.py`, `generate.py` and `encoding/`. Every sentence in this document that says no reference exists
is wrong, including the ones that justified three bank-building sessions. Section 7.4.7.

**Start at section 7.4.8.** It is the root cause and the fix, and it is why most of this document needs
re-reading rather than trusting. In short: a hosted endpoint serving the same model compiled 20 of 20 C++
blocks where Cachalot compiled 0 of 42 (section 7.4.6, which proved the fault was in this runtime rather
than in FP4, the model or the sampler); the defect turned out to be a transposed hyper-connection residual
mix; and with it fixed Cachalot compiles **20 of 20** and malforms **0 of 101** include lines — every column
equal to the reference. Section 7.4.7 is the evidence trail between those two points, and section 7.4.1 is
why every C++ syntax-error figure older than 2026-09-19 is withdrawn for an unrelated reason.

**Section 9 was frozen until quality was restored. That condition is now met** — the gate in section 7.4.8
is clean at 20/20 compiling and 0/101 malformed, equal to the reference — **so section 9 is unfrozen.** Two
conditions attach to reopening it. Every quality conclusion this document reached before 2026-09-20 was
measured through the transposed residual mix and has to be re-established before it is quoted, which is most
of sections 8, 9.0, 9.3 and 7.5. And any re-audit must use the repeat-copy number from
`benchmarks/continuation_rank.py`, not NLL or top-1, both of which were blind to the defect by construction.

**No speed optimization was ever the cause of the quality gap.** The fix reorders a contraction and cost
nothing: throughput during the clean gate was 2.1-2.5 tok/s, the same as before.

**What the 2026-09-19 session did, in one paragraph.** It profiled the bank that is actually mounted. Every
timing number this document carried had been measured on the 2-bit bank while FP4 is what runs, and correcting
that moved six conclusions, shipped one lever and closed four. Decode on FP4 is drive-bound almost end to end;
the drive is busy 80.5 % of decode rather than 45 % and is at its concurrency knee; the compute floor is
84.6 ms rather than 93; mirror striping across both drives was never harmful, only mis-tuned, and is now on;
and speculation, eviction policy, prefetch lead time and prefetch precision were each measured and closed for
a few hours of machine time and no runtime code.

**What the second half of 2026-09-19 did, and it is a correction rather than a measurement.** An outside
review of the repository found the coding quality gate scoring failed compilations as clean: `check_cpp`
grepped stderr for `": error: "` and never read the compiler's exit status, so a block aborting on a mangled
`#include` came back with an empty error list. Re-scored, **no bank this project has built has produced a
single compiling C++ block**, and the 8.9-against-31.6 that justified the standing decision's size was
largely measuring which arm aborted first. Section 7.4.1. The same review found that in-flight predictions
have no lifetime and are discarded for finishing early (section 9.13), which puts one cheap lever back on
the list.

**What the second live session at 52 GiB said, and one thing it corrected.** The shipped configuration
was read a second time, on a build with no runtime change against the one section 7.2.6 measured:
**9.59 tok/s on prose against 9.42, 8.15 on 1,484 tokens of Objective-C against 8.53, a 92.31 % session
hit rate against 92.37 %, and an MLX peak identical to the byte.** It replicates, so **52 GiB now has two
sessions** and the objection section 7.2.6 raised against making it the default is answered on the
footprint and hit-rate side (section 7.2.7). Section 7.1.9's price of a miss was checked against a
conversation for the first time and holds: 18.5 misses a token at 1.7 ms each on a 79.6 ms floor predicts
111 ms, against 104.3 and 122.7 observed on the session's two long turns. **And every MLX peak this
document has ever quoted was in GB against a limit in GiB** — the 52 GiB peak is **67.74 GiB, not 72.73,
against a 77.8 GiB wired limit, so the headroom is 10.1 GiB rather than 5.1** and a 60 GiB budget projects
to about 75.2. Nothing else moves; the peaks were only ever used to decide whether a budget fits, and the
error was in the safe direction. Section 7.2.8. **The third live coding turn also failed a compiler**, and
this one is the useful failure: repaired in one line it compiles, runs, exits 0 and still contradicts its
own documented column order, which is the first concrete case for a gate that runs what it builds
(`docs/live-turns/2026-09-21-json2csv-2/`).

**What the last session of 2026-09-21 did, in one paragraph.** It killed the one line in the ranking
that had no mechanism and then found that nothing else on the GPU side has one either. A streaming token
replayed a second time over the same continuation — same positions, same graph, same 240 experts per layer
scattered across a 40 GiB slot pool, but with everything already resident — costs **79.6 ms, which is the
all-resident floor to a tenth of a millisecond on every column**, so the "+20 ms inside `mx.eval`" that
three prompts called unexplained is simply part of a miss and is worth 0.31 ms of each one (section 7.1.9).
Read concurrency from 2 to 16 workers is a null and bypassing the page cache is a 16 ms loss, which is the
same answer from two more directions (section 9.26). Then the three blocks nobody had screened were
screened, and attention turned out not to be about attention: a reuse layer's 0.441 ms is 0.377 of weight
streaming and 0.064 of the sparse attention whose shapes three prompts wanted attacked. Underneath it is
one kernel, `fp8_gemv_decoded`, which moves **5.14 GB per token — more than twice the routed experts** —
and already runs at 79 % of what `mx.sum` gets over the same bytes (section 7.1.10). The shared expert is
at 80 % of the same ceiling, `wo_a` is faster in BF16 than in any FP8 form, and the lanes-per-row policy
nobody tuned is within 0.3 % of the tuned one (sections 9.27-9.29). **Every remaining millisecond is the
miss or the floor**, and the only lever on the miss is the budget.

**What 2026-09-20 did, in one paragraph.** It ran the independent reference arm that sections 7.4.4 and
7.4.5 had been asking for, and the answer is unambiguous. Two hosted arms — one FP4, one FP8, each with the
provider pinned and fallbacks off, no harness, matched sampling, 40 of 40 cases — compile every C++ block,
parse every Python block and emit not one malformed `#include` or `import`. Greedy on the same six tasks is
6 of 6 against Cachalot's 0 of 6. A separate Hermes harness arm was also scored, and it turns out to prove
nothing about harnesses: the corruption never appeared in any of its turns, including first drafts before a
tool ran, so there was nothing for it to repair. The runtime is the fault. Sections 7.4.6 and 2.

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

**None of those speed numbers is the headline any more.** Measured on the 40-case coding corpus against a
pinned hosted arm serving the same model, 2026-09-20:

| | Cachalot, corrupt | **Cachalot, fixed** | Cachalot, fixed, **2-bit bank** | hosted reference, FP4, no harness |
|---|---:|---:|---:|---:|
| C++ blocks that compile | 0 / 42 | **20 / 20** | **20 / 20** | 20 / 20 |
| Python blocks that parse | 5 / 26 | **18 / 18** | **18 / 18** | 18 / 18 |
| `#include` lines malformed | 49 / 154 (32 %) | **0 / 101 (0 %)** | **0 / 94 (0 %)** | 0 / 102 (0 %) |
| decode during the run, median of 40 | — | 2.28 tok/s | **4.47 tok/s** | — |

**The fourth column is the 2026-09-20 result that changes the work queue.** The 2-bit g128 bank, re-gated
through the fixed runtime, is equal to FP4 and to the hosted reference on every column of the gate while
generating at nearly twice the rate. Three sessions ranked it below FP4 on NLL and top-1, and section 7.4.8
proved both of those blind to the defect that was actually producing the artefacts. Section 7.6.

Section 7.4.6 opened that gap and **section 7.4.8 closed it the same day**: the cause was a transposed
hyper-connection residual mix, and with it fixed this runtime matches the reference on every column of the
gate. The decision quoted below is kept because it is the decision that produced the fix, and because the
premise it retired — that quantization set the quality ceiling — was wrong in a way worth remembering.

> ### The standing decision, replaced on 2026-09-20: reproduce reference quality first, at any speed
>
> **The old decision was "quality over speed", and it was built on a premise that is now false.** It assumed
> the ceiling was the quantized weights: that FP4 was the best quality this hardware could serve and the job
> was to pay for it in tokens per second. Section 7.4.6 killed that. A hosted endpoint serving the same model
> at FP4, with no harness and matched sampling, compiles **20 of 20** C++ blocks and emits **0 of 102**
> malformed `#include` lines. Cachalot compiles **0 of 42** and malforms **49 of 154**. The gap is not a
> quantization cost that had to be bought. It is a defect in this runtime, and it can be fixed rather than
> traded for.
>
> **The decision now: correctness is the only gate, and throughput is not a constraint while it is being
> found.** Hamed's instruction, 2026-09-20: it does not matter if the runtime produces 0.1 tok/s — the goal
> is to reproduce the model's original quality. A configuration that is slow and correct is a success; a
> configuration that is fast and corrupt is the bug.
>
> **How the culprit gets named: strip, then reinstate one at a time.** Every speed optimization this project
> has shipped is a suspect, because none of them was ever re-audited against a clean reference. The order of
> work is:
>
> 1. Build the slowest, most obviously correct path that exists — no prefetch, no prediction, no striping,
>    no speculation, no eviction cleverness, one expert read at a time, synchronous — and run the 40-case
>    corpus on it. If that is clean, the defect is in an optimization. If it is still corrupt, the defect is
>    in the expert kernel, the dequantization or the tokenizer, and none of section 9 matters.
> 2. Reinstate one optimization. Re-run the corpus. Score it.
> 3. Repeat until the corpus breaks. The optimization that broke it is the culprit.
>
> **This is a bisection, and it only works if the gate is trusted.** It is, now: `code_validity.py` reads the
> compiler's exit status (section 7.4.1), `include_integrity.py` tests a pre-registered prediction
> (section 7.4.4), and both have a clean reference arm to compare against (section 7.4.6). A step is "clean"
> when it matches the reference on compile rate and scores 0 malformed includes — not when it looks better
> than the step before.
>
> **What this retires.** "FP4 is the quality bank and 3.3 tok/s is what it costs" is no longer a trade that
> was made; it is a symptom that was mistaken for a trade. The three bank-building sessions that chased
> quality through quantization — the 3-bit bank, the searched fit, the activation weighting — were all
> answering the wrong question. None of them is wrong about quantization; all of them were aimed at a defect
> that is not in the weights. Sections 9.0 and 9.3.

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
produce. It collapsed on 2 of 6 long turns where FP4 collapsed on none (section 9.0), which is why the
table below still has four columns and not five.

| | **FP4, in use** | 2-bit g128, the fast option | 3-bit g64 (retired) | 3-bit oQ3e |
|---|---|---|---|---|
| bytes per expert | **18,800,640** | 9,953,280 | 15,482,880 | 15,482,880 |
| MiB per expert | **17.93** | 9.49 | 14.77 | 14.77 |
| experts per GiB of budget | **57.1** | 107.9 | 69.3 | 69.3 |
| bank total | **275.4 GiB** (experts only) | 142.4 GiB | 221.5 GiB | 331 GB |
| encoding | E2M1 nibbles, UE8M0 scales, group 32 | affine, searched fit | `mx.quantize`, bf16 scales | MLX affine, bf16 scales |
| C++ blocks that compile | **0 of 4** | not re-scored | not re-scored | not measured |
| where | `~/DeepSeek-V4.1-Flash-fp4-experts` | `~/...-q2g128` | **deleted** | `/Volumes/X10Pro/...` |

Deleted on 2026-09-18: the 2-bit g64 bank (strictly dominated, unused), the 3-bit g64 bank (built, gated,
adopted and retired the same day — see section 7.5) and the searched, activation-weighted 3-bit bank (built,
gated and rejected the same day — see section 9.0). All three rebuild from the FP4 checkpoint in under an
hour, and each one's evidence outlives it.

The oQ3e download is **not worth restoring**: our own 3-bit bank tied it on the production path (paired median
+0.0017, sign z −1.68), and a searched 3-bit fit now beats it on the screen (0.2600 against 0.2996, section
9.0).

## 4. The configuration to use

**Interactive chat, machine otherwise idle.** Hand him **`./chat.sh`**, not the one-liner below.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./chat.sh
```

`chat.sh` sets exactly the environment the one-liner sets, exports it, refuses to start a second runtime,
and passes any extra argument through to `cachalot.cli chat`. It exists because the one-liner cannot survive
being pasted into a terminal that wraps it — section 12 has what that costs, measured at 0.6 tok/s against
7.6 on 2026-09-21. The one-liner remains the record of what the configuration is:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && pgrep -fl "deepseek-v41/bin/python|cachalot" || CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json CACHALOT_HOTLIST_GIB=8 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat --expert-budget-gib 44 --max-seq-len 32768 --max-new-tokens 1024 --temperature 0.6
```

**Two things changed in that line on 2026-09-20 and both are measurements, not preferences.**

**The bank is now the 2-bit one.** Re-gated through the fixed runtime it compiles 20 of 20 C++ blocks and
malforms 0 of 94 include lines, equal to FP4 and to the hosted reference, while reading 804 MiB per token
against 2,080 (sections 7.6 and 6.2). Measured on `chat_turns.py` at a 44 GiB budget with the hotlist and
no mirror, it runs the six-turn chat at **7.03 to 7.40 tok/s** against FP4's 2.1-2.6 on the corpus — which
reproduces the 7.52 tok/s of section 7.2 that this bank was abandoned despite. Its
reply to "write a 100 word story of a small fish living in a greek" — the prompt that produced
"whitewas crumbling houses" and renamed Nikos to "Niks" on 2026-09-17, and that was the stated reason this
bank was abandoned — came back clean, with "Yiannis" spelled correctly throughout and the following turn's
haiku correctly quoting "olive oil" back out of it.

**`--frequency-penalty 0.2 --penalty-window 128` is gone.** It was adopted because 62 % of long code replies
collapsed into a loop; through the fixed runtime the rate is 0 of 12 with the penalty off, on both banks
(section 9.9). It distorts code that legitimately repeats, and nothing measurable pays for that any more.

**The mirror variables are gone, and that is a measurement, not an omission.** Striping was shipped on FP4
for −5 % decode; on the 2-bit bank it is an **18 % loss**, measured over five interleaved runs whose
ranges do not overlap and which missed the same experts and ended with the same resident set
(section 9.11.1). A 2-bit
expert is 9.49 MiB and a demand read is 4.15 ms, so the 10 % tail sent to the USB drive no longer fits under
the head — the second drive sets the critical path instead of adding to it. The copy at
`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128` is kept for a future sweep of smaller fractions.

**Nothing in that line changed on 2026-09-21, and one thing behind it did.** The Engram row reads now go
through the reader's worker pool and are issued at the top of the token
(`CACHALOT_ENGRAM_PARALLEL_MIN=8`, `CACHALOT_DECODE_ENGRAM_PREFETCH=1`, both defaults, section 9.24). They
need no variable in the command; setting either to `0`/`1000000` restores the old shape for an A/B. It is
worth +15 % on a 36 GiB benchmark and is expected to be worth less in a live session, which misses less.

If you do re-enable it, note the mirror is matched by shard filename inside the *bank*: pointing
`CACHALOT_MIRROR_PATH` at the FP4 checkpoint while serving the 2-bit bank silently disables striping with a
`lacks model-00001-of-00040.safetensors` line, which is what happened to the first 2-bit corpus run.

**To go back to FP4**, set `CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts` and
add back `CACHALOT_MIRROR_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_MIRROR_FRACTION=0.10`,
which does pay on that bank. FP4 is not worse on the gate; it is 1.6x slower for the same result.

**A 44 GiB budget needs about 73 GiB free, and on 2026-09-21 it was available again.** Four benchmarks ran
at 44 and two 310-second ones at 36, all with peak compressor under 3.2 GiB and no kill, after three days in
which both budgets were refused or killed. The hypothesis carried by prompt v22 — that the guard's wired
plan rather than the budget was what refused, because `guarded_run.sh` sets no `CACHALOT_MLX_WIRED_LIMIT_GIB`
and the runtime's auto limit is `trunk + budget + cache` — **is refuted**: the same 1,792-token run finished
in 310 s at a forced 72 GiB limit and in 313 s at the auto limit, 3.1 against 3.0 GiB of compressor. What
changed was how much the rest of the machine was holding. **Check `settle.sh`'s available figure, not the
calendar.**

**A 44 GiB budget needs about 73 GiB free and that is not always available.** `guarded_run.sh` computes
`need = budget + 29` and refused 44 four times on 2026-09-20 at 71.7-72.1 GiB available, with Firefox, Slack,
Mail and Stream Deck already closed — the last 1.6 GiB was the terminal emulator itself. 42 GiB passes and
costs about one point of hit rate. Never pass `--force`.

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

**Nothing new has to be set for the traced MoE block.** `CACHALOT_COMPILE_MOE` and
`CACHALOT_COMPILE_SHARED` default to on and the command above gets them; setting either to 0 restores the
per-expert loop and is for bisecting only (section 9.16).

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

`--frequency-penalty` and `--penalty-window` are no longer set: the 62 % collapse rate that justified them
was the transposed residual mix, and through the fixed runtime the rate is 0 of 12 with them off (section
9.9). The CLI still exposes all four sampling knobs if a loop is ever seen again. `--max-seq-len 32768` rather than something enormous:
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
12. **A curve drawn through the turns of a live session is not a curve.** Seven turns ordered by reply
   length looked monotone across four sessions and were not: the ordering was the working set each reply
   walked into, and every fast turn happened to be prose and every slow one Objective-C, always running
   second into a cache the prose had warmed. Two instruments took twenty minutes to refute it. Before
   reading a trend out of an interactive session, ask what else was ordered the same way. Section 6.4.
13. **Read a run's footprint line before reading its numbers.** A 1024-token anatomy returned 389 ms per
   token, twice the neighbouring context lengths, and its footprint carried 10.0 GiB of compressor and a
   wall-clock 2.90 GB/s against the usual 5.2-5.6. `guarded_run.sh` prints the footprint above the result
   for this reason.
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

### 6.2 Where the time goes on the 2-bit bank, re-measured 2026-09-20

Section 6's 2-bit anatomy is from 2026-09-17 and predates mirror striping, the predicted-load lifetime fix
and the hyper-connection fix. This is a fresh pair, both arms run the same evening at the same budget on the
same machine state, `decode_anatomy.py --prompt-tokens 512 --decode-tokens 64`, 32 GiB budget, mirror
striping off in both (the X10Pro holds the FP4 shards, so the 2-bit bank cannot be mirrored):

| | FP4 | **2-bit g128** |
|---|---:|---:|
| decode | 380 ms/token, 2.63 tok/s | **236 ms/token, 4.23 tok/s** |
| expert hit rate | 68.1 % | **77.9 %** |
| misses per token | 76.6 | **53.1** |
| bytes read per token | 2,080 MiB | **804 MiB** |
| expert wait | 233.3 ms (61.3 % of decode) | **74.1 ms (31.4 %)** |
| rest | 147.1 ms (38.7 %) | 162.3 ms (**68.6 %**) |
| drive busy | 20.57 s of 24.35 s (**84.5 %**) | 8.63 s of 15.13 s (**57.1 %**) |
| prediction precision | 53 % | 49 % |

**The budget is 32 GiB rather than section 6.1's 36 because the preflight refused 36**: 62.1 GiB available
against 65 needed, with other applications open. Rule 1 says lower the budget rather than force it, so both
arms were lowered. That is why the FP4 column here is 380 ms against section 6.1's 341.5 — a smaller budget
holds fewer experts and the machine was not idle. The pair is internally controlled; neither column should be
compared across sections.

**The shape of the problem inverts, and this is the finding.** Section 6.1's central fact was that decode on
FP4 is drive-bound almost end to end: the drive busy 80.5 % of decode, 84.6 ms of compute hiding underneath a
341 ms token, and every lever that saves compute worth nothing. On the 2-bit bank the drive is busy **57.1 %**
of decode and **68.6 % of the token is `rest`** — compute plus whatever is not a blocked expert read. Against
the 93.0 ms all-resident floor recorded for this bank, compute is about 39 % of a 236 ms token and the drive
now has idle capacity to hide things under.

**What that does to section 9's ranking.** Lever 2, dispatch count, was demoted on FP4 with the reasoning
"worth close to nothing until bytes come down: 84.6 ms is already hidden". Bytes have come down, by 61 %, and
the sentence expires with them. Conversely the prefetch-precision and lookahead work, which was priced
against a saturated drive, is worth re-reading against one that is idle 43 % of decode. **Nothing in section 9
below has been re-ranked yet**; this is the measurement that says it must be, and it is the third time this
document has had to re-rank after measuring the bank actually in use rather than the one it was writing about.

**Coverage is still the larger half of the blocked time on both banks.** 64.1 of the 2-bit bank's 74.1 ms of
expert wait, and 185.6 of FP4's 233.3, is a demand read for an expert that was never predicted. Timing is
7.3 ms against FP4's 46.5. So the case section 9.12 makes for a better predictor survives the bank change in
kind, at about a quarter of the size.

### 6.3 What a token is actually made of, measured 2026-09-21

Section 6.2 established that 68.6 % of a 2-bit token is `rest` rather than blocked expert reads. This section
opens `rest` up, and the first thing it does is withdraw a number this document has quoted since 2026-09-19.

**The 68.7 ms attributed to hyper-connections was an artifact of the instrument.**
`profile_decode_components.py` times each piece with `mx.eval` around every call, so every row pays a full
CPU-GPU round trip of roughly 0.15 ms. That is why its rows sum to 164.3 ms against a 93.6 ms token — the
script says so itself, and the sum was read as a breakdown anyway. `profile_decode_gpu.py`, which has been in
the repository since 2026-09-17 and launches each piece many times inside one lazy graph, prices the same
kernels the way a token pays for them. Both were run on the 2-bit bank at a 512-token context on 2026-09-21:

| piece | isolated, own barrier | **chained, as a token pays** |
|---|---:|---:|
| `hc_mixes` (sinkhorn) | 0.405 ms x 80 = 32.4 ms | 0.021 ms x 80 = **1.7 ms** |
| `hc_pre` + rms_norm | 0.263 ms x 80 = 21.0 ms | 0.015 ms x 80 = **1.2 ms** |
| `hc_post` | 0.251 ms x 80 = 20.0 ms | 0.021 ms x 80 = **1.7 ms** |
| compressed reuse attention, ratio 2 | 1.110 ms x 30 = 33.3 ms | 0.492 ms x 15 = 7.4 ms |
| compressed reuse attention, ratio 1 | not measured | 0.437 ms x 15 = 6.6 ms |
| router `route_topk` | 0.307 ms x 40 = 12.3 ms | 0.170 ms x 40 = 6.8 ms |
| shared expert (fp8, 3 gemv) | 0.403 ms x 40 = 16.1 ms | 0.125 ms x 40 = 5.0 ms |
| routed experts, 2-bit affine (6) | 0.627 ms x 40 = 25.1 ms | 0.348 ms x 40 = 13.9 ms |
| **hyper-connections, total** | **73.4 ms** | **4.6 ms** |
| sum of rows | 164.3 ms against a 93.6 ms token | 44.2 ms of an 86-92 ms token |

So the hyper-connection machinery is about 5 % of the token, not 80 %, and the whole argument that it is
"the largest addressable term in the product" is withdrawn. The same correction applies to any other reading
of the isolated profile: use it to compare two implementations of one piece, never to apportion a token.

**Where the token really goes: 65 % inside `mx.eval`, 35 % on the CPU.**
`benchmarks/profile_decode_sync.py`, new on 2026-09-21, wraps `mx.eval` and `mx.synchronize` for the duration
of one all-resident token and attributes every call to its site. Twelve repeats, 512-token context, 2-bit
bank at a 24 GiB budget:

    all-resident token, median 91.7 ms
      inside mx.eval    59.6 ms  (65.0%)
      CPU outside eval  32.1 ms  (35.0%)
      eval/synchronize calls per token: 44

    per call site, per token          calls   in eval   gap before
      moe_layer_metal.py:130           40.0   57.4 ms     29.1 ms
      engram_rows.py:59                 2.0    1.0 ms      1.9 ms
      text_decode_runtime.py:2546       1.0    2.4 ms      0.5 ms

`moe_layer_metal.py:130` is the `mx.eval(route.indices, route.weights, ...)` that every MoE layer has to run,
because routing must reach the CPU to address the resident store. It is not one barrier; it is **forty per
token**, and between two of them the CPU spends **0.73 ms building the next layer's graph with the GPU idle**
before waiting **1.44 ms** for that graph to run. The two sides are almost perfectly serialised: 29 ms of CPU
plus 57 ms of GPU against a 91.7 ms token. A runtime that overlapped them perfectly would decode this token
in about 60 ms.

**The per-dispatch floor is 4.72 microseconds**, measured by chaining a trivial Metal kernel 500 times in one
graph (`build + GPU`, so it is an upper bound on what fusing two kernels can return). A token issues on the
order of 230 substantial dispatches, which is about 1.1 ms of launch overhead in total. **That closes lever 2
on the arithmetic**: dispatch count is not what a decode token is spending its time on, on either bank, and
no amount of kernel fusion buys more than about a millisecond. Section 9.2.

**What is still unaccounted, and it is the largest single item left.** The 57.4 ms spent inside the router
eval contains the profiled pieces that precede it — attention, hyper-connections, the router itself — but
those add to roughly 25 ms. The rest is unprofiled: the four `SOURCE_LAYERS`, the four
`INDEX_ONLY_SOURCE_LAYERS` and their indexer, the two sliding-window layers, Engram, the final head, and the
40 extra `route_topk` launches the predictor issues (6.8 ms by itself). **Roughly 30 ms of a 90 ms token has
never been looked at**, and the index-source layers are the obvious suspects: a plain reuse layer's attention
is 0.44-0.49 ms and there are eight layers in the two source classes that nobody has timed.

### 6.3.1 The 30 ms is not concentrated anywhere, measured 2026-09-21

Both suspects named above were wrong, and the instrument that settles it adds no barrier of its own.
`benchmarks/profile_decode_layers.py` wraps the runtime's three per-layer entry points and `mx.eval`, so each
layer's wall time and the share of it spent waiting inside its own router eval are separated at the forty
drains a token already pays. One stage of GPU work is shifted forward — a layer's routed experts are queued
after its router eval and drained by the next layer's — but every layer queues the same top-6 work, so the
shift is uniform and the difference between two classes still belongs to the layer that caused it. Twelve
repeats, 2-bit bank, 24 GiB budget, 512-token context, an 88.1 ms median token:

| class | layers | wall per layer | total wall | inside eval | CPU |
|---|---:|---:|---:|---:|---:|
| reuse, compress ratio 2 | 15 | 1.96 ms | 29.5 ms | 20.7 ms | 8.8 ms |
| reuse, compress ratio 1 | 15 | 1.96 ms | 29.5 ms | 21.3 ms | 8.2 ms |
| index-only source (24, 28, 32, 36) | 4 | 2.65 ms | 10.6 ms | 6.6 ms | 4.0 ms |
| source, ratio 2 (2, 8, 14) | 3 | 2.47 ms | 7.4 ms | 5.1 ms | 2.4 ms |
| source, ratio 1 (20) | 1 | 2.58 ms | 2.6 ms | 1.9 ms | 0.7 ms |
| Engram, layers 1 and 14 | 2 | 0.78 ms | 1.6 ms | 0.7 ms | 0.8 ms |
| everything outside the wrapped layers | | | 7.0 ms | 5.0 ms | |

**The eight source and index-source layers cost about 4.5 ms more than eight plain layers**, not thirty, and
Engram is 1.6 ms. The unattributed time was never in a few expensive layers; it is spread evenly across all
forty at roughly 1.4 ms of GPU and 0.55 ms of CPU each, which is what section 9.15 already said about the CPU
side and had no per-layer confirmation of. The head, the embedding, the final hyper-connection, the sampler
and the two sliding-window layers are 7.0 ms together.

**One number in the table above is withdrawn.** The `route_topk` row — 0.170 ms per call, 6.8 ms per token —
was measured on `router_mlx.route_topk`, and the runtime has been on `router_fused_metal.route_topk_fused`
since before that profile was taken. Chained, the fused router is **0.025 ms per call**: the layer's own
routing and the predictor's together are 1.6 ms per token rather than 13.6, and fusing the two passes into
one 768-row scores launch is worth **0.6 ms**, not the 3 ms the v19 prompt priced it at
(`benchmarks/micro_router_dual_gate.py`).

### 6.4 Context length is not what makes a long turn slow — **the reply-length hypothesis is refuted, 2026-09-21**

Section 7.2.4 put seven long interactive turns between 7.95 tok/s at a 545-token reply and 5.61 at 1,788
and read the ordering as an effect of length. **It is not.** Two instruments say so.

**`decode_anatomy.py` at three context lengths**, 2-bit bank, 24 GiB budget, 64 decode tokens each:

| prompt tokens | ms/token | `rest` | blocked | hit rate | MiB/token |
|---:|---:|---:|---:|---:|---:|
| 512 | 191 | **124.0** | 66.7 | 73.4 % | 944 |
| 1024 | 202 | **128.7** | 73.7 | 68.0 % | 1,078 |
| 2048 | 196 | **126.0** | 69.9 | 72.1 % | 987 |

`rest` — the term that would carry attention's growing shapes — spans **4.7 ms across a fourfold change in
context, and it is not monotone**. The whole token spans 11 ms, 5.8 %, against the 29 % the sessions
showed, and the ordering follows the hit rate rather than the length. **A first 1024-token run was
discarded before it was read**: it returned 389 ms per token, and its footprint shows 10.0 GiB of
compressor and a wall-clock 2.90 GB/s against the 5.19-5.58 of every other run, with the mean demand read
at 7.10 ms against 3.25. That is the machine, not the runtime, and the rule about checking a confound
before reporting an effect is what caught it.

**`benchmarks/decode_rate_by_block.py`, new, decodes one 1,792-token continuation in a single pass** and
reports the rate in blocks of 256, which is the thing no instrument here could do — the anatomy stops at
64 tokens, and a rate averaged over a whole turn is exactly what needed taking apart.

| | first block | last block | hit rate | MiB/token |
|---|---|---|---|---|
| 24 GiB, code prompt | 5.57 tok/s at 768 context | **6.05 at 2,304** (+8.6 %) | 73.8 % → 78.3 % | 939 → 783 |
| 36 GiB, code prompt (4 of 7 blocks) | 5.82 tok/s | **6.85** (+18 %) | 83.0 % → 88.2 % | 652 → 474 |
| 24 GiB, prose prompt | 5.77 tok/s | 5.27 (−8.7 %) | 77.4 % → 72.2 % | 823 → 1,118 |

**Within a single long generation the rate does not fall with length; it follows the hit rate**, and on the
two code runs it *rises* as the reply converges on a working set the cache already holds. Where it falls,
the hit rate falls with it and the bytes per token rise — the prose run walks into experts it does not
hold, and its 512-block outlier at 3.96 tok/s coincides with 9.5 GiB of compressor, so even that arm is
partly the machine.

**What the numbers actually say.** At one budget and one reply length, two different prompts decode at
6.05 and 5.27 tok/s — **15 % apart from the working set alone**. That is the same size as the spread
section 7.2.4 attributed to length, and it needs no length to produce it. The live ordering is
between-turn working-set variation: every fast turn in those four sessions was a story and every slow one
was Objective-C, the code turn always ran second into a cache warmed by prose, and the one 5.61 remains
the extreme of a distribution rather than the end of a trend.

**Section 7.2.4's table stands as a record of what those sessions did; its explanation is withdrawn.** The
lever it implied — `mx.compile(shapeless=True)` on attention to stop the context from costing anything —
is worth at most the 4.7 ms of `rest` above, and only if all of it is attention.

**Two machine facts came out of these runs and both constrain what can be measured.** A **44 GiB budget was
refused for the third day running**, killed on swap growth after 325 s with 15.0 GiB of compressor. And
**36 GiB, which ran the A/Bs of 2026-09-17, is now killed too** — twice, at 133 s and 252 s, with 21 GiB of
compressor, and right-sizing the instrument's `max_seq_len` from 8192 to 4096 did not save it. A 24 GiB
budget is what this machine sustains for a long decode today. One difference is worth testing before
concluding the machine changed: `chat.sh` runs 44 GiB happily at a **72 GiB** MLX wired limit, while
`guarded_run.sh` plans `budget + 16`, so the benchmark at 36 GiB runs under a 52 GiB limit and churns where
the interactive session does not. The guard's wired plan, not the budget, may be what refuses.

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

### 7.1.1 Decode and prefill, FP4 bank — the bank in use

All measured 2026-09-19 at a 36 GiB budget with a 512-token prompt, machine idle, four runs per arm
interleaved in both directions with `settle.sh` between. Medians.

| arm | cold prefill, 512 | decode | tok/s |
|---|---:|---:|---:|
| FP4, as it stood on 2026-09-18 | 29.1 s | 341.5 ms/token | 2.93 |
| **FP4, mirror striping at 0.10** | **27.1 s** | **325 ms/token** | **3.08** |
| `CACHALOT_PREDICT_WORKERS=4` | — | 363 ms/token | 2.75 |
| `CACHALOT_PREDICT_WORKERS=8` | — | 369 ms/token | 2.71 |
| `CACHALOT_PREDICT_AHEAD=2` | — | 359.5 ms/token | 2.78 |
| `CACHALOT_EVICT=slru` | — | 326.5 ms/token | 3.06 |

Constant across every arm: hit rate 70.4–71.1 %, 69.5–71.0 misses per token, and — except where the knob
changes what is prefetched — 1,858–1,868 MiB read per token. **Only two of those arms change the bytes**, and
both changed them upward.

**A 44 GiB budget with the hotlist has not been measured on FP4.** Section 12.1's live-session figures are
from the 2-bit bank. Hamed's own configuration runs at an 87–90 % hit rate where this benchmark runs at 71 %,
so every number above is a lower bound on his hit rate and an upper bound on what a storage lever buys him.

### 7.1.2 The shipped bank's anatomy at a large budget, 2026-09-21

The 2026-09-20 pair in section 6.2 ran at a 32 GiB budget and the shipped configuration is 44 with the
hotlist. **44 could not be run**: `guarded_run.sh` needs `budget + 29` and the machine had 70.6 GiB available
against 73, with nothing left to close but the terminal and the agent session itself. Rule 1 says lower the
budget, so this is 40 GiB — closer to the shipped configuration than anything measured before it, and still
not it.

`decode_anatomy.py --prompt-tokens 512 --decode-tokens 64`, 2-bit g128, 40 GiB budget, hotlist on, no mirror:

    decode 64 tokens: 12.19 s = 5.25 tok/s (190 ms/token)
      expert hit rate 80.8% | 46.0 misses/token | 704 MiB read/token
      expert wait 3.83 s = 59.8 ms/token (31.4% of decode), 40.0 calls/token
      rest        8.36 s = 130.6 ms/token (68.6% of decode)
      demand    19.6 reads/token | mean 3.60 ms, p50 3.47, p90 5.61
      predict   54.7 reads/token | mean 2.84 ms, p50 2.80, p90 4.85
      drive busy 6.91 s of 12.19 s decode (56.7%); 2.09 reads in flight while busy
      blocked total            3.83 s = 59.8 ms/token
      a demand read in flight  3.27 s = 51.2 ms/token (85.6%) -- coverage
      only a predicted read    0.46 s =  7.2 ms/token (12.0%) -- timing
      no read outstanding      0.09 s =  1.4 ms/token ( 2.4%) -- store overhead
      prediction: 3498 loads, 1690 used (48% precision), 28.2 wasted loads/token

Everything section 6.2 concluded at 32 GiB survives at 40: the drive is busy 56.7 % rather than 57.1,
coverage is 85.6 % of the blocked time rather than 86.5, and `rest` is 68.6 % of the token in both. **The
extra 8 GiB of budget is worth 2.9 points of hit rate and nothing structural.** Note this benchmark's 190 ms
against the live session's 131 ms (section 7.2.2): the benchmark prompt runs at an 80.8 % hit rate where
Hamed's session runs at 90.2 %, so the benchmark carries roughly 20 ms per token more of expert wait. Neither
number is wrong; they measure different working sets, and the compute floor underneath both is the same.

### 7.1.3 The shipped configuration, profiled at last — 44 GiB, 2026-09-21

Section 7.1.2 could not run 44 and settled for 40; three sessions before it could not run 36 either, and the
last prompt told the next session to lower the budget again. **44 ran on 2026-09-21, and so did 36**, with
78 GiB available at `settle.sh` and nothing closed that had not been closed before. `decode_anatomy.py
--prompt-tokens 512 --decode-tokens 64`, 2-bit g128, 44 GiB budget, 72 GiB wired, hotlist on, no mirror:

    decode 64 tokens: 10.85 s = 5.90 tok/s (170 ms/token)
      expert hit rate 83.5% | 39.7 misses/token | 627 MiB read/token
      expert wait 3.46 s =  54.0 ms/token (31.9% of decode), 40.0 calls/token
      rest        7.39 s = 115.5 ms/token (68.1% of decode)
      demand    18.5 reads/token | mean 3.65 ms, p50 3.49, p90 5.84
      predict   47.6 reads/token | mean 2.75 ms, p50 2.55, p90 4.84
      drive busy 6.00 s of 10.85 s decode (55.3%); 2.12 reads in flight while busy
      blocked total            3.46 s = 54.0 ms/token
      a demand read in flight  2.97 s = 46.4 ms/token (85.9%) -- coverage
      only a predicted read    0.41 s =  6.5 ms/token (12.0%) -- timing
      no read outstanding      0.08 s =  1.2 ms/token ( 2.2%) -- store overhead
      prediction: 3046 loads, 1358 used (45% precision), 26.4 wasted loads/token

**Nothing about the shape changes at the budget that ships.** Against 40 GiB: 2.7 points of hit rate, 77 MiB
per token fewer, 20 ms per token faster, and every structural ratio within a point — coverage is 85.9 % of
blocked time against 85.6, the drive is 55.3 % busy against 56.7, `rest` is 68.1 % of the token against 68.6.
The extra budget buys hit rate and nothing else, exactly as 32 to 40 did. **The lever the profile points at
is unchanged and is now measured at the shipped budget: 46.4 ms per token of blocking on misses no
prediction covered, with the drive idle 45 % of the time.**

**This instrument is reproducible to ±0.3 % here**, which is what made the four A/Bs below readable: four
shipped-arm runs came in at 10.85, 10.85, 10.89 and 10.91 s.

| arm, 44 GiB | wall for 64 tokens | against shipped |
|---|---:|---:|
| **shipped** (top-6, one layer ahead, default GIL switch interval) | **10.85-10.91 s** | — |
| `CACHALOT_PREDICT_AHEAD=2` | 10.91 s | +37 % bytes (861 MiB/token), 2.5 fewer demand reads/token, **no time** |
| `CACHALOT_PREDICT_TOPK=8` | 10.96 s | 1.0 % slower |
| `CACHALOT_PREDICT_TOPK=4` | 11.10 s | 2.3 % slower |
| `sys.setswitchinterval(0.020)` | 10.85 s | null |
| `sys.setswitchinterval(0.001)` | 10.89 s | null |

**Top-6 is still the width**, now measured on a 170 ms token reading 627 MiB rather than the 341 ms token
reading 1,858 that tuned it, and the curve either side of it is shallow and symmetric. **Two layers ahead is
still a null on this bank**, and this time the drive was not the reason — it is busy 55 % here, not the
80.5 % that explained the FP4 result — so the mechanism is precision: `PREDICT_AHEAD=2` reads 51.1 wasted
loads per token against 26.4 and buys 2.5 demand reads. **The GIL switch interval does nothing**, which
rules out reader threads' scheduling as the reason a streaming token's `rest` (115.5 ms) stands 33 ms above
the all-resident floor (section 9.18's 77-82 ms). That 33 ms is now the largest unattributed block in the
token and nobody has a mechanism for it.

### 7.1.4 What the GPU actually spends a token on — corrected 2026-09-21

An all-resident token is **54.8 ms minimum inside `mx.eval`** and 21.6 outside it (section 9.18). The inside
is 71 % of the token, it is the largest block in the project, and it had never been decomposed on the code
the runtime currently runs. `profile_decode_gpu.py` decomposes it — and **two of its rows were timing paths
the runtime had stopped taking**, which is the fourth instance of the rule in section 11 and the reason this
section exists. Both are fixed; both old numbers are withdrawn.

| piece | per launch | launches | **ms/token** |
|---|---:|---:|---:|
| compressed reuse attention, ratio 2 | 0.504 ms | 15 | **7.6** |
| compressed reuse attention, ratio 1 | 0.446 ms | 15 | **6.7** |
| routed experts, 2-bit affine g128 (6), **traced** | 0.169 ms | 40 | **6.7** |
| shared expert (fp8, 3 gemv) | 0.158 ms | 40 | **6.3** |
| `hc_post_1d` | 0.040 ms | 80 | 3.2 |
| `hc_mixes_1d` (sinkhorn) | 0.022 ms | 80 | 1.7 |
| `hc_pre_norm_1d` | 0.015 ms | 80 | 1.2 |
| router `route_topk_fused`, **shipped** | 0.022 ms | 40 | **0.9** |
| | | | **34.3 ms** |
| *routed experts, per-expert loop — retired path* | *0.396 ms* | *40* | *15.8* |
| *router `route_topk` — retired path* | *0.186 ms* | *40* | *7.4* |

**The two withdrawn numbers.** The profiler timed `affine_expert_forward` in a loop, on the ground that
"moe_layer_metal calls affine_expert_forward once per routed expert" — true until the traced MoE block
shipped on 2026-09-21 (section 9.16). It also timed `route_topk`, where the runtime takes
`route_topk_fused`. **Routed experts are 6.7 ms per token, not 19.8; the layer's own router is 0.9 ms, not
7.5.** Anything ranked off either figure is ranked off a path nothing executes.

**What that does to the expert-kernel levers: it closes them further.** Section 11's fused-affine screen put
`mx.gather_qmm` 20-25 % ahead of the per-expert loop and called it "at most 3 ms per token of the 13.9 ms
routed-expert term". The term is **6.7 ms**, so the same 20-25 % is **≤1.5 ms**, and it still needs six LRU
slots made contiguous. Two model-free screens confirm there is nothing else there:
`benchmarks/micro_expert_roofline.py` runs the three quantized matmuls per expert over **240 distinct
experts** — 2.3 GiB of weights, so no launch reads what the last one left in cache — at 0.217 ms per layer,
8.7 ms per token, within 30 % of what the shipped traced block pays; and `benchmarks/micro_topk_core.py`
adds `topk_core`'s work back one layer at a time:

| arm | ms/layer | ms/token |
|---|---:|---:|
| the three `quantized_matmul` alone | 0.163 | 6.5 |
| + the fp32 casts | 0.239 | 9.5 |
| + the SwiGLU clamp | 0.272 | 10.9 |
| `topk_core`, uncompiled | 0.290 | 11.6 |
| **`topk_core` under `mx.compile` — what ships** | **0.169** | **6.8** |

`mx.compile` takes the whole 5 ms of elementwise work back out: the shipped block costs what its three
matmuls cost. **The routed experts are done.**

**Where the GPU time actually is, and what is missing.** The measured pieces sum to **34.3 ms** against
54.8 ms inside eval. The 20 ms difference is the head, the Engram rows, the **ten layers whose attention this
profile does not cover** — layer 0, the sliding-window layers and the source and index-source layers — and
whatever the 44 evals of a token cost to drain. **That 20 ms is unattributed**, and it is the GPU twin of
the 33 ms by which a streaming token's `rest` exceeds the all-resident floor (section 7.1.3). Between them
they are about 50 ms of a 128 ms live token, and neither has a mechanism.

**The largest named GPU item is attention**: 14.3 ms across the 30 layers measured, before the ten that are
not. It is also the only piece with an untried instrument — `mx.compile(shapeless=True)` — and section 6.4's
cap of 4.7 ms applies to how attention *grows* with the context, not to what it costs at a fixed one.

### 7.1.5 The 20 ms of GPU nothing accounted for, closed — 2026-09-21

Section 7.1.4 left the largest measurement gap in the project: the pieces it timed summed to 34.3 ms
against the 54.8 ms a token spends inside `mx.eval`, and it named four suspects — the ten layers whose
attention the profile did not cover, the head, the Engram rows, and whatever 44 evals cost to drain. All
four are now measured. `benchmarks/profile_decode_gpu.py` covers every layer of the token, and it has an
arm that prices an eval. **The gap is closed, and it was two things in roughly equal parts: about 12 ms of
work that had never been timed, and about 12 ms of per-eval synchronisation that chained timing excludes by
construction.**

**The ten layers.** `_decode_token_impl` runs layer 0 and layer 1 as sliding-window layers, 2/8/14/20 as
compressed sources, 24/28/32/36 as index-only sources and the remaining thirty as compressed reuse. Only
the reuse class had ever been timed. Measured on the same run, at a 513-position context, 40 GiB budget:

| piece | per launch | launches | **ms/token** |
|---|---:|---:|---:|
| compressed reuse attention, ratio 2 | 0.441 ms | 15 | 6.6 |
| compressed reuse attention, ratio 1 | 0.416 ms | 15 | 6.2 |
| **compressed attention, source, ratio 2** | **1.483 ms** | 3 | **4.4** |
| **compressed attention, index-only source** | **0.839 ms** | 4 | **3.4** |
| **compressed attention, source, ratio 1 (layer 20)** | **1.085 ms** | 1 | **1.1** |
| **layer-0 and layer-1 sliding-window attention** | 0.425, 0.430 ms | 1 each | **0.8** |
| **final norm + head logits** | **1.623 ms** | 1 | **1.6** |
| **Engram forward, layers 1 and 14** | 0.385 ms | 1 each | **0.8** |
| routed experts, traced | 0.172 ms | 40 | 6.9 |
| shared expert | 0.121 ms | 40 | 4.8 |
| `hc_post_1d`, `hc_mixes_1d`, `hc_pre_norm_1d` | — | 80 each | 4.2 |
| router `route_topk_fused` | 0.020 ms | 40 | 0.8 |
| | | | **41.6 ms** |

**Attention is not the largest named GPU piece. It is the largest GPU piece, by a factor of three**:
12.8 ms on the thirty reuse layers, 8.9 on the eight source layers and 0.8 on the two sliding-window ones,
**22.5 ms per token**, against 6.9 for the routed experts and 4.8 for the shared one. Section 7.1.4's
14.3 ms was the reuse class alone.

**A source layer costs about a millisecond more than a reuse layer, and that is the compressor and the
indexer.** Same attention, same weights, same context: 1.483 against 0.441 at ratio 2, 1.085 against 0.416
at layer 20, 0.839 against 0.416 for an index-only source. The difference — **5.5 ms per token across the
eight layers** — is the compressed-KV write and the index selection, and nothing in this project has ever
looked at it. It is now the third-largest named GPU block.

**What 44 evals cost to drain: 0.31 ms each, at least.** The instrument chains forty launches into one
graph and evaluates once, which is how every row above is measured and is *not* how a token pays. Timed
both ways on the same two pieces:

| piece | chained | one `mx.eval` each | delta | x 44 evals |
|---|---:|---:|---:|---:|
| compressed reuse attention, ratio 1 | 0.454 ms | 0.990 ms | **0.536 ms** | 23.6 ms |
| router `route_topk_fused` | 0.021 ms | 0.327 ms | **0.306 ms** | 13.5 ms |

The router row is the floor: its kernel is 0.02 ms, so 0.31 ms of the 0.33 is round trip — launch,
completion, and the CPU learning about it. **A token pays that 44 times** (one `mx.eval` per layer at
`moe_layer_metal.py:180`, plus the tail), and no chained measurement can see it. 41.6 ms of pieces plus
about 12 ms of drain is 54 ms against the 54.8 ms a token spends inside eval. **The sum was never allowed
to close, which is what candidate 4 of the v25 prompt asked.**

**Repeatability.** Two runs of the extended instrument, back to back at 40 GiB. The first stopped before
the two Engram rows, so the comparable sum is the one without them: **43.6 ms and 40.8 ms**, every row
within 7 % and the ordering identical, on whole-token minima of 76.1-83.2 ms against the 76.4 ms floor.
The 41.6 ms above is the second run with Engram included. Results in
`benchmarks/results/guarded/gpu40full2_*` and `gpu40full3_*`.

**Two things the instrument needed, and both are lessons about the timing of an isolated piece.** An
index-only source consumes the candidate mask layer 20 published while the *previous* token was decoded, so
it must be timed at that token's position; timed at the current one it raises `candidate mask length 513 is
smaller than compress_len 514`. And the Engram row read is I/O, so the row above times the forward on rows
already in memory — the read itself belongs to the store's accounting, not the GPU's.


### 7.1.6 What one `mx.eval` costs, and the first lever with a mechanism since 2026-09-19

`benchmarks/micro_eval_floor.py` is model-free and instant. It launches one router-shaped top-k over 384
logits and asks what it costs to find out the answer on the CPU:

| arm | min | x 40 layers |
|---|---:|---:|
| chained, no eval | 0.0017 ms | 0.07 ms |
| `mx.eval` | 0.2172 ms | 8.7 ms |
| `mx.eval` then `.tolist()` on the same array | 0.2013 ms | 8.1 ms |
| `.tolist()` alone | 0.2091 ms | 8.4 ms |
| `np.array` | 0.2145 ms | 8.6 ms |
| `.item()` on one scalar | 0.2228 ms | 8.9 ms |

**Every way of getting a value out of MLX costs the same 0.20-0.22 ms, so the readout is not the cost --
the synchronisation is.** The queue-depth arm says the same thing from the other side: one eval behind 1
launch is 0.218 ms and behind 40 launches 0.552 ms, so the round trip is mostly fixed and the work drains
into it. And three arrays in one `mx.eval` cost 0.28 ms against 1.03 ms for three separate evals, which is
why fusing this layer's routing with the predictor's look at L+1 into one call was right.

**The forty that remain are structural**: the store is addressed on the CPU, so the routing of layer L has
to be read before layer L's experts can be fetched. That is about 8 ms of a 76 ms all-resident token that
no kernel work can remove.

**What can be removed is the idleness.** While the CPU sits in that round trip the GPU has nothing queued,
and one piece of the layer does not depend on the routing at all: the shared expert. Priced on a stand-in
of the same size:

| arm | min |
|---|---:|
| the independent launch alone | 0.219 ms |
| sync first, then the work | 0.474 ms |
| **work submitted first, then the sync** | **0.198 ms** |

Two round trips collapse into one: **0.276 ms per layer, 11.0 ms per token at forty layers**, as an upper
bound on a launch the size of the shared expert.

### 7.1.7 The 33 ms a streaming token spent above the floor — it is the Engram row reads, 2026-09-21

Three prompts in a row named this the largest unexplained block in the project and none of them had a
mechanism for it: a streaming token spends about 115 ms outside the store's blocking calls against an
all-resident 76-82, and neither the GIL switch interval nor reader-thread scheduling explains the
difference. It is **the two Engram row reads, issued as 96 serial `pread`s from the decode thread while the
expert stream saturates the drive.**

**The instrument.** `benchmarks/profile_decode_sync.py` gained a streaming arm and two more columns. The
arm decodes a real continuation — every token a new position, nothing restored between tokens, the demand
path missing the way a live session's does — and reports the same per-call-site table as the all-resident
one, so the two can be read side by side from one process. The columns split every interval between two
evals three ways instead of two: time inside `mx.eval`, time the main thread spends blocked inside
`ResidentExpertStore.get`/`get_many`, and what is left, which is the only part that is really CPU. A fourth
counter times `_apply_engram`, because the Engram rows are read with synchronous `pread`s on the decode
thread and therefore land in the gap between two evals looking exactly like graph construction.

36 GiB budget, 512-token context, 48 tokens of continuation against 12 repeats of one token, medians,
`benchmarks/results/guarded/eg_base_*`:

| | all-resident | streaming | delta |
|---|---:|---:|---:|
| whole token | 77.3 ms | 188.3 ms | +111.0 |
| inside `mx.eval` | 56.9 ms | 77.3 ms | +20.4 |
| blocked in the expert store | 0.9 ms | 64.0 ms | +63.1 |
| **`_apply_engram` on the decode thread** | **1.7 ms** | **28.4 ms** | **+26.7** |
| CPU, none of those | 17.7 ms | 21.7 ms | +4.0 |
| expert hit rate | 100 % | 79.4 % | — |

**The CPU column barely moves. The Engram column moves by a factor of seventeen.** Of the 111 ms a
streaming token costs over an all-resident one, 63 are the priced miss cost, 27 are Engram, 20 are inside
eval and 4 are CPU. The question sections 7.1.3, 9.20 and 9.23 kept asking — whether the excess is in the
store's admission path or in the GPU waiting on memory the SSD DMA is also using — had a third answer
nobody had a column for.

**The mechanism, and it is arithmetic.** `EngramRowReader.read_rows` fetches a row with two `pread`s, one
for the 256 B of weights and one for the 8 B of scales, and it only hands the batch to its sixteen-worker
pool when the batch is 64 rows or more. Decode asks for **24 rows, twice a token** — layers 1 and 14 — so
every decode batch took the serial branch: **96 `pread`s issued back to back from the decode thread.**
That threshold was chosen for prefill, which asks for about 12k rows per layer and always went through the
pool. On an idle drive the serial branch costs 1.7 ms a token and nobody would find it. While the expert
store is reading 750 MiB a token through the same device each of those `pread`s waits behind the queue, and
0.28 ms x 96 is 27.

**One caveat about the new column.** `_apply_engram` is timed as a whole, and its total is charged to the
interval in which it *returned* — the eval that closes that interval, which is the same layer's MoE
`mx.eval`, not the `mx.eval` at `engram_rows.py:59` inside it. Read the per-token total, not the per-site
attribution, for the Engram row; the site table can show a negative CPU cell at
`moe_layer_metal.py:214` because of it.

What the lever built on this measures is section 9.24, and what is left of the block afterwards is about
15 ms — a 94 ms `rest` against an all-resident floor of 76-79.


### 7.1.8 What a source layer's extra millisecond is, and why `INDEX_TOPK` is not a lever — 2026-09-21

Section 7.1.5 measured a source layer's attention at 1.483 ms against a reuse layer's 0.441 and named the
difference — 5.5 ms per token across the eight source layers — the third-largest named GPU block and the
one nothing had looked at. `profile_decode_gpu.py` now times the two pieces inside it directly:
`compressor_forward` and `indexer_decode_base`, the second at four widths.

40-layer run at a 36 GiB budget, 513 positions, `benchmarks/results/guarded/gpu36ci_*`:

| piece | per launch |
|---|---:|
| compressed attention, source, ratio 2 (whole) | 1.657 ms |
| compressed reuse attention, ratio 2 (whole) | 0.602 ms |
| **the extra** | **1.055 ms** |
| of it: `indexer_decode_base`, `index_topk` 512 | 0.501 ms |
| of it: `compressor_forward`, closing token | 0.225 ms |
| of it: `compressor_forward`, open group | 0.101 ms |
| **left over: the compressed-KV write** (RoPE, FP4 round trip, cache store) | **~0.39 ms** |

A ratio-2 compressor returns a latent only on the closing token of each group, so a token pays one closing
and one open call per pair of positions: **0.163 ms on average, 0.5 ms per token** across layers 2, 8 and
14. The indexer is **0.501 ms per source layer, 2.0 ms per token** across the four base-indexer layers, and
the four index-only consumers pay 0.462 ms each over a reuse layer, 1.8 ms more. **The indexer family is
about two thirds of the source layers' extra; the compressor is a tenth of it.**

**`INDEX_TOPK` is not a lever.** The shipped width is 512 and the indexer costs the same at any of them:

| `index_topk` | 128 | 256 | **512** | 1024 |
|---|---:|---:|---:|---:|
| `indexer_decode_base` | 0.485 ms | 0.474 ms | **0.501 ms** | 0.540 ms |

An eightfold change in the width is worth 0.066 ms per layer, 0.26 ms per token across four layers, and it
is not monotone below 512. The indexer's cost is its projections, its own FP4 index-K cache and the
scoring, not the size of the selection — so narrowing it trades quality for nothing. Do not spend a session
on it.

**Read this run's absolute numbers only against itself.** Every row of it is 15-25 % above the
`gpu40full3_*` run of the same instrument — reuse attention at ratio 2 is 0.602 against 0.441, the routed
experts 0.189 against 0.172, the whole token 78.9 ms against 76.1 — so the machine was in a different state.
The shares within the run are what the section above quotes.


### 7.1.9 A streaming token at a 100 % hit rate is the all-resident floor — the "+20 ms inside `mx.eval`" is a per-miss cost, 2026-09-21

Sections 7.1.7, 9.25 and the v27 and v28 prompts all carried one line with no mechanism: against an
all-resident token, a streaming one spends about 20 ms more *inside* `mx.eval` doing the same arithmetic,
and the candidate nobody had tested was the GPU waiting on memory bandwidth the SSD DMA is also using.

**It is not a separate block at all. It is part of the miss cost, and it disappears exactly when the misses
do.**

**The instrument.** `benchmarks/profile_decode_sync.py` gained `--stream-passes N`. The snapshot is restored
before each pass and decoding is greedy, so every pass decodes the same token ids through the same graph at
the same positions over the same scattered working set. The only thing that differs is the store's
residency: pass 1 misses, pass 2 finds everything pass 1 read still resident. That isolates *reading from
the device* from *touching 240 experts spread across a 40 GiB buffer*, which no earlier arm separated —
the all-resident arm decodes one token twelve times and therefore touches the same 240 experts every time.

40 GiB budget, 512-token context, 24 tokens of continuation, medians,
`benchmarks/results/guarded/sync2pass_20260921-154341`:

| | all-resident | streaming, pass 1 | **streaming, pass 2** |
|---|---:|---:|---:|
| whole token | 79.6 ms | 152.3 ms | **79.6 ms** |
| inside `mx.eval` | 58.0 ms | 70.8 ms | **58.5 ms** |
| blocked in the expert store | 1.0 ms | 56.3 ms | **0.9 ms** |
| `_apply_engram` on the decode thread | 1.0 ms | 1.9 ms | 1.1 ms |
| CPU, none of those | 19.3 ms | 23.0 ms | **19.3 ms** |
| expert hit rate | 100 % | 83.4 % | **100 %** |
| misses per token | 0 | 39.8 | 0 |
| MiB read per token | 342 | 645 | 248 |

**Pass 2 sits on the all-resident floor to a tenth of a millisecond, on every column.** Same positions, same
graph, same 5,760 expert lookups spread over a 40 GiB slot pool, and the token costs 79.6 ms against the
all-resident arm's 79.6. So the excess is not the access pattern, not the graph, not the queue depth of a
token that has more work behind it, and not the store's bookkeeping. It appears only when experts are
actually read, and it scales with how many: 39.8 demand misses buy 12.3 ms inside eval, **0.31 ms per
miss**, on top of the 1.41 ms per miss the store already charges in its blocking call.

**What this does to the ranking.** There is no 20 ms block with no mechanism. There is one miss, costing
about 1.7 ms all in, and the fraction of it that lands inside `mx.eval` had simply never been attributed to
it because no arm had ever decoded a real continuation with nothing left to read. Section 7.1.7's +20.4 ms
was measured at a 36 GiB budget and a 79.4 % hit rate; the same figure at 40 GiB and 83.4 % is +12.3, and
the two are the same number per miss.

**And pass 2 is also the cleanest statement of what a larger budget is worth.** A streaming token that
never misses is an all-resident token. Everything between the 79.6 ms floor and a live token's 106 ms is
the miss, which is why section 9.4's budget lever is the only one that has moved the rate twice.

### 7.1.10 The FP8 GEMV family is the largest GPU path in the token, and attention is weight streaming — 2026-09-21

Section 7.1.5 named attention the largest GPU block at 22.5 ms and left "the shapes themselves have never
been attacked" open for three prompts, on the theory that `attention_kv` growing with the context and the
`start_pos % 128` window slice are what a reuse layer spends its 0.441 ms on. **They are not. A reuse
layer's attention is 86 % weight streaming**, and the KV it attends over is a rounding error beside the
weights it reads to build Q and project the output.

**The inventory, read off the checkpoint headers** (`layers.3.*`, and every layer is the same):

| tensor | shape | dtype on disk | MB on disk | MB resident |
|---|---|---|---:|---:|
| `attn.wq_a.weight` | (1280, 5120) | F8_E4M3 | 6.55 | 6.55 |
| `attn.wq_b.weight` | (32768, 1280) | F8_E4M3 | 41.94 | 41.94 |
| `attn.wkv.weight` | (512, 5120) | F8_E4M3 | 2.62 | 2.62 |
| `attn.wo_a.weight` | (8192, 4096) | F8_E4M3 | 33.55 | **67.11**, dequantized to BF16 at startup |
| `attn.wo_b.weight` | (5120, 8192) | F8_E4M3 | 41.94 | 41.94 |
| | | | **126.6** | **160.2** |

At 0.441 ms a reuse layer reads 160.2 MB, which is 363 GB/s — and the pieces add up:

| piece | per layer | measured by |
|---|---:|---|
| `wq_a`, `wq_b`, `wkv`, `wo_b` through `fp8_gemv_decoded` | 0.270 ms | `micro_fp8_gemv_kernel.py` |
| `wo_a` as a BF16 grouped `mx.matmul` | 0.107 ms | `micro_wo_a.py` |
| **the weights** | **0.377 ms** | |
| sparse attention over the KV, the norms, RoPE and inverse RoPE | 0.064 ms | the remainder against §7.1.5's 0.441 |

**So the shape idea is closed on the arithmetic, without writing anything.** A fixed-capacity `attention_kv`
with a mask instead of a growing slice changes the cost of the 14 % and pays for it with more positions of
arithmetic; `mx.compile`, which is the only thing a stable shape would unlock, is closed three ways already
(§9.21).

**What is underneath all of it is one kernel.** `fp8_gemv_decoded` — `fp8_fused_metal._gemv_v2_kernel`,
reached through `fp8_linear_quantized` whenever `CACHALOT_FUSED_FP8` is on, which is the default — serves
four of the five attention projections on all forty layers and all three of the shared expert's GEMVs on
all forty. That is **5.14 GB of weights per decoded token, more than twice the routed experts' 2.3 GiB**,
and it is the largest single GPU path in the runtime:

| shape | lanes/row | per launch | GB/s | launches/token | ms/token |
|---|---:|---:|---:|---:|---:|
| `wq_a` [1280, 5120] | 32 | 0.038 ms | 171 | 40 | 1.54 |
| `wq_b` [32768, 1280] | 8 | 0.092 ms | 456 | 40 | 3.69 |
| `wkv` [512, 5120] | 32 | 0.030 ms | 87 | 40 | 1.20 |
| `wo_b` [5120, 8192] | 32 | 0.110 ms | 383 | 40 | 4.39 |
| shared `w1`, `w3` [2304, 5120] | 32 | 0.049 ms | 243 | 80 | 3.89 |
| shared `w2` [5120, 2304] | 8 | 0.048 ms | 248 | 40 | 1.91 |
| | | | | | **16.61 ms** |

**And it is at 79 % of what the machine gives for the same bytes.** `mx.sum` over each weight buffer — no
arithmetic, no dequantization, just a full read — totals 13.13 ms against the kernel's 16.61. The gap is
3.48 ms per token and there is no obvious way to take it: the kernel already reads the weight as `uint4`,
the activation as pre-decoded `float4`, and decodes E4M3 out of a packed word through a constant table.

**Two shapes are far below the rest and it is not the kernel's fault.** `wkv` at 87 GB/s and `wq_a` at
171 are the two smallest output dimensions, 512 and 1280 rows; at 32 lanes per row they launch 512 and
1280 simdgroups, 64 and 160 threadgroups of 256 threads, on a machine with eighty cores. They are
occupancy-bound, and `mx.sum` over the same buffers only reaches 98 and 212 GB/s, so most of the shortfall
is the buffer's size and not the kernel. Together they are 2.74 ms per token and their own ceiling is 2.31.

### 7.1.11 Job 1 answered: the miss is at the drive's rated wall, and wired memory is a null — 2026-09-22

v29's Job 1 asked whether the 1.7 ms per-miss cost is the drive slowed by decode's own memory pressure —
`expert_read_scaling.py --wire-gib` has had the option since 2026-09-17 and nobody had run it near the
shipped budget, so every "6.6-6.8 GB/s cold" number in section 3.1 was measured on an idle machine while
decode holds 43-63 GiB wired and the docstring's own theory says the kernel must reclaim a page for every
page it reads.

**Two bugs in the instrument, fixed before it could answer anything.** `--wire-gib`'s ballast keeps itself
wired with a heartbeat thread, `mx.eval(beat + 1)` every 0.5 s, because macOS un-wires an idle Metal working
set in about six seconds. On MLX 0.32.2 that heartbeat raised `RuntimeError: There is no Stream(gpu, 0) in
current thread` on its very first tick and died silently — the exception prints to stderr but the thread
exits and `main()` carries on, so every arm after the first ran with the ballast already collapsing, and
three runs in a row showed the same signature: wired memory held for the first one or two arms of a
`--loaders 1,2,4,8,16` sweep, then fell to 6-9 GiB for the rest. The cause is not the documented six-second
idle window — it is that `beat = mx.zeros((1,))` is created lazily and never evaluated on the main thread, so
the *first* `mx.eval` MLX ever performs on that array happens on a background thread, and 0.32.2 cannot
resolve a default GPU stream for an array's first materialization from a thread other than the one that
built the graph. Fixed with one line, `mx.eval(beat)` on the main thread right after creating it
(`benchmarks/expert_read_scaling.py`); every `beat + 1` after that is a derived eval on an already-resident
array and never touches the broken path, on any thread. Confirmed by reproducing the crash standalone,
finding the exact statement that fixes it, then rerunning the sweep and getting a clean run with the
ballast holding steady wired GiB across all five arms — no traceback, no collapse.

**The measurement, once the ballast actually stays wired.** 28 GiB expert budget, 36 GiB ballast, 42.9-43.4
GiB system wired throughout (45 % of the machine's 96 GiB, the largest arm this session's free memory — 60-64
GiB — would admit; the shipped 52 GiB budget wires nearer 80 GiB and was not reachable today).
`benchmarks/results/guarded/rw_l1only_20260922-010102.out`, `rw_l8only_20260922-*.out`, and an unwired arm at
a disjoint offset for the baseline:

| loaders | wired 42.9 GiB | unwired (cold) | GB/s difference |
|---:|---:|---:|---:|
| 1 | 4.99 GB/s, 1.99 ms/expert | 5.04 GB/s, 1.97 ms/expert | 1 % |
| 8 (`io_workers`, the shipped concurrency) | 6.81 GB/s, 1.46 ms/expert | 6.76 GB/s, 1.47 ms/expert | 1 % |

**Wired memory pressure is a null, at both concurrencies, at 45 % of the machine wired.** The two arms are
inside each other's noise band, the same band the `io_workers` sweep already showed run-to-run (§9.26). The
reclaim-per-page theory the ballast option was built to test does not hold at this fraction of the machine;
it is not ruled out at the shipped budget's 80-83 % of RAM, which nobody could test today, but there is no
mechanism visible at 45 % that would suddenly appear at 83 with the same drive and the same reader.

**And it closes the miss's own arithmetic.** At `io_workers=8` the reader delivers experts at 1.46-1.47
ms each, wired or not, matching the top of section 3.1's "6.6-6.8 GB/s cold" rating. Section 9.30's miss
costs 1.41 ms blocked in the store — inside the isolated reader's own noise band, not above it — plus 0.31
ms inside `mx.eval` and ~0.05 ms of CPU, for ~1.7 ms total. **The blocked component is already at the
drive's rated ceiling; there is no store-side admission overhead hiding inside it and no memory-pressure
tax to remove.** The 0.31 ms inside `mx.eval` was already explained as part of the miss and not a separate
block (§7.1.9). **Job 1 is answered: the miss is at the wall, on both counts asked. The budget is the only
lever left on it**, exactly as v29 suspected, now measured rather than assumed.

### 7.2 Interactive chat, 44 GiB budget, 72 GiB wired

    ready in 16.4 s
    turn 1   prefill   5 tokens (2 reused)   0.8 s    51 tokens   6.03 tok/s
    turn 2   prefill  69 tokens (65 reused)  0.6 s    68 tokens   6.73 tok/s
    turn 3   prefill 157 tokens (152 reused) 0.7 s   117 tokens   7.52 tok/s
    session  87.3 % hit rate, 4,456 resident experts, 108.4 GiB read
             6,495 predicted loads, 2,991 used, 16 prefix-cache entries, 14 hits

A chat session beats the benchmark's hit rate — 87.3 % against 80.9 % — because it reuses experts across turns,
and 44 GiB now holds 4,456 experts where the 3-bit bank held 2,984.

### 7.2.1 Interactive chat re-measured on the fixed runtime, 2026-09-20

`benchmarks/chat_turns.py --max-new-tokens 160`, which replays six chat turns exactly as `cachalot chat`
does — same chat template, same prefix cache. 2-bit g128 bank, hotlist on, mirror striping off (section
9.11.1 explains why off):

| turn | reply | **44 GiB** | 42 GiB | resident at 44 |
|---|---|---:|---:|---:|
| 1, cold | 9 tok | 4.51 | 4.34 | 2,157 |
| 2 | 9 tok | **7.40** | 7.19 | 3,151 |
| 3, the 100-word story | 127 tok | **7.03** | 6.87 | 4,746 |
| 4, haiku | 20 tok | 6.97 | 6.41 | 4,746 |
| 5, two-sentence explanation | 44 tok | 6.21 | 5.86 | 4,746 |
| 6, Python one-liner | 11 tok | 4.72 | 4.38 | 4,746 |

**Section 7.2's 7.52 tok/s is reproduced.** That was a 117-token turn at a 44 GiB budget on this bank,
recorded on 2026-09-17 — before mirror striping existed, so it was unstriped too. This is a 127-token turn
at the same budget on the same bank at 7.03, with a peak of 7.40 on a short turn, so the configuration is
back where it was. What is different is that the output is now correct.

**The point of running turn 3 was not the throughput.** "Write a 100 word story of a small fish living in a
greek" is the prompt that, on this same bank on 2026-09-17, produced "whitewas crumbling houses" and renamed
its own character from Nikos to "Niks" — the observation that condemned the 2-bit bank and sent three
sessions after quantization. Through the fixed runtime it returns:

> A tiny fish lived in a Greek taverna's cracked marble sink. Every morning, the owner, Yiannis, poured olive
> oil down the...

and turn 4's haiku quotes "olive oil" back out of it. Copying out of recent context is the operation
`hc_post` destroyed, so that haiku is the cheapest possible end-to-end check that the fix holds in free
generation rather than only under teacher forcing.

**Budget note.** 44 GiB was attempted four times and refused every time at 71.7 to 72.1 GiB available against
the 73 that `guarded_run.sh` requires, with Firefox, Slack, Mail and Stream Deck already closed; the last
1.6 GiB was the terminal emulator hosting the session. 42 GiB is what passed. Section 9.4's curve prices that
step at roughly one point of hit rate.

### 7.2.2 The real interactive session, 2026-09-21 — 7.62 tok/s and a 90.2 % hit rate

Hamed's own six-turn session on the command in section 4: 2-bit g128, 44 GiB budget, 72 GiB wired, hotlist,
no mirror, no sampling penalties, temperature 0.6.

    ready in 16.6s; hotlist 863 experts preloaded (8.0 GiB in 2.0 s of reading)
    turn 1  "Hi"                            10 tokens    5.10 tok/s
    turn 2  "Answer only in english..."      9 tokens    6.27 tok/s
    turn 3  300-word story                 314 tokens   **7.62 tok/s**
    turn 4  TypeScript, Excel to PDF      1024 tokens    6.73 tok/s   stop=length
    turn 5  German poem                    234 tokens   **7.55 tok/s**
    session  90.2 % hit rate, 4,480 resident experts, 716.8 GB read,
             48,706 predicted loads / 15,898 used, 48 prefix-cache hits / 1 miss, 23,264 tokens reused

**This is the fastest this project has been, and it beats the number the configuration was abandoned at.**
Section 7.2's 7.52 tok/s was 117 tokens; this is 314 tokens at 7.62 and 234 at 7.55, so it holds over replies
three times longer. The session hit rate is **90.2 % against that session's 87.3 %**, which is the hotlist
plus the predicted-load lifetime fix of section 9.13.

**Quality, read turn by turn.** The 300-word story is coherent, correctly spelled and correctly structured.
The TypeScript is real working code — `exceljs` plus `pdfkit`, correct `path.basename(excelPath, ext)` to
build the output name, page-break handling, column-width scaling — and it stopped at `stop=length` because
`--max-new-tokens 1024` cut it mid-function, not because it broke. The German poem scans and rhymes; its one
error, "wer wir einst war" for "waren", is ordinary model quality and not the artefact class this project
has been chasing. **No malformed identifiers, no dropped characters inside words, no repetition loop with the
frequency penalty off.**

**Two observations that are not defects.** Turn 1 answered "Hi" in Chinese, which is this model's default and
is why `chat_turns.py`'s second turn exists at all. Turn 4 stopping at the token cap is the cap.

**One observation that might be.** Turn 2's reply begins **`wHi! How can I help you today?`** — a stray `w`
prepended to an otherwise perfect reply. It is one character at the very first position of a turn, and it is
the same *class* as the artefacts section 7.4 was built around, so it should not be waved away. Three things
argue it is sampling rather than the defect class: this session ran at temperature 0.6 where `chat_turns.py`
decodes greedily and produced `Hi! How can I help you today?` cleanly on the identical prompt; the gate that
follows it is clean at 0/94 malformed includes over 40 cases; and a single first-position token is where a
sampler is least constrained. **It is not established either way, and it is cheap to settle:**
`benchmarks/token_rank_probe.py` on a transcript containing that exchange reports where `Hi` ranked at that
position, and four greedy repeats of the same two turns say whether it reproduces. Neither has been run.

**The prediction waste is unchanged and is now the largest addressable term.** 48,706 predicted loads for
15,898 used is **32.6 % precision**, so 32,808 wasted loads at 9.49 MiB is about 311 GB of the session's
716.8 GB — **43 % of every byte read**, which matches section 9.10's 42.7 % measured on a different bank and
a different runtime. Whatever else changed tonight, that did not.

**Typing-time prefill deserves a second look.** 44 runs consumed 20.18 s to prefill 81 tokens, which is
249 ms per token against the 90-116 ms that section 7.2.1's turn prefills cost. It is hidden behind human
typing so it costs nothing observable, but it is doing far more work per token than a batched prefill and
nobody has asked why.

### 7.2.3 Two full sessions on the traced runtime, 2026-09-21 — and what a live session cannot resolve

Both sessions ran `./chat.sh`, 2-bit g128 at a 44 GiB budget, 72 GiB wired, hotlist, temperature 0.6. The
only difference is `CACHALOT_COMPILE_MOE`: session A is the shipped traced MoE block, session B is the
per-expert loop. Both opened with the correct banner — `9.49 MiB/expert`, 863 hotlist experts in about
2 seconds — which is the check section 12 now asks for.

| turn | session A, traced | session B, loop |
|---|---|---|
| "Hi" | 10 tokens, 5.14 tok/s | 9 tokens, 4.67 tok/s |
| "Answer only in english. Now, Hi!" | 9 tokens, 6.26 tok/s | — (not asked) |
| 500-word story | 643 tokens, **7.76 tok/s** (128.9 ms/token) | 889 tokens, **7.82 tok/s** (127.9 ms/token) |
| Objective-C, JSON to CSV | 1024 tokens, **6.95 tok/s** (143.9 ms), `stop=length` | 1024 tokens, **6.76 tok/s** (147.9 ms), `stop=length` |
| session hit rate | 90.3 %, 4,456 resident, 762 GB read | 91.1 %, 4,393 resident, 791 GB read |
| prediction precision | 17,353 / 53,110 = **32.7 %** | 18,212 / 56,039 = **32.5 %** |
| MLX peak | 64.0 GiB against the 72 GiB limit | 63.5 GiB |

**The live A/B is a null and it was always going to be.** The floor measurement says the traced block is
worth 5-8 ms of a token; a live token is 128-148 ms, so the expected effect is 4-5 %, and the two arms
differ by +2.8 % on the one matched prompt (the Objective-C turn, both capped at 1024 tokens) and −0.8 % on
the story turns, which were not matched in length. Session B also ran at 0.9 points more hit rate. **Neither
sign is evidence.** The change is quoted from `profile_decode_sync.py`, whose spread is about 1 ms, and this
section is here to record that the runtime behaves normally with it on, not to re-prove it. Section 9.16.

**No regression against the record**: 7.76-7.82 tok/s on long replies against the 7.55-7.62 of section 7.2.2,
at the same hit rate and the same prediction precision.

**Quality, read turn by turn, and it is the reference class on both arms.**

- The 500-word stories are coherent end to end, correctly spelled and correctly punctuated, with consistent
  names and tense across 643 and 889 tokens. Session B's coin is named George in its second paragraph and is
  still George, still a quarter, 700 words later — the failure mode that renamed Nikos to "Niks" does not
  appear. Session A's story overshoots into one small factual wobble: a 2004 cent is given "a small shield on
  one side", which is the post-2010 reverse. That is ordinary model error, not the artefact class.
- Both Objective-C programs are real, compiling-shaped code: `#import <Foundation/Foundation.h>` intact,
  `NSJSONSerialization`, `isKindOfClass:` guards, `NSError **` propagated, `writeToFile:atomically:encoding:error:`
  with the right selector. **No malformed `#include`/`#import` line, no broken identifier, no dropped
  character inside a word**, which are the three things sections 7.4.1 to 7.4.8 were built around. Session B's
  is the better program — free `CSVEscape`/`ValueToCSVString` functions with quote doubling, and a `main`
  that checks `argc` — but that is a sampling difference, not an arm difference.
- **Both coding turns stopped at `stop=length`**, cut mid-method by `--max-new-tokens 1024`. That is the cap,
  as in section 7.2.2's turn 4. A coding turn on this model wants 1500-2000.
- **The stray `w` of section 7.2.2 did not reproduce.** Session A's turn 2 is the same prompt shape at the
  same temperature and returned `Hi! How can I help you today?` clean, and session B's first turn answered in
  English rather than Chinese. One clean repeat is not a settlement — the thread in section 7.2.2 still wants
  its four greedy repeats — but nothing in two sessions and 3,600 tokens looked like it.

**Two numbers that did not move, and both are open levers.** Prediction precision is 32.5-32.7 %, so about
two thirds of every predicted load is still wasted — unchanged across three sessions, two banks and a
runtime fix (section 9.10). And typing-time prefill is still several times a batched prefill: 15.29 s for 53
tokens in session A (288 ms per token) and 6.96 s for 38 in session B (183 ms), against the 90-116 ms of a
turn prefill.

### 7.2.4 Two sessions on the memoised-constant runtime, 2026-09-21 — and the reply-length effect nobody had plotted

Both ran `./chat.sh --max-new-tokens 2000`, 2-bit g128 at a 44 GiB budget, 72 GiB wired, hotlist,
temperature 0.6, Cachalot 0.6.0, same four prompts. The only difference is `CACHALOT_KERNEL_CONSTS`:
session A is the shipped memoised parameters, session B rebuilds them per call. Both opened with the
correct banner — `9.49 MiB/expert`, 863 hotlist experts in 1.9 s, expert budget 44.0 GiB.

| turn | session A, memoised | session B, rebuilt |
|---|---|---|
| "Hi" | 10 tokens, 5.14 tok/s | 9 tokens, 4.78 tok/s |
| "Answer only in english. Now, Hi!" | 9 tokens, 6.26 tok/s | 9 tokens, 7.79 tok/s |
| 500-word story | 602 tokens, **7.86 tok/s** (127.2 ms/token) | 545 tokens, **7.95 tok/s** (125.8 ms/token) |
| Objective-C, JSON to CSV | 1,483 tokens, **6.94 tok/s** (144.1 ms), `stop=stop` | 1,788 tokens, **5.61 tok/s** (178.3 ms), `stop=stop` |
| session hit rate | **90.03 %**, 4,484 resident, 901.7 GiB read | **89.90 %**, 4,481 resident, 1,016.1 GiB read |
| prediction precision | 22,778 / 68,721 = **33.15 %** | 25,510 / 77,103 = **33.09 %** |
| MLX peak | 55.6 GiB against the 72 GiB limit | 55.5 GiB (§7.2.8) |
| typing-time prefill | 47 tokens in 7.95 s = **169 ms/token** | 48 tokens in 11.07 s = **231 ms/token** |

**The live A/B is a null, and this time it could not have been anything else.** The floor measurement puts
the memoised constants at 1.2-1.3 ms; a live token here is 126-178 ms, so the expected effect is **0.7-1.0
%**, an order of magnitude under what a chat session resolves. The one matched prompt is the story, and it
came out **1.1 % in favour of the arm without the change**. That is the sign of noise, not of a regression,
and it is why section 9.17 is quoted from `profile_decode_sync.py`. Hit rate agrees to 0.13 points and
prediction precision to 0.06.

**Quality is the reference class on both arms**, read turn by turn:

- Both stories are coherent end to end across 602 and 545 tokens, correctly spelled and punctuated, with a
  sustained conceit and consistent tense. Session A names its coin *Liberty* in the first line and it is
  still Liberty, still the grandfather's gift, 600 tokens later. Neither shows the renaming or the dropped
  character that sections 7.4.1 to 7.4.8 were built around.
- Both Objective-C programs are real code. `#import <Foundation/Foundation.h>` intact in both, correct
  `NSJSONSerialization` selectors, `isKindOfClass:` guards, RFC 4180 quote doubling, `NSError **`
  propagated, and a `main` that checks `argc`. **No malformed `#import`, no broken identifier, no dropped
  character inside a word.** Session B's is the better-organised program — a `JSONToCSVConverter` class with
  ARC and a matching `clang -fobjc-arc` build line — and session A's calls `[[NSString alloc] initWithData:]`
  under a build line with no `-fobjc-arc`, which leaks rather than fails to compile. Sampling difference,
  not an arm difference.
- **Both coding turns finished on `stop=stop` rather than `stop=length`.** Raising the cap to 2000 is what
  section 7.2.3 asked for and it worked: 1,483 and 1,788 tokens, both ending on a complete closing question
  rather than mid-method. The `chat.sh` default of 1024 should follow.
- **Session A answered a bare "Hi" in Chinese** — `你好！有什么我可以帮你的吗？😊`, well formed, with an
  emoji. `chat.sh` passes no system prompt (`--system` defaults to `None`), and at temperature 0.6 this
  model answers an unanchored one-word English greeting in Chinese some of the time; session B's first turn
  and both second turns are English. **This is sampling, not an artefact** — the text is clean Chinese, not
  corrupted English. Nobody should chase it. The stray `w` of section 7.2.2 did not reproduce in either
  session, which makes three clean sessions.

**What did move, and it is a finding rather than a check.** Decode rate falls monotonically with the length
of the reply, and this is the first time enough points existed to say so:

| tokens in the reply | context at the last token | tok/s | session |
|---:|---:|---:|---|
| 545 | ~600 | **7.95** | 7.2.4 B |
| 602 | ~660 | **7.86** | 7.2.4 A |
| 643 | ~700 | 7.76 | 7.2.3 A |
| 889 | ~950 | 7.82 | 7.2.3 B |
| 1,024 | ~1,700 | 6.95 / 6.76 | 7.2.3 A / B |
| 1,483 | ~2,160 | **6.94** | 7.2.4 A |
| 1,788 | ~2,410 | **5.61** | 7.2.4 B |

**A 1,788-token reply decoded 29 % slower than a 545-token one, in the same session, on the same runtime.**

> **The reading of that table as an effect of length is withdrawn, the same day, by section 6.4.** Context
> length is worth 4.7 ms of `rest` across 512 to 2048 tokens and is not monotone; a 1,792-token
> continuation decoded in one pass gets *faster* as it runs; and two prompts at the same length and budget
> decode 15 % apart on working set alone. **The table below is a record of what seven turns did, not a
> curve.** Its ordering has a confound it cannot separate: every fast turn was a story and every slow one
> was Objective-C, and the code turn always ran second, into a cache warmed by prose.

### 7.2.5 Two sessions on 0.7.0, 2026-09-21 — and the first live coding turn ever put through a compiler

Hamed ran both arms himself after the session that closed the prediction levers. `./chat.sh`, 2-bit g128 at
a 44 GiB budget, 72 GiB wired, hotlist, temperature 0.6, Cachalot **0.7.0**. The only difference is
`CACHALOT_KERNEL_CONSTS`, so this repeats section 7.2.4's A/B on a newer version: session A is the shipped
memoised parameters, session B rebuilds them per call. Session A was given three prompts rather than four.

| | session A, memoised | session B, rebuilt |
|---|---|---|
| "Hi" | 9 tokens, 4.81 tok/s, English | 11 tokens, 5.17 tok/s, **Chinese** |
| "Answer only in english. Now, Hi!" | not run | 8 tokens, 6.11 tok/s |
| 500-word story | 595 tokens, **7.78 tok/s** (128.5 ms/token) | 454 tokens, **7.92 tok/s** (126.3 ms/token) |
| Objective-C, JSON to CSV | 1,283 tokens, **6.90 tok/s** (144.9 ms), `stop=stop` | 1,518 tokens, **6.93 tok/s** (144.3 ms), `stop=stop` |
| session hit rate | **90.0028 %**, 4,495 resident (41.7 GiB) | **89.9166 %**, 4,490 resident (41.6 GiB) |
| prediction precision | 20,539 / 61,168 = **33.58 %** | 21,719 / 65,069 = **33.38 %** |
| bytes | 802.9 GiB, 429 MiB/token | 856.8 GiB, 433 MiB/token |
| misses | 24.0/token | 24.2/token |
| MLX peak | 59.17 GiB against the 72 GiB limit | 59.58 GiB |
| typing-time prefill | 30 tokens in 6.28 s = **209 ms/token** | 39 tokens in 7.28 s = **187 ms/token** |

**The live A/B is a null for the third time, and for the second time it leans against the change.** The
matched turns are 1.8 % and 0.4 % in favour of the arm *without* the memoised constants, where the floor
measurement says the change is worth 1.2-1.3 ms on a 126-145 ms token — 0.9-1.0 %. A chat session cannot
resolve it and this is now three sessions' worth of evidence that it cannot. Section 9.17 stands on
`profile_decode_sync.py` and nothing else.

**What a live session does resolve is its own invariants, and they do not move.** Across four sessions on
three runtime versions — traced MoE, memoised constants, 0.7.0 — the session hit rate is 90.03, 89.90,
90.00, 89.92 %; prediction precision is 33.15, 33.09, 33.58, 33.38 %; residents are 4,484, 4,481, 4,495,
4,490; MLX peaks at 55.6, 55.5, 55.1, 55.5 GiB (§7.2.8). **The runtime's steady state is reproducible to a tenth of
a point, which is why a 1 % change has to be measured somewhere else.**

**The byte accounting closes to the byte on both arms, and the waste is bigger than section 9.10 says.**

    A: (45,987 misses + 40,629 wasted predicted) x 9,953,280 B = 862,113,300,480 B = ssd_bytes_read
    B: (49,079 misses + 43,350 wasted predicted) x 9,953,280 B = 919,971,717,120 B = ssd_bytes_read

**46.9 % of every byte read in a live session is a prediction nobody used**, on both arms, against the
42.7 % that section 9.10 opened with. Section 9.19 closed the two ways of reclaiming those bytes on their
ceilings rather than on their size, so the larger denominator does not reopen either.

**And 12 GiB of the wired limit is unused, in every session so far.** 44 GiB of experts plus the trunk peaks
at 55.1-55.5 GiB against `chat.sh`'s 72 (§7.2.8). Section 9.4 closed a larger budget on a simulation run at a
different expert size; at 9.49 MiB the headroom is real and measurable now that the machine takes a 44 GiB
benchmark.

#### The coding turns, compiled

Section 7.2.4 read its two Objective-C programs by eye and called them both real code. **These two were put
through `clang -fobjc-arc -framework Foundation`, which is what the corpus gate does, and one of them does
not compile.**

- **Session A fails**, on one error: `csvEscape` is declared `NSString *csvEscape(NSString *value)` and is
  then handed the dictionary's `id` values, so the `NSNumber` branch inside it reads
  `no visible @interface for 'NSString' declares the selector 'stringValue'`. The program is otherwise
  complete and sane — correct `NSJSONSerialization` selectors, `isKindOfClass:` guards, RFC 4180 doubling,
  a `main` that checks `argc` — and the fix is one word in the signature.
- **Session B compiles clean and runs.** Given a two-row file with a comma, an embedded quote and a nested
  array, its output doubles every internal quote, leaves missing keys empty and serializes the nested array
  into one quoted cell — RFC 4180 correct. It has a behaviour bug its own usage text contradicts: the output
  path is always given a `.csv` extension, so `./json2csv in.json out.csv` writes `out.csv.csv`.

**Neither failure is this runtime's defect class.** `#import <Foundation/Foundation.h>` is intact in both,
there is no malformed include, no broken identifier and no dropped character inside a word — the things
sections 7.4.1 to 7.4.8 were built around. A type error in a helper signature and an extension appended
twice are what a 552B model at temperature 0.6 produces, and the hosted reference arm produces them too.
The finding is about the **protocol**: reading a program by eye passed one that a compiler rejects, so a
live coding turn should go through `clang` before it is called reference class. Three lines of shell, and
it is the same check the corpus gate has made since section 7.4.1 was written.

**Two loose threads moved.** A bare "Hi" was answered in Chinese again, in session B this time — **two of
six sessions**, always with no system prompt, always clean well-formed Chinese with an emoji. Section 7.2.4
called it sampling and it is. And the stray `w` of section 7.2.2 has now failed to reproduce in **five**
clean sessions.

### 7.2.6 The first live session on 0.9.0, at a 52 GiB budget — 9.42 tok/s and a 92.4 % hit rate

Hamed's own session, `./chat.sh` with `CACHALOT_MLX_WIRED_LIMIT_GIB=80 --expert-budget-gib 52`, 2-bit g128,
hotlist, temperature 0.6, ready in 17.3 s. **Two things changed against the four sessions of section 7.2.5
at once** — the Engram reads of section 9.24 and the budget of section 9.4 — so the attribution below is
arithmetic, not an A/B.

| | 0.7.0, 44 GiB, four sessions | **0.9.0, 52 GiB** |
|---|---|---|
| prose, long reply | 7.78-7.92 tok/s | **9.42 tok/s** (544 tokens) |
| Objective-C, 1,300-1,500 tokens | 6.90-6.93 tok/s | **8.53 tok/s** (1,493 tokens) |
| session expert hit rate | 89.92-90.00 % | **92.37 %** (463,862 hits, 38,333 misses) |
| resident experts | 4,490-4,495 | **5,314** |
| resident bytes | — | 52.89 GiB |
| MLX peak | 55.1-55.5 GiB | 67.74 GiB against a 77.8 GiB wired limit (both corrected, §7.2.8) |
| prediction precision | 33.38-33.58 % | 31.4 % (16,585 of 52,763) |
| prefix cache | — | 13 hits, 1 miss, 4,109 tokens reused |

**The budget's effect is exactly what the simulation said.** Section 9.4 predicted **+2.6 points** of decode
hit rate for 44 → 52 GiB by offline replay; the live session moved the *session* hit rate from 90.00 % to
**92.37 %, +2.37 points**, with 819 more residents and 13 GiB more MLX peak. That is the first prediction
from `simulate_policies.py` ever checked against a live session, and it lands within a quarter of a point.
**52 GiB fits**: peak 67.74 GiB against the 77.8 the flag wired, on a 96 GiB machine, with no pressure
event. That peak was quoted as 72.7 GiB here until §7.2.8 found the unit error; the headroom is 10.1 GiB.

**And the speed is up by far more than the hit rate explains, which is where the Engram change shows.**
A prose token went from 128.5 ms to **106.2 ms** and a coding token from 144.3 to **117.2**. Misses scale
with `1 - hit`, so 2.37 points takes the 25 ms of blocking a 90 % session pays (section 9.23) to about
19 ms — **6 ms of the 22-27 ms.** The remaining **16-21 ms per token is the Engram reads**, which is the
direction and the size section 9.24 predicted from the benchmark: 24 ms at an 83.7 % hit rate, less at a
92.4 % one because the drive is less busy. The coding turn now decodes faster than the prose turns of every
earlier session.

**This is above what a live session can and cannot resolve.** Four sessions repeated their steady state to
a tenth of a point and the standing rule is that a chat session cannot see a 5 ms change (section 12.1).
This is 22-27 ms, five times that. What has *not* been separated is the two causes; a 52 GiB session with
`CACHALOT_ENGRAM_PARALLEL_MIN=1000000 CACHALOT_DECODE_ENGRAM_PREFETCH=0` would do it and costs one
conversation.

**Two short turns and one language slip.** "Hi" came back in Chinese at 8 tokens and 5.54 tok/s; told to
answer only in English it complied for the rest of the session. Short turns carry the startup miss cost and
are not rate measurements — 5.54 and 6.71 tok/s on 8 and 9 tokens against 9.42 on 544. The Chinese reply to
a bare "Hi" is a model behaviour on a multilingual checkpoint, not an artefact of this runtime's defect
class; it did not recur.

**The coding turn does not compile, and that is the second live coding turn ever put through a compiler and
the second to fail.** `clang -fobjc-arc -framework Foundation`:

```
json2csv.m:9:26: error: no visible @interface for 'NSString' declares the selector 'stringValue'
    NSString *s = [value stringValue];
```

`CSVEscape` takes an `NSString *` and immediately sends it `-stringValue`, which `NSString` does not
declare. Repairing that one line compiles the program, and it then **crashes on the model's own example
input**, because the same mistake appears a second time where the compiler cannot see it — `[key
stringValue]` on an `id` from `-allKeys`:

```
*** Terminating app due to uncaught exception 'NSInvalidArgumentException', reason:
'-[NSTaggedPointerString stringValue]: unrecognized selector sent to instance'
```

Everything else about the program is sound: the RFC 4180 escaping is right, the nested-value serialisation
is right, the usage text, the build line and the example output are right, and the notes it appends
correctly warn that `NSDictionary` does not preserve key order. **It is one wrong method name, made twice.**
That is a model-level type error, exactly like section 7.2.5's `NSString *` handed `id` values, and it is
not this runtime's defect class — the defects this runtime produced were malformed `#include` lines and
in-context copy corruption, neither of which appears. The quality statement remains the 40-case corpus gate
of section 7.4.8 (20 of 20 C++ blocks compiling, 0 of 101 malformed includes, every column equal to the
hosted reference); **a hand-written Objective-C turn from a live chat is not that gate, and two of two now
say so.**

The turn is kept for the record, with the compiler output and the one-line repair, in
`docs/live-turns/2026-09-21-json2csv/`.


### 7.2.7 The second live session at 52 GiB, on 0.9.2 — the configuration replicates, 2026-09-21

Hamed's own session, the same command, on a build that carries **no runtime change** against 0.9.0:
0.9.1 and 0.9.2 are documentation and instruments, and nothing under `src/cachalot/` was touched. So this
is not a measurement of a change; it is the second independent reading of the shipped configuration, which
is what section 7.2.6 said it needed before 52 GiB could become the default.

| | 0.9.0, 52 GiB (§7.2.6) | **0.9.2, 52 GiB** |
|---|---|---|
| prose, long reply | 9.42 tok/s (544 tokens) | **9.59 tok/s** (548 tokens) |
| Objective-C | 8.53 tok/s (1,493 tokens) | **8.15 tok/s** (1,484 tokens) |
| session expert hit rate | 92.37 % | **92.31 %** (463,774 hits, 38,661 misses) |
| resident experts | 5,314 | 5,276 (48.91 GiB) |
| MLX peak | 72,734,620,776 B | **72,734,620,776 B** — 67.74 GiB, see §7.2.8 |
| prefix cache | 13 hits, 1 miss, 4,109 tokens reused | 13 hits, 1 miss, 4,192 tokens reused |
| ready | 17.3 s | 17.8 s (hotlist 863 experts, 8.0 GiB in 2.1 s) |
| startup language slip | "Hi" answered in Chinese | "Hi" answered in Chinese |

**It replicates.** The hit rate is within 0.06 points, the peak is identical to the byte, the residents are
within 0.7 %, and the two long turns bracket the earlier pair — prose 1.8 % faster, Objective-C 4.5 %
slower, both inside what section 12.1 says a single turn resolves. **52 GiB now has two sessions, not one**,
and the objection section 7.2.6 raised against making it the default is answered on the footprint and
hit-rate side. What remains unanswered is section 9.24's attribution: this session ran with the Engram
change on, so it is a second reading of the good arm and not the A/B.

**The per-miss cost of section 7.1.9 was checked against a conversation for the first time, and it holds.**
502,435 expert requests over 240 per token is about 2,093 token-equivalents, so the session missed **18.5
times per token**. At the 1.7 ms per miss section 7.1.9 measured on the bench — 1.41 blocked, 0.31 inside
`mx.eval`, about 0.05 of CPU — that is 31.4 ms on the 79.6 ms floor, **111 ms predicted against 104.3 ms
observed on the prose turn and 122.7 on the coding one.** The prediction lands between the session's own
two long turns. It is not a tight check, because the floor was measured at a 512-token context and a coding
turn runs out past 2,000 positions where attention costs more, but it is the first time the bench's model
of a miss has been priced against a real conversation and it does not contradict it.

**Half the drive traffic was not a demand miss, and nobody has measured what the rest bought.** The session
read **755.07 GB**. The 38,661 demand misses account for 384.8 GB of that at 9.953 MB an expert; the
hotlist is 8.6 GB and prefill was almost entirely prefix-cache reuse (13 hits, 4,192 tokens, 619 of 623 on
the coding turn), so **roughly 360 GB — about 48 % of everything read — was speculative prefetch.** That is
a large number at the shipped budget and section 9.10's 42.7 % figure was measured at 44 GiB on a different
shape of session.

**Do not read `predicted_used / predicted_loads` as the prediction hit rate.** It is 16,466 of 53,666 here
and 16,585 of 52,763 in section 7.2.6, and both sections' predecessors called it "prediction precision",
but `predicted_used` increments in exactly one place — `resident_store.py:629`, inside `get_many`'s pending
branch, when a demand request finds a prediction **still in flight and waits on it.** A prediction that
completes before it is demanded is counted as an ordinary hit and never touches this counter, so the ratio
is a lower bound on precision by an unknown margin, not a measurement of it. The 42.7 % of section 9.10 came
from offline replay and `predict_ghost.py`, which is a different and sounder instrument. **The live
counters cannot answer what fraction of 360 GB was wasted; that needs an instrument that does not exist.**

### 7.2.8 Every MLX peak in this document was quoted in GB against a limit in GiB — corrected 2026-09-21

`stats()["mlx_peak_bytes"]` is `mx.get_peak_memory()` in **bytes** (`model/api.py:162`), and the runtime's
own banner prints the wired limit as `rt.mlx_wired_limit_bytes / GiB`. Sections 7.2.3 to 7.2.6 divided the
peak by 10^9 and the limit by 2^30 and set the two side by side. **Every "peak against limit" pair in this
document before this section therefore understates the headroom by 7.4 %.**

| quoted | actual | against |
|---|---|---|
| "59.2-59.7 GiB" at a 44 GiB budget | **55.1-55.6 GiB** | a 72 GiB wired limit |
| "72.73 GiB" at a 52 GiB budget | **67.74 GiB** | a 77.8 GiB wired limit |

The arithmetic confirms which reading is right. At 52 GiB the session held 5,276 residents, which is
48.91 GiB of experts; the non-expert footprint is the 10.4 GiB trunk, the 2.4 GiB of transient slots and a
1.55 GiB MLX cache, about 14.4 GiB. **48.91 + 14.4 = 63.3 GiB, which 67.74 clears and 72.73 does not** —
72.73 GiB would need 23.8 GiB of non-expert memory that nothing in the runtime accounts for. The same
check passes at 44 GiB: 4,495 residents is 41.7 GiB, plus 14.4 is 56.1, against a corrected 55.6.

**What this changes.** Nothing about any speed or quality conclusion; the peaks were only ever used to
decide whether a budget fits, and a conservative error in that direction never let an unsafe budget
through. What it changes is the headroom, and therefore the next budget worth trying:

- At 52 GiB the peak is **67.74 GiB against a 77.8 GiB limit — 10.1 GiB of headroom, not 5.1.**
- Scaling the expert term alone, a **60 GiB budget projects to about 75.2 GiB**, which still fits under
  77.8 with 2.6 GiB to spare, and section 9.4's replay says 52 → 60 is worth another **2.3 points** of
  decode hit rate.
- That projection is linear in the resident bytes and ignores fragmentation, so it is a reason to screen
  60 GiB, not a reason to ship it. Read the peak out of `/stats` and compare it against this table, in the
  same units, before drawing any conclusion from it.

### 7.2.9 The 54 GiB session was slow, 60 GiB was slower, and 50 GiB replays clean — cause open, 2026-09-22

Job 1 asked whether 60 GiB is the next default. Hamed ran it. **60 GiB was abandoned as unusably slow, and a
54 GiB session ran at 4.9-6.2 tok/s against the 9.4-9.6 of the two 52 GiB sessions.** The 54 GiB session, on
0.9.5, temperature 0.6, `CACHALOT_MLX_WIRED_LIMIT_GIB=90`:

| | 52 GiB, two sessions | **54 GiB** | 50 GiB, guarded replay of the 54 GiB prompts |
|---|---|---|---|
| story, 526-606 tokens | 9.42 / 9.59 tok/s | **6.00** | **9.58** |
| Objective-C, 1,692-1,789 tokens | 8.53 | **4.87** | **8.52** |
| session hit rate | 92.37 / 92.31 % | 92.70 % | 91.9 % over the Objective-C turn |
| residents at the end | 5,314 | 5,503 | 5,393 |
| MLX peak | 67.74 GiB | 69.8 GiB (74.9 GB) | not read |
| pressure / swap / compressor | none seen | **not sampled** | 1 / flat 490 MB / flat 3.7 GiB |

**What is established.**

- The wired limit is not the variable. `resolve_wired_limit` (`src/cachalot/config.py:197`) returns the
  minimum of the request and the device's recommended working set, so the 90 in the command did nothing and
  the banner printed 77.8 GiB at 54 exactly as it did at 52. A 90 GiB request is not what 60 GiB ran under.
- The hit rate barely moved (+0.33 points against the ~+0.7 the replay predicts), so the 30-40 % speed loss
  is not the miss count. At 17.5 misses a token and 1.7 ms each, misses explain 30 ms of a 167 ms token, not
  the 60 ms of slowdown.
- Resident count is not the variable either: the replay held 5,393 residents at 50 GiB against the slow
  session's 5,503, and ran at the good speed.
- The 50 GiB replay used the same four prompts, temperature 0.6, hotlist, bank and wired request, on this
  machine right after the failed 60 GiB attempt: stale swap 490 MB before and after, compressor flat, pressure
  normal throughout, free memory touched 0.1 GiB (page cache filling it, which is normal) with 79.3 GiB
  counted available at the start against 79 the guard needs.

**What is not established, and the two candidates.** Nothing sampled memory during the 54 GiB session.
Either it crossed a physical-memory cliff that 50 does not (planned wired 65 GiB at 50, about 69 at 54, 75 at
60, on a 96 GiB machine that other applications already hold 12-17 GiB of), or the machine was degraded at
that moment (compressor full after the 60 GiB attempt, another process busy). **The replay cannot separate
them because `guarded_run.sh` will not start a budget above about 50 on this machine's 79 GiB available**
(54 needs 83), and adding ballast to reach 54 would step outside the guard's own plan, so it was not done.
The check is one conversation:

```bash
benchmarks/memwatch.sh 52
```
in a second terminal, then `CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52` and the same
four prompts, `/exit`, Ctrl-C the sampler; then again with `54`. A compressor that grows, swap that grows,
pressure above 1 or pageouts during the slow turns say cliff; a flat compressor at pressure 1 with slow decode
says the cause is not memory and the 54 GiB session was something else. Do not run 60 again until that answer
exists.

**Consequence.** No lever appeared and none was closed: the budget curve is flat-to-negative above 52 on this
machine as it stood, 52 GiB stays the shipped value, and the budget-as-lever claim in sections 9.31 and v30
now carries the caveat that its ceiling is physical memory, not the hit-rate table.

### 7.2.10 Six budget arms with a sampler running: no pressure cliff, one turn-4 collapse at 54 GiB — 2026-09-22

Job from §7.2.9. Hamed ran `benchmarks/memwatch.sh` beside six live `./chat.sh` sessions at 46, 48, 50, 52
(twice), 54 and 56 GiB. Session transcripts came back for four of the six (48, 50, 52-second, 54); 46, 52-first
and 56 have memory traces only, no saved transcript.

| budget | story tok/s (tokens) | Objective-C tok/s (tokens) | session hit rate | resident bytes | mlx peak / 77.8 GiB limit |
|---:|---:|---:|---:|---:|---:|
| 48 | 9.58 (530) | 8.39 (1707) | 91.06 % | 48.97 GiB | 63.72 GiB |
| 50 | 9.79 (699) | 8.75 (1492) | 92.07 % | 50.50 GiB | 65.72 GiB |
| 52 | 9.84 (418) | 8.69 (1526) | 91.93 % | 52.47 GiB | 67.74 GiB |
| **54** | 9.08 (577) | **4.83 (1565)** | 92.79 % | 54.35 GiB | 69.76 GiB |

**`Test-52-1.txt` and `test-54.txt` are byte-identical** — the same 54 GiB session saved twice, not two
independent runs. There is exactly one 54 GiB reading here, not a confirmed pattern.

**What the sampler shows, matched against the CSVs (`benchmarks/results/guarded/memwatch_*_20260922-1*.csv`):**

| budget | duration | min available | peak wired | compressor start→peak | swap | peak pressure |
|---:|---:|---:|---:|---:|---:|---:|
| 46 | 301 s | 14.3 GiB | 68.2 GiB | 3.5 → 3.5 GiB (flat) | flat | 1 (normal) |
| 48 | 353 s | 12.5 GiB | 70.5 GiB | 3.9 → 3.9 GiB (flat) | flat | 1 |
| 52 (first) | 452 s | 8.7 GiB | 74.3 GiB | 2.0 → 3.7 GiB | flat | 1 |
| 52 (second) | 322 s | 8.9 GiB | 74.2 GiB | 3.9 → 3.9 GiB (flat) | flat | 1 |
| **54** | 487 s | 7.9 GiB | 76.4 GiB | 3.5 → 4.2 GiB, gradual | flat | 1 |
| 56 | 621 s | 6.9 GiB | 78.1 GiB | 3.4 → 4.7 GiB | flat | 1 |

**§7.2.9's physical-memory-cliff hypothesis is refuted for this run.** At 54 GiB, available memory held
7.9-9.2 GiB the entire session, including the 325-second Objective-C turn that collapsed to 4.83 tok/s; the
compressor grew by 0.7 GiB gradually over the whole 487 s with no jump timed to that turn; swap never moved;
`kern.memorystatus_vm_pressure_level` read 1 (normal) at every one-second sample. **Nothing the OS reports
distinguishes the slow turn from the fast ones around it.** The MLX peak-against-limit headroom is smooth
too — 63.72, 65.72, 67.74, 69.76 GiB at 48/50/52/54, a plain ~2 GiB step per 2 GiB of budget, not a cliff
near 54.

**What actually collapsed, and where.** Not the session — turns 1-3 of the 54 GiB session (including the
577-token story, 9.08 tok/s) are indistinguishable from 48/50/52. Only turn 4, the longest single decode of
the six sessions compared here (1565 tokens, ~325 s of continuous generation), ran at 4.83 tok/s. That points
away from a budget-indexed memory effect and toward something that accumulates with **decode duration**
within a session — thermal throttling over a sustained multi-minute burst, or a one-off external interference
(Spotlight, Time Machine, another app waking) during that specific five-minute window — neither of which
`memwatch.sh` samples. Both are live hypotheses; neither is confirmed.

**Where this leaves the budget.** 52 GiB has two clean readings now (9.42-9.84 tok/s story, 8.53-8.69
Objective-C, 91.9-92.4 % hit rate) and stays the shipped value. 54 GiB is not disqualified — the one reading
that exists has a plausible non-budget explanation — but it is not confirmed either. **Do not adopt 54 or
higher from this data.** The next check is a repeat of 54 GiB itself (not 50, which already ran clean in
§7.2.9's guarded replay and does not bear on turn duration), watching wall-clock time into the turn rather
than memory: if the same long-turn collapse reproduces, decode duration or thermal state is the lever to
chase next with `sudo powermetrics --samplers cpu_power,gpu_power -i 1000`; if it does not reproduce, the one
reading was a one-off interference and 54 GiB should be screened again clean.

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
syntax-checks them with `ast.parse` or `clang++ -fsyntax-only`. **Every number this metric produced before
2026-09-19 was wrong, and section 7.4.1 is the correction — read it before quoting anything below.**

> **Correction, 2026-09-18.** An earlier version of this section reported "6.6x fewer syntax errors" for the
> 3-bit bank and a "4.6x" figure against the 2-bit one. **Both were confounded and are withdrawn.** The arms
> were not matched on what they generated: one produced three ~107-line C++ programs, the other ten ~19-line
> snippets, and long blocks accumulate errors while short ones do not. The metric was measuring block
> composition as much as correctness. Always compare arms on the same conversation *and* check the average
> block length before believing a ratio.

Matched properly — same three-turn conversation, same sampling, C++ blocks only. **This is the table as the
broken checker produced it, kept because the corrections attached to it are the useful part; section 7.4.1
withdraws every figure in the last column.**

| bank | blocks | avg block | lines | errors | ~~errors per 100 lines~~ |
|---|---:|---:|---:|---:|---:|
| **FP4**, 3 seeds, 2026-09-18 session 4 | 4 | 115 | 461 | 41 | ~~8.9~~ |
| FP4, the 2-seed arm the guard truncated | 3 | 54 | 163 | 1 | ~~0.6~~ |
| 3-bit g64, searched + activation-weighted | 3 | 138 | 414 | 131 | ~~31.6~~ |
| 3-bit g64, `mx.quantize` | 5 | 98 | 491 | 96 | ~~19.6~~ |
| 2-bit g128 | 13 | 18 | 233 | 59 | ~~25.3~~ |

> **Correction, 2026-09-18 session 4.** FP4's **0.6** was real but thin: 163 lines from an arm the memory
> guardian killed after two of three seeds, carried by one 153-line block the checker called clean. Re-run to
> three full seeds on the internal SSD — 461 lines, 4 blocks — FP4 scored **8.9**. Every ratio this document
> quoted against 0.6 is therefore too large by an order of magnitude.
>
> The lesson is the one section 7.4 already carried and did not apply to itself: a code-validity figure is
> only as good as the volume behind it, and a truncated arm is a small sample dressed as a measurement. Check
> that every arm ran to completion before comparing them, and prefer the guarded run's own log over the
> result file, which cannot tell you it is short.

> **Second correction, 2026-09-19, and it supersedes the first.** The 8.9 was not a smaller version of the
> same measurement; it was the same broken measurement. The checker never read clang's return code, so a
> block that aborted on a mangled `#include` scored one error or none. The "clean" block in the 2-seed arm
> and the "clean" block in the 3-seed arm are the same kind of false success. **Re-scored: 0 of FP4's 4
> blocks compile, and the whole column is withdrawn.** Section 7.4.1.

**The quantized banks are not meaningfully different from each other**, which is the finding that matters:
the +5.3 points of top-1 the `mx.quantize` 3-bit bank genuinely bought translated into no usable improvement
in code, and neither did the 28 % of output error the searched, activation-weighted fit bought after it.

> ~~**Python is not the discriminator; long C++ is.** On the same 2026-09-18 session-4 arms, Python blocks
> score **2.2** errors per 100 lines on *both* FP4 and the 3-bit bank.~~ **Withdrawn 2026-09-19.**
> `ast.parse` stops at the first `SyntaxError`, so 2.2 was counting broken files rather than defects. On the
> parse rate FP4 is **2 of 10** and the 3-bit bank **0 of 5**, and one 20-line FP4 block carrying five
> separate artefacts scored 1. Section 7.4.1.

The advice the withdrawn paragraph ended with is still right and is now better supported: **screen on the
hardest thing the model is asked to write.** Long C++ is where blocks run past 100 lines and an artefact
every few hundred tokens is certain to land inside one — but Python is not the clean control this document
took it for, and both languages must be reported on the compile rate rather than on a density.

**The three together.** Top-1 said 50.8 % against 44.5 %. The paired median said the 2-bit bank is worse on
62 % of tokens. The compiler was believed to say 15.2 errors against 2.3, and **it did not say that** — see
7.4.1. The standing decision in section 2 rests on the first two and on the collapse rate.

The repetition loops are a *separate* failure with a *separate* fix: they are sampling dynamics, they happen
on FP4 too, and the frequency penalty handles them. A better bank will not stop loops and the penalty will not
stop artefacts. Both are needed.

### 7.4.1 The compiler gate was broken, and every ratio it produced is withdrawn

**2026-09-19.** `check_cpp` collected the stderr lines containing `": error: "` and never read clang's return
code. Clang reports a missing header as `": fatal error: "` and then **stops**, so a block whose only defect
was a mangled `#include` came back with an empty error list and was scored **clean**. The gate is fixed,
`tests/test_code_validity.py` pins the failure, and the saved arms have been re-scored from the replies
themselves.

**What the saved arms actually are**, `benchmarks/code_validity.py` on `benchmarks/results/replies`:

| arm | lang | blocks | compile | aborted on a fatal | truncated | fully diagnosed | errors / 100 lines |
|---|---|---:|---:|---:|---:|---:|---:|
| FP4 | C++ | 4 | **0 of 4** | 3 of 4 | 2 | 1 | 33.6 |
| 3-bit searched + weighted | C++ | 3 | **0 of 3** | 1 of 3 | 3 | 0 | n/a |
| FP4 | Python | 10 | **2 of 10** | — | 0 | 10 | 2.2 (floor) |
| 3-bit searched + weighted | Python | 5 | **0 of 5** | — | 1 | 4 | 2.0 (floor) |

**Not one C++ block from either bank compiles.** The block this document called FP4's one clean block in four
is the 21-line block in seed 20260919 turn 3; it contains `#include <s>` and fails with
`fatal error: 's' file not found`.

**Three things follow, and the third is the one that matters.**

**The old error density measured which arm gave up first.** A block that aborts on its first mangled include
contributes one diagnostic; a block whose includes happened to survive contributes every diagnostic in the
file. Three of FP4's four blocks aborted and only one of the 3-bit bank's three did — so 39 of FP4's 41
errors came from its single fully diagnosed block, while the 3-bit bank's 131 came from two. **8.9 against
31.6 is very largely that asymmetry and not a quality difference.**

**Stratified onto comparable blocks there is nothing left to compare.** FP4 has exactly one block that
reached the end of the file and was not cut off at the token cap; the 3-bit bank has none, because all three
of its C++ blocks were truncated. The gate cannot separate these two banks on this evidence, in either
direction.

**`ast.parse` has the same flaw and it hid a worse result.** It raises on the first `SyntaxError` and never
sees the rest of the file, so "Python blocks score 2.2 errors per 100 lines on *both* banks, therefore Python
is not the discriminator" was reading a per-file indicator as a defect count. On the parse rate **FP4 is 2 of
10 and the 3-bit bank 0 of 5**. And the FP4 blocks are not marginal: seed 20260919 turn 2 carries
`c_csv_file_path`, `utfutf-8`, `csv.Dreader`, `utfutf-utf8` and `__name __` in twenty lines, and scored 1.
**FP4's own artefact rate is much higher than this document has ever recorded.**

**What survives, and what does not.**

- *Withdrawn:* 8.9, 31.6, 19.6, 25.3, 22.7, 4.9, "3.5x", "6.6x", "4.6x", "15.2 errors against 2.3", and
  "Python is not the discriminator". The 2-bit and `mx.quantize` 3-bit replies were not kept, so their
  numbers cannot be re-scored and must simply not be quoted.
- *Survives:* the standing decision. It does not depend on this metric. FP4 collapses on 0 of 9 free-running
  replies against the searched 3-bit bank's 2 of 9 (section 9.0); it wins top-1 by 5.3 points and the paired
  sign test beyond 5 sigma against every 2-bit bank (section 9.3.1). Two independent metrics, both intact.
- *Survives with its reason replaced:* lever 0 is still closed. Its closing condition was written in terms of
  a number that turns out not to mean anything, but the bank it was testing collapsed on a third of its long
  turns and produced no compiling code, and that is sufficient.

**The general lesson, and it is the fourth time this document has had to write a version of it.** A gate that
cannot fail loudly will fail quietly. `check_cpp` had no test, returned a list whose emptiness was read as
success, and ran for two sessions producing the headline number in a standing decision. **Every gate needs a
fixture that it is known to fail**, and a metric whose denominator is not fixed — errors per 100 lines,
over whichever blocks happened to be scoreable — will drift into measuring its own denominator.

### 7.4.2 The corpus the repaired gate needed, built 2026-09-19

Repairing the checker did not repair the evidence. Items 1 to 3 below were written as a to-do list and then
done the same day; item 4 is still open.

**`benchmarks/coding_tasks.json`** is twenty independent single-turn tasks: ten C++ programs, eight Python
programs and two snippets, covering complete programs, edits to supplied code and bug fixes, on subjects
other than file conversion — an LRU cache, a thread pool, interval merging, an INI parser, a retry decorator,
an atomic-write context manager, an argparse CLI, a mutable-default bug, an off-by-one binary search. Each
task declares the language it must be answered in and whether it is a complete program.

**`benchmarks/coding_quality.py`** runs them against one bank in one process with `rt.reset()` between tasks,
so a collapse cannot poison the next task and every bank enters every task from byte-identical context. The
multi-turn conversation test measures something else and `repetition_quality.py` keeps it.

Four things it does that the old arrangement did not:

1. **Every case is accounted for.** No code, wrong language, an untagged fence, a reply that stopped at the
   token cap — each is recorded as that rather than leaving the denominator.
2. **The run records what produced it.** An immutable directory per run holding the replies, a per-case row
   and a manifest: git SHA and dirty state, bank path and format, expert bytes, budget, every `CACHALOT_*`
   variable, all sampling parameters, the corpus hash, the seeds, and planned against completed cases.
3. **An unfinished run refuses to be scored.** The manifest is written before generation with
   `complete: false` and rewritten at the end. `code_validity.py` refuses a run whose manifest says it did
   not finish, or whose completed count falls short of its plan, unless `--allow-incomplete` is passed — and
   then it labels the arm `INCOMPLETE`. A guarded arm killed at its timeout leaving a result that looks whole
   is exactly what put a 0.6 in this document for a day.
4. **A snippet is scored as a snippet.** Tasks that ask for a fragment are told to omit their includes, so
   compiling one alone yields a page of `use of undeclared identifier 'std'`. They are counted in their own
   column and excluded from the compile rate, the fatal rate and the density. The smoke run showed why: one
   20-line snippet contributed 18 of 29 C++ diagnostics and moved the density from 14.9 to 31.5.

**Smoke-tested end to end on FP4**, three tasks, one seed, at a 24 GiB budget: all three finished on `stop`
with one correct-language block each and nothing truncated. Replies are much shorter than the old
conversation turns — 505, 595 and 127 tokens — because the tasks are scoped, so the full corpus at two seeds
costs about **1.7 hours** on FP4 rather than the nine a worst-case cap suggests.

**Still open, and it is item 4 from the old list.** Behavioural tests on the programs that compile, and
**successful tasks per wall-clock hour** as the product metric. Compiling is necessary and not sufficient,
and tokens per second alone rewards a fast stream of code that does not build. Nothing here measures whether
a program that builds also does what it was asked.

### 7.4.3 A second collapse mode, which `max_run` cannot see

**Found 2026-09-19 while smoke-testing the new corpus, and it touches the one pillar section 7.4.1 said was
intact.**

`repetition_quality.py` calls a reply collapsed when `max_run` -- the longest span covered by a k-gram
repeating **back to back** -- reaches 24 tokens. That detects the failure it was built for,
`"res.res.res.res..."`. It cannot detect a *paraphrased* loop, where the model writes bad code, notices,
apologises, and tries again in slightly different words. Nothing repeats exactly, so `max_run` stays small.

The new corpus produced one on its third task. `cpp-thread-pool`, FP4, 503 tokens, penalty on: **eighteen**
fenced C++ blocks, every one of them a mangled include list, separated by "I need to correct that", "I
apologize for the repeated errors", "I'm clearly stuck in a loop". **Its `max_run` is 7.** The model states
that it is looping and the detector scores it clean.

It is not new to the corpus. Two of the nine saved FP4 replies behind section 7.4's table do the same thing:

| saved reply | fenced blocks | `max_run` | what it is |
|---|---:|---:|---|
| `lc_fp4 … seed20260917_turn2` | 6 | 19 | four self-corrections, ending "I clearly need to reset" |
| `lc_fp4 … seed20260919_turn2` | 7 | 4 | seven, ending "I clearly cannot produce clean code in this response" |

Both are under the threshold of 24. **So "FP4 collapses on 0 of 9 replies" means "0 of 9 exact k-gram
loops", and there were at least two loops of the other kind in the same nine files.**

**What this does and does not change.**

- It does **not** re-rank FP4 against the 3-bit bank. The retry-phrase count is 2 of 9 on FP4 and 0 of 9 on
  the 3-bit bank, which looks bad for FP4 — but the 3-bit bank's replies died early *in exact loops* (two of
  them at 285 and 157 words), so they had far less opportunity to retry. The comparison is confounded in
  FP4's favour and against it at once, and nothing here separates them.
- It does mean the standing decision's free-running evidence is **narrower than recorded**. Section 2 rests
  on collapse rate, top-1 and the paired sign test. The collapse rate now covers one of two known failure
  modes; the other two metrics are teacher-forced and, by operating rule 4, structurally blind to both.
- It does **not** say FP4 is unusable. It says nobody has measured this mode on any bank.

**The frequency penalty does not stop it, and should not be expected to.** A retry loop is semantic, not
lexical: each apology is differently worded, so a per-token frequency penalty has almost nothing to bite on.
Section 9.9's "62 % to 12 %" is about exact loops and stands; it is not a claim about this mode.

**Do not fix this by adding a phrase list to the gate.** Matching "I apologize" is a screen, it is
English-specific, and it will be gamed by the next model that apologises differently — this document has
been wrong five times about screens. The robust signal available today is that the trigram rate separates
these cleanly (57.5 % on the thread-pool reply against 34.8 % on a healthy one from the same run), which
is awkward, because section 7.4 tells the reader to judge by `max_run` and **not** by the trigram rate. That
advice was right for exact loops and is wrong for these. **A collapse metric needs both**, with the
threshold for each set on replies that have been read.

**Measure it before believing any of it.** Three reply pairs is not a rate; rule 5 asks for four seeds. The
corpus run in flight is 40 cases and is the first sample large enough to put a number on this.

### 7.4.4 Where the corruption lands, and a cheap experiment nobody has run

**Diagnosis, not a metric. Read the warning at the end before using any number here.**

The failures in the new corpus cluster somewhere specific. Across the first six C++ replies, **26 of 70
`#include` lines are malformed**, and the dominant shape is a **dropped `<`**:

    #include escaping>          #include stdio.h>        #include unordered_map>
    #includequeue>              #include>                #include <ioman double>

That is fatal in a way an ordinary typo is not: clang stops at the first bad include (section 7.4.1), so one
dropped character costs the whole file and hides every other defect behind it.

Counting the same thing on the saved arms **separates the banks**, on a signal that is immune to the two
problems that broke the compile comparison — it does not care whether the block was truncated at the token
cap, and it does not care whether the compiler gave up:

| arm | `#include` lines | malformed | dropped `<` |
|---|---:|---:|---:|
| FP4 | 27 | **5 (19 %)** | 3 |
| 3-bit searched + weighted | 43 | **28 (65 %)** | 27 |

Python `import` lines are essentially clean on both (0 of 45 and 1 of 24), **which does not mean Python is
safe** — the same FP4 replies carry `utfutf-8`, `csv.Dreader`, `c_csv_file_path` and `__name __` in their
bodies, and 8 of 10 fail to parse (section 7.4.1). The corruption is general. The include line is simply
where it is densest and where one hit is fatal.

> **Why this is not in `benchmarks/` and must not be quoted as a gate result.** The pattern was written
> *after* looking at the replies, and fitted on nine of them per arm. That is a screen built on its own test
> set, and this document has been wrong about screens five times — most recently in section 9.12.1, where an
> offline table gave a count and hid its price. The 19 % against 65 % is a **hypothesis**. To use it,
> pre-register the pattern, then score the 40-case corpus, which was generated before the pattern existed.
> The script lives in the session scratchpad on purpose.

**The cheap experiment this suggests, and it is much cheaper than reference fixtures.** A 37 % failure rate
on one highly predictable token is more consistent with a systematic numerics fault than with quantization
blur, which should degrade everything roughly evenly. That can be tested **without generating anything**:
teacher-force the model over a file containing correct `#include <iostream>` lines and read the logits at
the `<` position. If FP4 ranks `<` far below where a clean path would, the fault is visible in one forward
pass with no sampling, no collapse and no compiler involved — and `nll_expert_precision.py` already has the
teacher-forced machinery to do it. `benchmarks/token_rank_probe.py` **is written and has not been run** -- the GPU was busy generating the
corpus. It teacher-forces a text dense in correctly spelled `#include <...>` lines, reports where the model
ranked the token that should follow `#include`, and carries Python `import` lines as a within-run control:
if `<` is mis-ranked and `import` is not, that is a fact about one token rather than about the whole model.
Run it on the production path and again with `--experts fp4`-style dense substitution to separate the
runtime from the bank.

It is an hour, and it is the first concrete argument this project has for the independent-reference work the
2026-09-19 review asked for: **run it first, and escalate to pinned reference fixtures only if the logits
look wrong.** Note what it cannot tell you: it answers "does the model know", not "does the model emit". A
token ranked 1 that still comes out wrong under sampling is a different bug, in the sampler.

### 7.4.5 Greedy decoding still corrupts, so sampling is not the cause

**Measured 2026-09-19, six C++ tasks at temperature 0 with the frequency penalty
off.** Greedy decoding takes the single most likely token at every step. If the
artefacts were sampling noise — an unlucky draw from a slightly blurred
distribution — greedy would remove them. It does not.

| task | tokens | finish | malformed / total `#include` |
|---|---:|---|---:|
| cpp-lru-cache | 529 | stop | **1 / 4** |
| cpp-matrix-transpose | 427 | stop | **1 / 6** |
| cpp-ini-parser | 355 | stop | **1 / 5** |
| cpp-template-stack | 649 | stop | **1 / 4** |
| cpp-csv-to-json | 2000 | length | 988 / 993 |
| cpp-thread-pool | 2000 | length | 988 / 992 |

**Six of six greedy replies contain at least one malformed include**, and the
two that ran to the token cap were looping on `#include>` itself — which is
what the anomalous 7.2–7.6 tok/s on those two arms was, against a normal 2.7:
a tight loop re-reads the same experts and the hit rate goes up.

**A greedy loop on `#include>` means that after `#include`, the argmax token is
`>`.** Deterministically, with no sampling anywhere in the path. Quantization
blur perturbs a distribution; it does not usually survive `argmax` on a token
this predictable, and it certainly does not do so repeatedly at the same
construction.

**And it is not only the delimiter.** From the greedy `cpp-lru-cache` reply:

    #include <iostream>      correct
    #include <list>          correct
    #include unordered_map>  the '<' is gone
    #include <utility>       correct
    ...
    explicit LRcache(size_t capacity)      LRUCache -> LRcache

Two well-formed includes, one with a dropped delimiter, one well-formed, and a
mangled identifier in the next declaration. The shape is **single-token drops at
high-confidence positions**, not general noise, and the same shape appears in
the 2026-09-18 saved replies as `std std::string`, `utfutf-8`, `csv.Dreader`
and `__name __`.

**What this establishes and what it does not.** It removes sampling from the
question entirely: the fault is in the numerics, the kernels or the weights.
It does **not** yet say which, and it does not say whether the reference
implementation does the same thing — `docs/HANDOFF.md` section 7.4.4's probe
and the OpenRouter reference arm are what separate those. But it does mean the
frequency penalty, the temperature and the sampler are all cleared, and any
future work that starts by tuning them is wasted.

**It also makes the probe trivial to interpret.** `benchmarks/token_rank_probe.py`
teacher-forces correct `#include <...>` lines and reads the rank of the token
that should follow `#include`. Greedy already tells us the argmax is wrong at
that position in real generation; the probe says by how much, and whether the
same happens on a dense-FP4 substitution path that bypasses the expert bank.

### 7.4.6 The reference arms are clean. The fault is in this runtime.

**Measured 2026-09-20. This is the section that closes the question sections 7.4.4 and 7.4.5 opened,
and it is the reason the standing decision in section 2 has changed.**

The 40-case corpus was run against a hosted endpoint serving the same model, with the provider pinned and
fallbacks off, by `benchmarks/run_reference.py`. That script is run A, the diagnostic: bare API calls, no
system prompt, no tools, no content retries, reasoning disabled, and the pack's own settings — temperature
0.6, top_p 1.0, max_tokens 2000, frequency_penalty 0.2, both seeds.

Two reference arms were run rather than one, so that the FP4 expert format could be separated from the
serving stack, and a third arm — the Hermes harness on the same endpoint — was scored alongside them:

| | Cachalot | run A, `relace/fp4` | run A, `deepinfra/fp8` | run B, Hermes harness |
|---|---:|---:|---:|---:|
| C++ blocks that compile | 0 / 42 | **20 / 20** | **20 / 20** | 20 / 20 |
| aborted on a fatal include | 8 / 42 | 0 / 20 | 0 / 20 | 0 / 20 |
| Python blocks that parse | 5 / 26 | **18 / 18** | **18 / 18** | 18 / 18 |
| `#include` lines malformed | 49 / 154 (32 %) | **0 / 102 (0 %)** | **0 / 103 (0 %)** | 0 / 112 (0 %) |
| `import` lines malformed | 4 / 57 (7 %) | **0 / 32 (0 %)** | **0 / 31 (0 %)** | 0 / 44 (0 %) |
| C++ errors per 100 lines | 42.2 | 0.0 | 0.0 | 0.0 |

All four arms are 40 of 40 cases. Both reference arms finished `stop` on every case and were served by
exactly one provider — `{'Relace': 40}` and `{'DeepInfra': 40}` — which is what `--provider` is for. The two
seeds produced twenty distinct replies per arm, so they are two genuine samples per task.

**Greedy, on the same six tasks section 7.4.5 used, at temperature 0:**

| | Cachalot greedy | run A `relace/fp4` greedy |
|---|---:|---:|
| C++ blocks that compile | 0 / 6 | **6 / 6** |
| aborted on a fatal include | 4 / 6 | 0 / 6 |
| `#include` lines malformed | 1980 / 2004 (99 %) | **0 / 34 (0 %)** |

**What this closes.** On RUN.md's own decision table this is "reference clean, Cachalot corrupt", and three
competing explanations are now dead:

- **Not FP4.** The `relace/fp4` arm is FP4 and is clean at 0 %. The checkpoint's `expert_dtype: fp4` is not
  the cause, and every sentence in this document that treated FP4 as an accepted quality cost is wrong.
- **Not the harness.** Run A has no harness at all. Run B's clean result was never the harness repairing
  anything: every assistant turn of all 40 Hermes sessions was scanned, including the first draft before any
  tool ran, and **zero malformed include lines appear anywhere**. Nine of the forty answered in one shot with
  no tool call. The harness had nothing to fix.
- **Not sampling.** Greedy on both sides keeps the gap at 99 % against 0 %, which section 7.4.5 had already
  established one-sidedly.

**What is left.** The defect is in Cachalot's own path: expert streaming, the FP4 expert kernel, the
prefetch and cache machinery, or detokenization. Its shape is a single-token drop at a high-confidence
position — 38 of 49 malformed lines in the sampled run and 1977 of 1980 in the greedy run are "missing
opening delimiter". That is not a quantization blur and it is not a model tendency. It is a bug.

**What this arm does not prove.** Neither Relace nor DeepInfra publishes whether its weights are the
official checkpoint byte for byte, so this is not a byte-identical pairing. It does not need to be: a 32 %
to 0 % gap on one dropped delimiter is not a weights difference.

**Reproducing the tables.**

```bash
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py \
  benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/ \
  ~/cachalot-runA-relace-fp4-scored ~/cachalot-runA-deepinfra-fp8-scored ~/cachalot-runB-hermes-scored
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/include_integrity.py \
  benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/ \
  ~/cachalot-runA-relace-fp4-scored ~/cachalot-runA-deepinfra-fp8-scored ~/cachalot-runB-hermes-scored
```

Each `-scored` directory carries `manifest.json` and `rows.json` so `code_validity.py` treats it as one arm
and honours `expect_compiles`, which keeps `cpp-string-split-snippet` — a task that forbids includes and
`main` — in the snippets column rather than counting it as a failed compile.

**Pinning a provider is fiddly, and the obvious choices do not work.** With `allow_fallbacks: false`,
OpenRouter returns `404 No endpoints found` if the pinned provider does not support every parameter in the
request. The first-party `deepseek` endpoint does not list `seed` and 404s. `sail-research/fp4` lists every
parameter and returned HTTP 429 on every attempt through five retries. `relace/fp4` also omits `seed` from
`supported_parameters` but accepts and serves the request; it is what the FP4 arm used, and the seed is
presumably ignored, which is why the two seeds were checked for being distinct.

**Defects in the returned Hermes run, all corrected before scoring.** Two of forty sessions were never
exported — `cpp-matrix-transpose` pass 1 and `cpp-ini-parser` pass 1 — and were recovered from
`~/.hermes/profiles/test-ds4-1/state.db`. Unrecovered, the denominator would have read 38 and the arm would
have looked complete, which is the third time a short run has nearly put a wrong number in this document.
Two files also carried the wrong task id, and the whole set was named `<task>-seed<n>.json` where the scorer
splits on `_seed` and reads `.txt`.

### 7.4.7 The defect is in-context copying, and six more suspects are cleared

**Measured 2026-09-20, second session. Raw tables and procedure in `HANDOFF-2026-09-20-quality.md`.**

Section 7.4.6 put the fault inside this runtime. This section narrows it, and retires the framing that the
next-session prompt v13 was built on.

**What the defect actually is.** Teacher-forced probes separate predictions that are open-ended from
predictions that are not. Choosing which header a file includes is fair to be uncertain about. Finishing a
word already begun is not, and copying a word the text spelled out ten tokens ago is not remotely.

| class | n | rank 1 | mean logprob | worst rank |
|---|---:|---:|---:|---:|
| continuation, first occurrence | 28 | 86 % | -0.699 | 19 |
| continuation, **repeat of an earlier word** | 101 | **51 %** | -3.578 | **23,989** |
| everything else | 1,371 | 88 % | -0.595 | 3,703 |

First-occurrence continuation is healthy and level with the baseline. Only repeats collapse. The model is
not failing to retrieve, it is retrieving the wrong thing: at `...the file lists allocator_t` it prefers
`' unordered'` and `' iterator'`, words from neighbouring sentences. This is a within-run control and needs
no reference arm — no correct model fails to copy a word it emitted ten tokens ago.

That one statement covers every artefact the project has collected, with no separate story per case:
`#include <stdex>` and `#include iostream>`, `LRUCache` written `LRcache`, `map_._.end()` with a duplicated
`_.`, `std std::string`, `utfutf-8`, `csv.Dreader`, `__name __`. Drops and duplications are both what
mis-retrieval looks like.

**What is cleared, and how.**

| suspect | verdict | evidence |
|---|---|---|
| detokenization | cleared | 34/34 `#include <x>` lines exact: batch, per line, and one token at a time |
| the sampler, again | cleared | greedy manifest records `temperature 0.0`, `frequency_penalty 0.0`; that path is plain `argmax` |
| the prefill path | cleared | 540 tokens batched vs fed one at a time: 60 % vs 65 %, within noise |
| depth and position | cleared | byte-identical block at token 568 scores 100 %; no decay bucketing 640 positions |
| **routed-expert kernel, streaming, cache** | **cleared** | `nll_expert_precision.py` dense fp32 vs production on the same FP4 weights: mean +0.0255 ± 0.0141 nats, **median +0.0000**, 96.0 % greedy agreement, same argmax at every failure |
| **Engram** | **cleared** | nine checks against the shipped reference; prime sums 384,006,168 and 384,016,682 match `engram_num_embeddings` exactly; gate math and `norm_eps` identical |
| fused attention decode, fused FP8, bf16 head | cleared | all three off together: repeat continuation 51 % against 51 %, unchanged |
| copy distance | not the variable | controlled probe, same identifiers, recall at 8/32/96/288 tokens: 73 %, 79 %, 62 %, 75 % against an 88 % run baseline |

**What this does to section 9 and to v13's plan.** Items 1 to 4 of the v13 bisection — the custom Metal
kernels and the prediction and prefetch family — are answered without running the corpus once. The
10-to-20-hour slow-path run that v13 made Job 2 is not the right next move; it was designed to bisect
optimizations, and the optimizations are not the fault.

**What is left, stated without overreach.** The trunk. Precise retrieval of one earlier position is what
attention governs, and V4.1's attention is not plain attention: `index_source_layer_ids = [2, 8, 14, 20, 24,
28, 32, 36]`, `kv_source_layer_ids = [2, 8, 14, 20]`, `candidate_source_layer_id = 20`, `index_topk = 512`,
`sliding_window = 128`, `compress_ratios` of 2 for layers 2-19 and 1 for 20-39. An `Indexer` scores
compressed positions and keeps 512 of them, behind a two-level candidate pre-filter. A selection that is
approximately right leaves fluency intact and breaks exact copying, which is the measured shape — but the
flat distance curve argues against the simplest version of that, because `index_topk` should bite at long
range rather than uniformly. **Which component causes it is not established and should not be asserted.**

**Update, later the same day: that job was done, and then the attention was measured.** The decode path
matches the reference at about forty points — selection, arithmetic, the KV it attends over, its
quantization, the MoE gate, Engram, and the hyper-connection machinery. And at a failing copy,
`benchmarks/attention_mass_probe.py` shows layer 3 putting **0.928 on position 321, the exact token the
model must emit**, alongside 0.845 on the `'_t'` that matches the current one — an induction circuit firing
correctly — while the model ranks that token **17,935th**. Two tokens earlier, at position 331, the same
machinery copies correctly at rank 0.

So retrieval is not the fault; something between the attention output and the logits discards what was
retrieved, at some positions and not others. It is not a static break downstream, not token identity, and
not the cache contents. `docs/HANDOFF-2026-09-20-quality.md` carries the tables and the next instrument.

**The superseded note, kept because it explains the method that worked.** `src/cachalot/model/attention_compressed.py` is 34 KB implementing
that scheme and had never been read beside `inference/model.py`. Engram was cleared by exactly that method
in under an hour. Already compared and matching: `get_window_topk_idxs` in prefill and decode, and the whole
Engram path. Uncompared: the indexer's scoring and top-k, the compressor's partial-group state, the
candidate pre-filter, and the RoPE positions the compressed latents are rotated with — the reference notes a
latent stands for the first token of its group at position `j * ratio`, with a decode-time index of
`start_pos + 1 - ratio`.

### 7.4.8 Root cause: the hyper-connection residual mix was transposed

**Found and fixed 2026-09-20. Tables in `HANDOFF-2026-09-20-quality.md`; the fix is commit `e37b73c`.**

`Block.hc_post` writes a sub-layer's output back into the four residual streams and mixes the incoming
streams through `comb`. The official implementation, `model.py:962`:

```python
torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
```

broadcasts to `elem[i, j, :] = comb[i, j] * residual[i, :]` and sums over `i`, so output stream `j` is
`sum_i comb[i, j] * residual[i]`. **comb is contracted over its first index.** Both of this runtime's
implementations contracted the second index — `comb @ residual` rather than `comb.T @ residual` — and did so
identically, the MLX path by summing the wrong axis and the Metal kernel by indexing `comb[h * hc + j]`
where it needed `comb[j * hc + h]`.

Verified numerically against explicit loops transcribed from `model.py`: ours matched `comb @ residual` to
6e-8 and differed from the reference by 2.0. No compensating transpose exists anywhere — `comb` is built as
`mixes[2*hc + j*hc + k]` with `j` the row in all three files that touch it, and the sinkhorn normalizes rows
then columns in that layout.

**The gate, 40 of 40 cases** — `benchmarks/results/coding/hcfix/`, scored against the reference arm:

| | corrupt | **fixed** | reference |
|---|---:|---:|---:|
| C++ blocks that compile | 0/42 | **20/20** | 20/20 |
| aborted on a fatal include | 8/42 | **0/20** | 0/20 |
| Python blocks that parse | 5/26 | **18/18** | 18/18 |
| `#include` lines malformed | 49/154 (32 %) | **0/101 (0 %)** | 0/102 (0 %) |
| `import` lines malformed | 4/57 (7 %) | **0/31 (0 %)** | 0/32 (0 %) |
| C++ errors per 100 lines | 42.2 | **0.0** | 0.0 |

Throughput was 2.1-2.5 tok/s during the gate, unchanged: the fix reorders a contraction and costs nothing.

**Result of the fix on the probe**, same 1,500 positions:

| | before | after |
|---|---:|---:|
| copying a word already spelled out | 51 % | **99 %** |
| its mean logprob | -3.578 | **-0.027** |
| its worst rank | 23,989 | **2** |
| everything else | 88 % | **95 %** |
| all positions | 85 % | **95 %** |
| the token after `#include` | 84 % | **100 %** |

**Why it survived months of work, which is the part worth remembering.** `comb` comes out of a sinkhorn
normalization and is close to doubly stochastic, so every residual stream receives about the right *total*
weight whichever way the matrix is applied. The model stayed fluent: 2.4 perplexity, 88 % top-1, prose and
code that read correctly. A transpose does not change how much information flows, it changes **which stream
it lands in**, and that only matters for a prediction depending on one specific earlier fact. Copying is
that prediction. Every aggregate metric this project owned — NLL, perplexity, top-1, hit rate — was blind to
it by construction.

It also defeated this project's own screens twice. The fused-decode screen cleared "the attention kernel"
because both implementations carried the same error, so switching between them moved nothing. And the
dense-versus-runtime expert comparison cleared expert arithmetic while leaving the shared code untouched.
A/B switching can only find a defect that differs between the arms.

**What this retires.** Every conclusion in this document that attributed quality loss to FP4, to the expert
bank, or to a speed optimization. Sections 9.0 and 9.3 chased quality through quantization against a defect
that was never in the weights, and section 7.4.6's "the defect is in this runtime" was right for a reason
nobody had guessed.

### 7.5 The 3-bit bank's gate, 2026-09-18

Three metrics, all against the 2-bit g128 bank it replaces, each on the protocol that metric was defined for.

| gate | 2-bit g128 | **3-bit g64** | how it was read |
|---|---:|---:|---|
| top-1, production path, 512 tokens | 44.5 % | **49.8 %** | +5.3 points |
| paired NLL against the 2-bit bank | — | median **−0.0208**, better on 58.4 % | **sign z +3.80** — the mean, at −0.0443 +- 0.0457, is z −0.97 and says nothing |
| free-running collapse rate, penalty on | 1/8 (12 %) | 1/8 (12 %) | **unchanged, and expected** |
| ~~syntax errors per 100 generated lines~~ | ~~22.7~~ | ~~**4.9**~~ | **withdrawn 2026-09-19** — the checker read a grep, not the compiler's exit status. Section 7.4.1. |
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

### 7.6 The 2-bit bank's gate, re-run through the fixed runtime, 2026-09-20

**The 2-bit bank is not worse than FP4. It never was, and three sessions concluded otherwise because the
transposed residual mix was in every arm.** The 40-case corpus, generated on the 2-bit g128 bank at the same
budget, corpus, sampling and seeds as the FP4 arm in section 7.4.8, scored against the same hosted reference:

| 40-case corpus | FP4, fixed | **2-bit g128, fixed** | hosted reference |
|---|---:|---:|---:|
| C++ blocks that compile | 20/20 | **20/20** | 20/20 |
| aborted on a fatal include | 0/20 | **0/20** | 0/20 |
| Python blocks that parse | 18/18 | **18/18** | 18/18 |
| `#include` lines malformed | 0/101 (0 %) | **0/94 (0 %)** | 0/102 (0 %) |
| `import` lines malformed | 0/31 (0 %) | **0/29 (0 %)** | 0/32 (0 %) |
| C++ errors per 100 lines | 0.0 | **0.0** | 0.0 |
| decode during the run, median of 40 | 2.28 tok/s | **4.47 tok/s** | — |

Every column is equal. The one C++ case that produces diagnostics, `cpp-string-split-snippet`, produces them
in all three arms including the hosted reference — it is a snippet the corpus asks for without headers, and
`code_validity.py` counts it as a snippet rather than a failure in every arm.

`benchmarks/results/coding/q2g128-v17/`, 40 of 40 cases, `code_validity.py` and `include_integrity.py`.

**The screen that justified spending the two hours.** `benchmarks/continuation_rank.py` on the recovered
repeat-identifier probe, the same 1,500 positions on both banks:

| bank | continuation, first | continuation, **repeat** | repeat worst rank | everything else |
|---|---:|---:|---:|---:|
| FP4 | 86 % | 99 % | 2 | 95 % |
| 2-bit g128 | 86 % | **99 %** | **1** | 95 % |

Indistinguishable, and the 2-bit bank is marginally better on the worst case. That is a screen and it was
read as one: it decided whether the corpus run was worth starting, and the corpus run is the gate.

**What this retires.** Section 7.5's "+5.3 points of top-1 and the price is 6.3-7.1 against 4.8-5.0 tok/s"
ranked the 3-bit bank above the 2-bit one on a metric that section 7.4.8 proved blind to the defect that was
actually producing the artefacts. Section 9.3's "recover the quality the 2-bit bank cost" was aimed at a cost
that, on the gate the project now trusts, is zero. Section 8's conclusion that quality lives in the weights
stands only for the NLL numbers it was computed from, and those numbers were never the thing a user sees.

**What it does not retire.** The paired NLL and top-1 differences between banks are real measurements and they
have not changed; what changed is that they do not predict the gate. The 2-bit bank still scores 44.5 % top-1
against the 3-bit bank's 49.8 % (section 7.3), and it still compiles 20 of 20. Both are true, and the second
is the one the standing decision is about.

**Two conditions on this result, stated so nobody quotes it past them.** The replies are slightly shorter on
the 2-bit bank — 1,423 C++ lines against 1,447, 1,128 Python lines against 1,246, 94 include lines against
101 — which the gate does not penalize and which nothing here explains. And this is one corpus at one
sampling setting; the collapse-rate question (section 9.9) is measured separately and was not re-measured
here.

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
produced no compiling C++ block and collapsed on 2 of 6 long turns against the retired bank's own record.
(The "31.6 against 19.6" this paragraph carried is withdrawn — section 7.4.1.) **Output error is not a proxy
for usable output at this resolution, however it is measured.** Use the screen to decide which formats
deserve a bank; never to predict what a bank will do.

### 8.4 Cost of a gate run

Each gate arm re-quantizes about 20,000 experts, not 15,360, because the 8 GiB packed cache evicts. At
`mx.quantize` speed (19 ms per expert) an arm is 540 s; with a 13x13 search grid (720 ms per expert) the first
attempt was still running when the guard's 5400 s timeout killed it. The 5x5 grid costs 125 ms per expert and
about 2200 s per arm, and buys 0.0005 of relative weight error against the finer grid. If a future arm needs a
better fit, raise the timeout or build the bank and gate it with `--experts runtime`; do not raise the grid.

---

## 9. Open levers, ranked

> **Frozen 2026-09-20, unfrozen the same day.** The freeze existed because the runtime's output did not
> match a reference at any speed. Section 7.4.8 found the cause — a transposed hyper-connection mix, not a
> speed lever — and the 40-case gate is now clean at 20/20 compiling and 0/101 malformed, equal to the
> reference. **The shipped optimizations are cleared: none of them was the defect, and the fix cost no
> throughput.** This is a work queue again.
>
> Two carried conditions. The quality arguments used to open or close levers below were measured through the
> defect and several of them ranked quantization formats, so they are provisional until re-established —
> section 2. And a lever's quality check must use the repeat-copy number rather than NLL or top-1, which
> stayed at 2.4 perplexity and 88 % while the runtime could not copy a word it had just written.
>
> **The first of those conditions is discharged for the bank question, 2026-09-20, and the answer inverts
> this section's premise.** The 2-bit g128 bank was re-gated on the 40-case corpus through the fixed runtime
> and is equal to FP4 and to the hosted reference on every column, at nearly twice the throughput
> (section 7.6). The whole ranking below was computed on a token that reads 1,858 MiB; the cheaper bank reads
> 804 MiB for the same gate result. **Re-measure the anatomy before trusting any ordering here** — section
> 9's own lesson is that a lever's rank is a property of the bank.

Ranked by expected value per unit of work, with the evidence, the cost and — most importantly — the
measurement that decides each one before any code is written.

**The ranking changed again on 2026-09-19, and this time because the bank in use was finally measured.**
Every timing number behind the 2026-09-18 ranking came from the 2-bit bank. On FP4 the token is 341 ms, of
which about 320 ms is drive time and 84.6 ms is compute that hides underneath it (section 6.1). That
demotes dispatch count and closes speculation:

**Re-ranked on 2026-09-21 against a measured token rather than an isolated profile.** The table below is
kept because its FP4 reasoning is still the record of what was tried; the ranking that applies to the bank
that ships is this one:

| lever | state on the 2-bit bank, 2026-09-21 | measured size |
|---|---|---|
| 18, what routing prediction costs the decode thread | **measured, and it is the largest named item left**: 7 ms of CPU and 4 of GPU, split evenly between computing a prediction and submitting it. Not a compile candidate | 11 ms per token |
| 16, tracing the graph with `mx.compile` | **the MoE block shipped**, 5-8 ms per token; **the hyper-connection glue and the router are screened and closed** at 0.2 and 0.3 ms; attention untried | under 1 ms outside attention |
| 17, memoising the kernels' scalar parameters | **shipped**, 1.2-1.3 ms of CPU, NLL identical | 1.2 ms per token |
| 15, the CPU third | **open, a third taken** — graph building with the GPU idle | 20.7 ms per token left |
| the unprofiled half of the eval window | **measured and closed as a suspect, 2026-09-21** — the eight source and index-source layers cost 4.5 ms more than eight plain ones; the time is spread evenly across all forty | section 6.3.1 |
| 12, prefetch precision | open, needs a different signal; the drive is now idle 43 % of decode so its price has fallen with its prize | 51.2 ms per token of coverage-blocked time |
| 2, dispatch count | **closed on the arithmetic**: the per-dispatch floor is 4.72 us | ~1.1 ms per token |
| a fused affine expert kernel | **screened and closed** | at most 3 ms per token |
| 8, below 9.49 MiB per expert | closed on the arithmetic and the quality | — |
| 11, mirror striping | closed on this bank, at any fraction | an 18 % loss |
| 1, DSpark speculation | closed | — |
| 0, 3, 4, 10 | closed | — |
| 5, 9, 13 | shipped | — |

| lever | state on FP4 |
|---|---|
| 11, mirror striping across both drives | **shipped 2026-09-19**: −5 % decode, −7 % cold prefill, no quality change |
| 12, prefetch precision | **bounded and mostly closed**: the missing 28.5 % is the router's selection boundary, and no cheap re-use of the stale scores beats plain top-k |
| 13, predicted-load lifetime | **fixed and shipped 2026-09-19**: a completed prediction for a later layer was released before it could be used. A null on decode at lead 1, as expected, and it made 9.12.1 interpretable |
| 12.1, non-cumulative lookahead | **closed 2026-09-19**: 2.0 % worse, ranges non-overlapping — but it prices the prefetch-timing term at about 24 ms per token, which is now the case for a better predictor |
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

**The state after 2026-09-19 is that nothing cheap is left, and this time the cheap things were looked for.**
The review found one real defect -- predictions with no lifetime -- and it is fixed, tested and a null on
decode, which is what it should be. The lever it unblocked was measured the same day and lost by 2.0 %. What
that experiment bought instead is a **price**: the prefetch-timing term is worth about 24 ms per token, 7 %
of decode, to anything that can reach two layers ahead without losing recall (section 9.12.1). That is the
first number section 9.12's "different signal" has ever had attached to it, and it is the argument for
spending a session on a predictor rather than on a knob. Mirror striping is taken. Speculation,
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

| arm | C++ blocks | avg block | lines | **compile** | aborted on a fatal | truncated |
|---|---:|---:|---:|---:|---:|---:|
| **FP4** | 4 | 115 | 461 | **0 of 4** | 3 of 4 | 2 |
| **3-bit searched + weighted** | 3 | 138 | 414 | **0 of 3** | 1 of 3 | 3 |

Re-scored 2026-09-19 with the repaired gate. The "1/4 clean, 8.9 errors per 100 lines against 31.6" this
table carried is withdrawn: the checker never read clang's exit status, and the density was largely measuring
which arm aborted on a mangled `#include` first. **Neither bank compiles anything, and the C++ blocks cannot
separate them.** Section 7.4.1. What closes the lever is the row below.

Free-running collapse, penalty on: **2 of 9 replies (22 %)** against FP4's **0 of 9**, which is **2 of 6**
long turns against none — the short first turn of each seed never collapses on either bank. Three seeds is
below the four this document asks for a rate (rule 5), so read the long-turn count rather than the
percentage; what is not in doubt is that FP4 produced six full 1400-token replies and the 3-bit bank produced
four, having run two of them into a loop at 279 and 129 tokens.

Block lengths are comparable — 138 against 115 — so this is not the block-composition artefact that withdrew
the "4.6x" earlier the same day. Python parse rates, re-scored 2026-09-19: FP4 **2 of 10**, the 3-bit bank
**0 of 5**. (The "2.2 on both arms" this paragraph carried is withdrawn — section 7.4.1.)

Section 9.0 set its own closing condition before the bank was built: *if it lands near 19.6, bits rather than
fit are binding and this lever closes for good.* **That condition was written in terms of a number that does
not mean anything**, and it cannot be evaluated as written. The lever closes anyway, on the evidence that did
survive: the bank produced no compiling code, parsed 0 of 5 Python blocks against FP4's 2 of 10, and ran 2 of
its 6 long turns into a repetition loop where FP4 ran none. **Closed.**

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

### 9.2 Lever 2 — Dispatch count — **closed on the arithmetic, 2026-09-21**

> **Closed, and for a better reason than the one that demoted it.** It was demoted on FP4 because compute hid
> under the drive; bytes then came down 57 % and the premise expired, which is what reopened it. Measured
> rather than assumed, **the per-dispatch floor is 4.72 microseconds** — a trivial Metal kernel chained 500
> times in one graph, build and GPU together. A decode token issues roughly 230 substantial dispatches, so
> **the whole prize is about 1.1 ms of a 90 ms token**, and the hyper-connection triple this section proposed
> fusing is 4.6 ms in total rather than the 68.7 ms it was ranked on (section 6.3). Two fusion screens run
> the same day agree: splitting `hc_mixes` across threadgroups returns about 1 ms per token, and a fused
> multi-expert affine path is worth 20-25 % of the 13.9 ms routed-expert term at best (section 11).
> **Do not spend days here.** The equivalent time in the CPU third (section 9.15) is worth an order of
> magnitude more. **Prefill remains unprofiled at the layer level** and is a separate question.

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

### 9.3 Lever 3 — Recover the quality the 2-bit bank cost — **closed 2026-09-20: there was no cost to recover**

> **Closed by section 7.6.** This lever existed to buy back "+0.019 nats and 6.3 points of top-1 against the
> 3-bit bank, visible as dropped and mangled tokens in output". The dropped and mangled tokens were the
> transposed residual mix, not the bank: re-gated through the fixed runtime the 2-bit bank compiles 20 of 20
> C++ blocks and malforms 0 of 94 include lines, equal to FP4 and to the hosted reference. The NLL and top-1
> differences below are real and unchanged; what is retired is the belief that they predicted anything a user
> sees. Everything after this line is kept as written, because the null it records about searched fits is
> still a null and still cost a bank build to establish.


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


**Re-simulated at the expert size that ships, 2026-09-21.** Section 9.4 closed a larger budget on a
simulation run at a different expert size. `simulate_policies.py` on `trace_routing_v7` at
`--expert-bytes 9953280` (9.49 MiB, the 2-bit g128 bank), first-come prefill, LRU decode:

| budget | prefill hit | **decode hit** | overall | SSD GiB over the trace |
|---:|---:|---:|---:|---:|
| 36 | 22.8 % | 81.9 % | 49.0 % | 409 |
| **44 (ships)** | 28.6 % | **85.1 %** | 53.7 % | 372 |
| 52 | 31.9 % | **87.7 %** | 56.7 % | 347 |
| 60 | 35.4 % | **90.0 %** | 59.6 % | 324 |

**44 to 52 GiB is worth 2.6 points of decode hit rate and 52 to 60 another 2.3**, which is the same shape
section 9.4 saw and now on the size that ships. At the shipped budget the anatomy measures 46.4 ms per
token of blocking on uncovered misses at an 83.5 % hit rate, so 2.6 points is roughly 6 fewer misses and
5-7 ms per token -- about 4 %, not the kind of number a chat session resolves, but the only lever on this
list that needs no code.

**It is a live-session experiment, not a benchmark one.** `guarded_run.sh` needs `52 + 29 = 81` GiB
available, which this machine has not had; a chat session peaks at 55.1-55.6 GiB against a 72 GiB wired
limit and has 12 GiB of headroom it never uses. The command is
`CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52`, and the number to read afterwards is
the session hit rate against 90.0 %.

**Run, 2026-09-21, and the simulation was right.** The live session hit rate went from 90.00 % to
**92.37 %** against a predicted +2.6, with 5,314 residents against 4,495 and an MLX peak of 67.74 GiB
against the 77.8 the flag wired — so **52 GiB fits on this machine** with no pressure event. This lever is
now shipped by hand: it needs the two flags on the command line, not a code change. Section 7.2.6.

**What is left of it.** The same table says 52 → 60 GiB buys another 2.3 points, which would wire about
85 GiB of 96 and is the configuration class that panicked this machine twice (section 3). Do not try it
without watching `kern.memorystatus_vm_pressure_level` the whole way.

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

### 9.8 Lever 8 — Below 9.49 MiB per expert — **closed on the arithmetic and on the quality, 2026-09-21**

**Asked directly: can we try 1 bit?** Measured before answering, because the last three sessions of this
project were spent on quantization questions that turned out to be about a transposed matrix.

**MLX cannot express it.** `mx.quantize` refuses: *"The requested number of bits 1 is not supported. The
supported bits are 2, 3, 4"*. A 1-bit bank therefore needs a custom Metal kernel for the fit *and* for the
matmul, which is a larger piece of work than the 2-bit bank was.

**And the weights do not survive it.** 24 real FP4 experts, group 128, the same min/max affine fit applied at
each width, 8 probes each, scored on the expert's own SwiGLU output:

| bits | MiB/expert | mean weight error | mean output error | vs 2-bit |
|---:|---:|---:|---:|---:|
| **1** | 4.75 | 2.1896 | **16.1498** | **15.4x** |
| 2 | 9.49 | 0.5330 | 1.0496 | 1.00x |
| 3 | 14.23 | 0.2116 | 0.3525 | 0.34x |

A *relative* output error of 16 is not a degraded expert, it is noise: the answer is sixteen times the size of
the thing being approximated. Section 8.3's warning that this screen ranks wrongly across quantizers does not
rescue it — that caveat covers tens of per cent, not 15x — and no better fit recovers it either, because at
one bit a group of 128 weights has two levels and there is nothing left to tune.

**Even if it worked, the arithmetic no longer wants it.** This lever was written when bytes were the
constraint. They are not any more. In the 2026-09-21 session the model ran at 7.62 tok/s, a 131 ms token,
against a 93.0 ms all-resident compute floor for this bank — so **at most 38 ms of the token is expert
streaming and everything else around it**, and halving the bytes again could not win more than about 19 ms.
That is roughly 16 %, bought with custom Metal kernels, against a 15x quality cliff.

**Where the same effort goes instead, ranked by what section 6.2 and section 7.2.2 actually measured.**

1. **Compute, which is now 69 % of the token** (section 6.2). The hyper-connection machinery is the largest
   block in `profile_decode_components.py`: `hc_mixes` 27.8 ms, `hc_pre + rms_norm` 22.5 ms, `hc_post`
   18.4 ms. Three kernels, one of which was rewritten last session and is now correct, and none of which has
   been optimized since it was written.
2. **Prefetch precision, at 32.6 % in the live session** — 32,808 wasted loads of 48,706, about 311 GB of the
   session's 716.8 GB. Section 9.12 priced better timing at 24 ms per token on FP4; the term is smaller here
   but the waste fraction is identical, and it is bytes the drive moves for nothing.
3. **Lever 2, dispatch count**, whose demotion said "84.6 ms is already hidden" and is no longer true.

**What stays true from the original entry.** 9.49 MiB at 2 bits and group 128 is the floor for any format the
existing MLX kernels can read, and going below it means a custom encoding. That is still correct. What has
changed is that it is no longer worth wanting.

### 9.9 Repetition collapse — **the defect, not the sampler. Closed 2026-09-20.**

> **The collapse was the transposed residual mix.** Re-measured through the fixed runtime on the protocol
> this section was written from — `repetition_quality.py`, 4 seeds, 900-token cap, short prompts — with the
> frequency penalty **off**:
>
> | arm | bank | penalty | collapses |
> |---|---|---|---:|
> | 2026-09-17, corrupt runtime | q2g128 | off | **5 / 8 (62 %)** |
> | 2026-09-17, corrupt runtime | q2g128 | 0.2 / 128 | 1 / 8 (12 %) |
> | **2026-09-20, fixed runtime** | **q2g128** | **off** | **0 / 12 (0 %)** |
> | **2026-09-20, fixed runtime** | FP4 | off | **0 / 12 (0 %)** |
>
> Against the original's own denominator — turns 2 and 3, the long replies — this is **0 of 8 against 5 of
> 8**, same bank, same seeds, same cap: two-sided Fisher exact **p = 0.026**. The FP4 arm is the control that
> says it is not a bank effect, which is what this section always claimed and now has both directions of.
> The worst repeat across all 24 replies was 7 tokens against a 24-token collapse threshold.
>
> **This was predictable from section 7.4.8 and nobody predicted it.** A repetition loop is a model that
> cannot tell it has already written something. `hc_post` applied the hyper-connection mix transposed, which
> is precisely a failure to read back what the recent context holds — the same defect that put a word the
> text had spelled out ten tokens earlier at rank 23,989. The sampler was treating the symptom.
>
> **What comes off.** `--frequency-penalty 0.2 --penalty-window 128` is removed from section 4's command and
> from the HTTP server defaults. It distorts code that legitimately repeats, which is most code, and the
> number that bought that distortion no longer exists.
>
> **What this does not establish.** 0 of 12 is not a proof of zero: the 95 % upper bound on the rate is
> 22 %, and P(0 of 12) is 0.216 even if the true rate were still the penalty-on 12 %. What is excluded, at
> P = 9.1e-6, is 62 %. If a loop is ever seen again, the knobs are still there and the CLI still exposes all
> four; the claim here is that nothing measurable justifies paying for them by default.

The original section, kept because its diagnosis of the *shape* was right and only its cure was wrong:

Free generation on code prompts falls into a repeating loop. It is **not** a quantization failure -- FP4 does
it too -- and it is not the prefix cache, which was tested and cleared. It tracks **generating a long reply
from a short prompt**: 62 % of replies collapse in that shape, against 0 of 7 when the same turn is generated
from about 1,700 tokens of context. A collapsed turn then poisons the next one.

`frequency_penalty` was adopted as the fix, because it grows with the count; a classic repetition penalty
fires once per unique token and a confident loop rides straight over it. At 0.2 with a 128-token window the
rate fell from 62 % to 12 %. The survivor had period 14, which only puts each of its tokens in the window
about nine times.

Note that "it tracks generating a long reply from a short prompt" and "0 of 7 from 1,700 tokens of context"
is, read through section 7.4.8, a description of the defect rather than of sampling: a long context gives the
model many more routes to the fact it is trying to recall, so a transposed mix hurts it less.

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

**Precision has now been read from seven live sessions and it does not move.** 32.5 %, 32.7 % (section
7.2.3), 33.15 % and 33.09 % (section 7.2.4), **33.58 % and 33.38 %** (section 7.2.5), against the 39.6 %
this section opened with, across two banks, four runtime changes and two expert sizes. **The waste share
has moved, and upwards**: the two sessions of section 7.2.5 close their byte accounting to the byte at
**46.9 %** of every byte read, not the 42.7 % above. **Two thirds of every predicted load is wasted, every
time.** A number that stable is either a property of the router's own uncertainty one layer ahead — which
`benchmarks/predictor_recall.py` says offline — or a property of the width, and section 9.18 now prices
what that waste costs the decode thread rather than only the drive.

**Keep the general lesson, and note it cuts both ways:** a large waste figure is not a lever unless the
wasted resource is the binding one — and when the bank changes, check whether it has become binding before
reusing the conclusion's reasoning for anything else.

### 9.11 Lever 11 — Mirror striping across both drives — **shipped, 2026-09-19**

**What.** The FP4 experts live on the internal SSD; the X10Pro holds the checkpoint they were copied from,
byte for byte, under the same shard names. `reader.py` has carried a mirror path since 2026-09-16: the tail
`CACHALOT_MIRROR_FRACTION` of every expert read is issued to a second drive concurrently, so one expert lands
sooner than either drive alone could deliver it.

> **This is a re-measurement, not a discovery, and the 2026-09-19 session initially wrote it up as one.**
> `HANDOFF-2026-09-16.md` §7.3 is precise about why the mirror was turned off, and section 11 of this document
> compressed it into "harmful; leave it off", which is what misled a later reader. The real finding was:
> **with the 3-bit stacked bank** decode fell from 3.22 to 2.74 tok/s, and the cause was a code interaction
> rather than bandwidth — `ExpertReader._read_pieces_concurrently` is skipped whenever a mirror is configured,
> so that bank's nine pieces per expert fell back to serial reads with USB tail latency on top. **An FP4
> expert is one contiguous range**, so it never takes that path and the interaction cannot bite. 09-16 also
> measured the positive case (10 % striping: 2.86 to 3.00 tok/s, 512-token prefill 32 to 28.5 s) and derived
> the optimum as `bandwidth_of_second_drive / total`, noting that past about 15 % the USB becomes the
> bottleneck. Everything below reproduces that on the current configuration and ships it.
>
> **The lesson is about this document rather than about storage:** a null compressed to its verdict loses the
> condition that made it true. §7.3 said "with the 3-bit bank" and "it could be fixed"; the summary said
> "harmful". Keep the condition in the null.

Decode is drive-bound on FP4 (section 6.1) and the gain is a cliff rather than a plateau. `decode_anatomy.py`,
36 GiB budget, 512-token prompt, four runs per arm interleaved in both directions with `settle.sh` between:

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

### 9.11.1 Mirror striping is a property of the expert size, and it is a **loss** on the 2-bit bank

**Measured 2026-09-20, and it reverses section 9.11 for the bank now in use.** `chat_turns.py` at a 44 GiB
budget with the hotlist, 2-bit g128, five runs interleaved with `settle.sh` between each:

| turn | reply | **no mirror x2** | mirror 0.10 x3 |
|---|---|---:|---:|
| 1, cold | 9 tok | **4.51, 4.36** | 3.28, 3.22, 3.25 |
| 2 | 9 tok | **7.40, 7.28** | 5.69, 5.63, 5.34 |
| 3, the story | 127 tok | **7.03, 6.88** | 5.69, 5.64, 5.75 |
| 4, haiku | 20 tok | **6.97, 6.58** | 5.50, 5.49, 5.51 |
| 5 | 44 tok | **6.21, 6.19** | 4.86, 4.82, 4.73 |
| 6 | 11 tok | **4.72, 4.54** | 3.38, 3.31, 3.40 |

On the story turn that is **6.96 mean against 5.69, an 18.1 % cost, with the two ranges not overlapping** —
spreads of 0.15 and 0.11 tok/s against a gap of 1.19. Every other turn separates the same way. Counting the
fraction sweep below, the mirror has now been measured five times across two budgets and three fractions and
has landed between 5.64 and 5.75 every time, against 6.87 to 7.03 without it.

The comparison is unusually clean: all five runs miss the same experts — 2,726 on turn 3, varying by at most
one — and end with the same 4,746 resident, so nothing about caching, routing or prediction differs between
the arms. The whole effect is in the read path.

**Why, and it is the same arithmetic section 9.11 used, run at the new expert size.** Striping issues the
tail `mirror_fraction` of every expert read to the second drive concurrently with the head. That wins when
the head is long enough for the two transfers to overlap the USB drive's much worse latency. On FP4 the tail
is 10 % of 17.93 MiB, 1.79 MiB, against a critical-path read of 9.37 ms — and it cut that to 8.10 ms. On the
2-bit bank the whole expert is 9.49 MiB and a demand read is about 4.15 ms (section 6.2), so the tail is
0.95 MiB and the USB round trip no longer fits underneath it. The second drive stops adding bandwidth and
starts setting the critical path.

**So `CACHALOT_MIRROR_PATH` and `CACHALOT_MIRROR_FRACTION` come off section 4's command.** They are kept
here, and in section 9.11, because they are still worth **−5 % decode and −7 % cold prefill on FP4**, which
is the arm they were measured and shipped on. The copy at
`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128` is left in place; it costs 143 GiB of a drive with
871 GiB free and it is what a future test of a smaller fraction would need.

**The fraction was swept and it changes nothing, which identifies the mechanism.** 0.10 was tuned on
17.93 MiB experts, where 0.08 to 0.12 were indistinguishable and 0.15 was worse than off, so the obvious
repair was a smaller tail. On the story turn:

| mirror fraction | tail per expert | tok/s |
|---|---:|---:|
| off | — | **6.87 - 7.03** |
| 0.02 | 0.19 MiB | 5.72 |
| 0.05 | 0.47 MiB | 5.70 |
| 0.10 | 0.95 MiB | 5.69 - 5.75 |

**A five-fold change in the tail moves the result by 0.03 tok/s.** The penalty is therefore not the tail's
transfer time — that would scale — but a **fixed per-read cost paid on every expert**, which is the USB
round trip itself. The striped read cannot complete before the second drive's latency has elapsed, however
few bytes it was asked for, so the mirror sets the critical path at any fraction above zero.

That also explains why the lever ever worked. On FP4 the internal read is 9.37 ms and the round trip fits
underneath it; on the 2-bit bank the whole read is 4.15 ms and it does not. The threshold is the second
drive's latency, not a bandwidth ratio, and the tuning guidance in `storage/reader.py` — "use
bandwidth_b / total" — is the right formula only once the head read is long enough to hide that latency.

**The lever is closed on this bank.** Not "small", not "needs a better fraction": there is no fraction above
zero that pays, and the sweep spans 5x.

**The one confound was checked and is dead.** The X10Pro absorbed the 143 GiB mirror copy shortly before the
first two mirror runs, so its write cache might have been recovering. The third mirror run was taken 90
minutes later, after the drive had been idle throughout, and returned 5.75 against the first two runs' 5.69
and 5.64. The drive was not warming up; the mirror is simply slower on this expert size.

**The general rule, for the third time in this document:** a lever's value is a property of the bank, not of
the runtime. Mirror striping joins dispatch count and prefetch timing on that list. Re-measure section 9
whenever the bank changes.

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

### 9.12.1 Non-cumulative lookahead — **built, measured and closed, 2026-09-19**

This section used to end: *"a non-cumulative version would double the lead time at constant bytes — worth a
code change and a 20-minute A/B."* The knob was built (`CACHALOT_PREDICT_LEAD`, which shifts the prediction
window where `PREDICT_AHEAD` widens it), and the A/B was run on the fixed store. **The premise was wrong and
the lever loses — but by much less than the bytes say, and that is the useful part.**

Four runs a side, interleaved in both directions, `settle.sh` between, FP4 at a 36 GiB budget with mirror
striping on and top-6 both sides:

| arm | ms/token | median | hit rate | MiB read/token | precision | wasted loads/token |
|---|---|---:|---:|---:|---:|---:|
| **lead 1**, predict L+1 (shipped) | 325, 328, 329, 329 | **328.5** | 70.8–70.9 % | 1,866 | **55 %** | 34.3 |
| lead 2, predict L+2 instead | 330, 334, 336, 344 | 335.0 | 70.9–71.0 % | 2,037 | 45 % | 43.8 |

**The ranges do not overlap** — every lead-1 run is at or under 329 ms and every lead-2 run at or over 330 —
so the 2.0 % is real at four runs a side.

**"Constant bytes" was the error, and it was an error about what a prediction costs.** Lead 2 predicts the
same *number* of experts as lead 1. It does not read the same number of *bytes*, because a wrong prediction
is a read that gets thrown away while a right one is either already resident or about to be needed. Precision
falls from 55 % to 45 % and wasted loads rise from 34.3 to 43.8 per token, so traffic rises 171 MiB per token
— **9.1 % more bytes on the resource that is 80.5 % busy.** The offline recall table's "6.5 points" is a
count; its price is bytes, and nothing in the table says so. Section 8.3, a fifth time.

**And yet the lead time is worth roughly 24 ms per token.** 171 MiB at the 5.97 GB/s this configuration
achieves is about 30 ms of drive time, and the arm is only 6.5 ms slower. By subtraction the extra layer of
lead recovers about 24 ms of the 41.4 ms the anatomy attributes to prefetch timing — which is what the recall
table predicted and is the largest confirmation this project has that timing, not only coverage, is worth
attacking on FP4. **Read that 24 ms as an inference, not a measurement**: it assumes the extra bytes are
fully exposed, which on a drive at its knee is close to true but is not measured directly.

**What it tells the next session, and it is worth more than the null.** A predictor that reached L+2 *at
lead-1 recall* would be worth about 24 ms per token, 7 % of decode, on the bank in use. That is no longer a
speculative prize attached to section 9.12's "different signal" — it is a measured lower bound on what one
would pay. Recall is what must improve; lead time is already known to convert.

`CACHALOT_PREDICT_LEAD` ships defaulted to 1, which is exactly the previous behaviour, and is kept because it
is three lines and it is how this was measured.

**The regression check that came with it.** Lead 1 on the fixed store is 328.5 ms against the 327 ms recorded
for the same configuration before the predicted-load lifetime fix (section 9.13) — inside the 7 % run-to-run
spread and inside this arm's own 325–329 range. **The lifetime fix costs nothing at lead 1**, which is what
it should do, since at lead 1 every prediction is aimed at the very next layer and no deadline is ever tested.

### 9.13 Lever 13 — Predicted loads had no lifetime — **fixed and shipped, 2026-09-19**

**Found by an outside review, reproduced on CPU with no model loaded, fixed and measured the same day.**

> **Shipped.** Each in-flight prediction now records the walk it belongs to and the layer it was issued
> for, and expires when that walk ends or when its layer has been requested. `TextDecodeRuntime.reset`
> expires them explicitly. `predicted_expired` counts the releases, so waste from a wrong prediction stays
> visible and is no longer mixed with work thrown away for being punctual. Nine deterministic store tests
> pin it, the first of which fails against the previous sweep.
>
> **It is a null on decode, which is the expected result and not a disappointment.** At the shipped lead of
> 1 every prediction is aimed at the very next layer, so no deadline is ever tested: lead 1 measures 328.5 ms
> per token against the 327 recorded before the fix, inside this arm's own 325-329 range. The fix is not a
> speed lever. It is what makes any experiment beyond lead 1 mean anything, and section 9.12.1 is the first
> one that did.

`ResidentExpertStore.get_many` opens with `self._sweep_inflight_locked(keep=requested)`, and that sweep
releases **every completed in-flight prediction that the current layer did not ask for**:

```python
def _sweep_inflight_locked(self, keep: set[Key]) -> None:
    """Release finished predicted loads that no request has claimed."""
    for key in [k for k, (f, _, _) in self._inflight.items() if k not in keep and f.done()]:
```

There is no target token or layer on an in-flight entry, so the sweep cannot tell a stale prediction from a
correct one that simply belongs to a later layer. With `CACHALOT_PREDICT_AHEAD=2`, layer L predicts both L+1
and L+2; when L+1 is then requested, any L+2 read that has already **finished** is dropped, its slot is
returned to the pool and its 18,800,640 bytes are counted in `predicted_wasted_bytes`. An L+2 read still in
flight survives, because the sweep only takes `f.done()` entries.

**So a prediction is punished for completing early.** That is the opposite of the intended behaviour, and it
falls hardest on exactly the predictions lead time is supposed to buy.

**What it invalidates.** Section 11 records `CACHALOT_PREDICT_AHEAD=2` as 8.8 % worse and attributes it to the
knob being cumulative — more bytes, lower precision, a queue on a saturated drive. That mechanism is real and
measured. But the arm was also discarding its own L+2 work, and the wasted-byte and precision figures it
reported (1,868 to 2,514 MiB, 55 % to 39 %) include those discards. **The null stands as "this knob loses";
it does not stand as evidence about lead time**, and the non-cumulative experiment in section 9.12 cannot be
interpreted until this is fixed.

**The fix.** Each in-flight speculative entry records the walk and the layer it was predicted for. Decode
visits layers in ascending order once per token, so a requested layer that does not advance is a new walk.
A prediction expires when its walk ends or when its layer has been requested -- at which point it was either
consumed, and is no longer in flight, or mispredicted, and is genuinely waste. Slot pressure needed nothing
new: `prefetch_decode` already refuses to start a load once the transient slots are down to
`PREDICT_SLOT_RESERVE`, so a longer lifetime costs prefetch depth rather than the demand path, and a test
pins that.

**Still not re-measured: the cumulative `CACHALOT_PREDICT_AHEAD=2` arm.** Section 11 keeps it as a null and
it will almost certainly stay one — it doubles the predicted count where lead 2 merely moved it, and lead 2
alone cost 171 MiB per token. But its recorded precision and wasted-byte figures were taken while the store
was discarding its own L+2 work, so those two numbers specifically should not be quoted. An hour of machine
time would settle it.

### 9.14 Miss reduction: what the offline replay says about the assessment's five recommendations

`docs/MISS-REDUCTION-ASSESSMENT-2026-09-19.md` proposes five experiments. Its own CPU replay was
**reproduced and is correct**: `simulate_policies.py` on `trace_routing_v7` gives decode hit 69.9 % at 36 GiB
and 73.3 % at 44, which is 72.2 and 64.1 misses per token and the 8.2-miss, 11.4 % gap it reports.

**Recommendation 2, predicted-expert lifetimes, is done** and is section 9.13.

**Recommendation 3, session-aware retention across suffix prefills, was screened and does not justify runtime
work yet.** `prefill_layer` evicts every resident of a layer the current prompt does not route to, so a short
new prompt wipes the previous turn's decode working set. Exempting a bounded, decaying set of
recently-decode-useful experts from that wipe, swept over the reserve per layer:

| reserve per layer | 36 GiB misses/token | 44 GiB misses/token |
|---:|---:|---:|
| 0 (baseline) | 72.319 | 64.106 |
| 1 | 72.206 | 64.013 |
| 2 | 72.006 | 63.919 |
| 4 | 71.969 | 63.663 |
| 8 | **71.731** (−0.59) | **63.138** (−0.97) |

The best case is **0.97 misses per token, 1.5 %**, which is 18 MiB and about **3 ms of a 328 ms token**. The
run-to-run spread is 7 %. Even if the runtime delivered the full simulated gain it could not be measured, and
the simulator has already over-predicted once by more than this whole effect — it put segmented LRU at
+0.9 points where the runtime delivered −0.25 (section 11).

**But the screen is weak in the direction that matters, and that is the finding.** This trace has **32 decode
tokens per segment**. Session retention is a mechanism for carrying a working set *built during decode* across
the next prefill, and 32 tokens barely builds one. A real coding turn generates 256 to 1,024. The assessment
says as much about its own replay. So the honest reading is not "session retention does not work" but **"this
trace cannot tell, and it is the wrong trace to ask."**

**The cheap prerequisite nobody has done: record a longer routing trace.** `src/cachalot/metrics/routing_trace.py`
already records one, and recording it during a real multi-turn chat session costs nothing but the session.
Until a trace exists with realistic decode lengths, recommendations 3, 4 and 5 are all being screened on five
32-token segments, and none of their results will mean much. **Record the trace first; it makes three
experiments interpretable for the price of one chat.**

**Recommendation 1, 36 against 44 GiB on the runtime**, is unscreened because it needs no screen — the
replay's 8.2 misses per token is 154 MiB, about 26 ms, which is large enough to measure. Hamed already runs
44 GiB, so this settles the benchmark's budget rather than his configuration.

---

### 9.15 Lever 15 — The CPU third: 29 ms per token of graph building with the GPU idle — **opened 2026-09-21, one tenth taken**

**This is the largest measured block in the decode token and no session before 2026-09-21 had looked at it.**
Section 6.3 has the measurement: an all-resident token is 65 % inside `mx.eval` and 35 % outside it, and the
outside is Python and MLX constructing the next layer's operations while the GPU has nothing queued. Per
layer it is 0.73 ms of CPU followed by 1.44 ms of GPU, serialised by the router eval that every MoE layer
must run.

**Why it cannot simply be overlapped.** Layer L's routing has to reach the CPU before layer L's experts can
be addressed, layer L's experts have to run before layer L+1's input exists, and MLX is define-by-run, so
there is no graph to build ahead. `ASYNC_MOE` already does the one overlap available — it dispatches the
expert and shared work before the CPU starts building the next layer — and it measures as a wash
(85.3-91.1 ms with it off against 87.6-88.9 with it on, four runs). The lever is therefore to make the CPU
side cheaper, not to hide it.

**What it is made of, so far.** Prediction costs about 3.9 ms of it: with `CACHALOT_PREDICT_TOPK=0` the gap
falls from 29.1 ms to 25.2 ms and the eval from 57.4 ms to 54.9 ms. That is not an argument for turning
prediction off — it serves 41.8 of 46 misses early on the streaming path — but it is the first price anyone
has put on issuing it, and it is paid on the main thread inside the decode loop. Moving the submission off
that thread is an unexplored three-figure-millisecond-per-session idea.

**What was taken: the affine slot views, worth 2.5 ms per token.** The affine expert path rebuilt every
projection's `(weight, scales, biases)` views on every call — nine `.view().reshape()` pairs per expert, six
experts, forty layers, **4,320 MLX op constructions per decoded token**. Those views are zero-copy on a
contiguous buffer and keep aliasing the slot's memory when a different expert is read into it, so they can be
built once per slot and reused for the slot's whole life. `ExpertSlot.typed` now holds them and
`affine_views` memoises into it.

Three runs per side, interleaved with `settle.sh` between, 2-bit bank, 24 GiB budget, 512-token context,
twelve tokens per run:

| | baseline | **cached views** |
|---|---|---|
| whole token, min | 88.1, 87.1, 87.0 ms | **83.9, 84.4, 85.1 ms** |
| whole token, median | 91.7, 93.2, 92.9 ms | **87.9, 90.3, 89.0 ms** |
| CPU outside eval, min | 30.6, 30.4, 30.7 ms | **27.1, 27.7, 28.2 ms** |
| CPU outside eval, median | 32.1, 32.3, 33.6 ms | **28.4, 30.4, 31.1 ms** |

**The ranges do not overlap on any of the four statistics**, the effect is 2.5-3 ms on the token and 2.6 ms
on the CPU side, and a micro-benchmark predicted 2.95 ms of construction time before the change was written
(`benchmarks/micro_affine_cpu.py`). That is 3 % of the all-resident floor and about 2 % of a live token,
which is below what a throughput A/B can resolve — the reason it is quoted from the CPU-gap measurement,
whose run-to-run spread is about 1 ms rather than 7 %.

**It is numerically inert and that was checked rather than asserted.** The cached arrays are the same MLX
arrays the old path built, fed to the same operations in the same order. `nll_expert_precision.py --experts
runtime --tokens 512` on both arms returns mean NLL 2.2356 nats, perplexity 9.352, median 1.3024, top-1
52.0 %, worst 14.53 at token 333 — identical to four decimals. `tests/test_expert_bank.py` gained a test that
pins the aliasing property itself, so if a future MLX makes `.view().reshape()` materialise a copy the suite
fails instead of the model quietly serving one expert's weights under another's name.

**That run also re-establishes the production arm on the fixed runtime**, which section 7.3 still carries
from before 2026-09-20: the 2-bit bank is **2.2356 nats and 52.0 % top-1** at 512 tokens, against the 2.5187
and 44.5 % measured through the transposed residual mix.

**What is left here.** 26 ms per token when this was written; 21.5 ms after section 9.16 and **20.7 ms
after section 9.17**, of which routing prediction is **7 ms** — section 9.18 re-measures the 3.9 ms below
and splits it, and withdraws the suggestion in the paragraph above it that moving the submission off the
decode thread would help, which section 11 records as a null. `benchmarks/profile_decode_cpu.py` runs the
token under cProfile and is the starting point for the rest, with the caveat that MLX's nanobind calls are
invisible to it and their time lands in the calling Python function's `tottime`.

### 9.16 Lever 16 — Trace the MoE block instead of rebuilding it — **shipped 2026-09-21, 5-8 ms per token**

**The CPU third is mostly construction, and construction is what `mx.compile` removes.** Section 9.15 took
one piece of it by caching the affine slot views; this takes the rest of the MoE block by never building it
twice. `mx.compile` traces a function once and replays the traced graph, so the Python and nanobind cost of
every operation inside it is paid on the first token rather than on all forty layers of every token.

**The screen came first** (`benchmarks/micro_compile_moe.py`, six experts at the shipped shapes, no model
load):

| | graph construction, one layer | per token | chained, CPU + GPU |
|---|---:|---:|---:|
| the shipped loop, views cached | 0.0854 ms | 3.42 ms | 11.16 ms/token |
| the same math, router weights as an array | 0.0347 ms | 1.39 ms | |
| **`mx.compile`** | **0.0104 ms** | **0.41 ms** | **6.23 ms/token** |

and it answered a second question on the way: **MLX's custom Metal kernels survive `mx.compile`**. The FP8
shared expert and the fused router both trace and return bit-identical output, which is what made it worth
putting the shared expert inside the same traced block.

**Two conditions make the trace reusable rather than rebuilt.** The router weights have to enter as the
router's own fp32 array: as Python floats they are traced constants and every token retraces. And the expert
count is part of the cache key, so the miss-budget approximation — which drops experts and rescales — gets
its own trace instead of silently reusing a six-expert one.

**What it measures on the runtime.** `profile_decode_sync.py`, 2-bit bank, 24 GiB budget, 512-token context,
twelve tokens per run, arms interleaved with `settle.sh` between them:

| | whole token, min | whole token, median | CPU outside eval, min | CPU outside eval, median |
|---|---|---|---|---|
| per-expert loop, 3 runs | 82.2, 85.4, 83.3 ms | 87.4, 87.2, 87.0 ms | 26.6, 27.6, 27.9 ms | 28.9, 28.7, 29.1 ms |
| experts traced, 5 runs | 77.6-80.8 ms | 82.1-86.6 ms | 22.4-23.6 ms | 23.4-26.7 ms |
| **whole block traced, 3 runs** | **76.9, 77.1, 77.3 ms** | **81.3, 81.8, 82.9 ms** | **21.4, 22.0, 21.9 ms** | **23.4, 23.7, 23.7 ms** |

The loop and the traced block do not overlap on any of the four statistics. The token falls **5-8 ms** and
the CPU side **5.5 ms, a fifth of it**; folding the shared expert in after the experts is worth about 1 ms
more on the token and 1 ms on the CPU side, non-overlapping on the minima and overlapping on the medians.
The all-resident floor is therefore **77 ms at its minimum against 82-85 before**, and the live decode
anatomy at the same budget agrees in the term that should move: `rest` falls from 133.8 to 125.2 ms per
token while the blocked term stays inside its own run-to-run spread.

**It is numerically inert and that was checked on the model.** `nll_expert_precision.py --experts runtime
--tokens 512` returns mean NLL 2.2356 nats, perplexity 9.352, median 1.3024, top-1 52.0 %, worst 14.53 at
token 333 — identical to four decimals to the production arm measured earlier the same day, and the screen
reports bit-identical output on synthetic inputs. `tests/test_expert_bank.py` gained a test that the traced
block and the per-expert loop agree exactly on two different router weight vectors.

**Kill switches, for bisecting only.** `CACHALOT_COMPILE_MOE=0` restores the per-expert loop;
`CACHALOT_COMPILE_SHARED=0` keeps the experts traced and leaves the shared expert outside.

**What this opens.** Everything inside a decode layer is a candidate for the same treatment, and the pieces
that are left are larger than the one taken: the hyper-connection glue, the attention block and the
per-layer graph around them account for most of the remaining 21.5 ms of CPU. Attention is the hard case
because its shapes grow with the context, so each token would retrace — `mx.compile(shapeless=True)` exists
and has not been tried here. The MoE block was the easy half; it was also the half nobody had to reshape to
compile.

### 9.17 Lever 17 — The kernels' scalar parameters were rebuilt 1,600 times a token — **shipped 2026-09-21, 1.2 ms**

**Every fused Metal kernel in the decode path built its own parameter buffers on every call.** A wrapper
around `mx.fast.metal_kernel` hands the kernel its row count, its epsilon, its softmax scale as one-element
arrays, and each of those was a fresh `mx.array([...])` on the main thread, in the gap where the GPU has
nothing queued. Counted from `profile_decode_cpu.py`'s call counts, a decoded token builds about **1,600 of
them**: 240 for the hyper-connection glue, 240 for rope, 316 for the two router passes, 340 for the FP8
shared expert, 160 for the 1-D norms, 80 for sparse attention.

They are constants, and there are a few dozen distinct values in a session. `cachalot/model/kernel_consts.py`
memoises them, evaluated once at construction, and the five decode-path kernel modules call `u32()` and
`f32()` instead of `mx.array`. The safety argument is the one the affine slot views already rest on: an MLX
array is an immutable value, so feeding the same object to many graph nodes is indistinguishable from
feeding equal copies of it.

`profile_decode_sync.py`, 2-bit bank, 24 GiB budget, 512-token context, twelve tokens per run, three runs a
side interleaved with `settle.sh`:

| | rebuilt every call | **memoised** |
|---|---|---|
| CPU outside eval, min | 22.3, 22.4, 21.6 ms | **21.0, 21.1, 20.7 ms** |
| CPU outside eval, median | 23.5, 23.1, 21.8 ms | **21.7, 21.7, 21.6 ms** |
| gap before the router eval | 21.1, 21.1, 20.4 ms | **19.6, 19.5, 20.0 ms** |
| whole token, min | 78.1, 77.9, 77.1 ms | 77.3, 76.3, 76.8 ms |

**The CPU side is 1.2-1.3 ms cheaper and the three statistics that measure it do not overlap.** The token
itself does overlap, which is what a 1 ms change looks like against a token whose run-to-run minimum moves
by 1 ms — the same reason section 9.15 quoted the affine-view result from the CPU gap rather than from
throughput. A screen predicted 2.1 ms from the per-call construction cost and the measured figure is 1.2;
the screen times the construction with nothing else contending for the CPU.

**It is numerically inert, and both halves of that were checked.** `nll_expert_precision.py --experts
runtime --tokens 512` returns mean NLL 2.2356 nats, perplexity 9.352, median 1.3024, top-1 52.0 %, worst
14.53 at token 333 — identical to four decimals to the production arm. `tests/test_kernel_consts.py` pins
that the cache really returns one object, that `CACHALOT_KERNEL_CONSTS=0` really rebuilds, and that the
hyper-connection, norm and rope kernels produce bit-identical output either way. The live anatomy at the
same budget agrees in the term that should move and in no other: `rest` 125.2 to 123.4 ms per token, hit
rate 73.3 to 73.4 %, 947 to 945 MiB read per token.

**Kill switch, for bisecting only.** `CACHALOT_KERNEL_CONSTS=0` rebuilds every parameter array on every call.

**It was then run interactively, one session per arm, and the live A/B is a null by construction.** 1.2 ms
on a 126-178 ms live token is 0.7-1.0 %, and the one matched prompt came out 1.1 % in favour of the arm
without the change. Hit rate agrees to 0.13 points, prediction precision to 0.06, quality is the reference
class on both, and both coding turns finished cleanly at the raised 2000-token cap. Section 7.2.4, which
also carries the reply-length effect those two sessions exposed.

### 9.18 What routing prediction costs the decode thread, measured 2026-09-21

Section 9.15 priced issuing the prediction at 3.9 ms of CPU from a single `CACHALOT_PREDICT_TOPK=0`
comparison and called moving it off the decode thread an unexplored idea. Re-measured on the traced runtime
with the constants memoised, with a third arm that separates computing a prediction from submitting it
(`CACHALOT_PREDICT_SUBMIT=0` computes it, evaluates it and throws it away). `profile_decode_sync.py`, 2-bit
bank, 24 GiB budget, 512-token context, two runs an arm:

| | whole token, min | inside eval, min | CPU outside eval, min |
|---|---|---|---|
| no prediction (`PREDICT_TOPK=0`) | 64.6, 66.8 ms | 50.8, 51.8 ms | 13.8, 14.3 ms |
| predicted, not submitted | 70.8, 71.3 ms | 52.8, 53.0 ms | 17.0, 17.6 ms |
| **shipped** | 76.3, 76.8, 77.3 ms | 54.6, 55.0, 55.6 ms | 20.7, 21.0, 21.1 ms |

**Prediction is 11 ms of the 77 ms all-resident floor** — about 7 ms of CPU and 4 ms of GPU — split almost
evenly between computing it and submitting it. That is by a wide margin the largest single named item left
in the CPU third, and it is three times the 3.9 ms the old measurement gave.

**None of it is a `mx.compile` candidate, and that was checked rather than assumed.**
`benchmarks/micro_compile_router.py` prices the whole router pass, both gates, at 0.40 ms per token of graph
construction with the constants memoised, and 0.10 ms compiled. The 3.3 ms of CPU that computing the
prediction costs is therefore not construction; it is inside MLX, in the work of carrying forty more outputs
through each token's evals, and no amount of tracing the Python around it will take it.

**This is not an argument for turning prediction off.** It serves 41.8 of 46 misses early on the streaming
path, and the arms above are all-resident tokens that pay its cost and collect none of its benefit. What the
table says is where to look: the lever is a cheaper prediction, not a faster one, and the two candidates are
admitting mispredicted bytes instead of discarding them (section 9.10) and predicting fewer, better experts.

### 9.19 Three ways to reclaim the prediction's waste, all closed, 2026-09-21

Section 9.18 left prediction as the largest named item in the CPU third and named three candidates. All
three were measured the next session and none survives. `benchmarks/predict_ghost.py`, new, wraps the store
from the outside and logs every prediction the sweep expires unused and every read the demand pool performs,
with the decode pass each happened on; 24 GiB budget, 512-token context, 96 decode tokens. Its counts are
deterministic — two runs agreed to within five reads of 6,800 — so the churned second run's numbers are used
here for the correlations and its timing is discarded.

**1. Admitting the mispredicted bytes instead of dropping them: the ceiling is 4.3 %.** The run reads 71.0
experts per token speculatively, drops 35.7 of them unused, and issues 26.9 demand reads. For each demand
read, the distance back to the most recent drop of the same expert:

| a dropped expert is demanded within | reads | of all demand reads |
|---:|---:|---:|
| 1 token | 31 | 1.2 % |
| 4 tokens | 71 | 2.7 % |
| 8 tokens | 110 | 4.3 % |
| 16 tokens | 151 | 5.8 % |
| 64 tokens | 243 | 9.4 % |

Holding a drop for eight tokens means holding 286 experts, 2.7 GiB of a 24 GiB budget, to save 1.15 demand
reads per token of 26.9 — and the 2.7 GiB comes out of a cache running at a 74 % hit rate. **The bytes are
in memory and they are not worth the residency.** v21's Job 2 and v22's Job 1.1 are closed.

**2. A blocklist on re-reads: the trade is 2.4 wasted reads saved per useful one lost, and it is a loss.**
The same logs answer a question nobody had asked: **29.2 % of all speculative reads re-read an expert a
prediction had already read and dropped**, because the router barely moves between adjacent tokens, so a
wrong prediction is made again and paid for again. Refusing to re-read an expert dropped within the last N
passes would have blocked, per token:

| TTL | blocked | would have been wasted again | would have been used |
|---:|---:|---:|---:|
| 1 | 3.9 | 2.7 | 1.2 |
| 8 | 11.6 | 8.1 | 3.4 |
| 64 | 20.0 | 13.9 | 6.1 |

At every lifetime about 30 % of the blocked reads would have been right this time — worse than the arm's
overall precision, but not worthless. Blocking them at TTL 8 trades 8.1 speculative reads per token, off the
critical path, for 3.4 extra demand misses per token, on it, against a baseline of 26.9. **A wrong
prediction is not reliably wrong.**

**3. Making the submission cheaper: the Python is 0.04 ms per token, not 3.6.**
`benchmarks/micro_predict_submit.py`, new and model-free, times one token's worth of the submission
bookkeeping — forty `.tolist()`s off evaluated 6-element arrays, 240 tuple constructions, 240 `int()` calls
and 240 lookups in a 15,360-entry dict — at **0.040 ms**. Replacing the dict with a per-layer list indexed
by expert id takes it to 0.025. Concatenating the layer's routing and its prediction into one array to save
a `.tolist()` costs **8.4 ms**, because the concatenate is another eval per layer. **There is nothing to win
here**, and the 3.6 ms section 9.18 attributed to submission is not Python.

**What it is instead, and it corrects the floor.** `profile_decode_sync.py` now reports what the arm read.
Its "all-resident" token is all-resident on the *demand* path only: a mispredicted expert was never
demanded, so it is not resident, and because the probe restores the same snapshot twelve times the same
wrong prediction is read, dropped and read again on every repeat. The shipped arm issues **40.0 speculative
reads per token, expires all 40, and reads 380 MiB per token off the SSD**. That is what
`CACHALOT_PREDICT_SUBMIT=0` removes, and it is why submission looked like 3.6 ms of CPU: it is the cost of
issuing forty wasted reads — pool submits, slot acquisition, the sweep, and whatever the reader threads take
from the decode thread — not of building the list handed to them. **The 77 ms all-resident floor is not a
floor without I/O**, and any future arm that changes prediction changes what that number contains.

### 9.20 The ranking, rebuilt on measured sizes, 2026-09-21

**The decode rate has not moved in four sessions.** 7.6-7.9 tok/s on prose, 5.6-6.9 on code, across the
traced MoE block, the memoised kernel constants and 0.7.0. The reason is visible once the token is laid out
by measured size rather than by which lever was next on a list: **every lever attacked since 2026-09-21 was
worth 1-5 ms, and the two largest blocks in the token have never been attacked at all.**

A live prose token is **128.5 ms** (section 7.2.5). A benchmark token at the same budget is 170 ms with a
colder working set (section 7.1.3). The all-resident floor is 76.4 ms, of which 54.8 is inside `mx.eval`
and 21.6 is CPU (section 9.18, corrected by 9.19: that floor itself issues 40 speculative reads).

| block | size | state |
|---|---:|---|
| **GPU time no measured piece accounts for** | **~20 ms** | head, Engram, ten unprofiled attention layers, 44 eval drains. No mechanism. §7.1.4 |
| **`rest` above the all-resident floor when streaming** | **~33 ms** | not the GIL switch interval, not reader scheduling. No mechanism. §7.1.3 |
| blocking on misses no prediction covered | 46.4 ms at 83.5 % hit, ~25 ms at a session's 90 % | drive idle 45 %; width, lead and precision all closed. §7.1.3, §9.19 |
| attention, 30 of 40 layers | 14.3 ms | largest *named* GPU piece; `mx.compile(shapeless=True)` never tried. §7.1.4 |
| routed experts, traced | 6.7 ms | at the cost of their three matmuls; fusion ≤1.5 ms. **Closed.** §7.1.4 |
| shared expert | 6.3 ms | never screened |
| hyper-connection glue | 6.1 ms | tracing it is 0.2 ms. **Closed.** §11 |
| routing prediction, compute and submission | 11 ms | every named way of making it cheaper is closed. §9.18, §9.19 |
| the layer's own router | 0.9 ms | nothing there. §7.1.4 |

**The two unattributed blocks are together about 50 ms of a 128 ms token** — larger than everything closed
since 2026-09-19 put together, and larger than any lever anyone has proposed. They are measurement jobs,
not engineering jobs, and neither needs a design decision to start.

**What was re-checked this session and did not reopen.** Speculative decoding closed in section 9.1 on FP4
constants, every one of which has moved, so the replay was re-run at 9.49 MiB against the current trace:
bytes per accepted token rise from 515 MiB at width 1 to 600 at width 2 and 876 at width 5
(`speculation_bytes_q2_44.json`), and `verify_forward_cost.py` on the 2-bit bank measures a K-position
forward at **242.6 ms + 26.9 ms per extra position** — because the only multi-position path this runtime has
is the *prefill* path, which costs three times a decode forward for the same single token. At width 5 that
is 350 ms for 2.85 tokens, 123 ms per accepted token against a live 128.5, while reading twice the bytes.
**It stays closed, and now on a measured cost rather than on retired byte constants.** It would need a
decode-shaped multi-position forward — the fused kernels, batched over K — before the economics are worth
recomputing, and that is a session of kernel work whose payoff is bounded by the byte table above.


### 9.21 Lever — `mx.compile` on attention, **closed three ways, 2026-09-21**

Attention is the largest GPU block in the token (section 7.1.5: 22.5 ms across all forty layers) and
`mx.compile(shapeless=True)` was the last instrument in the repository nobody had pointed at it.
`benchmarks/micro_compile_attention.py` points it at `compressed_attention_decode_reuse`, the function the
thirty reuse layers call, over twelve consecutive decode positions -- because a screen that holds
`start_pos` fixed cannot see a retrace, and this function's graph moves with the context:
`attention_kv` is `[128 + (start_pos + 1) // compress_ratio, 512]` and `window_slot = start_pos % 128`
changes the slice bounds of the window cache on every token.

**It is closed three times over.**

1. **`shapeless=True` does not run.** `ValueError: [Primitive::output_shapes] CustomKernel cannot infer
   output shapes.` The decode attention path is built out of `mx.fast.metal_kernel` custom kernels whose
   output shapes are concrete Python values; shapeless tracing hands them symbolic shapes and they cannot
   infer anything from them. There is no flag that fixes this from the outside.
2. **A plain trace is not bit-identical.** 2 of 12 positions matched, worst element 7.0e-02 apart. The MoE
   block and the hyper-connection glue trace bit-identically (sections 9.16, 11); attention does not, and
   this project does not ship a numerics change for a speed gain.
3. **And it would be a loss anyway.** A trace saves 1.5 ms per token of construction -- 0.074 ms per call
   against 0.023, 2.22 against 0.69 ms over thirty layers -- and costs 0.13 ms per call the first time it
   sees a position, 1.089 ms against the shipped 0.963. **Every token is a new position**, so every reuse
   layer retraces every token: 36.3 ms/token against 31.8 for the shipped path, a net loss of about 4 ms.

Section 6.4's cap is about how attention *grows* with the context and did not bound this; the screen did,
in twenty minutes and one run.



### 9.22 Lever — give the GPU work during the routing round trip, 2026-09-21

Section 7.1.6 prices what a layer's `mx.eval(route.indices, ...)` costs: about 0.20 ms of round trip,
forty times a token, with the GPU idle throughout. The shared expert does not depend on the routing, so it
can be issued before the wait instead of after it. `CACHALOT_PRELAUNCH_SHARED` does that, in two shapes,
and the difference between them is the whole result.

`moe_layer_metal.py`, `0` (shipped), `1` submits the shared expert with `mx.async_eval` before the routing
eval, `2` adds it to the routing eval's own argument list. Both take the shared expert out of the traced
MoE block and back onto the pre-2026-09-21 shape (`CACHALOT_COMPILE_SHARED=0`), so the arithmetic and its
order are unchanged, and a 16-token greedy fingerprint confirms that: `benchmarks/decode_fingerprint.py`
prints identical token ids and identical fp32 logit sums to six decimals on all three arms.

`profile_decode_sync.py` at a 40 GiB budget, 512-token context, all-resident, fastest token of twelve:

| arm | token | inside eval | CPU outside |
|---|---:|---:|---:|
| **0 — shipped** | **79.6, 79.1 ms** | 57.4, 56.1 ms | 21.5, 22.2 ms |
| 1 — `mx.async_eval` before the sync | 81.8, 82.6 ms | **47.3, 48.5 ms** | 34.5, 34.1 ms |
| 2 — in the routing eval | 83.5, 85.7 ms | 61.8, 60.9 ms | 21.7, 24.3 ms |

**The mechanism works and the accounting is brutal.** Arm 1 takes **10 ms per token out of `mx.eval`** --
almost exactly the 11.0 ms the model-free screen said was available -- and puts **13 ms back on the CPU**,
for a net loss of 2.9 ms on the fastest token of each run. Submitting a graph is not free: `mx.async_eval`
walks and enqueues it on the decode thread, forty times a token, and that is Python on the critical path in
the same way section 9.18's prediction is.

**Arm 2 loses differently, and the difference is the explanation.** Adding the shared expert to the
routing's own `mx.eval` costs little on the CPU — 21.7 and 24.3 ms against the shipped 21.5-22.2 — and puts
**5 ms into eval**, because the sync now waits for the shared expert and nothing else is queued behind it. The
routed experts cannot start until the routing is known, so what arm 2 buys is serialisation: the shared
expert runs alone inside the wait instead of alongside the routed ones afterwards.

**So the opportunity is real and neither way of taking it is cheap.** Arm 1 buys the overlap and pays for
the submission; arm 2 avoids the submission and loses the GPU-side overlap between the shared expert and
the routed ones that the traced block gives for free. The flag ships at `0`. What is not closed is the
count of syncs — section 7.1.6's three-arrays-in-one-eval row is the shape of a lever, and this is the
first arm in the project to move 10 ms of anything.

### 9.23 The ranking, after the GPU side closed — 2026-09-21

Section 9.20's version of this table had two lines with no mechanism, worth about 50 ms of a 128.5 ms live
token. One of them is now decomposed to the millisecond and the other is untouched.

A live prose token is **128.5 ms** (section 7.2.5); a benchmark token at the same budget is 170 ms with a
colder working set (7.1.3); the all-resident floor is **76.4-79.6 ms**, of which about 56 is inside
`mx.eval` and 21.5 outside it.

| block | size | state |
|---|---:|---|
| blocking on misses no prediction covered | 46.4 ms at 83.5 % hit, ~25 at a session's 90 % | drive idle 45 %; width, lead, precision, admission and a blocklist all closed. **A larger budget is the only untried lever and it needs no code.** §9.4, §9.19 |
| **`rest` above the all-resident floor while streaming** | **~33 ms** | not the GIL switch interval, not reader scheduling. **Still no mechanism, and now the only block without one.** §7.1.3 |
| **attention, all forty layers** | **22.5 ms** | 12.8 reuse + 8.9 source + 0.8 sliding. `mx.compile` closed three ways; the shapes themselves have never been attacked. §7.1.5, §9.21 |
| routing prediction, compute and submission | 11 ms | every named way of making it cheaper is closed. §9.18, §9.19 |
| **the 44 `mx.eval` round trips** | **~9-12 ms** | 0.20 ms each, fixed, whatever they evaluate. Overlapping them with the shared expert moves 10 ms out of eval and costs 13 on the CPU. §7.1.6, §9.22 |
| routed experts, traced | 6.9 ms | equals its three matmuls. **Closed.** §7.1.4 |
| **compressor and indexer, eight source layers** | **5.5 ms** | the difference between a source layer's attention and a reuse layer's. **Never examined.** §7.1.5 |
| shared expert | 4.8 ms | never screened |
| hyper-connection glue | 4.2 ms | tracing it is 0.2 ms. **Closed.** §11 |
| head | 1.6 ms | one bf16 gemv against 130k rows; never attacked |
| Engram forwards | 0.8 ms | two layers |
| the layer's own router | 0.8 ms | **Closed.** §7.1.4 |

### 9.24 Lever — the Engram row reads were serialised on the decode thread — **shipped 2026-09-21, 15 % of a streaming token**

Section 7.1.7 found the mechanism: a decode token asks `EngramRowReader.read_rows` for 24 rows twice, the
reader's sixteen-worker pool only took batches of 64 or more, and so 96 `pread`s were issued one after
another from the decode thread while the expert stream saturated the same drive. Two changes, both of which
read the same bytes in the same order:

- **`CACHALOT_ENGRAM_PARALLEL_MIN`** (default **8**, `engram_reader.py`) is the pool threshold. Each worker
  writes into its own slice of the output buffer, so the assembled rows do not depend on the scheduling.
- **`CACHALOT_DECODE_ENGRAM_PREFETCH`** (default **1**, `text_decode_runtime.py`) issues both layers' reads
  at the top of the token instead of at the layer that needs them. The row ids come from
  `engram_hash.push(token_id)` and depend only on the token being decoded, which is known before layer 0
  runs, so the read for layer 14 has thirteen layers of compute to hide behind and the one for layer 1 has
  one.

`profile_decode_sync.py`, 36 GiB budget, 512-token context, 48 tokens of real continuation, median token,
two passes of all four arms:

| arm | streaming token | `_apply_engram` | blocked in store | inside eval | CPU | all-resident token |
|---|---:|---:|---:|---:|---:|---:|
| **0 — shipped before today** | **188.3, 189.5 ms** | 28.4 ms | 64.0 ms | 77.3 ms | 21.7 ms | 77.3 ms |
| parallel `pread`s only | 167.8, 165.7 ms | 9.5 ms | 62.3 ms | 74.7 ms | 21.9 ms | 79.6 ms |
| prefetch only | 165.3, 164.1 ms | 8.6 ms | 63.8 ms | 72.9 ms | 21.3 ms | 77.2 ms |
| **both — shipped now** | **162.4, 157.7 ms** | **2.4 ms** | 64.0 ms | 77.5 ms | 21.7 ms | **76.1 ms** |

**Every other column is unchanged and so is what the arm read**: 79.4 % hit rate and 753-754 MiB per token
on all four arms, 49.4 misses per token, and an all-resident token that does not regress. The whole
difference is the Engram column, and the two mechanisms compose — parallel reads make each batch cheap,
the prefetch hides what is left.

**In the instrument the 33 ms was originally quoted from.** `decode_anatomy.py`, 36 GiB, 96 tokens of
continuation:

| | before | after |
|---|---:|---:|
| decode rate | 5.65 tok/s | **6.51 tok/s** |
| per token | 177 ms | **154 ms** |
| expert wait | 59.5 ms | 59.7 ms |
| **`rest`** | **117.7 ms** | **94.0 ms** |
| expert hit rate | 81.6 % | 81.6 % |
| read per token | 695 MiB | 694 MiB |
| drive busy | 58.0 % | 66.7 % |
| prediction precision | 45 % | 45 % |

**+15.2 % on the decode rate**, 23.7 ms of it out of `rest`, with the hit rate, the byte count, the miss
count, the blocking time and the prediction precision all held to within a tenth of a point. The drive
busies harder because the same bytes are now read in less wall clock, which is what a bytes-bound token
looks like when a serial CPU-side cost comes off it.

**And the same A/B at a second budget, 40 GiB, which is where this document's reference profiles live:**

| | before | after |
|---|---:|---:|
| decode rate | 5.69 tok/s | **6.61 tok/s** |
| per token | 176 ms | **151 ms** |
| expert wait | 55.6 ms | 55.4 ms |
| **`rest`** | **120.1 ms** | **95.9 ms** |
| expert hit rate | 83.7 % | 83.7 % |
| read per token | 632 MiB | 632 MiB |
| misses per token | 39.2 | 39.2 |
| prediction precision | 43 % | 43 % |

**+16.2 %, 24.2 ms of `rest`, and every other figure identical to the digit** — the same bytes, the same
misses, the same waiting, 25 ms less wall clock. `anat40_base_*` and `anat40_new_*`. 44 GiB was not
available on the machine at the time (70 GiB free against the 73 the guard needs), so the shipped budget
itself has not been run on either arm.

**Numerics.** `benchmarks/decode_fingerprint.py`, 16 greedy tokens, the shipped-before and shipped-now arms:
**identical token ids and identical fp32 logit sums and maxima to six decimals**, which is expected — the
change moves who issues a `pread` and when, not what is read.
`benchmarks/results/guarded/fpeg_base_*` and `fpeg_both_*`.

**Tests.** `tests/test_engram_reader_parallel.py` pins the parallel and serial paths against each other and
against the table itself at seven batch sizes including 24, pins that repeated row ids keep their
positions, and pins that the default threshold is at or below a decode batch. 229 tests pass.

**What it is worth to a live session, and what it is not.** The gain is the removal of a serial cost that
is paid per token and grows with how busy the drive is, so it is largest where the hit rate is lowest. At
36 GiB and a 79-82 % hit rate it is 23-26 ms of a 177 ms token. A live session holds a 90 % hit rate and
reads about half the bytes, so expect less — but expect it in the direction of the floor, because the
all-resident arm also moved, 1.7 ms to 1.0.

**Run interactively the same day, at a 52 GiB budget, and it is worth 16-21 ms of a live token.** Section
7.2.6: prose 7.78-7.92 → **9.42 tok/s**, Objective-C 6.90-6.93 → **8.53**, session hit rate 90.00 →
92.37 %. The budget accounts for about 6 ms of the 22-27 ms per token by the miss arithmetic; the rest is
this change, which is the size the benchmark predicted once the drive is less busy. **The two causes were
not separated** — a 52 GiB session with `CACHALOT_ENGRAM_PARALLEL_MIN=1000000
CACHALOT_DECODE_ENGRAM_PREFETCH=0` would do that and costs one conversation.

### 9.25 The ranking, after the Engram reads came off the decode thread — 2026-09-21

Section 9.23's version of this table had one line left with no mechanism, worth about 33 ms of a 128.5 ms
live token. It had a mechanism all along and it was not in the list: the Engram row reads were never a row
in any profile, because `decode_anatomy.py` buckets them into `rest` and `profile_decode_gpu.py` excludes
the row read by construction.

A live prose token was **128.5 ms** on 0.7.0 (section 7.2.5) and has not been re-measured since this
change; a benchmark token at a 36 GiB budget is 154 ms after it and was 177 before; the all-resident floor
is **76.1-79.6 ms**, of which about 56 is inside `mx.eval` and 21.5 outside it.

| block | size | state |
|---|---:|---|
| blocking on misses no prediction covered | 59.7 ms at 81.6 % hit, ~25 at a session's 90 % | every prediction lever closed. **A larger budget is the only untried one and needs no code.** §9.4, §9.19 |
| **attention, all forty layers** | **22.5 ms** | 12.8 reuse + 8.9 source + 0.8 sliding; `mx.compile` closed three ways; the shapes never attacked. §7.1.5, §9.21 |
| **`rest` above the all-resident floor while streaming** | **~15 ms**, was ~33 | 27 of it was the Engram `pread`s and is taken; what is left is 4 ms of CPU and about 11 unaccounted. §7.1.7, §9.24 |
| routing prediction, compute and submission | 11 ms | every named way of making it cheaper is closed. §9.18, §9.19 |
| the 44 `mx.eval` round trips | ~9-12 ms | 0.20 ms each, fixed. Overlapping them with the shared expert moves 10 ms out of eval and costs 13 on the CPU. §7.1.6, §9.22 |
| routed experts, traced | 6.9 ms | equals its three matmuls. **Closed.** §7.1.4 |
| **compressor, indexer and the compressed-KV write, eight source layers** | **5.5 ms** | now decomposed: two thirds indexer, a tenth compressor, the rest the KV write. **`INDEX_TOPK` is not a lever.** §7.1.8 |
| shared expert | 4.8 ms | never screened |
| hyper-connection glue | 4.2 ms | tracing it is 0.2 ms. **Closed.** §11 |
| head | 1.6 ms | one bf16 gemv against 130k rows; never attacked |
| **Engram row reads on the decode thread** | **2.4 ms**, was 28.4 | **taken, and it is the only thing that has moved the rate in five sessions.** §9.24 |
| Engram forwards | 0.8 ms | two layers |
| the layer's own router | 0.8 ms | **Closed.** §7.1.4 |

### 9.26 Lever — read concurrency and the page cache, **both null, 2026-09-21**

The v27 and v28 prompts named two experiments for the in-eval excess, both on the theory that the GPU is
competing with the device. Section 7.1.9 answered the question from a third direction; these two were run
anyway, because a null is only worth the run that produces it.

**`io_workers` is a null from 2 to 16.** `profile_decode_sync.py` gained `--io-workers`. Eight arms at a
40 GiB budget, 512-token context, 64 tokens of continuation, two reps of each width, run interleaved so
drift cannot line up with a width (`benchmarks/results/guarded/iow{2,4,8,16}_r{1,2}_*`):

| `io_workers` | whole token, rep 1 / rep 2 | inside `mx.eval`, rep 1 / rep 2 |
|---:|---|---|
| 2 | 144.1 / 144.5 ms | 65.0 / 65.0 ms |
| 4 | 147.9 / 147.1 ms | 66.4 / 65.9 ms |
| **8, ships** | **141.5 / 147.0 ms** | **64.2 / 65.2 ms** |
| 16 | 144.0 / 148.2 ms | 64.8 / 63.1 ms |

Every arm is inside the 141.5-148.2 ms band and the two reps of the shipped width span 5.5 ms of it, which
is the whole range across an eightfold change in the queue depth. The split between in-eval and blocked
does not move either. **Lowering the device queue depth does not move time out of `mx.eval`, so the
in-eval excess is not something read concurrency controls.**

**Bypassing the page cache is a 16 ms loss.** `CACHALOT_PAGE_CACHE=0` turns `F_NOCACHE` back on, so the
reads land straight in the slot buffers with no kernel copy. At 24 tokens of continuation against two
matched baselines run the same way (`ie_nocache_*`, `ie_w8a_*`, `ie_w8b_*`):

| arm | whole token | inside `mx.eval` | store-blocked | CPU |
|---|---:|---:|---:|---:|
| page cache on, a | 144.2 ms | 67.0 ms | 50.8 ms | 21.5 ms |
| page cache on, b | 145.5 ms | 72.9 ms | 48.6 ms | 21.9 ms |
| **page cache off** | **161.7 ms** | 63.5 ms | **57.4 ms** | **26.6 ms** |

Same hit rate, same 645 MiB read. The in-eval column is inside the baselines' own spread, and what moves
is the blocking and the CPU. **Removing the kernel's copy out of the read path does not give the GPU
anything back**, which is the same answer the concurrency arm gives, and it confirms the shipped
`CACHALOT_PAGE_CACHE=1` from a direction nothing had tried.

**A warning about the instrument.** `--mode both` and `--mode stream` do not produce the same streaming
arm: the all-resident arm leaves 240 experts pinned in the store and changes what the continuation evicts.
The first baseline of this session was read off a `--mode both` run and was 7 ms slower than the matched
`--mode stream` ones. **Compare arms that were launched the same way.**

### 9.27 Lever — the shared expert, screened at last, **closed 2026-09-21**

Carried unscreened through the v26, v27 and v28 prompts. It is three FP8 GEMVs per layer over 33.78 MiB of
E4M3 weights, 0.121 ms per layer and 4.8 ms per token in section 7.1.5's profile.
`benchmarks/micro_shared_expert_roofline.py` runs it on the real shapes across forty distinct layers —
1.32 GiB, so no launch reads what the previous launch left in cache — chained inside one `mx.eval`:

| arm | per layer | GB/s | x 40 | launches |
|---|---:|---:|---:|---:|
| **shipped, three GEMVs** | **0.116 ms** | **307** | **4.6 ms** | 120 |
| `w1` and `w3` stacked into one [4608, 5120] GEMV | 0.106 ms | 335 | 4.2 ms | 80 |
| `mx.sum` over the same bytes | 0.092 ms | 385 | 3.7 ms | 120 |
| the two activation quantizations alone | 0.028 ms | — | 1.1 ms | 80 |

The screen reproduces the profiler to 4 % (0.116 against 0.121 ms per layer), which is the check that it is
measuring the shipped thing. **The block is at 80 % of what the machine gives for its own bytes and the
whole headroom is 0.9 ms per token.**

**The one structural idea the shape allows is worth 0.4 ms.** `w1` and `w3` consume the same quantized
activation and neither depends on the other, so the two [2304, 5120] weights concatenate into one
[4608, 5120] and their E8M0 block scales concatenate with them exactly (2304 / 32 = 72 rows each). `K` is
unchanged, so `_lanes_per_row` picks the same 32 and every output row is summed in the same order as
before: **the fused form is bit-identical by construction, not by luck.** It is 0.106 against 0.116 ms per
layer, **0.4 ms per token, and 80 launches instead of 120.**

**Implemented behind a kill switch, still not the default, 2026-09-22 — and the first version of it crashed
Hamed's own live session.** `CACHALOT_FUSED_SHARED_EXPERT=1` concatenates `w1`/`w3` once per layer, cached by
weight-object identity (the same pattern `_get_wo_a` already uses), and issues one `[2I, H]` GEMV;
`=0`, still the default, is the shipped two-GEMV path unchanged.

**The bug.** The first version built and evaluated the concatenation *inside* `shared_expert_forward` itself.
On the shipped 2-bit affine bank that function is called from inside `moe_layer_metal._compiled_moe_block`'s
`mx.compile`d trace (§9.15's whole-MoE-block compile), and `mx.eval` inside a trace is refused outright:
`ValueError: [eval] Attempting to eval an array during function transformations like compile or vmap is not
allowed.` Every offline check before this session's edit ran the fused path eagerly — the micro-benchmark,
the bit-identical unit test, the full 236-test suite — and none of them called it from inside `mx.compile`,
so nothing caught it before Hamed's own `./chat.sh` did, on the very first decode step (`warmup()`).
**The rule this repeats: a bit-identical unit test proves the arithmetic, not the calling convention — the
two are independent claims, and only a live session, or a test that reproduces the real call shape, checks
the second one.**

**The fix moves the fusion out of the trace.** `shared_expert_forward_fused(x, w13=..., w13_scales=..., w2=...,
w2_scales=...)` takes an already-concatenated pair and does no `mx.eval` of its own.
`moe_layer_metal._shared_args` builds `w13`/`w13_scales` via `_fused_w13` in eager Python *before* calling
`_compiled_moe_block`, which now also takes `fused_shared: bool` as part of its `lru_cache` key so the fused
and unfused arms get separate traces (a compiled function's Python control flow is fixed at first trace; it
cannot branch on the flag per call, only vary its array arguments). The two other call sites
(`CACHALOT_PRELAUNCH_SHARED`, the default post-routing fallback) call `_fused_w13` directly — they run in
plain eager Python, never inside a trace, so this was never their bug.

**Re-verified, this time against the call shape that broke and on the real checkpoint, not a synthetic one.**
`tests/test_shared_expert_fused.py` gained `test_fused_forward_survives_being_called_inside_mx_compile` (the
fix, reproduced) and `test_building_w13_inside_a_trace_is_the_bug_this_guards_against` (the original bug,
pinned so it can't silently return) — 238/238 project tests pass. Beyond the unit tests, `V41Model.from_pretrained`
was run directly against `/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128` (the shipped 2-bit affine bank,
not FP4 — the affine `fmt.kind` gate is what selects `_compiled_moe_block` at all) at a 24 GiB budget with
`CACHALOT_FUSED_SHARED_EXPERT=1`: warmup and eight further `decode_token` calls completed with no traceback,
the per-layer `_fused_w13` cache settled at 40 entries (one per layer) and stayed there across all eight
tokens, and the same eight greedy token ids — `5, 223, 939, 21, 695, 736, 1266, 856` — came out with the flag
on and off, run separately. That is bit-identical output through the real compiled path on the real bank,
not just the isolated kernel.

Re-measured through `shared_expert_forward` / `shared_expert_forward_fused` directly, not the benchmark's own
hand-fused arm: 0.115 → 0.106 ms/layer min, 4.60 → 4.23 ms per token across forty layers, reproducing this
section's number exactly.

**Tried live by Hamed, 2026-09-22, clean, then shipped as the only path.** `CACHALOT_FUSED_SHARED_EXPERT=1
CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52` — no crash, `/exit` normal. A 642-token
story and a 1,744-token Objective-C json-to-CSV turn (`NSMutableOrderedSet` for first-seen key order, the
exact property the coding gate's ObjC tasks check) both read correct. 9.91 and 8.27 tok/s, 92.35 % hit rate —
inside or a few percent of the shipped-52-GiB baseline range (9.42-9.84 / 8.53-8.69 / 91.9-92.4 %, §7.2.10),
on different prompts than that table's fixed four, so this was not a controlled A/B and was never going to
be: 0.4 ms/token is below what any live session resolves, on purpose. **What the run actually established is
that the crash class was gone and nothing read wrong** — correctness and quality live, the part a live
session *can* check.

With the blocking risk (the crash) resolved and no further live signal obtainable by design, Hamed asked to
flip the default and drop the kill switch in the same change, 2026-09-22. `CACHALOT_FUSED_SHARED_EXPERT` is
gone; `moe_layer_metal.py`'s three decode call sites and `_compiled_moe_block` now unconditionally call
`shared_expert_forward_fused` with `_fused_w13`'s output. `shared_expert_forward` (the original two-GEMV
form) stays in `shared_expert_metal.py` only because `moe_prefill_grouped.py` still calls it for its
per-token fallback — that path was never benchmarked fused and is out of scope. Re-verified after removing
the switch: 238/238 tests, and the same real-bank warmup-plus-eight-tokens check as above, now with no env
var at all, produced the identical eight token ids. The shared expert now always issues two GEMVs per layer
instead of three, unconditionally.

**And a quarter of the block is the two activation quantizations**, 0.028 ms per layer for two launches
that move 15 KB between them — pure launch cost, 1.1 ms per token. `CACHALOT_FUSED_FP8` already collapsed
each of them from an eight-op MLX chain into one kernel; what is left is the launches themselves.

### 9.28 Lever — `wo_a` is held dequantized, and it is the right call, **closed 2026-09-21**

`attn.wo_a.weight` ships as F8_E4M3 [8192, 4096], 33.55 MB, and `text_decode_runtime._get_wo_a`
dequantizes it to BF16 for all forty layers at startup. What the GPU reads every token is therefore
67.11 MB per layer, **2.50 GiB resident against 1.25 GiB**, and it is the only projection in the model that
is not read as FP8 bytes. The reason is the shape and not the precision: `wo_a` is applied as eight
independent [1024, 4096] matvecs, one per output group, and the FP8 GEMV kernel takes a flat 2-D weight.

On the face of it that is 33.5 MB per layer of avoidable traffic, 1.34 GB per token, and 1.25 GiB of wired
memory that could be expert budget. `benchmarks/micro_wo_a.py` prices it on the real shapes across forty
distinct layers:

| arm | per layer | GB/s | x 40 | launches |
|---|---:|---:|---:|---:|
| **BF16 grouped `mx.matmul`, ships** | **0.107 ms** | **625** | **4.3 ms** | 320 |
| FP8, one `fp8_gemv_quantized` per group | 0.183 ms | 184 | 7.3 ms | 320 |
| FP8, one GEMV over the flat [8192, 4096] | 0.114 ms | 294 | 4.6 ms | 40 |
| `mx.sum` over the FP8 bytes | 0.070 ms | 479 | 2.8 ms | 40 |

**The shipped arm reads twice the bytes and is the fastest of the four.** MLX's BF16 batched matmul runs
this shape at 625 GB/s; a grouped FP8 kernel, in the one-launch upper bound that ignores the indexing work
it would have to do, would be 0.114 ms, and the honest eight-launch version is 0.183. **Quantizing `wo_a`
back down is a loss of 0.3 to 3.0 ms per token before any numerics question is asked**, and the 1.25 GiB
of memory it would return is worth about 1.2 points of decode hit rate by section 9.4's curve — a real
trade, but one that starts from behind.

This is also where the FP8 GEMV kernel's own ceiling became visible: MLX's BF16 matmul reaches 625 GB/s on
this shape while the custom kernel reaches 294 on half the bytes. That is what section 7.1.10 went and
measured properly.

### 9.29 Lever — the FP8 GEMV kernel's lanes-per-row policy, **null, 2026-09-21**

`_lanes_per_row(n_blocks)` returns the first of 32, 16, 8 that divides the block count. The policy is
written for load balance and **`N` is not an input to it at all**, so it was worth asking what it costs.
Every legal split on every shape the runtime issues, `benchmarks/micro_fp8_gemv_kernel.py`:

| shape | 8 lanes | 16 lanes | 32 lanes | ships | `mx.sum` |
|---|---:|---:|---:|---|---:|
| `wq_a` [1280, 5120] | 0.063 ms | 0.037 ms | 0.038 ms | 32 | 0.031 ms |
| `wq_b` [32768, 1280] | 0.092 ms | 0.108 ms | 0.120 ms | 8 | 0.078 ms |
| `wkv` [512, 5120] | 0.042 ms | 0.036 ms | 0.030 ms | 32 | 0.027 ms |
| `wo_b` [5120, 8192] | 0.119 ms | 0.116 ms | 0.110 ms | 32 | 0.079 ms |
| shared `w1`/`w3` [2304, 5120] | 0.051 ms | 0.050 ms | 0.049 ms | 32 | 0.037 ms |
| shared `w2` [5120, 2304] | 0.048 ms | 0.048 ms | 0.050 ms | 8 | 0.038 ms |

**The divisibility rule picks the fastest split on five of the six shapes**, and on the sixth — `wq_a`,
where 16 lanes is 0.037 against 0.038 — the difference is one microsecond per launch, 0.05 ms per token,
and the 16-lane arm is *not* bit-identical, because the split determines the order in which a row's blocks
are summed. Across the token: 16.61 ms for the shipped policy, 16.56 for the best split per shape,
17.79-18.62 for any single width applied everywhere. **A policy nobody tuned is within 0.3 % of the tuned
one.** Do not spend a session here.

### 9.30 The ranking, after the in-eval excess found its mechanism — 2026-09-21

Section 9.25's table had one line with no mechanism and three blocks that had never been screened. The line
is gone — it was the miss, all along — and the screens came back null or nearly so, which means **the GPU
side of this runtime is now closed end to end and every remaining millisecond is the miss or the floor.**

A live prose token is 106.2 ms at 9.42 tok/s (§7.2.6); a benchmark token at 40 GiB is 141-152 ms; **a
streaming token that does not miss is 79.6 ms, which is the all-resident floor exactly** (§7.1.9).

| block | size | state |
|---|---:|---|
| **the miss** | **~1.7 ms each**: 1.41 blocked, 0.31 inside `mx.eval`, ~0.05 CPU | 39.8/token at 83.4 % hit, ~19 ms at a session's 92.4 %. **The budget is the only lever and it is taken by hand at 52 GiB.** §9.4, §7.1.9, §7.2.6 |
| **the FP8 GEMV family** | **16.6 ms** | wq_a, wq_b, wkv, wo_b on forty layers plus all three shared-expert GEMVs: 5.14 GB/token, the largest GPU path. **At 79 % of `mx.sum` over the same bytes.** Lanes-per-row null. §7.1.10, §9.29 |
| of it: attention's four FP8 projections | 10.8 ms | §7.1.10 |
| of it: the shared expert | 5.8 ms | screened; the w1/w3 fusion is 0.4 ms and bit-identical. §9.27 |
| `wo_a`, BF16 grouped matmul | 4.3 ms | 625 GB/s, the fastest arm in the model. **Quantizing it is a loss.** §9.28 |
| routing prediction | 11 ms | closed. §9.18, §9.19 |
| the 44 `mx.eval` round trips | ~9-12 ms | 0.20 ms each; the overlap arm moves 10 and costs 13. §7.1.6, §9.22 |
| routed experts, traced | 6.9 ms | equals its three matmuls. Closed. §7.1.4 |
| compressor, indexer, compressed-KV write | 5.5 ms | decomposed; `INDEX_TOPK` is a null. §7.1.8 |
| sparse attention over the KV itself, all forty layers | ~2.6 ms | 0.064 ms per reuse layer. **This is what three prompts called "attention's shapes".** §7.1.10 |
| hyper-connection glue | 4.2 ms | closed. §11 |
| streaming CPU above the floor | ~3.7 ms | pass 2 puts the CPU column back on the floor, so this too is the miss. §7.1.9 |
| head | 1.6 ms | never attacked |
| Engram row reads | 1.9 ms, was 28.4 | taken. §9.24 |

### 9.31 Lever — the miss's own drive wall, screened at last — **closed, no lever, 2026-09-22**

Job 1 of v29 asked whether the 1.7 ms per-miss cost above this table is the drive slowed by decode's own
wired memory. It is not: `io_workers=8` reads experts at 1.46-1.47 ms each whether the machine has 42.9 GiB
wired or nothing at all, matching section 3.1's cold rating, and the runtime's own 1.41 ms blocked component
sits inside that same band rather than above it. **There is no memory-pressure tax and no store-side
admission overhead to remove from the blocked component of a miss.** Tested at 45 % of the machine's RAM
wired, not the shipped budget's 80-83 %, because available memory this session (60-64 GiB free) did not
admit a bigger ballast; nothing in the mechanism suggests that changes at the higher fraction, but nobody has
measured it. §7.1.11.

**One instrument bug closed with it.** `expert_read_scaling.py --wire-gib`'s ballast-heartbeat thread died on
its first tick on MLX 0.32.2 (`RuntimeError: There is no Stream(gpu, 0) in current thread`) because the array
it pings was never evaluated on the thread that created it; three sweeps in a row silently measured an
unwired machine after the first one or two arms without printing anything wrong. Fixed with one
`mx.eval()` call. Any session that used this flag before 2026-09-22 was not measuring what it said it was.
§7.1.11, §12.

### 9.32 Job 3 taken: the coding gate compiles and runs Objective-C — **gate built, not yet run against a model, 2026-09-22**

The v30 prompt named this the highest-value job that needs no live session, and it is not a speed lever: it
is the check that lets any later speed change claim "quality unchanged" on the language Hamed actually
prompts in. Three live Objective-C turns had been read by eye or compiled by hand; the corpus gate held
C++ and Python only and executed nothing. **The gate now has six Objective-C tasks (26 in all), compiles
Objective-C with `clang -fobjc-arc -fsyntax-only`, and runs any block that compiles against an expected
stdout.** The tasks were chosen to include the exact failure of the third live turn: the CSV header must be
in first-seen key order, and the data is given as ordered pairs so the model cannot excuse an unordered
`allKeys`. Every reference program builds and prints its stated output (a test enforces it), and the gate
rejects the live turn that called `-[NSMutableArray map:]` while accepting its one-line repair, which is the
point at which running, not compiling, becomes the check. **What is not established:** no bank has been
scored on the new cases, so there is no Objective-C compile or output-match rate yet, and the corpus hash
changed so an interrupted 20-task run cannot be resumed. Cost of the full run is about two hours.

## 10. Retired premises — conclusions whose reasons expired

These were correct when written and are now misleading. Anyone reading the older logs will meet them.

| claim | where | why it no longer holds |
|---|---|---|
| The routed experts cost 19.8 ms of GPU per token, and the router pass 7.5 | HANDOFF §6.3, §9.2, §11 | Both were timed on paths the runtime had stopped taking — `affine_expert_forward` in a loop, retired by the traced MoE block, and `route_topk`, retired by `route_topk_fused`. Measured on what ships: **routed experts 6.7 ms, router 0.9 ms**. Every lever ranked off either number shrinks with it. §7.1.4. |
| A fused multi-expert affine path is worth up to 3 ms per token | HANDOFF §11 | The 20-25 % it wins is 20-25 % of **6.7 ms**, not of 13.9. **≤1.5 ms**, still needing six LRU slots made contiguous. §7.1.4. |
| Speculative decoding closes on the arithmetic | HANDOFF §9.1 | It closed on FP4 constants — 18.8 MB experts, 5.97 GB/s, a 325 ms baseline. All three moved. Re-replayed at 9.49 MiB: bytes per accepted token rise from 515 to 600 MiB at width 2, and a K-position forward **through the prefill path** measures 243 ms + 27 ms per extra position, which loses outright. It is not reopened; what changed is that the closure now rests on a measured cost of the only multi-position path this runtime has, rather than on retired byte constants. §9.20. |
| FP4 collapses on 0 of 9 free-running replies | HANDOFF §2, §9.0, §7.4 | True of the failure `max_run` detects, which is an *exact* k-gram loop repeating back to back. It cannot see a **paraphrased retry loop** -- bad code, an apology, another attempt -- because nothing repeats exactly. Two of those same nine FP4 replies are retry loops scoring `max_run` 19 and 4 against a threshold of 24, and the new corpus reproduced one at `max_run` 7 in which the model writes "I'm clearly stuck in a loop". Does not re-rank the banks; does narrow what the collapse rate covers. §7.4.3. |
| Judge a collapse by `max_run`, never by the trigram rate | HANDOFF §7.4, §9.9, rule 4 | Right for exact loops and wrong for retry loops, where `max_run` stays under 10 and the trigram rate separates cleanly (57.5 % against a healthy 34.8 %). A collapse metric needs both, each thresholded on replies someone has read. §7.4.3. |
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
| FP4 scores 0.6 syntax errors per 100 lines, 30 to 40x better than any quantized bank | HANDOFF §2, §3.3, §7.4 | Twice wrong. First: 163 lines from an arm the guardian truncated after two of three seeds. Then **8.9 was wrong too** — the checker never read clang's exit status. §7.4.1. |
| FP4 writes roughly 3.5x fewer C++ syntax errors than a quantized bank | HANDOFF §2, §7.4, §9.0 | **The gate was broken.** `check_cpp` grepped for `": error: "` and ignored the return code, so a block aborting on a mangled `#include` scored clean. Re-scored: **0 of 4 FP4 blocks compile and 0 of 3 from the 3-bit bank**, and the density was measuring which arm aborted first. No C++ ratio between any two banks is established. The standing decision survives on collapse rate and top-1. §7.4.1. |
| Python is not the discriminator; both banks score 2.2 errors per 100 lines | HANDOFF §7.4, §9.0 | `ast.parse` stops at the first `SyntaxError`, so 2.2 counted broken files, not defects. On the parse rate FP4 is **2 of 10** and the 3-bit bank **0 of 5**; one 20-line FP4 block with five artefacts scored 1. §7.4.1. |
| A finished predicted load survives until the layer that asked for it | implicit in HANDOFF §9.12, §11 | `_sweep_inflight_locked(keep=requested)` releases every *completed* in-flight prediction not requested by the current layer, so an `L+2` read that finishes before `L+1` is discarded before `L+2` can use it. Finishing sooner makes a prediction less useful. This invalidates reading the `PREDICT_AHEAD=2` result as evidence about lead time. §9.13. |
| A better affine fit above 2 bits is the only route left to a smaller quality bank, and the next session's first job | HANDOFF §2, §9.0 | Taken. The fit was improved 28 % on a screen made faithful with real activations, a 221.5 GiB bank was built and gated, and it collapsed on 2 of 6 long turns where FP4 collapsed on none and parsed 0 of 5 Python blocks against FP4's 2 of 10. Bits, not fit, are binding. **Lever 0 closed** — on that evidence, not on the withdrawn C++ densities (§7.4.1). |
| Activation-weighted fitting is the untried part of the calibration idea worth keeping | HANDOFF §9.0.1, §9.3 | Tried. Worth 5.1 % of routed-expert output error at 3 bits, transfers across texts, costs no bytes — and worth nothing a compiler can see. The *capability* is kept; the lever it was meant to open is closed. |
| The drive is not saturated during decode, so a speculative read is nearly free | HANDOFF §9.10 | True on 9.49 MiB experts, where the drive was busy 45 % of decode. On FP4 it is busy **80.5 %** and at its knee: raising `CACHALOT_PREDICT_WORKERS` from 2 to 8 *lowers* achieved bandwidth from 5.70 to 5.34 GB/s. The width conclusion survives on a different mechanism. §6.1, §9.10. |
| Prediction from an earlier activation is free and useless: timing is not the problem, coverage is | 09-17 §13, HANDOFF §11 | Measured where timing was 2.2 % of blocked time and worth 1.6 ms per token. On FP4 timing is **20.5 % and 41.4 ms per token**. Still unbeaten, but the premise has expired and `CACHALOT_PREDICT_AHEAD` has never been swept on FP4. §6.1. |
| Mirror striping is harmful; leave it off | HANDOFF §11 | **The null lost its condition.** 09-16 §7.3 measured it harmful *with the 3-bit stacked bank*, because a mirror disables concurrent piece reads and that bank had nine pieces per expert; an FP4 expert is one contiguous range. 09-16 also measured the positive case and derived the optimum. Reproduced and shipped 2026-09-19 at 0.10: −5 % decode, −7 % cold prefill. §9.11. |
| DSpark is parked pending arithmetic on FP4's expert size | HANDOFF §9.1, prompt v9 | Arithmetic done. **1.03x** on measured constants against its own 1.15x bar. Closed. §9.1. |
| The all-resident compute floor is 93 ms | HANDOFF §6, §9.2 | 93.0 ms is the 2-bit bank's. On FP4 it is **84.6 ms**, and the FP4 expert kernel costs 23.5 ms per token against the affine path's 24.6. Compute is bank-independent in fact. §6.1. |
| Dispatch count is the largest open lever and the only large one depending on nothing else | HANDOFF §9.2, prompt v9 | True by size, misleading by value on the bank in use: 84.6 ms of a 341 ms token that is already hidden under ~320 ms of drive time. It pays after bytes come down, not before. §9 ranking. |
| FP4 is the quality bank, and 3.3 tok/s is what quality costs on this hardware | HANDOFF §2, §3.3, §7.4 | The premise was that the quantized weights set the ceiling. A hosted FP4 arm with no harness compiles 20 of 20 C++ blocks and malforms 0 of 102 includes; Cachalot compiles 0 of 42 and malforms 49 of 154. The quality was never bought with speed — it was lost to a runtime defect. §7.4.6. |
| The corpus results are evidence about the model or about FP4 | HANDOFF §7.4.3, §7.4.4, §7.4.5 | They are evidence about this runtime only. Both a pinned FP4 provider and a pinned FP8 provider are clean on the same 40 prompts at the same settings. §7.4.6. |
| Quality work means building a better bank | HANDOFF §9.0, §9.3, §8 | Three sessions of bank building — 3-bit, searched fit, activation weighting — were aimed at a defect that is not in the weights. The quantization findings stand; the lever they were meant to open never existed. §7.4.6, §2. |
| The next quality step is `benchmarks/token_rank_probe.py` on the production path | HANDOFF §7.4.4 | Still worth running, but it is no longer the first step and no longer the cheapest decisive one. The reference arm answered the question the probe was a proxy for. The probe's remaining value is localising *where* in the forward pass the token is lost, which is step 1 of the bisection in §2. |
| There is no `pack_3bit`, and MLX's packing above 2 bits is the blocker | HANDOFF §9.0, §8.1 | `pack_bits` writes MLX's layout at any width and is pinned against `mx.quantize` at 2, 3, 4, 5, 6 and 8 bits. The layout is one contiguous little-endian bit stream with no padding. |

## 11. Null results — do not repeat these

Each was measured and rejected, and the reasoning still holds. Re-running them costs hours and returns nothing.

**The GPU side, screened 2026-09-21 — every one of these is a null or nearly one**
- **`io_workers`.** Eight arms from 2 to 16 at a 40 GiB budget, two reps each, interleaved: every whole
  token inside 141.5-148.5 ms and the two reps of the shipped width span 5.5 ms of that. The in-eval /
  blocked split does not move. §9.26.
- **`CACHALOT_PAGE_CACHE=0`.** Same hit rate, same bytes, **+16 ms per token** — the loss is in the
  blocking and the CPU, and the GPU gets nothing back. The shipped `=1` is right. §9.26.
- **Attention's shapes.** A reuse layer's 0.441 ms is 0.377 of weight streaming and 0.064 of everything
  the shape idea would touch. A fixed-capacity `attention_kv` with a mask can only make the 14 % worse,
  and the `mx.compile` it would unlock is closed three ways already. §7.1.10, §9.21.
- **Quantizing `wo_a` back to FP8.** The shipped BF16 grouped `mx.matmul` reads twice the bytes at
  625 GB/s and is faster than every FP8 arm, including a one-launch upper bound that ignores the indexing
  a grouped kernel would need. §9.28.
- **The FP8 GEMV kernel's lanes-per-row policy.** A rule written for load balance picks the fastest split
  on five of six shapes; the best split per shape is 16.56 ms against the shipped 16.61. §9.29.
- **The shared expert.** 307 GB/s against 385 for `mx.sum` over the same bytes; the whole headroom is
  0.9 ms per token and the one structural idea in the shape is 0.4. §9.27.

**The expert kernel, measured 2026-09-21 on the path that ships**
- **Anything that makes the routed-expert matmuls faster.** The traced block costs 0.169 ms per layer,
  6.7 ms per token, which is what its three `mx.quantized_matmul` calls cost on their own (0.163 ms) — every
  cast, clamp and accumulate around them is fused away by `mx.compile`. `gather_qmm` over pre-stacked
  weights is 20-25 % better and therefore **≤1.5 ms**, and it needs six LRU slots made contiguous.
  `benchmarks/micro_topk_core.py`, `benchmarks/micro_expert_roofline.py`. Section 7.1.4.

**Routing prediction, all measured 2026-09-21 at the shipped 44 GiB budget unless stated**
- **Admitting a mispredicted load instead of dropping it.** A dropped expert is demanded again within eight
  tokens on 4.3 % of demand reads; keeping drops that long costs 2.7 GiB of a 24 GiB cache. Section 9.19.
- **A blocklist on re-reading a recently dropped expert.** 29.2 % of speculative reads are such re-reads, but
  30 % of them turn out right; at a lifetime of 8 it saves 8.1 wasted reads per token and adds 3.4 demand
  misses. Section 9.19.
- **Making the submission bookkeeping cheaper.** The whole per-token loop — forty `.tolist()`s, 240 tuple
  keys, 240 dict lookups — is 0.040 ms; the best rewrite saves 0.015; fusing the two `.tolist()`s with
  `mx.concatenate` costs 8.4. Section 9.19.
- **`CACHALOT_PREDICT_AHEAD=2` on the 2-bit bank.** 10.91 s against 10.85-10.91 for 64 tokens, with bytes up
  37 % to 861 MiB/token. The FP4 null said the drive was at its concurrency knee; here it is 55 % busy and
  the arm still returns nothing, so the reason is precision (51.1 wasted loads/token against 26.4), not
  bandwidth. Section 7.1.3.
- **Prediction width other than top-6.** top-8 is 1.0 % slower and top-4 is 2.3 % slower, on a 170 ms token
  reading 627 MiB — a different token from the 341 ms one that tuned the width, same answer. Section 7.1.3.
- **`sys.setswitchinterval`.** 0.001, 0.005 (default) and 0.020 s all decode 64 tokens in 10.85-10.89 s.
  Reader-thread scheduling is not what puts a streaming token's `rest` 33 ms above the all-resident floor.
  Section 7.1.3.

**Attention, measured 2026-09-21 on the thirty reuse layers**
- **`mx.compile(shapeless=True)` on decode attention.** It does not run: the path is built from
  `mx.fast.metal_kernel` custom kernels and shapeless tracing cannot infer their output shapes
  (`[Primitive::output_shapes] CustomKernel cannot infer output shapes`). Section 9.21.
- **A plain `mx.compile` trace of decode attention.** Not bit-identical — 2 of 12 positions matched, worst
  element 7.0e-02 — and a net loss even if that were acceptable, because `window_slot` and the compressed
  cache length change on every token, so every reuse layer retraces every token: 36.3 ms/token against the
  shipped 31.8. `benchmarks/micro_compile_attention.py`. Section 9.21.

**Storage and I/O**
- **Closing the achieved-bandwidth gap.** While the drive is busy the runtime already moves about 6 GB/s of the
  6.6–6.8 GB/s the drive delivers on cold experts.
- **The OS page cache under a large wired set.** A full 2x2 at 2 and 8 concurrent loaders, every arm on disjoint
  cold experts: all eight readings fall between 6.62 and 6.81 GB/s. Both variables null. `CACHALOT_PAGE_CACHE=1`
  stays on because it costs nothing and helps genuine repeat hits.
- **`F_RDAHEAD 0`** alongside `F_NOCACHE`: null.
- **Chunked expert reads**: a single expert read already saturates a stream.
- **Mirror striping with a stacked bank**: harmful, and the condition is the point. A configured mirror disables `_read_pieces_concurrently`, so a bank with nine pieces per expert falls back to serial reads. **An FP4 expert is one contiguous range and this does not apply**; mirror striping is shipped on FP4 at fraction 0.10 (section 9.11). Past about 15 % the USB drive becomes the bottleneck on any bank.
- **Engram on the internal SSD** for speed: null (it survives as lever 7 for robustness only).

**Kernels and dispatch, all screened 2026-09-21 on the 2-bit bank**
- **Splitting `hc_mixes` across threadgroups.** The shipped kernel does 24 dot products of length 20,480 plus
  a sum of squares in one threadgroup, so one GPU core reads the whole 0.98 MiB `hc_fn` matrix at about
  25 GB/s. A two-stage version (partial reductions in G threadgroups, then a tiny finish kernel) peaks at 8-16
  groups and 38 GB/s — **3.2 ms per token becomes 2.2 ms**, and past 32 groups the second dispatch costs more
  than the parallelism returns. Outputs agree to 6e-08. A millisecond is not worth a second kernel in the
  decode path; `benchmarks/micro_hc_mixes_parallel.py` keeps the measurement.
- **A fused multi-expert affine path.** `mx.gather_qmm` over pre-stacked weights beats the shipped per-expert
  loop by **20-25 %** and `concat`+`gather` by about 10 %, consistently across four runs (the first run of the
  series showed 79 % and was a cold-arm outlier — read the repeats). That is at most 3 ms per token of the
  13.9 ms routed-expert term, it ignores what it would cost to make six LRU slots contiguous, and the only
  way to get it without stacking is a custom kernel. **Screen, not a gate**: it says do not spend a session
  here. `benchmarks/micro_affine_expert_fusion.py`. This is the same conclusion 2026-09-16 reached on the
  FP4/affine-8 bank, now confirmed on the bank that ships.
- **Issuing the six experts projection-by-projection instead of expert-by-expert.** Identical output, same 18
  dispatches, 1-3 % — inside the noise.
- **Tracing the fused hyper-connection glue with `mx.compile`.** The instruction the v20 and v21 prompts
  both carried was to start here, on the strength of `benchmarks/micro_compile_hc.py`. That script times
  `hyper_connection_mlx`, and the runtime has been on the single-launch Metal kernels in
  `decode_fused_metal` since before it was written — the same mistake section 6.3.1 caught in the
  `route_topk` row. Re-asked on the path the runtime takes (`benchmarks/micro_compile_hc_fused.py`), the
  glue's whole graph construction is **1.13 ms per token** across all 240 launches, of which `mx.compile`
  takes 0.84 ms — and 0.63 ms of that 0.84 is taken by memoising the kernels' scalar parameters
  (section 9.17), which is a smaller change with no trace to keep valid. **The incremental value of
  compiling the glue is 0.2 ms per token.** Fused kernels were already the fix for this problem; there is no
  second helping. Outputs bit-identical on all three arms.
- **Tracing the router pass with `mx.compile`.** 0.40 ms per token of construction for the layer's own pass
  and the predictor's together, 0.10 ms compiled, so **0.30 ms per token**, bit-identical.
  `benchmarks/micro_compile_router.py`.
- **Submitting predicted loads from a dedicated thread instead of the decode thread.** Section 9.15 called
  this "an unexplored three-figure-millisecond-per-session idea" and section 9.18 prices what it was aiming
  at: 3.6 ms per token of CPU spent taking the store lock, checking slots and calling `submit` once per
  predicted expert. Handing the whole submission to one worker thread and returning is a **null, and worse
  on the median**: CPU outside eval 20.0, 19.3, 19.0 ms in line against 19.6, 20.4, 19.4 off-thread on the
  minimum — fully overlapping — and 20.6, 20.7, 20.9 against 21.6, 22.3, 21.2 on the median, which does not
  overlap in the wrong direction. **The GIL is the reason and it generalises**: the submission is Python, so
  moving it to another thread does not take it off the decode thread's critical path, it only adds a handoff.
  Do not reach for a thread to hide Python work from the decode loop; make the Python cheaper or make it
  disappear.
- **Fusing the layer's router with the predictor's into one scores launch.** Both passes read the same input
  vector with different gate matrices, so the two 384-row launches stack into one 768-row launch — and the
  768-row launch costs 0.018 ms against 384's 0.015, which is the occupancy argument working. It is still
  worth only **0.6 ms per token**, because the fused Metal router is 0.025 ms per call, not the 0.170 ms of
  the MLX router the 3 ms estimate came from. `benchmarks/micro_router_dual_gate.py`. **Screen, not a gate**:
  it says the second router pass is not where a token's time is.

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
- **Non-cumulative lookahead** (`CACHALOT_PREDICT_LEAD=2`, predict L+2 instead of L+1 at top-6, so the
  same number of experts is predicted). 328.5 against 335.0 ms per token, four runs a side interleaved,
  non-overlapping ranges, 2.0 % worse. **Its premise was wrong**: the same prediction count is not the same
  bytes, because precision falls from 55 % to 45 % and traffic rises 171 MiB per token on a drive that is
  80.5 % busy. The lead time itself works -- 171 MiB is about 30 ms of drive time and the arm loses only
  6.5 -- so a predictor with lead-1 recall two layers early is worth roughly 24 ms per token. Measured on the
  store *after* the lifetime fix, so unlike the `PREDICT_AHEAD=2` arm it is evidence about lead time.
  Section 9.12.1.
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
  recorded activations and shown to transfer across texts, and **2 of 6 long turns collapsing against FP4's
  none**, **0 of 5 Python blocks parsing against FP4's 2 of 10**, and no compiling C++ block on either side.
  A 221.5 GiB bank was built in 47 minutes and gated on matched generations to find this out. Bits, not fit,
  are binding above 2 bits. Sections 9.0 and 7.4.1 — the "31.6 against 8.9" this entry used to quote came
  from a broken checker and is withdrawn. Do not build another affine expert bank above 2 bits expecting
  quality.
- **A 9x9 wide grid for the affine fit at 3 bits**: 0.2588 against the 5x5 grid's 0.2587, for 378 ms per
  expert against 143. The 2-bit finding that a finer grid does not pay holds at 3 bits too.
- **Storing affine scales and biases in fp32 instead of bf16**: 0.5767 against 0.5789 of routed-expert output
  error, inside the noise. Scale precision is not where the 2-bit error lives.
- **A grid finer than 9x9 for the affine fit**: a 17x17 grid plus least-squares refinement buys 0.05 %.
- **Narrowing `INDEX_TOPK`** (512 shipped): the indexer costs 0.485, 0.474, 0.501 and 0.540 ms per source
  layer at widths 128, 256, 512 and 1024 — an eightfold change in the selection is worth 0.066 ms per layer
  and 0.26 ms per token, and it is not monotone below 512. The indexer's cost is its projections, its FP4
  index-K cache and the scoring, so a narrower index trades quality for nothing. Section 7.1.8.
- **Segmented LRU, decayed frequency and popularity-ordered prefill admission, re-run at the 9.49 MiB
  expert**: 0.9 and 0.5 points respectively over plain LRU, and popularity ordering is worse than first-come.
  The 2026-09-17 null survives the smaller bank.

### 9.33 `kernel_consts.py:39`'s store-blocked time, explained — **attribution artifact, not a cost, 2026-09-22**

The v31-v33 prompts carried this forward as unexplained: `benchmarks/profile_decode_sync.py --mode stream`
attributes 0.5 calls per token to `kernel_consts.py:39` (the `mx.eval(a)` inside `_u32_cached`/`_f32_cached`
on an LRU-cache miss), almost all of it — 31.2 of 31.6 ms per token in a fresh 28 GiB, 48-token streaming
run (`benchmarks/results/guarded/kcheck_20260922-132812.out`) — landing in the "store-blocked" column rather
than CPU.

**The mechanism is in the instrument, not in the kernel.** `profile_decode_sync.py` replaces `mx.eval`
globally and charges the whole wall-clock gap since the *previous* `mx.eval` call — including any time the
main thread spent blocked inside `expert_store.get_many()` — to whichever call site's `mx.eval` ends that
gap (`_site()` reads two frames up the stack, `_record` keys on it). It does not check whether that call
site caused the block. `kernel_consts.u32`/`f32` are called from `fp8_fused_metal.py`, `moe_fused_metal.py`
and `router_fused_metal.py` — kernels that run inside the same layer, immediately after `get_many()` returns
from its own blocking wait. On the rare token where one of those kernels needs a distinct scalar this
session has not built yet, its `mx.eval` is the first one to fire after the block ends, and the profiler
pins the block's whole duration to `kernel_consts.py:39` instead of to `moe_layer_metal.py:214`, where the
`get_many()` call that actually waited lives.

**And the 0.5 calls per token is cache warm-up, not a recurring cost.** `kernel_consts.py`'s own docstring:
"there are only a few dozen distinct ones in a session." A few dozen misses over a 48-token stream is
exactly the observed ~0.5/token average; on a live reply of several hundred tokens the same few dozen misses
land almost entirely in the first handful of tokens and the rate falls toward zero. There is nothing to
fix — the underlying wait is real and already counted correctly, once, at whichever site's `mx.eval` happens
to end it; only its label is sometimes wrong. **No lever, and the miscount does not point at a second
uncounted cost** — it's the same store-blocked time named twice in different rows of the same table, not
extra time.

Confidence: verified structurally (the instrument's charging rule, and which modules call `kernel_consts`
from which point in the per-layer sequence) and confirmed the effect reproduces at all, not confirmed by
tracing one specific miss end to end.

## 12. Pitfalls worth knowing before touching the code

- **Read which function an instrument calls before ranking a lever off it — fifth occurrence,
  2026-09-21.** A whole screen of four kernel variants was written, run and found bit-identical against
  `fp8_gemv_metal.fp8_gemv_quantized`, which the runtime does not call: `fp8_linear_quantized` dispatches
  to `fp8_fused_metal.fp8_gemv_decoded` whenever `CACHALOT_FUSED_FP8` is set, and it is set by default.
  The retired kernel reads the weight a byte at a time through a constant table; the shipped one reads
  `uint4` weights against a pre-decoded `float4` activation with a tuned lanes-per-row split, and was
  already 30 % faster than the best variant of the other. The two live in files whose names differ by one
  word. **Grep for the caller, not the definition.**
- **`mlx_peak_bytes` is bytes; divide it by 2^30, and check the units on both sides of a comparison.**
  Four sections quoted the MLX peak in GB against a wired limit in GiB and understated the headroom by
  7.4 % for five sessions. The arithmetic catches it in one line: resident experts plus the 10.4 GiB trunk,
  2.4 GiB of transient slots and a 1.55 GiB cache is the whole footprint, and if a quoted peak needs more
  non-expert memory than that, the units are wrong. §7.2.8.
- **`predicted_used` is not the numerator of a precision.** It counts only the predictions a demand
  request catches **still in flight** (`resident_store.py:629`); one that lands before it is demanded is
  an ordinary hit and never increments it. Two sections called `predicted_used / predicted_loads`
  "prediction precision". It is a lower bound by an unknown margin. Use `predict_ghost.py` and the offline
  replay. §7.2.7.
- **`--mode both` and `--mode stream` do not produce the same streaming arm.** The all-resident arm leaves
  240 experts pinned and changes what the continuation evicts; a baseline read off one and compared
  against arms read off the other is 7 ms out. §9.26.
- **A background thread's exception can die silently in the middle of a benchmark and the run still exits
  0.** `expert_read_scaling.py --wire-gib`'s ballast-heartbeat thread crashed on its first tick on MLX
  0.32.2 — `RuntimeError: There is no Stream(gpu, 0) in current thread`, because the array it pings was
  never `mx.eval`'d on the thread that created it, and 0.32.2 cannot resolve a default GPU stream for an
  array's first materialization from a different thread. Python prints the traceback to stderr and moves
  on; `main()` never sees it, the script's own exit code is 0, and every arm after the first ran on a
  ballast that had already unwired itself, with nothing in the benchmark's own output saying so. Three
  sweeps in a row showed the tell — wired GiB held for one or two arms, then collapsed to single digits for
  the rest — before it was read as a bug rather than as the result. **Read a background thread's own
  output, not just the parent's exit code, before trusting what a benchmark with a heartbeat or a ballast
  measured.** Fixed by evaluating the pinged array once on the main thread before spawning the thread.
  §7.1.11, §9.31.


- **`settle.sh` counts the script that calls it as a live runtime, and the A/B then waits forever.** Its
  guard is `pgrep -f "deepseek-v41/bin/python|cachalot"`, and `-f` matches whole command lines, so a shell
  whose argv *contains* the arm's command — which is what happens when a multi-arm script is passed to
  `bash` as text rather than written to a file — makes `runtime_alive 1` true for as long as the script
  lives. On 2026-09-21 two such scripts sat in `settle` with 73 GiB available and pressure 1, each waiting
  for the other and for itself; the arms that had already run were fine and the rest never started. **Write
  a multi-arm A/B to a file and run `bash the-file`**, and read `settle`'s own line — `runtime_alive` is
  printed on every tick.
- **The interactive one-liner does not survive line wrapping, and the failure is silent.** Pasted into zsh
  with the terminal's own wrapping, each wrapped line runs as its own command: `CACHALOT_MODEL_PATH=...` and
  `CACHALOT_EXPERT_BANK=...` become shell parameters that are never exported, the interpreter starts from a
  later line without them, and the trailing flags come back as `zsh: command not found: --max-seq-len`. The
  runtime then serves **FP4 off the USB drive** — which it will happily do, because the checkpoint holds a
  bank of its own. On 2026-09-21 that read **0.6 tok/s against the configuration's 7.6**, and both arms of an
  A/B were affected identically, so the A/B looked like a null instead of a broken command. The tells are in
  the banner: `hotlist: 456 experts preloaded (8.0 GiB in 9.1 s)` is **17.96 MiB per expert at 0.9 GB/s**,
  where the 2-bit bank on the internal SSD is 863 experts, 9.49 MiB each, in 2.0 s; a missing
  `--expert-budget-gib` shows up as an expert budget of 47.5 GiB rather than 44.0. **Use `./chat.sh`.** The
  bank line is now printed unconditionally rather than only when `CACHALOT_EXPERT_BANK` is set, so a session
  that is serving the wrong bank says so in its first three lines.
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

A healthy chat session **at a 52 GiB budget with an 80 GiB wired limit**, which is what 0.9.0 is run at,
looks like this (section 7.2.6): **92.4 % hit rate, 5,314 resident experts, 8.5-9.4 tok/s on replies past
500 tokens, MLX peaking at 67.74 GiB (§7.2.8), 16,585 of 52,763 predictions caught in flight.** At the older 44 GiB budget the same
session shape reads 90.3-91.1 % hit rate, 4,393-4,456 residents, 6.8-7.8 tok/s and 63.5-64.0 GiB of MLX
peak (section 7.2.3). The older reading below is kept because the two ways of misreading the machine that
follow it are still the ones people make:

    expert_hit_rate 90.7 %       better than the 87.3 % on record; the hotlist is part of it
    resident 4,431 experts       93.4 % of the budget, and 4,431 x 9,953,280 B exactly
    decode 6.3 to 7.1 tok/s      the upper half of the recorded 6.0-7.5 range
    prefix cache 19 hits, 1 miss
    mlx peak 55.5 GiB            under the 72 GiB wired limit (divide mlx_peak_bytes by 2^30, see 7.2.8)

Two ways to misread the machine while it runs:

**"RAM is at 81 %, so there is headroom."** There is not much, and 0.9.0 has now spent most of it. MLX
peaked at 55.5 GiB at a 44 GiB budget and **67.74 GiB at 52** (§7.2.8), against an 80 GiB wired limit on a 96 GiB
machine; the session ran clean with no pressure event, which is the evidence that 52 fits (section 7.2.6).
The next step on that curve, 52 to 60 GiB for another 2.3 points, would wire about 85 of 96 and is the
configuration class that panicked this machine twice. Section 9.4.

**"A live session cannot resolve a speed change."** It cannot resolve 5-8 ms, which is what that rule was
calibrated on and what made three A/Bs nulls. It resolved 22-27 ms on 2026-09-21 without difficulty, on
long turns whose rate four earlier sessions had repeated to a tenth of a point. Read the hit rate first,
then read long turns only, and do not read a short one at all — the same session gave 5.54 tok/s on 8
tokens and 9.42 on 544. Section 7.2.6.

**"The GPU is only at 56 %, so there is compute headroom."** That idle *is* the expert stall. At 7.08 tok/s a
token is 141 ms and the measured all-resident floor is 93 ms, so about a third of every token is the GPU
waiting on the drive. You cannot convert it by giving the GPU more of the same work. Only two things use it:
removing stalls (sections 9.10 and 9.4) or filling them with speculative work (section 9.1).

## 13. Reference commands

All are copy-paste ready and assume nothing about the current directory.

**The reference arm, run A — the diagnostic that says whether a quality problem is ours**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && set -a && . ~/.hermes/.env && set +a && python3 benchmarks/run_reference.py --pack ~/cachalot-corpus-pack --out ~/cachalot-runA-relace-fp4 --model deepseek/deepseek-v4.1-flash --provider relace/fp4 --attempts 5 --sleep 1.5
```

**Score any arm against Cachalot's, and against the reference**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/ ~/cachalot-runA-relace-fp4-scored ~/cachalot-runA-deepinfra-fp8-scored
```
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/include_integrity.py benchmarks/results/coding/20260919-085931_DeepSeek-V4.1-Flash-fp4-experts/ ~/cachalot-runA-relace-fp4-scored ~/cachalot-runA-deepinfra-fp8-scored
```

**Generate a Cachalot arm on the same 40 prompts**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 14400 --tag coding -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/coding_quality.py
```

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

**Where a streaming token's time goes, against an all-resident one — the instrument that found the Engram reads**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh --budget-gib 36 && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 1200 --tag sync -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/profile_decode_sync.py --prompt-tokens 512 --mode both --stream-tokens 48
```

**Put the Engram change back in its box, for an A/B against it**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh --budget-gib 36 && benchmarks/guarded_run.sh --budget-gib 36 --max-seconds 1200 --tag anateg_base -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 CACHALOT_ENGRAM_PARALLEL_MIN=1000000 CACHALOT_DECODE_ENGRAM_PREFETCH=0 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/decode_anatomy.py --prompt-tokens 512 --decode-tokens 96
```

**Whether a streaming token's excess survives when the misses do not — the two-pass arm**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh --budget-gib 40 && benchmarks/guarded_run.sh --budget-gib 40 --max-seconds 1500 --tag sync2pass -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=72 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/profile_decode_sync.py --prompt-tokens 512 --mode both --stream-tokens 24 --stream-passes 2
```

**The three GPU screens added 2026-09-21 — no model, no guardian, seconds each**
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/micro_fp8_gemv_kernel.py
```
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/micro_shared_expert_roofline.py
```
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/micro_wo_a.py
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

**Does the generated code compile?** — the paired bank comparison, no GPU needed. Read the **compile**
column first; the error density is only defined over blocks that reached the end of the file and were not
cut off at the token cap, and a density marked `(floor)` comes from a checker that stops at the first error.
Section 7.4.1 is why.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py /tmp/replies_bank_a /tmp/replies_bank_b
```

**Generate the coding corpus against a bank** — twenty independent tasks, about 1.7 hours on FP4 at two
seeds. Swap `CACHALOT_EXPERT_BANK` for the arm under test and run both arms with `settle.sh` between.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 14400 --tag cq-fp4 -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 CACHALOT_MIRROR_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_MIRROR_FRACTION=0.10 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/coding_quality.py --seeds 2 --max-new-tokens 2000
```

**Score a run, or two against each other** — no GPU. Pass the run directories, not the replies: the manifest
is what lets the scorer refuse an unfinished arm and tell a snippet from a program.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py benchmarks/results/coding/<run-a> benchmarks/results/coding/<run-b>
```

**Is one token mis-ranked, or is everything blurry?** — the section 7.4.4 experiment, one forward pass, no
sampling and no compiler
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 3600 --tag rank-runtime -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/token_rank_probe.py --tokens 600
```

**Smoke the harness in ten minutes** before committing hours to it
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 3600 --tag cq-smoke -- env CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/coding_quality.py --seeds 1 --max-new-tokens 1400 --only cpp-lru-cache,py-retry-decorator,cpp-string-split-snippet
```

**The prefetch-lead A/B** — the 2026-09-19 lever, four runs a side interleaved both ways, `settle.sh`
between. Put it in the background and kill it by the recorded PID, never by a `pgrep` pattern.
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && nohup benchmarks/ab_predict_lead.sh > /tmp/ab_predict_lead.log 2>&1 & echo $! > /tmp/ab_predict_lead.pid
```

**Re-score the two saved arms** — no generation, seconds, and the command behind section 7.4.1
```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/code_validity.py benchmarks/results/replies/lc_fp4 benchmarks/results/replies/lc_q3g64act
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
| `benchmarks/micro_compile_hc_fused.py` | what tracing the **fused** hyper-connection glue is worth, and what memoising its constants is worth |
| `benchmarks/micro_compile_router.py` | what tracing the router pass is worth |
| `benchmarks/nll_expert_precision.py` | the quality gate; dense arms and the production arm |
| `benchmarks/decode_anatomy.py` | blocked time split by cause, reads split by worker pool |
| `src/cachalot/model/fp8_fused_metal.py` | **the FP8 GEMV the runtime actually calls** (`fp8_gemv_decoded`, `_gemv_v2_kernel`) and the fused activation quantization; 5.14 GB of weights per token go through it |
| `src/cachalot/model/fp8_gemv_metal.py` | the earlier FP8 GEMV, still the fallback when `CACHALOT_FUSED_FP8=0`. **Not the shipped path** |
| `src/cachalot/model/wo_a_dequant.py` | why `wo_a` is BF16 resident, and the conversion it reproduces |
| `benchmarks/micro_fp8_gemv_kernel.py` | the largest GPU path priced per shape against `mx.sum`, every lanes-per-row split, bit-identity checked |
| `benchmarks/micro_shared_expert_roofline.py` | the shared expert against the memory wall, and the `w1`/`w3` fusion |
| `benchmarks/micro_wo_a.py` | `wo_a` BF16 against every FP8 form of the same projection |
| `benchmarks/profile_decode_sync.py` | the three-way split per call site, `--stream-passes` for a streaming token that does not miss, `--io-workers` for the device queue depth |
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
| `benchmarks/decode_rate_by_block.py` | **the rate over one long generation, in blocks of 256**: tok/s, hit rate, misses and bytes per token as a turn runs, which is the only way to see a within-turn effect |
| `benchmarks/predict_ghost.py` | **what happens to a mispredicted expert after it is dropped**: how often it is demanded again, how often it is speculatively re-read, and what a blocklist of each lifetime would have done |
| `benchmarks/micro_predict_submit.py` | the prediction submission bookkeeping, four arms, no model — 0.040 ms per token |
| `benchmarks/micro_expert_roofline.py` | the routed-expert matmuls against `gather_qmm`, a dense matvec and 240 distinct experts — how far the expert kernel is from the memory wall, no model |
| `benchmarks/micro_topk_core.py` | `topk_core` built back one layer at a time, ending at the compiled block the runtime calls, no model |
| `benchmarks/micro_eval_floor.py` | **what one `mx.eval` costs**: six ways of reading a value back, the cost against queue depth, k arrays in one eval against k evals, and whether a launch submitted first hides the round trip — no model |
| `benchmarks/micro_compile_attention.py` | the `mx.compile` and `shapeless=True` screen for decode attention, over consecutive positions so a retrace is visible; parity, construction, first call and chained |
| `benchmarks/decode_fingerprint.py` | **the cheap numerics check for a speed arm**: 16 greedy tokens with their ids and fp32 logit checksums, to be diffed between two arms |
| `benchmarks/profile_decode_sync.py` | **the instrument that found the CPU third and then the Engram reads**: wraps `mx.eval`/`mx.synchronize` for one token and attributes every wait, and the gap before it, to its call site; `--mode both` runs an all-resident arm and a real streaming continuation side by side and splits each gap into store-blocked, Engram `pread` and CPU |
| `src/cachalot/storage/engram_reader.py` | the random-access Engram row reader; `CACHALOT_ENGRAM_PARALLEL_MIN` (8) is the batch size above which its sixteen-worker pool is used instead of the calling thread |
| `tests/test_engram_reader_parallel.py` | pins the parallel row path against the serial one and against the table itself at seven batch sizes, and pins that the default threshold is at or below a decode batch |
| `benchmarks/profile_decode_gpu.py` | per-piece GPU time the way a token pays it — many launches in one lazy graph, one eval — plus the whole-token time with `ASYNC_MOE` on and off; takes `--prompt-tokens` and handles an affine bank |
| `benchmarks/profile_decode_components.py` | per-piece time with a barrier around each call. **Use it to compare two implementations of one piece, never to apportion a token** — its rows sum to 164 ms against a 94 ms token |
| `benchmarks/profile_decode_cpu.py` | the same token under cProfile, for the CPU third; MLX's nanobind calls are invisible to it and land in the caller's `tottime` |
| `benchmarks/micro_affine_cpu.py` | what the affine expert path costs to *construct*, with and without cached slot views |
| `src/cachalot/model/kernel_consts.py` | the memoised one-element parameter arrays every fused Metal kernel is handed; `CACHALOT_KERNEL_CONSTS=0` rebuilds them per call |
| `tests/test_kernel_consts.py` | pins that the cache returns one object, that the switch really rebuilds, and that the fused kernels are bit-identical either way |
| `benchmarks/profile_decode_layers.py` | **per-layer attribution with no barrier added**: wall time, time inside that layer's own evals and the CPU remainder, by layer and by layer class |
| `benchmarks/micro_compile_moe.py` | the `mx.compile` screen for the MoE block: construction time, chained time, and whether the traced output is bit-identical |
| `benchmarks/micro_router_dual_gate.py` | the layer's router pass against the predictor's, and both stacked into one 768-row scores launch |
| `benchmarks/micro_affine_expert_fusion.py` | the fused-expert screen: per-expert loop against `concat` and `gather_qmm` upper bounds |
| `benchmarks/micro_hc_mixes_parallel.py` | `hc_mixes` against a multi-threadgroup version, output-checked |
| `benchmarks/quant_fit_screen.py` | candidate affine fits at one format, in minutes |
| `tests/test_dspark_draft.py` | pins the draft's attention index set, whose failure mode is a false null |
| `benchmarks/repetition_quality.py` | free-running collapse rate; canned-context and no-prefix-cache arms |
| `benchmarks/code_validity.py` | compiles generated code blocks; compile rate first, fatal rate second, density only over comparable blocks |
| `tests/test_code_validity.py` | pins the missing-header false success that made the gate report 8.9, and the block accounting behind it |
| `benchmarks/coding_tasks.json` | the coding corpus: twenty independent tasks, each declaring its language and whether it must compile alone |
| `benchmarks/coding_quality.py` | generates the corpus against one bank, one process, with the run manifest that makes a result traceable and an unfinished run refusable |
| `tests/test_coding_quality.py` | pins the corpus shape and the per-reply accounting: no code, wrong language, untagged fence, truncation |
| `benchmarks/ab_predict_lead.sh` | the interleaved prefetch-lead A/B, four runs a side both ways |
| `benchmarks/token_rank_probe.py` | teacher-forced rank of one suspected token, with a within-run control; separates a mis-ranked token from general blur without sampling or a compiler |
| `tests/test_sampling_penalties.py` | pins that the frequency penalty grows with the count and survives greedy |
| `benchmarks/run_reference.py` | run A, the diagnostic: the corpus against a hosted endpoint, no harness, one pinned provider, transport retried and content never retried |
| `benchmarks/export_corpus.py` | writes `~/cachalot-corpus-pack`: the prompts, the settings, the run script and the instructions another runtime is handed |
| `benchmarks/include_integrity.py` | the pre-registered include/import check; the one quality signal that is immune to truncation and to a compiler giving up |
| `~/cachalot-runA-relace-fp4-scored/` | the FP4 reference arm, 40 of 40, with `ANALYSIS.md` and the notes that say how the provider was pinned |
| `~/cachalot-runA-deepinfra-fp8-scored/` | the FP8 reference arm, 40 of 40 — the control that rules the expert format out |
| `~/cachalot-runA-relace-fp4-greedy-scored/` | six tasks at temperature 0, paired with Cachalot's greedy run |
| `~/cachalot-runB-hermes-scored/` | the Hermes harness arm, with the two sessions recovered from the Hermes database and an `ANALYSIS.md` explaining why it answers a different question |
| `benchmarks/quant_affine.py` | the fits; `fit_search`/`refine_lsq` work at any width, `dequantized()` screens without packing |
| `tests/test_bank_writer.py` | pins that the quantizer's output fills exactly what the shard header reserved |

## 15. The Hermes HTTP server — more done than tracked, smoke-tested at the shipped configuration, 2026-09-22

**Priority above the next speed lever, at Hamed's request.** He wants Cachalot behind an OpenAI-compatible
HTTP endpoint so Hermes Agent Desktop can drive it directly, the way it drives a hosted model.

**It already existed and was further along than this document tracked.** `src/cachalot/server/app.py` and
`engine.py` (695 lines) implement `/health`, `/v1/models`, `/v1/stats`, `/v1/chat/completions` (streaming
SSE and non-streaming), `/v1/completions`, Bearer auth, thinking mode, reasoning effort, and OpenAI-style
tool calls, built on the same `V41Model` / `stream_tokens` path as `cachalot chat`. README already listed it
as "✅ working, tested" from a 2026-09-15 smoke run (`benchmarks/results/server_smoke.log`), on the checkpoint
path and budget of that day. Nothing in `docs/HANDOFF.md` referenced it before this section.

**One real bug found and fixed: the server's default frequency penalty was still 0.2.** `ServerConfig` in
`app.py` set `default_frequency_penalty = 0.2` with a comment citing the "62 % collapse / 12 % with the
penalty" finding — the exact claim section 9.9 retracted once the transposed residual mix was found: through
the fixed runtime the collapse rate is 0 of 12 with the penalty off, on both banks, and `cachalot chat`
dropped the flag entirely in section 4. The server was the one remaining place still defaulting to the old,
retracted value, so any unmodified OpenAI client — including a default Hermes connection — would have had
its code replies distorted by a penalty nothing pays for any more. **Fixed**: `default_frequency_penalty =
0.0`. Two tests in `tests/test_server.py` asserted the stale 0.2 as the *default* for an unmodified client
and for `/v1/completions`; both updated to assert 0.0, and `test_the_server_default_is_configurable` now
sets `default_frequency_penalty=0.2` explicitly to prove the knob still works. 233 tests still pass.

**`serve.sh` added**, mirroring `chat.sh`: same env vars, same pgrep guard against a second runtime, the
shipped 52 GiB budget and 80 GiB wired limit, port 8011.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ./serve.sh
```

**Smoke-tested against the real shipped configuration this session** (52 GiB, 2-bit g128, hotlist, `serve.sh`
unmodified) — `/v1/models`, a non-streaming completion, a streaming completion with `stream_options.
include_usage`, and a tool-calling request:

```json
{"role": "assistant", "content": "I'll check the current weather in Paris for you.",
 "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
                  "id": "call_5133741_0"}]}
```

finish_reason `tool_calls`, arguments valid JSON matching the declared schema, content produced before the
call the way a client expects. Streaming usage reported `decode_tok_per_s` alongside token counts, matching
`/stats`' own accounting. All four checks passed; the server was stopped afterward (`pkill -f "cachalot.cli
serve"`) so it does not hold the machine's only GPU slot.

**What is not yet checked, and is Job 1 for the next session.** This was a curl smoke test, not a Hermes
Agent Desktop connection. Unverified: whether Hermes's own client sends anything `ChatCompletionRequest`
does not model (a `content` list of parts rather than a string, an unexpected `tool_choice` shape, a
`response_format` Cachalot does not implement), whether Hermes's tool-call parser accepts the exact shape
above, and how Hermes reacts to a single-flight engine — `engine.py`'s docstring is explicit that one model
instance serves one request at a time on a lock, so a second Hermes conversation, or a background task,
queues rather than running concurrently. Point Hermes at `http://127.0.0.1:8011/v1`, model id
`deepseek-v4.1-flash`, any placeholder API key (no `--api-key` was set), and read what actually breaks.

## 16. Vision — scoped, not started, second priority at Hamed's request

**Not implemented.** `src/cachalot/model/resident_trunk.py:39-51` deliberately filters every `vision.*`,
`aligner.*` and image-delimiter tensor out of the resident trunk, with the comment "Vision + MTP paths are
deliberately excluded from the first text-only runtime." That filter is the entire extent of vision-awareness
in this codebase today.

**The weights are real and already on disk.** The checkpoint's own `model.safetensors.index.json` carries
263 `vision.*`/`aligner.*` tensors, all in the first of 48 shards (`model-00001-of-00048.safetensors`) — no
extra download. `config.json` has the full vision block: 32 ViT layers, 1024-dim, 16 heads, patch size 14,
downsample ratio 3, `image_token_id` 129264, plus 43 `layers.N.ffn.gate.bias_vl` tensors — a second,
image-specific routing bias for the 43 MoE layers, selected per token rather than per request.

**The reference implementation is in the checkpoint directory and is small.**
`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/vision.py` (114 lines, PyTorch) is a standard ViT:
`PatchEmbed` (one linear layer), a stack of pre-norm `Block`s (bidirectional attention with 2D RoPE, no
causal mask, then a SwiGLU MLP), a final `RMSNorm`, and an `Aligner` that space-to-depth downsamples by 3x3
and projects into the text model's 5120-dim embedding space with a two-layer MLP.
`image_processor.py` (its docstring is exact about the contract) turns an image into a `n_vit_h x n_vit_w`
patch grid, picks the largest aspect-preserving resize whose token count fits `vision_max_n_token` (1024),
and emits the token sequence `[IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]`,
every position carrying `image_token_id` in `input_ids` with a separate `token_types` array distinguishing
the four roles. `model.py`'s `merge_image_embeddings` overwrites each image's token span in the embedded
sequence with the aligner's output rows in reading order, and the delimiter positions get three learned
embedding vectors (`image_start`, `image_end`, `image_newline`) — no MoE routing for those three, only for
the `[IMAGE]` positions, which is what the per-token `bias_vl` selects.

**Cost estimate.** Roughly 480 M parameters across the ViT and aligner (32 layers x ~13 M + ~73 M in the
aligner) — under 1 GiB even at BF16. This competes with nothing on the expert budget; it is trunk-sized
memory, alongside the 10.4 GiB the text trunk already holds resident.

**What porting it means, concretely, as four pieces:**
1. **The ViT itself in MLX** — patch embed, 2D RoPE bidirectional attention, SwiGLU MLP, final norm. No
   routed experts, no streaming, no cache; every piece already exists in this codebase in a text-attention
   form to copy the pattern from (`sparse_attn_mlx.py`, `norm_rope_mlx.py`) but none of it is directly
   reusable — vision attention is dense and bidirectional, not compressed or causal.
2. **`image_processor.py` ported near-verbatim** — it is pure NumPy/PIL preprocessing with no PyTorch
   dependency in the parts that matter (the resize-ratio solver, the patchify), so this is mostly a straight
   port plus removing the one `torch.Tensor` return type.
3. **The splice into prefill** — `merge_image_embeddings` writing aligner rows into the embedded sequence at
   `image_token_id` positions before layer 0, and threading `image_mask` through to the 43 MoE layers with a
   `bias_vl`. The router kernel (`router_fused_metal.py`, `router_mlx.py`) already takes one `bias` array
   applied uniformly to a whole batch; per-token bias selection between `bias` and `bias_vl` by `image_mask`
   is a real change to that kernel, not a parameter swap, and is the trickiest piece architecturally.
4. **The server's `/v1/chat/completions` accepting image content** — OpenAI's `content: [{"type":
   "image_url", ...}]` message shape, decoding the URL or base64 payload, and running it through the ported
   `image_processor.py` before building the prompt. `image_processor.py`'s own image-loading path already
   handles a URL via `urlopen` and a data URI via `base64`, so the decode side is close to done; the message
   shape has to be added to `ChatCompletionRequest`, which currently types `content` as part of an untyped
   `dict`.

**Order of attack for the next session that picks this up:** (1) and (2) together as a standalone feasibility
spike — load the ViT and aligner weights, run one image through them in MLX, and diff the aligner output
against the reference `vision.py` run in PyTorch on the same image and patches, bit tolerance aside. That
proves the port before anything is wired into the text model. (3) is the real engineering work and should not
start until (1) is numerically checked. (4) is server plumbing and can happen in parallel with (3) once (1)
is done, since a hand-built prompt with a fake image span can exercise the server path without a working ViT.

### 16.1 Piece 1+2 numerically checked — 2026-09-22

The feasibility spike above is done and passed. `src/cachalot/model/vision_mlx.py` ports the ViT (patch
embed, 2D-RoPE bidirectional attention via `mx.fast.scaled_dot_product_attention`, SwiGLU MLP, final
RMSNorm) and the `Aligner` (space-to-depth downsample reproduced by reshape/transpose since MLX has no
`unfold`, then the two-layer projection). `src/cachalot/model/image_processor_mlx.py` ports the resize-ratio
solver and patchify, PIL replacing torch for image I/O and `ml_dtypes` supplying the BF16 patches. Neither
touches `resident_trunk.py`'s filter or `TextDecodeRuntime` — piece 3 is still not started, as instructed.

`benchmarks/vision_parity_check.py` runs one real image
(`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/assets/dsv41_kv_cache.png`, downsized to 700x486 for a
tractable CPU reference forward — the full-resolution image drives the ViT to ~9,200 patches and a
dense-bidirectional-attention CPU forward pass over that many tokens on 32 layers did not finish in a
reasonable time; a smaller image exercises the identical code paths) through both the MLX port and the
official PyTorch `vision.py`/`image_processor.py`, loaded dynamically from the checkpoint's own
`inference/` directory the same way `generation.py.load_official_encoding` loads `encoding.py`. Two checks:

- **Image preprocessing.** MLX and reference grids agree exactly — `n_vit=(35,50) n_llm=(12,17)` both sides
  — and the patch values are bit-identical, `max|diff| = 0.0`.
- **ViT+Aligner, same bit-identical patches, same checkpoint weights, both cast to BF16** (matching the
  checkpoint's shipped precision): `max|diff| = 3.78e-2` against `max|value| = 0.86` (one outlier element
  out of 204x5120 = 1,044,480), `mean|diff| = 6.9e-4`. Rerun with both paths upcast to FP32 (isolating BF16
  rounding noise from an actual defect): `max|diff|` drops to `6.7e-6`, `mean|diff|` to `1.2e-7` — machine
  precision. The BF16 diff is accumulated rounding across 32 layers, not a bug.

Needs `torch` and `pillow` in the venv — dev-only, not a `cachalot` runtime dependency; `pillow` is a real
runtime dependency of `image_processor_mlx.py` once piece 3/4 wire it in and is declared as the `vision`
optional extra in `pyproject.toml`. `~/venvs/deepseek-v41/bin/pip install torch pillow` before running the
parity check. 238 existing tests still pass; no test suite entry was added for the vision port itself since
it is unwired and the parity script is the checkpoint.

**Next session on this thread starts piece 3**: the splice into prefill (`merge_image_embeddings`,
per-token `bias_vl` in the router kernel) — the real engineering HANDOFF section 16 flagged as the trickiest
piece, now unblocked since piece 1 is numerically checked.

### 16.2 Piece 3, step 1 — `merge_image_embeddings` written and tested, not wired — 2026-09-22

Step 1 of piece 3's three-part list (§16, step 2 is per-token `bias_vl` in the router kernel, step 3 is
threading `image_mask` through `TextDecodeRuntime`) is done in isolation. The splice point is
`text_decode_runtime.py:1826-1841`, inside `_prefill_tokens_impl`:

    x = mx.stack(
        [embed_token_decode(token_id, self._global("embed.weight"), hc_mult=HC_MULT)
         for token_id in token_ids],
        axis=0,
    )

**The "trickiest piece architecturally" turned out simpler than feared at this one step.** The worry going
in was that `hc_mult`'s hyper-connection expansion might mean an image row needs some learned transform to
enter the pre-mix stream correctly — the exact defect class (a residual-mix bug) that cost this project
three sessions before, section 7.4.8. Reading `model_boundary_mlx.embed_token_decode` settles it:  the
official input boundary is `h = self.embed(input_ids); h = h.unsqueeze(2).repeat(..., hc_mult, ...)` — a
bare broadcast of the embedding row, no learned expansion, no pre-mix arithmetic at this boundary. An image
row from `vision_embed()` lives in the same text-embedding space and needs exactly the same broadcast, not a
different numerical path.

`vision_mlx.merge_image_embeddings(token_ids, embed_weight, image_token_id, image_rows, hc_mult=4)` — where
`image_token_id=129264` is the reference's default (`inference/model.py:128`) — walks `token_ids`, and at
each position either calls `embed_token_decode` unchanged (text) or broadcasts the next row of `image_rows`
hc_mult times, cast to `embed_weight`'s dtype, consumed in reading order (image). It raises if the prompt's
`image_token_id` count and `image_rows`' row count disagree in either direction — an image-bearing prompt
that is malformed on either side fails loud here rather than silently misaligning image and text tokens
later. Returns `[n_tokens, hc_mult, dim]`, the exact shape the `mx.stack(...)` above already produces, so it
is a drop-in replacement for that line once piece 3's other two steps exist to route to it.

**Not wired into `_prefill_tokens_impl` yet, on purpose.** Wiring the call site now would either be inert
(no code path ever passes image content in) or, worse, silently wrong (an image-bearing prefill would run
with `bias` instead of the per-token `bias_vl` step 2 has not built yet, and no `image_mask` to select with).
`tests/test_vision_prefill_splice.py`, 5 cases: an all-text prompt is bit-identical to the unmodified
`embed_token_decode` stack; an interleaved text/image prompt places each image row correctly in reading
order and leaves text positions untouched; too few image rows, too many image rows, and image rows supplied
for a prompt with no `image_token_id` positions all raise `ValueError` rather than silently misaligning.
247/247 project tests pass (242 before this session's wq_a/wkv fusion, +5 here).

**What's left for piece 3, unchanged in shape from §16's original plan**: per-token `bias_vl` selection in
`router_fused_metal.py`/`router_mlx.py` (today takes one uniform `bias`; needs to select `bias_vl` for
tokens where `image_mask` is set, for the 43 MoE layers that carry a `bias_vl` — a real signature change,
not a parameter swap) and threading `image_mask` itself from wherever piece 4's server-side image-content
parsing will eventually produce it, through `_prefill_tokens_impl`, to reach both this splice and the router
change. Piece 4 (the server's `ChatCompletionRequest` image-content parsing) still should not start before
piece 3 has a full working splice, per §16 — there is still no image-bearing prompt to test piece 4 against
until steps 2 and 3 exist too.

### 16.3 Piece 3, steps 2-3 — per-token bias_vl and image_mask threaded end to end, wired and live-smoke-tested — 2026-09-22

**Correction to §16/§16.2's "43 MoE layers" figure.** `model.safetensors.index.json` carries 43 `bias_vl`
tensors, but three of them are `mtp.{0,1,2}.ffn.gate.bias_vl` — the MTP layers `resident_trunk.py`'s filter
already excludes from this text-only runtime (§16's own opening line: "Vision + MTP paths are deliberately
excluded"). The real count for this runtime is **40**, one per main layer, exactly matching `bias`
(`layers.{0..39}.ffn.gate.bias`) 1:1. Both are `F32 [384]`, same shape, confirmed by reading the shard header
directly rather than trusting the tensor-name count. `bias_vl` was never filtered by `resident_trunk.py` (it
matches none of that function's exclusion patterns), so all 40 are already resident and have been since piece
1+2 landed — nothing to load, only something to use.

**The router-kernel risk named in §16/§16.2 turned out to apply to exactly one function.**
`router_fused_metal.route_topk_fused` (decode) and `router_mlx.route_topk` (prefill's per-token Python-loop
fallback) already take one `bias` per call, one token at a time — a decode token is never an image position
(images only ever appear in a prompt, i.e. prefill), so neither needed touching. The real batched router —
`moe_prefill_batched.route_topk_rows`, the only path `moe_prefill_grouped`'s default (`batched=True`) ever
calls in production, and the one every caller uses (`batched=False` is set nowhere in this codebase) — is the
one function that broadcasts one `bias` array to every row via `scores + bf`. That is where per-token
selection is a real change, exactly as flagged, and it is pure MLX (`mx.where`, no Metal kernel), not the
Metal-kernel rewrite the "not a parameter swap" language might have suggested.

**The change.** `route_topk_rows` gained `bias_vl`/`image_mask`, both optional and enforced both-or-neither
(`ValueError` if only one is given): `selection_bias = mx.where(image_mask[:, None], bias_vl[None, :],
bias[None, :])` replaces the uniform `bf[None, :]` broadcast when both are given, otherwise the old single
line runs unchanged — bit-identical to before this existed (`tests/test_route_topk_bias_vl.py`, 5 cases,
checked against `router_mlx.route_topk` called once per row with the bias that row should have used; scores
and weights compared with a tolerance because the batched matmul and the per-row reference matmul sum in a
different order, the same caveat this file's own docstring already carries — indices are compared exactly).
`moe_prefill_grouped` gained matching `gate_bias_vl`/`image_mask` kwargs, forwarded to `route_topk_rows`, and
raises `NotImplementedError` if `image_mask` is given with `batched=False` before touching
`expert_index`/`expert_store` (`tests/test_moe_prefill_grouped_image_mask.py`). All five
`block_*_prefill.py` modules (`layer0`, `sliding_window`, `compressed_source`, `compressed_reuse`,
`compressed_index_source`) gained the same two keyword-only parameters, threaded straight through to their
`moe_prefill_grouped` call — mechanical, one line each, no existing unit tests for these block-level
functions to extend (none existed before this session; the project's own precedent for this layer is a live
smoke test, not a synthetic-weight unit test, since a real block call needs the full attention/indexer/HC
weight set to mean anything).

**Step 3, the actual splice.** `_prefill_tokens_impl` and `prefill_tokens` gained `image_rows: mx.array |
None = None` and `image_token_id: int = IMAGE_TOKEN_ID` (`129264`, `config.json`'s value, now a module
constant). The old `mx.stack([embed_token_decode(...) for token_id in token_ids], axis=0)` is replaced
unconditionally by piece 3 step 1's `merge_image_embeddings(...)` — unconditionally, not behind an `if`,
because the function is already bit-identical to the old stack when `image_rows` is `None` (§16.2). A local
`image_mask` is built once (`None` when `image_rows is None`, else the per-token `token_id ==
image_token_id` boolean array) and passed to all five layer-block call sites through a small closure,
`_gate_bias_vl_for(layer_id)`, that returns `None` — not the tensor — when there is no image, so the
overwhelmingly common text-only prefill never even fetches `ffn.gate.bias_vl` and `route_topk_rows` takes its
old no-`bias_vl` branch. Decode's own `_common_block_kwargs`/block functions are untouched; images cannot
appear at decode time, so `route_topk_fused` never needed this.

**Live-smoke-tested on the real 4-bit bank** (`benchmarks/vision_piece3_prefill_smoke.py`, same pattern as
`qkv_fusion_live_smoke.py`): a plain-text prefill first, confirming the new kwarg threading through all five
block variants has not disturbed ordinary text prefill (first decode token `'Hello'` on the same greeting
prompt prior smoke tests used — unchanged); then the same token ids with three interior positions overwritten
to `IMAGE_TOKEN_ID` and random `[3, 5120]` `image_rows` (no vision encoder is wired yet — piece 4 is still not
started, so these are not real aligner rows, the same standard the qkv fusion smoke test used before its
HANDOFF section shipped: proving no crash and finite output, not real-image correctness). Both prefills and
eight follow-on decode tokens produced finite logits, no crash. 253/253 project tests pass (247 before this
session, +5 `test_route_topk_bias_vl.py`, +1 `test_moe_prefill_grouped_image_mask.py`).

**Piece 3 is now fully wired end to end and piece 4 (the server's `ChatCompletionRequest` image-content
parsing) is unblocked**, per §16's original ordering — piece 4 can now build against a real, working
`prefill_tokens(image_rows=..., image_token_id=...)` call, and once piece 4 produces real `image_rows` from
`vision_mlx.vision_embed()` on an actual image, this session's live smoke becomes a real numerical check
rather than a plumbing check.

### 15.1 Job 1 done: Hermes Agent drives Cachalot, tools and images included — 2026-09-23

**Run from this session, not by Hamed.** The Hermes CLI is installed (`~/.local/bin/hermes`, the same
agent loop and client as Hermes Agent Desktop), so Job 1 no longer needed Hamed's own session. To leave his
Hermes configuration untouched, it ran against an isolated `HERMES_HOME`: an empty directory whose only
file is a `config.yaml` naming a `custom:cachalot` provider at `http://127.0.0.1:8011/v1`
(`docs/integrations.md` has the file). The toolset was the stock `hermes-cli`: 24 tools, including
`terminal`, `read_file`, `search_files`, `write_file`, `patch` and `vision_analyze`. The server ran with
`CACHALOT_SERVER_DUMP` set, which appends every request body to a JSON-lines file, and each request left one
`[request]` line on stderr (both new this session).

**Four things broke, in the order Hermes hit them. All are fixed.**

1. **Hermes refused to connect at all.** `/v1/models` reported `max_context_length` 32,768, and Hermes
   rejects any model below 64,000 tokens before sending anything. `serve.sh` now serves 65,536; `chat.sh`
   is unchanged at 32,768. At 65,536 the preallocated compressed-KV caches cost about 84 MB more.
2. **Every main request died in prefill with a Metal out-of-memory**, retried eleven times. Hermes's
   first prompt is 13,504 tokens (a 17,359-character system prompt plus 24 tool schemas), and one-shot
   layer-major prefill of that many tokens does not fit at the shipped budget. Section 15.2.
3. **Hermes's session-title call sends `reasoning_effort: "none"`**, and the official encoding asserts on
   anything but an int in [1, 100] or `low`/`high`/`max`, so the server returned 500. `app.py` now maps
   OpenAI-style values: `none`/`off`/`minimal` turn thinking off, `low`→`low`, `medium`→50, `high`→`high`,
   `xhigh`/`max`/`ultra`→`max`, a numeric string → int. Anything else is a 400.
4. **A client that disconnected still cost a full prefill.** The SSE loop only noticed a disconnect when a
   token arrived, so every abandoned retry, queued behind the single-flight lock, ran its whole prefill
   once the lock freed. The loop now polls for a disconnect every second, `stream_chat` drops a request
   already cancelled when it gets the lock, and prefill checks `cancel` between chunks. The same loop sends
   an SSE `: keep-alive` comment every 15 s of silence, so proxies and stale-stream detectors do not cut a
   stream that is quiet through a minutes-long prefill. Hermes's own read timeout for a local endpoint is
   already 1800 s (`chat_completion_helpers._stream_timeouts`).

**Then it worked.** The shapes v37 feared never came up. Hermes sends plain-string `content`, and the
official encoding already handles content-part lists anyway. `tool_choice` never appeared. The session
title's `response_format: json_schema` is passed through to the encoding and answered in plain text,
which Hermes accepts. The exchange, verbatim from the server log and the request dump:

| request | prompt | reused | prefilled | prefill | decode | result |
|---|---:|---:|---:|---:|---|---|
| session 1, turn 1 | 13,504 | 0 | 13,504 | 157.5 s | 98 tok, 7.66 tok/s | two **parallel** tool calls: `search_files(pattern="*.txt")` and `read_file("notes.txt")`, valid JSON arguments |
| session 1, turn 2 | 13,693 | 13,504 | 189 | 5.3 s | 25 tok | *"3 .txt files (a.txt, b.txt, notes.txt). The secret word in notes.txt is PLANKTON."* — correct |
| session 1 resumed, write task | 13,741 | 13,718 | 23 | 1.6 s | 85 tok | `write_file`, then a read-back; `summary.txt` on disk contains `plankton` — correct |
| session 2 (new), turn 1 | 13,495 | **12,288** | 1,207 | **18.9 s** | 38 tok | a `read_file` call; answer *"c.md contains exactly one line: `x`"* — correct |
| `hermes chat --image` | 243 | 0 | 243 | 7.5 s | 391 tok | Hermes's `vision_analyze` pre-pass, served by Cachalot's own vision path (`images=1`); final answer *"A solid red circle on the left, and the black uppercase text "CACHALOT" on the right."* — correct |

The session-2 row is section 15.3's change. Before it, a new session's first request reused nothing and
re-prefilled the same system prompt in 206-285 s. The title-generation side calls (326-335 tokens, 8-13 s)
queue behind or ahead of the main request as v37 predicted. Nothing broke on them once item 3 was fixed.

**Decode speed in these runs, 3.7-7.7 tok/s, is not a clean number.** These are single short replies at a
13.5k-token context on a cold-to-warm expert cache, with the machine swapping (2.9-3.6 GB of swap in use).
Section 7.2's live chat readings are still the decode reference. What is left for Hamed is the Desktop
app itself (the same client code, but its UI flows were not exercised) and a long real session.

### 15.2 Long prompts ran the Metal heap out; prefill is now chunked — 2026-09-23

**Found by Hermes, not by any benchmark.** Hermes Agent's first request is a 13,504-token prompt (a
17,359-character system prompt plus 24 tool schemas). Every one of its attempts died in prefill with
`[METAL] Command buffer execution failed: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`,
and Hermes retried it eleven times before giving up. No benchmark in this project had prefilled more than a
few thousand tokens at the shipped configuration, and `cachalot chat` never builds a prompt that long.

**Where the memory goes** (`benchmarks/prefill_memory_sweep.py`, the shipped config, real prose, peak MLX
memory over the ~67 GiB resident baseline, measured per layer block):

| prompt | result | worst pre-MoE block (attention, indexer, hyper-connections) | worst MoE |
|---:|---|---:|---:|
| 1,024 | ok, 21.0 s | 2.44 GiB | 0.55 GiB |
| 4,096 | ok, 52.9 s | 5.28 GiB | 1.39 GiB |
| 8,192 | ok, 149.6 s | 7.09 GiB | 2.77 GiB |
| 13,504 | **OOM in layer 1** | 7.74 GiB (layer 0 alone) | 4.89 GiB |

Every block's working set grows with the prompt, because layer-major prefill holds every token's activations
for a layer at once, so there is no one kernel to fix. (The 149.6 s at 8,192 is a cold-cache ordering artifact,
not a superlinear cost: the A/B below ran whole-prompt 8,192 in 85-96 s.) The indexer's own scoring is
quadratic on top of that: `index_scores_chunk` builds `[T, 32, cmax]` bf16 per-head scores, 5.8 GB at
T = 13,504 before its two same-sized temporaries.

**Two changes, both default-on:**

1. **`generation.prepare_prompt` prefills in chunks of `CACHALOT_PREFILL_CHUNK` tokens (default 4096).**
   A chunk continues the sequence exactly the way a prefix-cache hit continues a snapshot, so no new
   runtime path is involved. A chunk never splits an image span (the span is pushed whole into the chunk it
   starts in), and the server's `cancel` is checked between chunks, so a client that gave up no longer costs
   the rest of a prefill. A prompt of at most 4096 tokens is one call, exactly as before.
2. **The source layers' indexer scores in query-row slices of `CACHALOT_INDEX_Q_CHUNK` rows (default 1024)**,
   inside each prefill call, keeping the full `cmax` width so top-k and candidate selection see the same
   arrays. Slices are balanced and never below half a chunk: MLX picks a different matmul kernel for a
   handful of rows (7-row slices were *not* bit-identical in the test), while 512- and 1024-row slices are.
   A prompt of at most 1024 tokens is one slice, bit-identical to before. `tests/test_index_query_chunking.py`.

**Measured on the real bank** (`benchmarks/prefill_chunk_check.py`, `benchmarks/prefill_chunk_quality.py`):

| prompt | arm | time | peak MLX | last-position logits vs first arm |
|---:|---|---:|---:|---|
| 8,192 | whole | 96.2 s | 75.70 GiB | reference |
| 8,192 | 4096 chunks | 94.4 s | 73.15 GiB | KL 3.4e-2, top-1 differs, top-5 same |
| 8,192 | 2048 chunks | 122.1 s | 70.29 GiB | KL 3.0e-2, top-1 differs, top-5 same |
| 13,504 | 4096 chunks | 233.9 s | 73.16 GiB | reference — **was OOM** |
| 13,504 | 2048 chunks | 204.0 s | 70.24 GiB | KL 6.1e-5, same top-1 |

Arms run in order on the same prompt, so a later arm starts with a warmer expert cache. The times are not a
clean speed comparison, and no speed claim is made here. The quality check is the one that matters,
because one position's logits are one sample. After prefilling, it teacher-forces 64 real continuation
tokens through `decode_token` and scores them. Token-by-token decode serves as ground truth: the prefill
path's documented contract is that its state equals decode's.

| prompt | arm | NLL of 64 continuation tokens | mean KL vs reference (max) |
|---:|---|---:|---|
| 1,536 | sequential decode | 1.8681 | reference (ground truth) |
| 1,536 | whole | 1.9376 | 1.48e-2 (0.100) |
| 1,536 | 512 chunks | 1.9184 | 1.37e-2 (0.092) |
| 8,192 | whole | 1.5415 | reference |
| 8,192 | 4096 chunks | 1.5290 | 1.99e-2 (0.170) |
| 8,192 | 2048 chunks | 1.5452 | 1.98e-2 (0.093) |

**Chunked prefill is as close to ground truth as whole-prompt prefill: slightly closer at 1,536, and at
8,192 its NLL falls on both sides of whole's.** The KL between arms is the same size as whole prefill's
own distance from sequential decode, which is bf16 accumulation order. The module docstring already
names that ("only fp32/bf16 accumulation order differs from the per-token path"). **What it costs:** where a
whole prefill still fits, 4096-chunking was 94.4 s against 96.2 s in one A/B and 99.5 s against 85.0 s in the
other, so somewhere between free and ~17 %, unresolved at this sample size. It buys 2.5 GiB of peak headroom
at 8,192, and it is the only way a prompt past about 12k tokens completes at all. 4096 stays the default.
`CACHALOT_PREFILL_CHUNK=8192` is the knob if a later session shows the 17 % is real and the headroom is not
needed.

**New open item, not a regression: batched prefill sits ~0.014-0.02 mean KL from sequential decode**,
with NLL 1.94 against 1.87 on the one 1,536-token sample. That is older than this session and is a
property of every prefill this runtime has ever run. It has not been sized on more than 64 tokens. If it
holds on a larger sample, it is a quality lever on prefill's accumulation dtypes, not a speed lever.

### 15.3 Prefix-cache snapshots are 6.5x smaller, and a new agent session reuses the system prompt — 2026-09-23

**Snapshots were ~215 MB each, and a Hermes session fills the cache fast.** `/v1/stats` after the first
Hermes runs showed 12 snapshots holding 2.59 GB, the OS compressor active, 13 % memory free and 2.9 GB of
swap. A snapshot stored every position-indexed cache at its preallocated size: the compressed-KV caches
and the indexer's K cache, sized for `max_seq_len`, which is 65,536 on the server now. That was true even
when the sequence had written a few thousand rows. **`snapshot()` now keeps only the written rows (+1 for a
partial compression group), and `restore()` pads the zeros back**, because every row past the written ones
is zero by construction: caches are updated functionally from a zeroed allocation. The published
`shared_attn.compress_kv` is stored by source layer when it aliases a compressed cache, and re-pointed at the
padded copy on restore, the same way `shared_index_k_layer` already worked.

`benchmarks/prefix_snapshot_exactness.py` prefills a real-prose prompt, snapshots, teacher-forces 12 tokens,
then does `reset()` + `restore()` and forces the same 12 again. **Bit-identical at 3,000 and 9,000 tokens (max
|diff| 0).** Snapshot size is 14.7 MiB at 3,000 tokens and 33.0 MiB at 9,000. In the live replay, 12 entries
held **514 MB against 2.59 GB**, MLX active memory was 72.1 GB against 74.0, and 18 % of memory was free
against 13 %.

**With snapshots that small, prefill now snapshots at every chunk boundary** (every 4,096 tokens, section
15.2), not only at the prompt's end, and the cache evicts least-recently-*used* instead of oldest-added. An
agent's next session shares the long system prompt and tool schemas but diverges at its first user
message, and no end-of-prompt snapshot is a prefix of that. The chunk-boundary snapshot at 12,288 is.
**Measured: a new Hermes session's first request reused 12,288 of 13,495 tokens and prefilled in 18.9 s,
against 206.3 s cold in the same server run, 10.9x.** The whole second session took 46 s wall time. This is
the largest speed change of the session. It is exact (a restore is bit-identical) and it is for agent use,
not for `cachalot chat`, whose prompts are short. Up to 4,095 tokens of a shared prefix are still
re-prefilled, because boundaries fall on chunk multiples rather than on the message boundary. Snapshotting
at the last message boundary instead would recover that remaining ~15 s per new session, and it is the next
lever on this path.

### 15.4 An agent's system prompt, reused whole and kept across restarts — 2026-09-23

**Both levers of v38's Job 2, done and measured live with the stock Hermes CLI** (isolated `HERMES_HOME`,
section 15.1's setup, `./serve.sh` at the shipped configuration).

**Lever 1: snapshot where the system prompt ends, not only at chunk multiples.** Section 15.3's chunk-boundary
snapshots left up to 4,095 tokens of a shared system prompt to re-prefill in every new session (1,207 of
13,495 for Hermes, 18.9 s). `Engine.system_prefix_len` renders the request's leading system message on its
own with the official encoding (tools, `response_format` and the reasoning-effort header included, exactly
as `_encode_chat` attaches them), tokenizes it, and uses its length only when those tokens are a prefix of
the full prompt's. `prefill_chunks` takes that position as a `cut`: a prefill call ends there, and
`prepare_prompt`'s existing snapshot-at-every-call-boundary takes the snapshot. A cut inside an image span is
ignored, and the server skips the cut for image-bearing requests, whose positions shift on expansion.

Before relying on the prefix property it was checked against the real tokenizer and the official encoding
(a scratch script, not kept): a long system prompt, with and without 24 tool schemas, in chat and thinking
modes, reasoning effort none/`high`/50, one-user and tool-call histories, and a system prompt ending in
blank lines. In all 32 cases the rendered system block was a token prefix of the prompt, and a second
conversation with a different user message shared exactly `system_end + 1` tokens (the `<｜User｜>` token).
It holds because every following message starts with a special token, which the tokenizer never merges
across. If it ever fails (another template, a tokenizer that merges), the prefix check turns the cut off.

**Lever 2: keep the boundary snapshots on disk.** `src/cachalot/model/snapshot_store.py` writes a snapshot as
one safetensors file (every cache group, the Engram history as int64, `tokens`, and the layer ids and image
spans as metadata), write-then-rename, and keeps the four newest files. `PrefixCache.persist` is called only
for boundary snapshots, so per-turn snapshots never touch the disk; a failing write is logged and ignored.
`cachalot serve --snapshot-dir` (or `CACHALOT_SNAPSHOT_DIR`; `serve.sh` sets
`~/.cache/cachalot/prefix-snapshots`, empty disables) loads the matching files into the prefix cache at
startup. A file is only loaded when its identity matches: runtime version, `max_seq_len`, size and mtime of the
checkpoint's `config.json` and index, and of the expert bank's `config.json` and every shard. A mismatched
file is skipped, not deleted, so switching banks back and forth keeps each bank's snapshot until it ages out.
Bumping the version invalidates every file, deliberately.

**Exactness.** `benchmarks/prefix_snapshot_exactness.py` gained a disk arm: the same snapshot is saved, read
back, restored, and 12 teacher-forced decode steps must match the pre-snapshot run bit for bit. **Bit-identical
at 3,000 and 9,000 tokens for both the in-memory and the disk restore (max |diff| 0).** Save 0.01 s, load under
0.01 s, files 14.7 and 33.1 MiB. The cut itself is one more prefill-call boundary, the same operation section
15.2 checked against token-by-token decode (512-token chunks sat 1.37e-2 mean KL from sequential decode,
whole-prompt prefill 1.48e-2), so it is quality-neutral within that measurement.

**Measured live** (`[request]` lines; Hermes system block = 13,456 tokens):

| run | prompt | reused | prefilled | prefill | result |
|---|---:|---:|---:|---:|---|
| session 1, fresh server, empty snapshot dir | 13,478 | 0 | 13,478 | 163.1 s | tool calls, then the answer "a.txt and notes.txt ... NARWHAL", correct |
| session 1, turn 2 | 13,664 | 13,478 | 186 | 5.1 s | — |
| session 2 (new), turn 1 | 13,468 | **13,456** | 12 | **1.09 s** | `read_file`; "c.md contains a single line: `x`", correct |
| session 2 title call | 316 | 304 | 12 | 0.78 s | was 9.8 s cold |
| **server restarted**; startup loaded 2 snapshots (13,456 and 304 tokens) in 0.02 s | | | | | |
| session 3, turn 1 | 13,479 | **13,456** | 23 | **3.31 s** | `write_file` + read-back; `out.txt` contains `ORCA`, correct |
| session 3, turn 2 | 13,898 | 13,479 | 419 | 7.62 s | — |

Session 2's whole run took 15 s wall against 46 s in section 15.1; session 3's took 49 s, most of it decoding
191 tokens. The 3.31 s after the restart against 1.09 s in the warm server is the expert cache, which a
restart empties (23 new tokens still route to experts that have to come off the SSD). Decode was 6.2-8.4 tok/s
at a 13.5k context in these short replies, with no swapping this time (2.9 GB swap in use from before, not
growing); still not a clean decode number.

**What is left on this path.** The Hermes title call has its own 304-token system prompt and takes one of the
four disk slots; harmless. A different agent (or a changed Hermes toolset) adds its own file; four is enough
for a handful of harnesses. Nothing else in the first-request cost is prefix work any more: what remains is
the cold expert cache after a restart, a few seconds.

### 15.5 A long Hermes session, the reply splice, and decode speed at agent context lengths — 2026-09-23

**Run from this session with the stock Hermes CLI** (v0.21.3, updated by Hermes itself at 16:49 the same day),
an isolated `HERMES_HOME`, `./serve.sh` at the shipped configuration and `CACHALOT_SERVER_DUMP` on. Four
sessions: a 10-turn one (file reads and writes, running scripts, a CSV aggregation, a test file, a grep-style
search, a rename, a summary), a 4-turn and a 6-turn repeat, and a 5-turn one with compression lowered. Plus a
two-image vision conversation through the server.

**The long session worked.** All 10 answers were correct (f37(2) = 161, fib(20) = 6765 and fib(25) = 75025,
the per-city averages, both tests passing, f93/f102 tied at 99, the rename). The context grew from 13.7k to
26.3k tokens over 35 main requests. After the first request `reused` never fell back to 0 or to a
4,096-multiple, and swap stayed flat (3.2 → 2.9 GB). One slip in content: a README bullet guessed
`cpython-37` for a `__pycache__` file name.

**What changed in Hermes since 15.1, found from the request dump:**

- It now sends `reasoning_effort: "medium"` on every main request, so its sessions run in thinking mode
  (effort 50), not chat mode.
- Its system prompt carries `Current working directory: …` at token 3,924 of the 13,698-token system block,
  ahead of ~9,700 tokens of tool schemas, and wording that changes between Hermes versions (the scratch-dir
  line did). A new project or a Hermes update is a new system block and one cold prefill (163-190 s on a
  quiet machine). The disk store now keeps eight snapshots instead of four so a handful of projects stay warm.
  Changing the prompt order to move that line would change what the model sees; not done.
- 0.11.0's disk snapshot did not load this morning because it was written under version 0.10.0 before the
  bump, which the identity is designed to reject. Not a bug.

**The reply splice (new, `Engine._splice_own_replies`).** After a reply, the prefix cache holds prompt + reply.
The next turn reuses it only if the client sends the reply back token for token, and Hermes does not:

1. Tool arguments come back with the keys in a different order. The model wrote `write_file` as
   `path, content`; Hermes returned `{"content":"narwhal","path":"out.txt"}`, so the re-rendered DSML block
   diverged at the first parameter (token ids 9860 vs 9326), 18 tokens into a 57-token reply.
2. A reply whose thinking block was empty (`<think></think>`) comes back as `reasoning_content: " "`, which
   renders as `<think> </think>`.

The splice keeps the last 64 replies (the prompt as prefilled, the reply's token ids, and the reply parsed
with the official parser). For each incoming prompt it finds each stored prompt that is a prefix. If the
client's copy of that reply, up to the next end-of-sentence token, parses to the same message, the model's
own tokens replace it. "Same message" means content and reasoning equal up to surrounding whitespace, the
same tool names, and argument values equal as JSON. Anything else is left exactly as sent. A splice never
moves an image span: only replies after the last image are replaced. The model then conditions on what it
actually wrote, which is what it was conditioned on when it wrote the next token anyway; the text differs from
the client's rendering only in key order and a space.

Measured by replaying the 10-turn session's dump offline through the same code, with and without the splice
(`benchmarks/reply_splice_replay.py`; its baseline reproduces the live `reused` numbers exactly): 8 of 13 warm
turns had re-prefilled part of the model's own reply; the splice removes 499 of 4,453 warm prefill tokens
(11 %), at most 111 on one turn. At the warm short-suffix rate measured in the same sessions (about 1 s + 19 ms
per token) that is ~0.7 s per turn on average and ~2 s at most. It is small here because Hermes's replies were
short tool calls; it is proportional to reply length, so a `write_file` whose body is re-serialized, which
re-prefilled the whole file before, now costs nothing. Live, with the splice on, every turn after a tool call
reused exactly prompt + reply (`reused` = previous `prompt` + previous `completion`), and the `[request]` line
reports `spliced=N`, the number of the model's replies in the prompt that were put back.

**Decode speed does not depend on context length.** `benchmarks/decode_vs_context.py`, the shipped
configuration, the same fixed coding task behind N tokens of filler, one arm per process, order alternated:

| context | prefill | decode tok/s | misses/token |
|---:|---:|---:|---:|
| 54 | 6.2 s | 6.23 | 23.7 |
| 16,054 | 223.1 s | 7.38 | 29.1 |
| 54 | 5.9 s | 8.45 | 23.7 |
| 16,054 | 191.6 s | 7.45 | 29.1 |

The two 54-token arms had identical misses and routing and still differ by 35 %: that is run-to-run noise in
what a miss costs, larger than any context effect. The server adds nothing to this (the same task through
`serve.sh`: 8.6-9.2 tok/s, chat and thinking mode alike, 19-22 misses/token), and agent turns decode at the
same speed (8.1-8.8 tok/s at 13.7-14.2k context in the second session). **This closes v39's open question
about decode at 13-30k context: it is the same as at short context.**

**But decode alternates between two speeds, and the slow one is not explained.** The first session (17:22-
18:00) ran its whole length at 3.8-4.5 tok/s, and its cold 13.7k prefill took 598 s against 163-190 s
otherwise. The third session went from 8 tok/s to 3.7-4.2 for about ten minutes (18:25-18:36) and back to
6.2-7.8 with no restart, at the same 20-30 misses/token throughout. So a miss sometimes costs twice as much.
Measured during a slow window: raw uncached reads from the bank ran at 5.0 GB/s single-stream (1.9 ms per
9.49 MiB expert) and 7.0 GB/s with 8 streams, so the drive was not throttled. A bf16 matmul probe got 15.3
TFLOP/s beside the running server, and `pmset -g therm` recorded no thermal or performance warning. One apparent
correlation turned out void: the server's RSS was 30.5 GB at 18:33 and 50.5 GB by 18:35, when speed recovered,
but RSS then fell to 8.5 GB during an ordinary cold prefill while system-wide wired memory stayed at 77-78 GB.
RSS does not measure residency for a process whose memory is wired Metal buffers; do not use it. System-wide memory free was 16-19 %
throughout, swap 3-4.7 GB, and other apps were busy (the Codex service at ~60 % CPU for hours). This is the
same shape as the 54 GiB "turn-4 collapse" (memory: duration/thermal suspected). It is now Job 1 of the next
session, with the `miss/tok` field on the `[request]` line to separate "cold routing" from "a miss costs more".

**Vision through the server, beyond the first check (v39 Job 2, first item).** Two synthetic images in one
message (a blue square labelled "ALPHA 17", a green triangle labelled "BRAVO 58"), then two text follow-ups
that resend both images in the history:

| turn | prompt | reused | prefill | answer |
|---|---:|---:|---:|---|
| 1 (two images) | 440 | 0 | 8.64 s | both shapes, colours and labels exactly right |
| 2 | 532 | 511 | 1.63 s | "The larger number is 58, and it is written next to a triangle." |
| 3 | 564 | 548 | 0.99 s | "75" |

The prefix cache reused the image spans by digest across turns, as unit-tested in 16.4. The ViT and aligner
used to run again for every image in the history on every turn. `VisionEncoder` now keeps span rows by
content digest (16 images), and `/v1/stats` reports `vision_rows_reused`. The ablation of 16.4's three fixes
is still open; it needs debug switches that do not exist yet.

**A bug that one long session exposes: the system-block snapshot was evicted from memory.** The third and
fourth sessions used the same `HERMES_HOME` and working directory, so their system prompts were identical
(checked in the dump), yet the fourth session's first request prefilled all 13,734 tokens cold (185.9 s).
The prefix cache holds 16 snapshots in LRU order, and every request adds two (after the prompt and after the
reply) plus one per 4,096-token chunk boundary. After a 20-request session the system block had been pushed
out; it was still on disk, but the disk is only read at startup. **Fix: boundary snapshots are evicted only
after every per-turn snapshot** (`PrefixCache.max_pinned`, 8; beyond that the least recently used one becomes
an ordinary entry), and snapshots loaded from disk at startup are pinned the same way.
`tests/test_system_boundary_snapshots.py` has the 20-turn case.

**Hermes compression, live.** Hermes raises `compression.threshold` to at least 75 % for any window under
512k tokens, and to 85 % when its 64k floor binds (`agent/context_compressor.py`,
`_effective_threshold_percent`, `_MIN_CTX_TRIGGER_RATIO`), so on Cachalot's 65,536 it compresses at ~49-56k
whatever the config says. `compression.threshold_tokens: 18000` lowers it, and that is how it was triggered
here. What it cost:

| request | prompt | reused | prefill | decode | note |
|---|---:|---:|---:|---:|---|
| main, before | 17,801 | 14,874 | 33.5 s | 93 tok | read `big.py` |
| **summary** | 2,145 | 0 | 35.8 s | **1,595 tok, 213 s** | a separate prompt (a summarization instruction + the middle turns as text), chat mode |
| **main, after** | 19,108 | **0** | **254.4 s** | 89 tok | a new system block, see below |
| main, next | 17,059 | 14,814 | 36.5 s | 229 tok | reuse resumes, on the new block |

The whole turn took 806 s of wall time. **The request after compression is cold for a reason the server
cannot fix: Hermes adds a tool.** Every main request before it carried 24 tool schemas; from the summary on,
25, with `skill_manage` inserted at position 13 of the alphabetical list. The system message text is
byte-identical, but the tools are rendered into the system block, which grows from ~13.7k to 14,814 tokens
and diverges ~9k tokens in. Nothing cached is a prefix of it, so it is prefilled whole once; the next request
reuses the new block (14,814). The rerun on 0.12.0, with pinning, is the same shape: summary 2,145 tokens
prompt and **1,774 tokens decoded, 46.3 + 296.4 = 342.7 s**, then the next main request 19,161 tokens at
`reused=0`, 254.4 s, then 14,814 reused. That summary ran past Hermes's 300 s auxiliary budget; the turn
still completed (858 s wall) with the answer correct. **Candidate lever, not built:** pin the chunk-boundary
snapshots that fall inside a system block (4,096 and 8,192 here) as well as the block itself. They survive a
mid-list tool insertion, so that one cold re-prefill would reuse ~8k of its 14.8k tokens (~100 s). It happens
once per compressed session and costs 2-3 more pinned snapshots (tens of MB each), so it waits for Hamed's
long Desktop session to show how often it matters. The summary itself is the bigger cost, 4-6 minutes, almost
all decoding. Pointing Hermes's auxiliary `compression` provider at a hosted model takes it off the machine;
that is Hamed's call and is noted in the manual test. Also seen: after compression the prompt was *larger*
than before (19,108 against 17,801), since at this small scale the summary replaced less than it added.

**The eviction fix, verified live on 0.12.0** (same `HERMES_HOME` and working directory throughout):

| run | prompt | reused | prefill | session wall |
|---|---:|---:|---:|---:|
| session A turn 1, fresh server, version bump so no disk snapshot | 13,734 | 0 | 195.0 s | 270 s |
| session A: 15 requests, including a compression and a 25-tool block | | | | 1,128 s |
| **new session B after it** (the case that cost 185.9 s before the fix) | 13,711 | **13,702** | **1.07 s** | **9 s** |
| server restarted; startup loaded 3 snapshots (13,702, 304 and 14,814 tokens) in 0.03 s | | | | |
| **session C** | 13,711 | **13,702** | **1.69 s** | **6 s** |

### 15.6 Hamed's first Hermes Agent Desktop session: two server bugs and a volatile system prompt — 2026-09-23/24

**Run by Hamed** in Hermes Agent Desktop (v0.21.4, profile `careerlens`), `./serve.sh` 0.12.1 with the request
dump on, following `docs/manual-tests/hermes-desktop.md`. The first attempt never reached the server: the
start guard matched the test's own `tee /tmp/cachalot-serve.log` (fixed in 0.12.1). The second:

| request | prompt | reused | prefill | decode | what |
|---|---:|---:|---:|---:|---|
| 1 | 22,299 | 0 | 420.1 s | 142 tok, 5.8 tok/s | the Desktop system block is **22,281 tokens** (MCP servers, memory, skills): cold |
| 2-7 | 27.8k-39.4k | prompt + reply every time | 1-119 s | 4.2-4.5 tok/s | list the Desktop folder (four tool rounds, large results), write and run `hello.py` (5050): correct |
| 8 | 39,279 | **0** | 917.7 s | — | `finish=cancel`; next turn's system block had changed at token 6,708 |
| 9-14 | 39.3k | — | — | — | six retries, all **"There is no Stream(gpu, 12) in current thread"**, no `[request]` line |
| 15 | 22,173 | 4,096 | 441.9 s | 9 tok | a new "Say hi" chat: system block changed again at token 6,141 |

**Bug 1: MLX thread affinity.** MLX 0.32 ties an array that is not evaluated yet to the thread that built
it; evaluating it from another thread raises exactly the error Desktop showed (reproduced in isolation, and
for arrays built on the main thread too). The server ran every streaming request on a new thread. A request
cancelled between prefill chunks returns right after taking a chunk-boundary snapshot, and `snapshot()`
evaluated only the arrays it copies: the attention windows and the published indexer state it holds by
reference were still unevaluated from the last chunk. Each retry of the same prompt found that snapshot as
the longest prefix, restored it on another thread and failed; a new chat, sharing only 4,096 tokens, did
not. **Fix:** all generation runs on one persistent thread (`generate_pool`, one worker; the engine is
single-flight anyway), and `snapshot()` now evaluates everything it refers to.

**Bug 2: the 900 s cancel.** Hermes's stream stale detector gives a local endpoint 900 s without a parsed
chunk (`agent.local_stream_stale_timeout`, `chat_completion_helpers._local_stream_stale_timeout_default`).
The server's `: keep-alive` SSE comments keep proxies and read timeouts happy but never reach an OpenAI SDK
client, so a 39k-token cold prefill (917 s) was cut. **Fix:** every 15 s of silence the server now also sends
an empty-delta `chat.completion.chunk`, which Hermes counts (`_count_chunk` runs on every parsed chunk).

**Why that prompt was cold at all: Hermes's system prompt is volatile.** The request dump holds only two
distinct system-message texts, differing in one line (`Provider: custom` against `Provider: custom:cachalot`,
from re-selecting the provider), but the rendered system block also carries the tool schemas, and
`browser_exec`'s description is chosen per session by `tools/browser_use_cli._description_header()`:
"Screenshots are attached to your context automatically…" when Hermes believes the main model has native
vision, "Your model cannot view images, so work text-first…" otherwise. That belief is read from the
*profile's configured* main model (`_should_use_native_vision_fast_path`), which for `careerlens` is
`openai/gpt-5.6-sol` until the Desktop selection changes it, so the text flipped between turns of the same
conversation. Either change diverges the 22k-token block ~6-7k tokens in, and the whole block is prefilled
cold (7 minutes). What stabilizes it is config, not server code: declare the Cachalot model vision-capable
(`models: {deepseek-v4.1-flash: {supports_vision: true}}` under the `cachalot` entry of `custom_providers`),
which is true, and the official encoding renders an image inside a tool result (`<tool_result>…<｜deepseek_image｜>`),
checked. Then images also go to Cachalot natively instead of through `vision_analyze`.

**The vision check did not test Cachalot.** The "transcribe this image" chat was answered by
`openai/gpt-5.6-sol` over OpenRouter (the exported session's `reasoning_details` carry that endpoint slug; the
server log has no `images=` request for it). The new chat had started on the profile's default model. The
transcription was correct, but it says nothing about Cachalot; the manual test now says to check the model
selector on every new chat.

### 16.4 Piece 4 — images through the server, end to end, and three things piece 3 had missed — 2026-09-23

**Vision works end to end through `./serve.sh`.** An OpenAI `image_url` content part (a URL, a file path or
a base64 data URI) goes through the official encoding, the MLX preprocessing, the ViT and aligner, and the
piece 3 splice into prefill, and produces a correct answer. Two real checks on the shipped 2-bit bank:

- A synthetic 640x480 PNG, a red disc on the left and "CACHALOT 42" on the right: *"A red circle is on the
  left, and the black text "CACHALOT" is on the right."* Shape, colour, position and word are right; it
  dropped the "42". 234 prompt tokens.
- The checkpoint's own `assets/dsv41_kv_cache.png` (1,016 prompt tokens, 990 of them the image span): the
  model read the title "Global KV Cache Per Token (Bytes)", all four values (389,120 / 48,068 / 3,514 / 890),
  the four model names with their dates, and all three ratios (8.1x, 13.7x, 3.9x). Every figure matches
  the image.

This is the first real-image check of the whole chain; §16.3's live smoke used random rows. It is a
qualitative check. A numerical end-to-end diff against the PyTorch reference is not possible, because the
reference cannot run the 552B text model on this machine. Pieces 1 and 2 are numerically checked in §16.1.

**Three things piece 3 had missed, found by reading the reference's `Transformer.forward` and
`image_processor.py` rather than trusting §16's summary of them:**

1. **The span's delimiter positions take learned vectors, not aligner rows.** Every position of an image span
   carries `image_token_id`, including `[IMAGE_START]`, the `[IMAGE_NEW_LINE]` after each grid row, and
   `[IMAGE_END]`. The reference fills those three with the checkpoint's `image_start`/`image_newline`/
   `image_end` vectors (filtered out of the trunk by `resident_trunk.py`, so never loaded before). Piece 3's
   `merge_image_embeddings` consumes one row per `image_token_id` position, so piece 4 builds the full span,
   delimiters included (`vision_prompt.image_span_rows`), and piece 3 needed no change.
2. **Image positions take no part in Engram.** The reference computes `engram_mask = ~image_mask`. An image
   position enters the n-gram hash history as DEAD, which also blocks every n-gram reaching back across the
   span, and its Engram gate is forced to zero. `EngramHashState.push` already had `alive=False` and nobody
   passed it. `_prefill_tokens_impl` now does, and `engram_forward_batched` gained `token_mask`. Both are
   `None` for text, and bit-identical then (`tests/test_vision_prompt.py`).
3. **The prefix cache could have reused one image's KV for another.** Every image position is token 129264,
   so two different pictures of the same size tokenize identically. Snapshots now carry each span's
   `(start, length, sha256)`, and `PrefixCache.find` only matches a snapshot whose image spans are exactly
   the requested ones inside it. Tested with a cat/dog pair that must not reuse each other.

**What piece 4 added.** `src/cachalot/model/vision_prompt.py`: `VisionEncoder` loads the tower on first use
(about 1 s and 0.9 GiB, so a text-only server never pays for it), and `expand_prompt_images` ports
`prepare_vl_inputs`. `Engine.encode_chat_images` calls the official `encode_messages(...,
return_multi_modal_data=True)`, which already accepted OpenAI content-part lists and collected their
images; the server's message type never needed to change. A malformed image request (bad data, placeholder
count mismatch) is a 400, not a 500.

**Cost.** The ViT and aligner take 0.14 s for a 206-row image and 1.16 s for the 990-row diagram. An image
prompt's prefill time is the text model's, because it is expert streaming like any other prompt of that
length (the diagram request prefilled 1,016 tokens in 65.8 s on a cold expert cache).

### 9.36 The FP8 GEMV family, re-measured post-fusion — a methodology trap, then a self-consistent number — 2026-09-22

v36's Job 5 asked for the 3.48 ms gap figure (§7.1.10) to be re-measured against the post-fusion baseline
before being quoted again. **First attempt was wrong and is recorded here so nobody repeats it.** Reran
`micro_qkv_fusion_roofline.py` and `micro_shared_expert_roofline.py` fresh, plus `micro_fp8_gemv_kernel.py`
for the two shapes no fusion ever touched (`wq_b`, `wo_b`, §9.35). `micro_fp8_gemv_kernel.py`'s isolated,
raw-kernel-launch measurement of the unfused shared triplet (`w1`+`w3`+`w2` via its own `launch()` helper)
gives 5.79 ms/token; `micro_shared_expert_roofline.py`'s "shipped" arm, calling the real
`shared_expert_forward` function for the same three GEMVs, gives 4.60 ms/token — a ~20% gap between two
scripts both claiming to measure the identical unfused operation, from real-function overhead (dtype casts,
SiLU, clip) that the raw-kernel-only script excludes. **The two numbers are not comparable, and neither is
directly comparable to §7.1.10's original 3.48 ms, which used yet the isolated-kernel method.** An early
composed "family total" that mixed sources this way is not in this document; it was discarded once this
mismatch was found, not published.

**The number that is trustworthy: each shape group's own before/after pair, from the one script that measures
both under identical conditions, summed across non-overlapping groups.**

| group | fused (ships today) | unfused (same script) | this session's win |
|---|---:|---:|---:|
| `wq_a`+`wkv` (`micro_qkv_fusion_roofline.py`) | 1.62 ms | 1.86 ms | 0.24 ms |
| shared `w1`/`w3`/`w2` (`micro_shared_expert_roofline.py`) | 4.30 ms | 4.60 ms | 0.30 ms |
| `wq_b` (`micro_fp8_gemv_kernel.py`, untouched) | 3.67 ms | 3.67 ms | — |
| `wo_b` (`micro_fp8_gemv_kernel.py`, untouched) | 4.41 ms | 4.41 ms | — |
| **total** | **14.00 ms** | **14.54 ms** | **0.54 ms** |

Combined win this session, 0.54 ms/token, is the same order of magnitude as the two fusions' individually
documented deltas (0.35 ms §9.34 + ~0.4 ms §9.27 ≈ 0.75 ms) — sub-millisecond GPU timing varies run to run,
and this is a bundled re-run under load from three scripts executed back to back, not each one's own isolated
session. Ceiling, same per-group source, unaffected by fusion: 1.51 + 3.70 + 3.18 + 3.25 = 11.64 ms/token.
**Gap today: 14.00 − 11.64 = 2.36 ms/token, against 14.54 − 11.64 = 2.90 ms/token for the same shapes unfused,
measured the same way, the same session.** Both are smaller than §7.1.10's original 3.48 ms, but that
comparison crosses methodologies (isolated-kernel vs full-function) the way the discarded first attempt did,
so **"3.48 → 2.36" is not a clean before/after; "2.90 → 2.36," measured today, in one session, is.**

**Rule for whoever re-measures this again:** diff numbers only within one script's own arms, never across
scripts, even when both claim to measure the same named shape — `launch()`-a-raw-kernel and
call-the-real-function are different operations with different overhead, and the gap between them (here,
~20%) can be larger than the fusion win being sized. Section 9.35's own "a live session cannot resolve 5 ms"
rule has a offline-microbenchmark cousin: a cross-script diff cannot resolve anything either, no matter how
many decimal places it prints.

### 9.34 Lever — wq_a/wkv fusion, screened and shipped — 2026-09-22

Section 7.1.10 named `wq_a` [1280, 5120] and `wkv` [512, 5120] the two worst-throughput shapes in the FP8
GEMV family — 171 and 87 GB/s against `wo_b`'s 383 and the shared expert's 243 — and pinned it on occupancy:
512 and 1280 output rows launch too few simdgroups (64 and 160 threadgroups of 256 threads) to fill eighty
GPU cores. Together they cost 2.74 ms/token against a 2.31 ms `mx.sum` ceiling on the same bytes, 0.43 ms of
the family's 3.48 ms total gap. `attention_compressed.py` (all three decode variants — `_source`, `_reuse`,
`_index_source`), `attention_layer0.py` and `attention_sliding_window.py` all read both `wq_a` and `wkv`
from the same `x`, the layer's hidden state, independently of each other — `qr = fp8_linear(x, wq_a, ...)`
then, unconnected, `window_kv = fp8_linear(x, wkv, ...)` — exactly the shape section 9.27's shared-expert
`w1`/`w3` fusion exploited, and confirmed uniform across all 40 layers by reading the real checkpoint header
(`layers.{0,1,3,39}.attn.{wq_a,wkv}.weight` all `[1280,5120]`/`[512,5120]`). The shipped path also quantized
`x` to FP8 twice, once per call, for no reason — both calls quantize the same array.

`src/cachalot/model/attention_qkv_fusion.py` concatenates `wq_a` and `wkv` into one `[1792, 5120]` weight
(same `_fused_w13`-style identity-keyed cache, `mx.eval`d eagerly at concatenation time — attention is never
called from inside an `mx.compile`d trace, unlike the MoE block, so this fusion carries none of the
shared-expert fusion's eval-inside-a-trace hazard) and issues one GEMV instead of two, one activation
quantization instead of two, then slices the output back into `qr`/`window_kv`. `benchmarks/micro_qkv_fusion_roofline.py`,
40 distinct layers chained in one `mx.eval`: shipped 2.01 ms/token (80 launches), fused 1.66 ms/token (40
launches) against a 1.55 ms/token `mx.sum` ceiling — **0.35 ms/token recovered, bit-identical by
construction** (`mx.all(fused == shipped)` true on real-shaped random weights). `tests/test_attention_qkv_fusion.py`
covers the bit-identity and the cache's identity-keying. All five call sites were switched to
`fused_qr_kv_linear` directly, no kill switch — the shared-expert precedent (section 9.27) showed the risk
that mattered was the `mx.compile`-eval hazard, which does not apply here, and the fusion is bit-identical
by the same construction argument that held there. 242/242 tests pass. A live smoke test on the real 2-bit
bank (`benchmarks/qkv_fusion_live_smoke.py`, prefill + 12 greedy decode tokens on "Say hello in one short
sentence.") ran clean: no crash, first decode token `'Hello'`, then `'!'`, then EOS — correct.

### 9.35 Lever — wq_b/indexer wq_b fusion, screened and rejected: bit-identical but 0.038 ms/token — 2026-09-22

Job 5 of v34/v35's next-session prompt asked for one more session hunting a second same-activation,
independent-output GEMV pair before calling the FP8 GEMV family's remaining 3.1 ms gap (`wq_b`, `wo_b`,
shared `w1`/`w3`/`w2`) fully closed. `attention_compressed.py` names the candidate itself, in a comment
directly above `qr = fused_qr_kv_linear(...)`:

    # qr is shared conceptually between:
    #   attention wq_b
    #   indexer wq_b

Both read `qr` — the post-`rms_norm`, 1280-dim low-rank query, computed once per layer — and neither depends
on the other's output: `q = fp8_linear(qr, wq_b, wq_b_scales)` (`[32768, 1280]`) and, inside
`indexer_mlx.py`'s `indexer_decode_base`/`indexer_decode_candidate_source`/`indexer_decode_candidate_consumer`,
`index_q = fp8_linear(qr, indexer_wq_b_weight, ...)` (`[4096, 1280]`, `INDEX_N_HEADS=32 * INDEX_HEAD_DIM=128`).
Same shape as `wq_a`/`wkv` and the shared expert's `w1`/`w3` — on paper, a fusion candidate.

**Two differences from the two prior fusions, both found before writing any wiring code.** First, unlike
`wq_a` (171 GB/s) and `wkv` (87 GB/s), `wq_b` is not occupancy-bound — section 7.1.10 already put it at
456 GB/s, near its own `mx.sum` ceiling, so there is no occupancy shortfall to recover here; whatever a
fusion buys has to come from one fewer activation quantization and one fewer kernel launch, not from filling
idle GPU cores. Second, unlike `wq_a`/`wkv` (all 40 layers, every reuse and source layer alike), the indexer
only runs on the 8 layers that compute `topk_idxs` — `text_decode_runtime.py`'s `SOURCE_LAYERS = {2: 2, 8: 2,
14: 2, 20: 1}` and `INDEX_ONLY_SOURCE_LAYERS = {24, 28, 32, 36}` — so a fused kernel only replaces two launches
on 8 of 40 layers, not 40.

`benchmarks/micro_qb_indexer_fusion_roofline.py`, 8 distinct layers chained in one `mx.eval`, real-shaped
random FP8 weights: fused output **bit-identical** to the shipped two-call path (`mx.all(fused == shipped)`
true). Shipped 1.107 ms total (16 launches) against fused 1.069 ms (8 launches) against a 0.974 ms `mx.sum`
ceiling on the same bytes — **0.038 ms/token recovered**, a fifteenth of `wq_a`/`wkv`'s 0.35 ms and two
orders of magnitude below the rule this project has used all session to size a live check: "a live session
resolves a change of 20 ms or more; it cannot resolve 5 ms" (v34/v35 rules). The redundant-quantization arm
alone measured 0.339 ms across the 8 layers, well above the 0.038 ms actually recovered — the second
`quantize_fp8_activation` call the fusion removes turns out not to be additively serial with the GEMV it
precedes, so most of that cost was already hidden.

**Rejected on the measurement, not implemented.** The candidate is real — the code's own comment names it,
the same-activation/independent-output shape holds, and the fused output is provably bit-identical — but at
0.038 ms/token it is not worth the risk of touching `indexer_mlx.py`'s three decode variants (the module
that carries this project's `mx.compile`-eval crash class precedent, section 9.27) for a win no live session
could ever confirm. **Job 5's ask is now answered: one more session spent looking, one candidate found, sized,
and correctly left unshipped.** The FP8 GEMV family's remaining ~3.1 ms gap (`wq_b`, `wo_b`, shared
`w1`/`w3`/`w2`) has no same-activation fusion candidate left unexamined and can be called closed.

### Session logs, for history

| document | what it holds |
|---|---|
| `docs/HANDOFF-2026-09-16.md` | the five changes that reached 4.3–5.6 tok/s; budget sweeps on the 3-bit bank; the panic postmortem |
| `docs/HANDOFF-2026-09-17.md` §1–9 | concurrency diagnosis, Engram null, prediction-knob nulls, segmented LRU |
| `docs/HANDOFF-2026-09-17.md` §10–15 | what decode blocks on; coverage versus timing; the three measurement traps; the real drive speed |
| `docs/HANDOFF-2026-09-17.md` §16–19 | the 2-bit bank, the fit, the gate, the width sweep, and the ranking this document replaces |
| `docs/HANDOFF-2026-09-17.md` §20–27 | the compute floor, DSpark and its acceptance, speculation's economics, the refined 2-bit fit, and lever 4 closed |
| `docs/HANDOFF-2026-09-20.md` | the reference-arm session: how the arms were run and pinned, the raw tables behind §7.4.6, and the Hermes arm's own analysis |
| `docs/HANDOFF-2026-09-20-quality.md` | the session that found the defect: every suspect eliminated and how, the three reading passes over the decode path against the shipped reference, the attention and value-delivery measurements, the transposed `hc_post`, and the clean gate. Also the two hypotheses that were wrong — a distance effect that was rarity confounded, and a constants mismatch that came from reading the wrong config file |
