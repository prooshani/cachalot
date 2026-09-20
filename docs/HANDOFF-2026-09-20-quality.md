# Session log — 2026-09-20, second session: localizing the defect

Session log only. The authoritative state is `docs/HANDOFF.md`. This file carries the raw tables and the
reasoning behind section 7.4.7.

The previous session established that a hosted endpoint serving the same model is clean where Cachalot is
corrupt (section 7.4.6), which put the fault inside this runtime. This session was spent narrowing where.

## What was eliminated, and how

### Detokenization

`AutoTokenizer` from the checkpoint, 34 `#include <header>` lines covering the headers the corpus actually
uses. Three round trips: the whole block at once, each line on its own, and each line decoded one token at a
time and concatenated, which is the path a streaming detokenizer would take.

| round trip | result |
|---|---|
| whole block, batch decode | exact |
| per line, batch decode | 34 / 34 exact |
| per line, one token at a time | 34 / 34 exact |

Detokenization is not the fault.

The same run established the shape of the construction under suspicion. `#include <iostream>` is four
tokens: `6201 '#include'`, **`818 ' <'`**, `25428 'iostream'`, `32 '>'`. The malformed line
`#include iostream>` is one missing token 818, and `#include <stdex>` is a header name that stopped early.
Those are two different failures and the corpus scores both as one number.

### The sampler

The greedy arm's own manifest, `benchmarks/results/coding/20260919-140553_*/manifest.json`, records
`temperature 0.0` and `frequency_penalty 0.0`. With both penalties zero, `stream_tokens` sets
`penalties_on` false and passes `recent=None`, so `sample_logits` reaches its `temperature == 0` branch with
the logits untouched and returns `argmax`. The emitted token in that arm is the argmax by construction.
Section 7.4.5 was right to close sampling, and this is the second, independent reason.

### The model not knowing

`benchmarks/token_rank_probe.py`, written weeks ago and never run until today, teacher-forces text and reads
where the correct token ranks. On the production path, the real FP4 bank mounted, 24 GiB budget:

```
teacher-forced 262 positions in 76.4 s
after #include:     16/16 at rank 1 (100%), mean logprob -0.014
after import/from:   4/12 at rank 1  (33%), mean logprob -3.236
all:               207/262 at rank 1 (79%), mean logprob -0.855
```

The token after `#include` is at rank 1 every time with about 98.6 % probability. Under greedy that token is
what comes out. **So the model knows, and the corrupt greedy arm was not in the same state as this probe.**

## Four arms, to find what that state difference was

All four are the same script on the same runtime, bank and budget, differing only in the probe text and in
how much of it is consumed by `prefill_tokens` rather than `decode_token`.

| arm | probe text | prefill | positions |
|---|---|---|---|
| `rankprobe-prod` | the script's built-in text, includes at the top | 16 | 262 |
| `rankprobe-decode16` | 640-token file: prose, then C++ code, then the include block | 16 | 623 |
| `rankprobe-prefill540` | the same 640-token file | 540 | 99 |
| `rankprobe-deep` | unrelated meeting-notes prose, then the *byte-identical* built-in text | 16 | 830 |

### Prefill is not the fault

`decode16` and `prefill540` differ only in whether 540 tokens of identical context went through the batched
prefill path or one at a time through decode.

| | after `#include` at rank 1 | mean logprob |
|---|---:|---:|
| `decode16` | 13 / 20 (65 %) | -1.142 |
| `prefill540` | 12 / 20 (60 %) | -1.238 |

Within noise of each other. The prefill path, its batched attention, its grouped MoE and its speculative
variant are all exonerated for this defect.

### Depth is not the fault

`deep` puts the byte-identical include block at token 568 — the same depth at which `decode16` scored 65 % —
behind 568 tokens of unrelated prose.

```
after #include: 20/20 at rank 1 (100%), mean logprob -0.008
```

Bucketing every position of `decode16` by depth, 64 wide, shows no decay either: 44 %, 62 %, 86 %, 88 %,
94 %, 81 %, 86 %, 88 %, 73 %, 71 %. Position is not the variable.

## What is left, and it is the finding

The summary line of that probe measures only the delimiter, the `' <'` slot. Splitting every position inside
an include line into the *choice* of header — genuinely open-ended — and the *continuation* of a header
already begun — which has exactly one correct answer — gives this:

| arm | header choice | header continuation | mean logprob | worst rank |
|---|---:|---:|---:|---:|
| `rankprobe-prod` | 8/17 (47 %) | 22/28 (79 %) | -1.085 | 524 |
| `rankprobe-decode16` | 8/20 (40 %) | 27/32 (84 %) | -1.211 | 3260 |
| `rankprobe-prefill540` | 9/20 (45 %) | 26/32 (81 %) | -1.319 | 2450 |
| `rankprobe-deep` | 9/20 (45 %) | 27/33 (82 %) | -1.092 | 812 |

Continuation is about 80 % across every arm, and when it misses it misses by thousands of ranks. The same
two positions fail in all four arms, including the one that scored 100 % on the delimiter:

```
want 'cept'  after '#include <stdex'   rank 767 / 2450 / 3260 / 524
             model preferred '>\n'  '\n\n'  '<|end of sentence|>'
want 'ip'    after '#include <ioman'   rank 812 /  738 /  740 /  72
             model preferred '\n\n'  '\n'   '<|end of sentence|>'
```

Mid-word, the model wants to close the bracket or end the sequence. `#include <stdex` followed by `>` is
`#include <stdex>`, which is a malformed include line, and it is produced without any token being dropped.

Generalizing the same measurement to every position in the probe, a *word continuation* being a token that
starts with an alphanumeric character where the preceding character is also alphanumeric — the model
finishing a word it has already committed to:

| arm | word continuation | mean logprob | everything else | all positions |
|---|---:|---:|---:|---:|
| `rankprobe-prod` | 12/19 (63 %) | -1.807 | 80 % | 79 % |
| `rankprobe-decode16` | 10/20 (50 %) | -3.502 | 79 % | 78 % |
| `rankprobe-prefill540` | 5/11 (45 %) | -3.761 | 75 % | 72 % |
| `rankprobe-deep` | 16/23 (70 %) | -1.877 | 76 % | 76 % |

**Finishing a word is the easiest prediction a language model makes, and on this runtime it is the hardest.**
It sits below the average of all positions in all four arms, not above it.

That single statement accounts for every artefact this project has collected without needing a separate
explanation for each: `#include <stdex>` and `#include iostream>`, `LRUCache` written `LRcache`,
`map_._.end()` with a duplicated `_.`, `std std::string`, `utfutf-8`, `csv.Dreader`, `__name __`. Drops and
duplications alike are word-internal token errors.

**The honest caveat.** The word-continuation buckets hold 11 to 23 positions. The direction is consistent
across four arms and the effect is large, but the sample is small and a probe built for this measurement —
hundreds of continuation positions rather than twenty — should be run before the number is quoted anywhere.
That probe is the obvious next instrument and it costs one forward pass.

## Engram, checked and cleared so far

Word-internal continuation is what a token n-gram memory carries, which makes Engram the natural suspect,
and `LRUCache` collapsing to `LRcache` looks like a casing artefact of a normalizer that lowercases.

The checkpoint declares `engram_compressed_vocab_size = 99092` in `config.json`. `build_compressed_token_map`
computes the map from the tokenizer, and it returns exactly 99092 over a 129,280-entry map. The
normalization and compression map is correct.

That clears the map, not the rest of Engram: the primes, the offsets, the hash multipliers and the row
lookup are untested, and `engram_num_embeddings` is `[384006168, 384016682]` over layers `[1, 14]` with
`engram_max_ngram_size = 4`.

## The expert path, runtime against dense reference math

`benchmarks/nll_expert_precision.py` runs the same runtime twice over the same text and the same FP4
weights. `--experts runtime` is the production path: the resident store, the streaming reader and the fused
expert kernel. `--experts fp4` substitutes dense fp32 reference arithmetic for every routed expert in
decode. Attention, the shared expert, Engram and the head are identical in both, so the two differ only in
how routed-expert arithmetic is done.

The runtime arm, 620 tokens of the same 640-token probe:

```
mean NLL 0.8761 nats | ppl 2.402 | median NLL 0.0156 | top-1 78.2% | worst 14.67 at position 570
```

The median is 0.0156 nats and the mean is 0.8761, which is the same story the rank table tells: most
positions are confident and correct, and a small number are catastrophically wrong. The worst eight:

| position | NLL | wanted | argmax | context |
|---:|---:|---|---|---|
| 570 | 14.67 | `'cept'` | `'>\n'` | `...#include <stdex` |
| 315 | 14.46 | `' fields'` | `' split'` | `...std::vector<std::string>` |
| 273 | 12.86 | `'pace'` | `'class'` | `...(end > begin && std::iss` |
| 10 | 11.33 | `' header'` | `'\n'` | `...without consulting the` |
| 18 | 11.25 | `' bounds'` | `' the'` | `...reviewer should check` |
| 576 | 11.07 | `'ip'` | `'\n\n'` | `...#include <ioman` |
| 33 | 10.18 | `' Style'` | `' The'` | `...handling of empty input.` |
| 87 | 8.84 | `'struct'` | `'The'` | `...about its arguments.\n\n` |

Three of the eight are word continuations — `stdex` -> `cept`, `ioman` -> `ip`, `std::iss` -> `pace` — and
in each the argmax is a terminator or an unrelated keyword rather than the rest of the word. `isspace` and
`stdexcept` are not obscure. The rest of the list is ordinary open-ended prose prediction, where being wrong
is unremarkable.

Position 570 also matches the rank probe exactly: rank 3260, logprob -14.670 there, NLL 14.67 here. The two
instruments agree, which is worth stating because they share no scoring code.

### The dense arm, and what it clears

```
fp4 (dense fp32 reference math): tokens 620 | mean NLL 0.9016 | ppl 2.463 | median NLL 0.0136 | top-1 77.7%
runtime (production path):       tokens 620 | mean NLL 0.8761 | ppl 2.402 | median NLL 0.0156 | top-1 78.2%
paired: mean +0.0255 +- 0.0141 nats (z +1.81) | median +0.0000 | dense better on 290/620 (sign z -1.61)
greedy agreement 595/620 = 96.0%
```

Dense reference arithmetic is not better than the production path; it is a hair worse, the median difference
is exactly zero and neither test reaches significance. **The fused expert kernel, the resident store and the
streaming reader are cleared.** So are items 1 to 4 of the v13 bisection order, which were aimed at them.

The failures do not move either:

| position | wanted | runtime NLL / argmax | dense NLL / argmax |
|---:|---|---|---|
| 570 | `'cept'` | 14.67, `'>\n'` | 14.48, `'>\n'` |
| 576 | `'ip'` | 11.07, `'\n\n'` | 11.45, `'\n\n'` |
| 273 | `'pace'` | 12.86, `'class'` | 9.11, `'class'` |
| 315 | `' fields'` | 14.46, `' split'` | 14.72, `' split'` |

and neither does the class:

| | continuation top-1 | continuation NLL | other top-1 | other NLL |
|---|---:|---:|---:|---:|
| runtime | 50 % | 3.502 | 79 % | 0.789 |
| dense | 45 % | 3.432 | 79 % | 0.818 |

Whatever is losing word-internal information is upstream of routed-expert arithmetic: the trunk attention,
Engram, the head, or the non-expert weights, none of which either arm changed.

## The question this cannot answer on its own

Nothing above establishes that 50 % continuation top-1 is *wrong*. There is no reference set of
teacher-forced ranks to compare against, and the hosted endpoint will not produce one. The claim rests on a
prior — that finishing a word already begun should be near-certain — and a prior is not a measurement.

The reference-free control is a probe in which every long identifier appears three times in its own
sentence. On the second and third occurrence the continuation is a copy from a few tokens back, which no
correct model fails, so a low rank there is a defect by construction rather than by assumption. That probe
is 1,534 tokens with 133 continuation positions and it was running when this was written.

## The defect, stated correctly: in-context copying is broken

The repeat-identifier probe settles the question the section above could not. 1,534 tokens in which every
long identifier appears three times in its own sentence — *"The allocator_traits header is required, so the
file lists allocator_traits before anything else, and allocator_traits appears again later."* On the second
and third occurrence, finishing the word is a copy from a few tokens back.

```
teacher-forced 1500 positions in 583.6 s
```

| class | n | rank 1 | mean logprob | worst rank |
|---|---:|---:|---:|---:|
| continuation, first occurrence | 28 | **86 %** | -0.699 | 19 |
| continuation, **repeat** | 101 | **51 %** | -3.578 | **23,989** |
| everything else | 1,371 | 88 % | -0.595 | 3,703 |
| all positions | 1,500 | 85 % | — | — |

**First-time continuations are fine.** They sit at 86 %, level with the 88 % of everything else. It is only
the repeats that collapse, to 51 %, with a worst case of rank 23,989 on a token the text spelled out in full
ten tokens earlier.

That reverses yesterday's reading of the defect. Sub-word prediction is not degraded; **copying from recent
context is**. And the tokens the model prefers instead are not noise — they are words from *other* sentences
of the same probe:

```
rank 23989  want 'raits'        after '...the file lists allocator_t'   preferred ' unordered' ' ...' ' iterator'
rank  9165  want 'ip'           after '...anything else, and ioman'     preferred ' header' ' is' ' unordered'
rank  4144  want 'ographical'   after '...anything else, and lexic'     preferred '_' ' lex' '_com'
rank  3024  want 'ap'           after '...ile lists unordered_multim'   preferred ' unordered' 'unordered' '...'
rank   995  want 'cept'         after '...anything else, and stdex'     preferred 'ex' 'exception' ' is'
```

It is retrieving from the wrong place, not failing to retrieve. This is a within-run control and needs no
reference arm: no correct model fails to copy a word it emitted ten tokens ago, so the prior the previous
section rested on is now a measurement.

It also explains the corpus artefacts without needing a separate story for each. `LRUCache` written
`LRcache` is a failed copy of an identifier the reply had already defined; `map_._.end()` is a duplicated
retrieval; `std std::string` and `utfutf-8` are the same. Drops and duplications are both what mis-retrieval
looks like.

## Engram, checked against the shipped reference and cleared

The checkpoint ships its own implementation — `inference/engram.py` and `inference/model.py` under
`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash`. Nobody on this project had read it. It is the reference this
document has been describing as unavailable, and it is 184 lines plus one class in `model.py`.

Engram is the obvious suspect for in-context copying, being an n-gram hash memory. It is not the fault:

| checked | result |
|---|---|
| compressed token map | computes 99,092, exactly `engram_compressed_vocab_size` |
| prime generation, layer 1 | sum = 384,006,168, exactly `engram_num_embeddings[0]` |
| prime generation, layer 14 | sum = 384,016,682, exactly `engram_num_embeddings[1]` |
| offsets, flatten order | n-gram major, head minor, matching `torch.cat(..., dim=-1)` |
| hash multipliers | same `np.random.default_rng(10007 * layer_id)`, same bound from the *compressed* vocab |
| rolling hash | XOR per lookback, modulo the per-(n-gram, head) prime, matching |
| `blocked` propagation | cumulative once set, matching |
| gate math | same normalized dot, same signed sqrt before the sigmoid, same `1e-6` clamp |
| `norm_eps` | ours 1e-20; the reference's `ModelArgs` default is also 1e-20 |

Two nine-digit sums matching exactly is not a coincidence, and the gate is line-for-line the reference.

## Where that leaves it

Cleared: detokenization, the sampler, the prefill path, depth, the routed-expert kernel and its streaming,
and Engram. What remains is the trunk, and the symptom points inside it: precise retrieval of one earlier
position is exactly what attention does, and V4.1's attention is not plain attention. `config.json` declares
`index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]`, `kv_source_layer_ids = [2, 8, 14, 20]` and
`candidate_source_layer_id = 20` — an indexed scheme that selects *which* earlier positions a layer attends
to. A selection that is approximately right would leave general fluency intact and break exact copying,
which is the measured result.

`src/cachalot/model/attention_compressed.py` is 34 KB of hand-written implementation of that scheme and has
never been compared against `inference/model.py`. That comparison is the next job, and it can now be done
against a real reference rather than by reasoning.

## The three hand-written kernels, cleared together

The same repeat-identifier probe with `CACHALOT_FUSED_DECODE=0`, `CACHALOT_FUSED_FP8=0` and
`CACHALOT_BF16_HEAD=0` — the attention decode path, the dense FP8 linears and the final logit GEMV all off
their Metal kernels at once.

| class | fused (default) | unfused |
|---|---:|---:|
| continuation, first occurrence | 86 % | 86 % |
| continuation, repeat | **51 %** | **51 %** |
| everything else | 88 % | 87 % |
| worst repeat rank | 23,989 | 34,385 |

Nothing moves. All three are cleared, and with them the last of the custom-kernel suspects. The fault is in
the attention *algorithm*, not in an implementation of a kernel — which is the expensive answer, but it is
now the only one left standing.

## A distance result, and why it is not yet reportable

Bucketing the 101 repeat positions by how far back the word they copy was last spelled gives this:

| distance back | n | rank 1 | mean logprob | worst |
|---|---:|---:|---:|---:|
| under 16 tokens | 70 | 40 % | -4.781 | 23,989 |
| 256 or more | 22 | 77 % | -0.971 | 13 |

Read naively that says near copies fail and far copies succeed, which would be backwards for every ordinary
degradation and would point hard at the most recent positions. **It is confounded and must not be quoted.**
In that probe the near repeats are the rare identifiers — `allocator_traits`, `unordered_multimap` — and the
far repeats are the frame words every sentence shares — `header`, `the file lists`. Distance and rarity move
together, so the table measures both at once.

The controlled version holds the words fixed and varies only the distance: each rare identifier is defined
once and recalled once, with filler in between sized to put the recall 8, 32, 96 or 288 tokens later. Six
identifiers per bucket, 2,718 tokens. `sliding_window` is 128, so three buckets sit inside the recent window
and one outside it. That run was in flight when this was written, and until it lands the honest statement is
the one in the section above: in-context copying is broken, mechanism not yet localized.

## The distance-controlled run, and the hypothesis it kills

Same rare identifiers in every bucket, each defined once and recalled once, with filler sized to put the
recall 8, 32, 96 or 288 tokens later. 2,700 positions teacher-forced in 1,162 s; the copied tokens are every
token of the identifier after its first, so each one is a pure copy of a word spelled out earlier in the
same sentence.

| distance back | n | rank 1 | mean logprob | worst |
|---|---:|---:|---:|---:|
| 8 | 15 | 73 % | -1.026 | 8 |
| 32 | 14 | 79 % | -2.608 | 463 |
| 96 | 13 | 62 % | -3.050 | 31 |
| 288 | 4 | 75 % | -1.401 | 4 |
| all positions in the run | 2,700 | 88 % | -0.552 | 2,890 |

**There is no distance effect.** The 40 %-against-77 % split in the section above was the rarity confound
and nothing else, exactly as suspected; it is now retired and must not be carried forward. With it goes the
story that the sliding window or the window-to-compressed handoff is where the fault lives, which was the
most attractive hypothesis this session produced. The 288 bucket holds four positions and says nothing on
its own either way.

What survives the control is smaller but real: copying a rare identifier sits at 62 to 79 % against this
run's own 88 % baseline. Degraded, uniformly, at every distance tested.

## Honest summary of the state

Established, each by its own measurement:

- The defect is in this runtime, not FP4, not the model, not the sampler (previous session, section 7.4.6).
- It is not detokenization, not the sampler a second time, not the prefill path, not depth.
- It is not the routed-expert kernel, its streaming or its cache: dense fp32 reference math on the same
  weights gives a median NLL difference of exactly 0.0000 and 96 % greedy agreement.
- It is not Engram: nine checks against the implementation the checkpoint ships, including two nine-digit
  prime sums that match `engram_num_embeddings` exactly.
- It is not the fused attention decode kernel, the fused FP8 linears or the bf16 head GEMV: all three off
  together moves the number by zero.
- **In-context copying is degraded.** Repeat continuations 51 % against an 88 % baseline in one probe and
  62-79 % against 88 % in another, with misses as deep as rank 23,989 on a word spelled out ten tokens
  earlier, and wrong retrievals that pull words from neighbouring sentences.
- The degradation does not depend on copy distance between 8 and 288 tokens.

Not established, and not to be asserted: which component of the attention path causes it. The sparse
selection is the remaining structural suspect because precise retrieval is what it governs, but the flat
distance curve is evidence *against* the simplest version of that story, since `index_topk = 512` should
bite at long range rather than uniformly.

## The next job, and why it is now cheap

`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/` contains `model.py`, `engram.py`, `kernel.py`,
`convert.py`, `generate.py` and an `encoding/` directory. **This is the official implementation, it ships
with the weights, and no session before this one had opened it.** Sections of this handoff written on the
premise that no reference existed — including three bank-building sessions that chased quality through
quantization — were written beside a copy of the answer.

Engram was cleared against it in under an hour today by reading the two implementations side by side. The
same treatment of `src/cachalot/model/attention_compressed.py`, 34 KB against `model.py`'s `Compressor`,
`Indexer`, `select_candidate_blocks` and `Attention`, is the next job and needs no machine time at all.

Two pieces are already compared and match: `get_window_topk_idxs` in both prefill and decode, and the
Engram path end to end. The uncompared surface is the indexer's scoring and top-k, the compressor's group
handling and partial-group state, the candidate pre-filter rooted at layer 20, and the RoPE positions the
compressed latents are rotated with — the reference comments that a latent stands for the first token of its
group and takes position `j * ratio`, with a decode-time index of `start_pos + 1 - ratio`, which is the kind
of expression an independent implementation gets subtly wrong.


