<div align="center">

# 🐋 Cachalot

**Run DeepSeek V4.1 Flash (552B MoE) on a single Apple Silicon Mac, streaming experts from SSD.**

[![License: MIT](https://img.shields.io/badge/License-MIT-f5de53.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![MLX](https://img.shields.io/badge/MLX-0.32%2B-black.svg)](https://github.com/ml-explore/mlx)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20Silicon-lightgrey.svg)](#hardware)
[![CI](https://github.com/prooshani/cachalot/actions/workflows/ci.yml/badge.svg)](https://github.com/prooshani/cachalot/actions/workflows/ci.yml)

*A cachalot is a sperm whale: it dives deeper than anything else its size and comes back up with what it went for. This runtime does the same with a 475 GB checkpoint on a 96 GB machine.*

</div>

---

## What it is

Cachalot is an inference runtime, written from scratch in Python + [MLX](https://github.com/ml-explore/mlx) + hand-written Metal kernels, for the official
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) checkpoint. The model has 552B parameters,
289 GB of which are FP4 routed experts that never fit in unified memory. Cachalot keeps the dense trunk resident,
holds a bounded working set of experts in GPU-visible memory, and streams the rest from SSD on demand,
prefetching ahead of the compute that needs them.

It exposes the model as an **OpenAI-compatible HTTP server**, so agent harnesses such as OpenCode, Hermes, Continue, aider,
or any `openai` SDK client can use it as a drop-in local model.

The configuration that ships serves the routed experts from a **2-bit affine g128 bank built here from the FP4
checkpoint** — 9.49 MiB per expert against FP4's 17.93, which halves the bytes a token reads. On the 40-case
coding gate it is equal to a hosted FP4 and a hosted FP8 reference arm on every column.

It is **not** a port of the PyTorch reference and it is **not** a generic MLX model loader. Every component was
implemented against the released `inference/model.py` semantics and validated token-for-token, including the parts
that make V4.1 Flash unusual:

| V4.1 Flash component | Cachalot implementation |
|---|---|
| Single-pass mHC hyper-connections (Sinkhorn mixing) | Fused single-dispatch Metal kernel |
| Compressed Sparse Attention 2: Full / Reindex / Reuse layer modes | Exact source / index-only / reuse block topology, shared cross-layer KV and top-k |
| Hierarchical sparse indexer with candidate blocks (layer 20 → 24/28/32/36) | Implemented, per-token candidate replay in layer-major prefill |
| FP4 E2M1 expert weights with E8M0 block scales | Custom Metal GEMV, bit-exact dequantization tables |
| FP8 E4M3 trunk with dynamic activation quantization | Custom Metal GEMV, official `act_quant` semantics |
| Engram conditional memory (196B params, 2 × 384M-row tables) | Sparse `mmap` row reads, exact n-gram hashing and token normalization |
| Sliding-window attention over a bounded 128-position window, FP4 KV cache (E2M1 + E4M3 scale per 16) | Implemented |
| Official chat protocol (`encoding.py`), thinking mode, reasoning effort | Loaded from the checkpoint, not reimplemented |

Vision input and DSpark speculative decoding are not implemented yet (see [Roadmap](#roadmap)).

**Prefill runs the full prompt through all 40 layers.** The row above used to read "SWA bounded replay",
which a 2026-09-19 review reasonably read as the decoder replay shortcut in §3.2.2 of DeepSeek's technical
report -- process the whole prompt through the encoder, build the decoder's global KV from its output, and
replay only the final 128 positions through the decoder. **That is not implemented**, it is approximate by
DeepSeek's own account because sliding-window dependencies accumulate across layers, and it would be an
opt-in research branch rather than a drop-in speedup. The bounded window the row refers to is the attention
window itself.

## Why this exists

A 552B MoE activates only 8B parameters per token during prefill and 16B during decode, so the *compute* fits a Mac
Studio comfortably. The *bytes* do not: each token touches 240 routed experts (40 layers × top-6), and the
15,360 of them total **289 GB as FP4** — 145 GB re-quantized to the 2-bit bank that ships, which is still
more than the machine has. Any runtime for this class of machine is therefore an exercise in **caching and
I/O scheduling** before it is an exercise in kernels. Cachalot is built around that fact:

```
                       ┌──────────────────────────────────────────────┐
  HTTP (OpenAI API) ─▶ │  cachalot serve   ── request queue ──▶ V41Model │
                       └──────────────────────────────────────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     │                 TextDecodeRuntime                    │
                     │  layer-major prefill (expert-major MoE)  ·  decode   │
                     └──────────────┬───────────────────────────┬──────────┘
                                    │                           │
              ┌─────────────────────▼──────────┐   ┌────────────▼─────────────────┐
              │ Resident trunk (~12 GiB, MLX)  │   │ ResidentExpertStore           │
              │ attention · router · shared    │   │ wired slot pool (auto-sized)  │
              │ expert · norms · heads · RoPE  │   │ per-layer quotas ·            │
              └────────────────────────────────┘   │ deterministic admission ·     │
                                                   │ cross-turn reuse              │
                                                   └────────────┬─────────────────┘
                                                                │ miss
                                                   ┌────────────▼─────────────────┐
                                                   │ Loader workers (8 threads)    │
                                                   │ preadv() straight into the    │
                                                   │ slot's unified memory         │
                                                   └────────────┬─────────────────┘
                                                                │
                                                   ┌────────────▼─────────────────┐
                                                   │ safetensors shards on SSD     │
                                                   │ 289 GB routed experts         │
                                                   │ Engram tables (mmap)          │
                                                   └──────────────────────────────┘
```

## Status

Cachalot is **alpha**. It produces reference-quality output and runs multi-turn sessions at 9.4–9.6 tok/s
on the configuration in [Performance](#performance). Decode is no longer bound by SSD bandwidth — the drive is
idle 45 % of the time — and is now limited by the share of experts that are already resident. Read
[Performance](#performance) before deciding whether it fits your use.

| Area | State |
|---|---|
| Text generation, official chat protocol, thinking mode | ✅ working |
| Layer-major prefill with expert-major MoE scheduling | ✅ working |
| Auto-sized, wired expert slot pool with zero-copy SSD loads | ✅ shipped |
| Cross-turn expert residency | ✅ working, validated |
| MLX allocator tuning (2 GiB free-buffer cap) | ✅ shipped, removed 100–380 ms allocation stalls |
| Routing trace + offline cache-policy analysis | ✅ `benchmarks/` |
| Unit tests without checkpoint | ✅ `pytest -q` |
| OpenAI-compatible server (`/v1/chat/completions` SSE, tools, thinking, `/v1/completions`) | ✅ working, tested |
| Prefix cache (only new tokens are prefilled per turn) | ✅ working |
| `cachalot serve / chat / doctor / bench` CLI | ✅ working |
| Parallel loading of a decode layer's expert misses | ✅ shipped |
| Fused top-k expert Metal kernels, bf16 head GEMV | ✅ shipped |
| Fused decode path: router top-k, sparse attention, hyper-connection mixes, RoPE/RMSNorm, FP8 quantization + vectorized FP8 GEMV | ✅ shipped, all-resident token 0.10 → 0.068 s |
| FP4 expert GEMM on simdgroup matrix units for prefill | ✅ shipped, 2.5× less GPU time per expert |
| Speculative next-layer expert loads + background Engram rows in prefill | ✅ shipped, SSD busy 59 % → 91 % of a 2048-token prefill |
| Auto budget capped by memory available at start | ✅ shipped |
| Kernel warm-up at load (first token 1 s → 0.35 s) | ✅ shipped |
| Second checkpoint copy on another drive, byte-striped expert reads (`CACHALOT_MIRROR_PATH`) | ⚠️ shipped, **off on the 2-bit bank**: an 18 % loss at 9.49 MiB per expert where it was a gain at 17.93 |
| 2-bit affine g128 expert bank, half the bytes of FP4 at equal gate quality | ✅ shipped, `benchmarks/build_affine_bank.py` |
| Traced decode MoE block and memoised kernel parameters | ✅ shipped, all-resident token 85 → 76.4 ms |
| Coding-quality gate against a hosted reference arm (40 cases) | ✅ 20/20 C++ compile, 0/101 malformed includes |
| `./chat.sh` launcher for the shipped configuration | ✅ shipped |
| Batched prefill (attention for all 40 layers, HC, router, routed + shared experts, Engram) | ✅ shipped |
| DSpark / MTP speculative decoding | ⛔ measured and closed twice; needs a decode-shaped multi-position forward first |
| Vision | ❌ not planned for v1 |

## Hardware

Tested on a **Mac Studio M3 Ultra, 96 GB unified memory, 60-core GPU**, with the checkpoint on the internal SSD
and, earlier, on a Crucial X10 Pro over USB 3.2 Gen 2.

Requirements:

- Apple Silicon Mac with **≥ 64 GB unified memory**. The expert budget auto-sizes to the machine:
  ~14 GiB of experts on 64 GB, ~50 GiB on 96 GB, ~73 GiB on 128 GB, the full 270 GiB on 512 GB
  (at which point the SSD is only touched at load time).
- macOS 14+ with Metal 3 or newer.
- **~480 GB** of storage for the checkpoint. Storage speed is the single largest performance factor;
  `cachalot doctor` measures yours:

| Storage path | Sequential read | Measured effect |
|---|---:|---|
| USB 3.2 Gen 2 SSD | ~1.0 GB/s | FP4 decode 1.3 s/token, cold 512-token prefill 8–9 min |
| Thunderbolt 4/5 NVMe enclosure | 3–6 GB/s | Proportionally faster misses, no code change |
| Internal Mac SSD (tested) | 5.2 GB/s | **2-bit bank: 0.128 s/token interactive, 16.4 s cold 512-token prefill**; FP4 0.33 s/token |

The loader saturates a USB SSD at queue depth 1 (17.7 ms per expert read) and reads at 5.7 GB/s from the internal
disk, so more threads do not help; faster storage does.

## Install

```bash
git clone https://github.com/prooshani/cachalot.git
cd cachalot
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[server]"
```

Download the checkpoint (≈475 GB) to a fast disk:

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /Volumes/FastSSD/DeepSeek-V4.1-Flash
```

## Quick start

```bash
# Check hardware, memory budget, storage speed and checkpoint layout
cachalot doctor --model /Volumes/FastSSD/DeepSeek-V4.1-Flash

# One-shot prompt
cachalot chat --model /Volumes/FastSSD/DeepSeek-V4.1-Flash "Explain unified memory in two sentences."

# Interactive session (/clear, /exit)
cachalot chat --model /Volumes/FastSSD/DeepSeek-V4.1-Flash

# OpenAI-compatible server on http://127.0.0.1:8000
cachalot serve --model /Volumes/FastSSD/DeepSeek-V4.1-Flash --port 8000
```

Then from any OpenAI client:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "Write a haiku about sperm whales."}],
    "stream": true
  }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="cachalot")
for chunk in client.chat.completions.create(
    model="deepseek-v4.1-flash",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Python API

```python
from cachalot import V41Model

with V41Model.from_pretrained("/Volumes/FastSSD/DeepSeek-V4.1-Flash") as model:
    reply = model.chat(
        [{"role": "user", "content": "What is a Sinkhorn iteration?"}],
        max_new_tokens=256,
        temperature=0.6,
        thinking_mode="thinking",
    )
    print(reply.content)
```

`V41Model` is the stable public boundary. Everything below it (`TextDecodeRuntime`, caches, kernels) may change
between minor versions.

Harness setup (OpenCode, Hermes, aider, Continue, OpenAI SDK): [docs/integrations.md](docs/integrations.md).

## Configuration

All knobs live in `cachalot.config.RuntimeConfig` and can be overridden on the CLI or via `CACHALOT_*` environment variables
(`CACHALOT_EXPERT_CACHE_BUDGET_GIB=48`, `CACHALOT_MODEL_PATH=...`, `CACHALOT_MAX_SEQ_LEN=...`, `CACHALOT_PORT=...`).
If you keep a second identical copy of the checkpoint on another drive, `CACHALOT_MIRROR_PATH=/Volumes/.../DeepSeek-V4.1-Flash`
makes every expert read fetch its tail from that drive concurrently (`CACHALOT_MIRROR_FRACTION`, default 0.10 = the
share of bytes for the second drive; use its bandwidth divided by the total). **Whether this helps is a property of
the expert size, not of the drives**: with 17.93 MiB FP4 experts it is worth ~5 % on decode and ~12 % on short
prefills, and with the 9.49 MiB 2-bit experts that ship it is an 18 % *loss*, because the per-read latency it adds
stops being amortized. It is off in the shipped configuration. See `docs/HANDOFF.md` section 9.11.1.

| Setting | Default | Meaning |
|---|---:|---|
| `expert_cache_budget_bytes` | auto | Routed experts kept resident. Auto = unified memory − 12 GiB trunk − 2 GiB MLX cache − 32 GiB system reserve, capped at the Metal recommended working set. `--expert-budget-gib 40` for a fixed value. |
| `mlx_wired_limit_bytes` | auto | Trunk + experts + cache are wired via Metal residency sets so macOS cannot compress them under pressure (without this decode was 3× slower, see [docs/performance.md](docs/performance.md)). |
| `system_reserve_bytes` | 32 GiB | Memory left for macOS, page cache and other applications when auto-sizing. Lower it on a dedicated machine. |
| `mlx_cache_limit_bytes` | 2 GiB | Cap on MLX's free-buffer cache. Larger values recreated allocator stalls under concurrent materialization. |
| `io_workers` | 8 | Loader threads. Bandwidth-bound; more threads do not raise throughput on USB SSDs. |
| `CACHALOT_ENGRAM_PARALLEL_MIN` | 8 | Engram row batches at or above this size are read through the reader's 16-worker pool instead of by the calling thread. A decode token asks for 24 rows twice; at the old threshold of 64 those 96 `pread`s went out one at a time from the decode thread and cost 28 ms per token behind a busy drive. |
| `CACHALOT_DECODE_ENGRAM_PREFETCH` | 1 | Issue both Engram layers' row reads at the top of the token rather than at the layer that consumes them. The row ids depend only on the token being decoded, so layer 14's read hides behind thirteen layers of compute. |
| `max_seq_len` | 32768 | Sequence capacity for KV and compressed caches (a few hundred MB; CSA2 keeps KV tiny). |

**Memory budget guidance.** More resident experts is the only software lever that materially cuts SSD bytes:
in the routing trace, 40 → 50 GiB removed 6 % of bytes and 40 → 64 GiB removed 13 %, while smarter eviction
policies were worth 1–2 % (see [docs/performance.md](docs/performance.md)). The auto budget takes what the machine
has minus a 32 GiB reserve; shrink the reserve with `CACHALOT_SYSTEM_RESERVE_GIB` on a dedicated box, or pin a budget
with `--expert-budget-gib` if you run other memory-hungry software alongside.

## Performance

Two numbers matter and they are measured differently. **A live interactive session** is what a user sees; an
**all-resident token** is the compute floor the runtime reaches when every expert it needs is already in
memory. Everything below is the hardware above, checkpoint and expert bank on the internal SSD, 512-token
context, official chat protocol.

### The shipped configuration

2-bit affine g128 expert bank (9.49 MiB per expert, built from the FP4 checkpoint by
`benchmarks/build_affine_bank.py`), 44 GiB expert budget, startup hotlist, 72 GiB wired, no mirror, no
frequency penalty. Launch it with `./chat.sh`.

| what | result |
|---|---|
| Interactive decode, prose | **9.4–9.6 tok/s** at a 52 GiB budget, two sessions (7.6–7.9 at 44) |
| Interactive decode, ~1,500 tokens of Objective-C | **8.2–8.5 tok/s** at a 52 GiB budget, two sessions (5.6–6.9 at 44) |
| Session expert hit rate | **92.3–92.4 %** at 52 GiB, two sessions; 89.9–90.2 % at 44, repeated across four sessions and three runtime versions |
| Resident experts, MLX peak | 5,276–5,314 experts, 67.7 GiB at 52 GiB (4,480–4,495 and 55.1–55.6 at 44) |
| Follow-up prefill (prefix cache) | 90–116 ms per prompt token |
| Cold 512-token prefill | 16.4 s |
| Quality, 40-case coding corpus | 20/20 C++ blocks compile, 18/18 Python blocks parse, **0 of 101 malformed `#include` lines** — every column equal to a hosted FP4 and a hosted FP8 reference arm |

The same configuration as a benchmark, with a colder working set than a conversation builds: **170 ms per
token (5.90 tok/s)** at an 83.5 % hit rate, reading 627 MiB per token, drive busy 55 % of decode,
reproducible to ±0.3 %.

**The 52 GiB row is two sessions now, on 0.9.0 and 0.9.2, and they replicate**: 9.42 and 9.59 tok/s on
prose, 8.53 and 8.15 on Objective-C, 92.37 % and 92.31 % hit rate, and an MLX peak identical to the byte.
Run it with `CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52`. It is 22–27 ms per token
faster than 0.7.0 at 44 GiB, of which the larger budget explains about 6 ms by the miss arithmetic and the
Engram change the rest; the two have still not been separated by an A/B. The budget's effect is what
`benchmarks/simulate_policies.py` predicted offline — +2.6 points of hit rate, +2.37 measured — and MLX
peaked at **67.7 GiB against the 77.8 GiB the flag wires**, 10.1 GiB of headroom, with no memory-pressure
event. `chat.sh` still defaults to 44 GiB.

### Where a token's time goes

An all-resident token is **76.4 ms** (13.1 tok/s): about 56 ms inside `mx.eval` and 21.5 ms of CPU building
the next graph. A streaming token adds what it blocks on.

| block | ms/token |
|---|---:|
| **The miss**, ~1.7 ms each — 1.41 blocked, 0.31 inside `mx.eval`, ~0.05 CPU | 55–67 at an 83.4 % hit rate, ~19 at a session's 92.4 % |
| The FP8 GEMV family: four attention projections and all three shared-expert GEMVs, 5.14 GB/token | 16.6 |
| Routing prediction, computed and submitted | 11 |
| The 44 `mx.eval` round trips, ~0.20 ms each | 9–12 |
| Routed experts, six per layer, traced | 6.9 |
| Compressor, indexer and compressed-KV write, eight source layers | 5.5 |
| `wo_a`, a BF16 grouped matmul at 625 GB/s | 4.3 |
| Hyper-connection glue | 4.2 |
| Head, Engram forwards and the layer's own router | 3.2 |
| Sparse attention over the KV itself, all forty layers | ~2.6 |
| Engram row reads on the decode thread | 1.9, and 28.4 before 0.9.0 |

**The ceiling is the expert hit rate, not the kernels**, and 0.9.2 measured that rather than assuming it.
The same continuation decoded a second time, with everything it needs already resident, costs **79.6 ms a
token — the all-resident floor to a tenth of a millisecond on every column**, so every millisecond between
the floor and a live token is a miss and nothing else. Read concurrency from 2 to 16 workers moves nothing;
bypassing the page cache costs 16 ms. And the miss itself is at the drive's rated wall: at `io_workers=8`
experts read at 1.46–1.47 ms each whether 43 GiB is wired or nothing, so there is no memory-pressure tax
hiding inside it either (0.9.4).

**And the GPU side is at the machine.** Attention's 22.5 ms turned out to be 86 % weight streaming: a reuse
layer spends 0.377 ms of its 0.441 reading the five projections that build Q and project the output, and
0.064 on the sparse attention whose shapes had been the standing suggestion. Underneath four of those five
is one kernel, `fp8_gemv_decoded`, which moves 5.14 GB per token — more than twice the routed experts — at
**79 % of what `mx.sum` gets over the same bytes**. The shared expert is at 80 % of the same ceiling,
`wo_a` in BF16 is faster than any FP8 form of itself, and the lanes-per-row policy nobody ever tuned is
within 0.3 % of the tuned one. The remaining lever is the hit rate: a larger budget is worth 2.6 points of
decode hit from 44 to 52 GiB by offline replay, and the live session moved 2.37.

**What 0.9.0 took, and how it was hiding.** A streaming token used to spend 33 ms more outside the store's
blocking calls than an all-resident one, with no mechanism for it in three successive analyses. It was the
Engram row reads: a token asks for 24 rows twice, each row is two `pread`s, and the reader only used its
worker pool for batches of 64 or more — so 96 reads went out one at a time from the decode thread, behind
a queue the expert stream was filling. They now go through the pool and are issued at the top of the token
rather than at the layer that needs them. Same bytes, same order, an identical 16-token greedy
fingerprint, and 15 % off the benchmark token.

### The FP4 bank the checkpoint ships with

For reference, and because it is what runs without `CACHALOT_EXPERT_BANK`: 17.93 MiB per expert, 325–341 ms
per token (2.9–3.1 tok/s) at a 36 GiB budget with a 70–71 % hit rate, reading about 1,860 MiB per token, the
drive busy 80 % of decode. The 2-bit bank is faster because it reads half the bytes, and after the
hyper-connection fix in 0.6.0 it is not worse: both are equal to the hosted reference on the coding gate.

Method, instruments and the measurement behind every line above are in [docs/HANDOFF.md](docs/HANDOFF.md);
the older FP4-era analysis is in [docs/performance.md](docs/performance.md).

## How it works

**Layer-major, chunk-batched prefill.** The prompt is processed one layer at a time. Attention runs as one batched pass per
layer (windowed causal attention over a concatenated key pool plus the source layer's per-token compressed top-k),
the MoE phase is regrouped *expert-major* (each routed expert is dequantized once and applied to all of its tokens
with a GEMM), and hyper-connections, router, shared expert, Engram, compressor and indexer are row-batched with
per-token visibility masks. Per-token top-k accumulation order is preserved.

**Deterministic admission.** Before a layer's prefetch starts, `ResidentExpertStore.prepare_prefill_layer()` decides
the layer's resident set from logical work order and per-layer quotas. Asynchronous SSD completion order therefore
cannot change what ends up cached, which makes runs reproducible and cache behaviour debuggable.

**Cross-turn residency.** `reset()` clears sequence state but keeps experts. Returning to a previously seen task after
two unrelated ones saved 31 GiB of SSD reads in the validation benchmark.

**Zero-copy, zero-allocation expert loads.** Every routed expert has the same six tensor sizes, so the resident cache
is a pool of slots allocated once at start-up and wired with `mx.set_wired_limit`. A miss is a `preadv()` from the
shard straight into a writable view of the slot's unified memory; the GPU reads it in place. There is no per-expert
`mx.array`, no memcpy, and no allocator or Metal residency churn on the hot path.

**Memory pressure is the enemy.** Before wiring, macOS compressed cold expert buffers and every GPU access paid a
decompression fault: decode was 3× slower than its SSD bytes implied. Expert reads also bypass the page cache
(`F_NOCACHE`) so a 300 GB stream cannot push the rest of the system out. Details and the measurements that led here
are in [docs/performance.md](docs/performance.md).

**Allocator hygiene.** MLX's free-buffer cache is capped at 2 GiB; larger values produced hundreds of 100–380 ms
allocation stalls under concurrent materialization.

**Exactness.** Every kernel has a NumPy reference (`fp8_ref.py`, `fp4.py`) and the routing, sampling
(Gumbel-max equivalent of the official exponential-race sampler), prompt encoding and completion parsing use the
checkpoint's own code paths.

## Benchmarks

```bash
# Everything below auto-discovers ~/DeepSeek-V4.1-Flash or CACHALOT_MODEL_PATH

# Multi-turn locality benchmark (A → B → C → A, 256 tokens each)
PYTHONPATH=src python benchmarks/bench_multiturn.py

# Routing trace on four realistic prompts + offline policy analysis
PYTHONPATH=src python benchmarks/trace_routing.py --prompt-tokens 512 --decode-tokens 32
PYTHONPATH=src python benchmarks/analyze_trace.py benchmarks/results/trace_routing.trace.npz \
    --out benchmarks/results/trace_routing.md
```

`analyze_trace.py` reports the static coverage upper bound per memory budget, per-layer routing skew, simulated
hit rates for LRU / frequency-aware / static-hot policies, and adjacent-token expert overlap. Use it before
changing the cache policy or budget.

## Roadmap

Ordered by measured size in a token, not by expected difficulty. The full ranking, with what closed each
line, is `docs/HANDOFF.md` section 9.25.

1. ~~Prefix cache, OpenAI-compatible server, batched prefill, memory auto-sizing~~ shipped.
2. ~~A smaller expert bank~~ shipped in 0.4.0: 2-bit affine g128, 9.49 MiB per expert, half the bytes of FP4
   and equal quality once the hyper-connection defect was fixed.
3. ~~The hyper-connection residual mix~~ fixed in 0.6.0 — it was transposed, and it was the cause of every
   quality artefact this project had blamed on quantization.
4. ~~Graph construction on the decode thread~~ largely shipped in 0.6.0: the MoE block is traced once per
   process instead of rebuilt forty times a token, and the fused kernels' scalar parameters are memoised.
5. ~~DSpark / MTP speculative decoding~~ measured and closed twice, most recently on current constants: a
   K-position forward costs 242.6 ms + 26.9 ms per extra position because the only multi-position path is
   the prefill path. It needs a decode-shaped batched forward before the economics change.
6. ~~Mirror striping across two drives~~ shipped in 0.5.0 and withdrawn in 0.6.0: it is a property of the
   expert size, an 18 % loss at 9.49 MiB where it was a gain at 17.93.
7. **The expert hit rate**, which is the only lever a live session resolves. Offline replay at the shipped
   expert size says 44 → 52 GiB is worth 2.6 points of decode hit and 52 → 60 another 2.3; an interactive
   session peaks at 55.6 GiB against a 72 GiB wired limit and 67.7 at 52 against 77.8, so the headroom exists
   and 60 GiB projects to about 75.2 GiB. **Confirmed the only lever in 0.9.4**: the drive delivers the same
   6.8 GB/s whether the machine has 43 GiB wired or nothing, so there is no memory-pressure tax hiding in
   the miss to remove first.
8. ~~The 33 ms a streaming token spends above the all-resident floor~~ closed across 0.9.0 and 0.9.2: 27 of
   it was the Engram row reads, serialised on the decode thread behind the expert stream, and the rest is
   the miss. A streaming token that does not miss is the all-resident floor on every column.
9. ~~Attention's shapes~~ closed on the arithmetic in 0.9.2: a reuse layer is 0.377 ms of weight streaming
   and 0.064 ms of everything a fixed-capacity `attention_kv` would touch, and the `mx.compile` such a shape
   would unlock is closed three ways already. What is left is the FP8 GEMV kernel under it, which is already
   at 79 % of `mx.sum` over its own bytes — 3.5 ms per token, with `uint4` loads, pre-decoded activations
   and a tuned lane split all in place.
10. ~~The compressor and the indexer on the eight source layers~~ decomposed in 0.9.0: about two thirds of
    the 5.5 ms is the indexer, a tenth the compressor and the rest the compressed-KV write, and `INDEX_TOPK`
    is not a lever — an eightfold change in the width is worth 0.066 ms per layer.
11. ~~Whether the miss itself is slowed by decode's own wired memory~~ closed in 0.9.4: `io_workers=8` reads
    experts at 1.46–1.47 ms each whether 42.9 GiB is wired (45 % of the machine) or nothing, matching the
    drive's cold rating, and the runtime's own 1.41 ms blocked-per-miss component sits in that same band.
    No store-side overhead and no memory-pressure tax to remove — the budget (item 7) is the only lever
    left on the miss.

## Project layout

```
src/cachalot/
  model/        transformer blocks, attention variants, MoE, kernels, generation, public API
  cache/        resident expert store (LRU, quotas, deterministic admission)
  io/           prefetch executors
  storage/      safetensors indexing, expert reader, Engram row reader
  metrics/      routing tracer, system snapshots
  cli.py        `cachalot` command
benchmarks/     reproducible benchmark and analysis scripts
tests/          checkpoint-free unit tests (`pytest -q`), model tests (`pytest --model`)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: one architectural change per PR, every performance
claim comes with its benchmark JSON, and nothing may change model output without saying so.

## License

Cachalot is released under the [MIT License](LICENSE). The DeepSeek-V4.1-Flash weights are licensed separately by
DeepSeek under their [MIT model license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE).

## Acknowledgements

- [DeepSeek](https://www.deepseek.com/) for releasing V4.1 Flash with a complete reference implementation and
  protocol code, which made exact reimplementation possible.
- [Apple MLX](https://github.com/ml-explore/mlx) for a lazy array framework with first-class custom Metal kernels.
