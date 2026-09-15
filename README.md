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
| SWA bounded replay, FP4 KV cache (E2M1 + E4M3 scale per 16) | Implemented |
| Official chat protocol (`encoding.py`), thinking mode, reasoning effort | Loaded from the checkpoint, not reimplemented |

Vision input and DSpark speculative decoding are not implemented yet (see [Roadmap](#roadmap)).

## Why this exists

A 552B MoE activates only 8B parameters per token during prefill and 16B during decode, so the *compute* fits a Mac
Studio comfortably. The *bytes* do not: each token touches 240 routed experts (40 layers × top-6) at 18.8 MB each,
and the 15,360 experts total 289 GB. Any runtime for this class of machine is therefore an exercise in
**caching and I/O scheduling**, not kernels. Cachalot is built around that fact:

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

Cachalot is **alpha**. It produces correct output and runs multi-turn sessions, and the numbers below are real,
but decode is still bound by SSD bandwidth. Read [Performance](#performance) before deciding whether it fits your use.

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
| FP4 expert GEMM on simdgroup matrix units for prefill | ✅ shipped, 2.5× less GPU time per expert; prefill wall time already at the SSD floor |
| Batched prefill (attention for all 40 layers, HC, router, routed + shared experts, Engram) | ✅ shipped |
| DSpark / MTP speculative decoding | 🔜 planned |
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
| USB 3.2 Gen 2 SSD | ~1.0 GB/s | Decode 1.3 s/token, cold 512-token prefill 8–9 min |
| Thunderbolt 4/5 NVMe enclosure | 3–6 GB/s | Proportionally faster misses, no code change |
| Internal Mac SSD (tested) | 5.2 GB/s | Decode 0.45 s/token, cold 512-token prefill 86 s |

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

| Setting | Default | Meaning |
|---|---:|---|
| `expert_cache_budget_bytes` | auto | Routed experts kept resident. Auto = unified memory − 12 GiB trunk − 2 GiB MLX cache − 32 GiB system reserve, capped at the Metal recommended working set. `--expert-budget-gib 40` for a fixed value. |
| `mlx_wired_limit_bytes` | auto | Trunk + experts + cache are wired via Metal residency sets so macOS cannot compress them under pressure (without this decode was 3× slower, see [docs/performance.md](docs/performance.md)). |
| `system_reserve_bytes` | 32 GiB | Memory left for macOS, page cache and other applications when auto-sizing. Lower it on a dedicated machine. |
| `mlx_cache_limit_bytes` | 2 GiB | Cap on MLX's free-buffer cache. Larger values recreated allocator stalls under concurrent materialization. |
| `io_workers` | 8 | Loader threads. Bandwidth-bound; more threads do not raise throughput on USB SSDs. |
| `max_seq_len` | 32768 | Sequence capacity for KV and compressed caches (a few hundred MB; CSA2 keeps KV tiny). |

**Memory budget guidance.** More resident experts is the only software lever that materially cuts SSD bytes:
in the routing trace, 40 → 50 GiB removed 6 % of bytes and 40 → 64 GiB removed 13 %, while smarter eviction
policies were worth 1–2 % (see [docs/performance.md](docs/performance.md)). The auto budget takes what the machine
has minus a 32 GiB reserve; shrink the reserve with `CACHALOT_SYSTEM_RESERVE_GIB` on a dedicated box, or pin a budget
with `--expert-budget-gib` if you run other memory-hungry software alongside.

## Performance

Measured on the hardware above with the checkpoint on the **internal SSD (5.2 GB/s)**, auto expert budget
(50 GiB, 2,855 experts = 19 % of the routed set), 512-token prompts through the official chat protocol, greedy
decode. Wall clock, single request. `benchmarks/trace_routing.py` reproduces the table.

| Phase | Throughput | Expert hit rate | SSD read |
|---|---:|---:|---:|
| Cold prefill, 512 tokens (first prompt after start) | 13.2 tok/s (39 s) | 0 % | 173 GiB |
| Warm prefill, 512 tokens, unrelated task | 18.5–19.2 tok/s (27 s) | 21–25 % | 122–133 GiB |
| Return to a previous task, 512 tokens | 18.3 tok/s (28 s) | 22 % | 134 GiB |
| Decode after prefill | **2.3–2.6 tok/s** (0.39–0.43 s/token) | 74–78 % | ~1 GiB / token |
| Decode, every expert resident | 0.068 s/token (14.7 tok/s) | 100 % | 0 |
| Multi-turn follow-up (prefix cache) | 3.9 s prefill vs 9.9 s from scratch | | |

Prefill runs within 10–20 % of the SSD floor (173 GiB at 5.5 GB/s ≈ 33 s cold, ~24 s warm).

Same code on the **USB 3.2 external SSD (1.0 GB/s)**: decode 1.3 s/token, cold 512-token prefill 8–9 min.
The starting point of this project (before the memory, loader, kernel and batching work) was 2.7 s/token decode and
a 270 s cold prefill on that USB disk.

Where the time goes now: decode is ~80 % SSD bytes (misses × 18.8 MB at 5.7 GB/s) and ~20 % compute (0.07 s/token).
Prefill runs at 80–86 % SSD occupancy; the rest is per-expert GEMM launch overhead and per-layer route syncs. The decode ceiling on this machine is set by the
expert hit rate, not by code: 10 tok/s single-stream needs ~96 % hits, the static bound at the largest wireable budget
is ~74 %, and consecutive tokens share only 30 % of their experts so speculative decoding cannot amortize loads.
A machine that holds the routed experts resident (256–512 GB) decodes at the 0.068 s/token compute floor. Details and the measurements behind every design decision are in
[docs/performance.md](docs/performance.md).

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

Ordered by measured impact on bytes read per generated token.

1. ~~Prefix cache~~ shipped.
2. ~~OpenAI-compatible server~~ shipped.
3. ~~Cache policy~~ measured: SLRU/LFU worth 1–2 %, not adopted; per-layer quotas already optimal. Memory budget
   auto-sizing shipped instead.
4. ~~Batched prefill~~ shipped for all 40 layers; prefill runs within 10–20 % of the SSD floor.
5. **DSpark / MTP speculative decoding.** Amortizes expert loads across drafted tokens; the standard answer for
   bandwidth-bound decode.
6. **More kernel fusion** (attention projections, shared expert) now that decode compute is 25 % of the token time.

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
