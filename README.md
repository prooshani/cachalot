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
              │ Resident trunk (~12 GiB, MLX)  │   │ ResidentExpertStore (MLX)     │
              │ attention · router · shared    │   │ 40 GiB default · per-layer    │
              │ expert · norms · heads · RoPE  │   │ quotas · deterministic        │
              └────────────────────────────────┘   │ admission · cross-turn reuse  │
                                                   └────────────┬─────────────────┘
                                                                │ miss
                                                   ┌────────────▼─────────────────┐
                                                   │ Prefetch workers (8 threads)  │
                                                   │ pread → NumPy view → mx.array │
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
| 40 GiB resident expert cache, cross-turn locality | ✅ working, validated |
| MLX allocator tuning (2 GiB free-buffer cap) | ✅ shipped, removed 100–380 ms allocation stalls |
| Routing trace + offline cache-policy analysis | ✅ `benchmarks/` |
| Unit tests without checkpoint | ✅ `pytest -q` |
| OpenAI-compatible server (`/v1/chat/completions` SSE, tools, thinking, `/v1/completions`) | ✅ working, tested |
| Prefix cache (only new tokens are prefilled per turn) | ✅ working |
| `cachalot serve / chat / doctor / bench` CLI | ✅ working |
| Parallel loading of a decode layer's expert misses | ✅ shipped |
| Frequency-aware expert admission, RAM byte tier | 🚧 in progress |
| Batched prefill attention/MoE GEMM | 🔜 planned |
| DSpark / MTP speculative decoding | 🔜 planned |
| Vision | ❌ not planned for v1 |

## Hardware

Tested on a **Mac Studio M3 Ultra, 96 GB unified memory, 60-core GPU**, with the checkpoint on a
**Crucial X10 Pro 4 TB over USB 3.2 Gen 2**.

Requirements:

- Apple Silicon Mac with **≥ 64 GB unified memory** (96 GB recommended; the default budget targets 96 GB).
- macOS 14+ with Metal 3 or newer.
- **~480 GB** of storage for the checkpoint. Storage speed is the single largest performance factor:

| Storage path | Sequential read | Effect on Cachalot |
|---|---:|---|
| USB 3.2 Gen 2 SSD (tested) | ~1.0 GB/s | Decode is I/O-bound at ~1 s per 1 GB of expert misses |
| Thunderbolt 4/5 NVMe enclosure | 3–6 GB/s | 3–6× faster expert misses, no code change |
| Internal Mac SSD | 5–7 GB/s | Best case; needs ~300 GB free for the expert shards |

The runtime already saturates a USB SSD at queue depth 1 (17.7 ms per expert read), so more threads do not help;
faster storage does.

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
| `expert_cache_budget_bytes` | 40 GiB | Routed experts kept resident in MLX memory. Sized so the whole runtime fits a 96 GB Mac with headroom for the OS and an agent harness. |
| `mlx_cache_limit_bytes` | 2 GiB | Cap on MLX's free-buffer cache. Larger values recreate allocator stalls under 8-way concurrent `mx.array` materialization. |
| `mlx_memory_limit_bytes` | 64 GiB | MLX working-set limit. |
| `io_workers` | 8 | Prefetch threads. Bandwidth-bound; more threads do not raise throughput on USB SSDs. |
| `max_seq_len` | 32768 | Sequence capacity for KV and compressed caches (a few hundred MB; CSA2 keeps KV tiny). |

**Memory budget guidance.** The 40 GiB default is deliberately conservative so that more machines can run the model.
On a 96 GB Mac the runtime uses ≈52 GiB active; the remaining ≈35 GB is used by macOS as page cache for the expert
shards, which acts as an implicit second tier. If you run nothing else heavy, raising the budget increases hit rate
roughly in proportion to the coverage curve in `benchmarks/results/` (see [Benchmarks](#benchmarks)).

## Performance

Measured on the hardware above, 40 GiB expert budget, 512-token prompts through the official chat protocol,
greedy decode. Numbers are wall clock.

| Phase | Throughput | Expert cache hit rate | SSD read |
|---|---:|---:|---:|
| Cold prefill, 512 tokens | 1.9 tok/s | 0 % | 167 GiB |
| Warm prefill (different task), 256 tokens | 2.2–2.5 tok/s | 25–31 % | 67–78 GiB |
| Return to a previous task, 256 tokens | 2.2 tok/s | 28 % | 79 GiB |
| Decode after prefill | 0.45–0.5 tok/s (2.1–2.4 s/token) | 64–71 % | 1.2–1.5 GiB / token |

Where the time goes: a 512-token prompt touches 62 % of all 15,360 experts, so cold prefill is 167 GiB of SSD reads
at ~1 GB/s. Decode touches 240 experts per token; with a 65 % hit rate that is ~84 misses × 18.8 MB ≈ 1.6 GB per token.
**The bottleneck is bytes, not compute.** The engineering roadmap below is ordered by bytes saved.

## How it works

**Layer-major prefill.** The prompt is processed one layer at a time. Attention, compressor, indexer and Engram state
evolve token-sequentially inside the layer (exactly as in decode), while the MoE phase is regrouped *expert-major*:
every routed expert is acquired once per layer and applied to all tokens that selected it. Per-token top-k
accumulation order is preserved, so results are bit-identical to sequential decode.

**Deterministic admission.** Before a layer's prefetch starts, `ResidentExpertStore.prepare_prefill_layer()` decides
the layer's resident set from logical work order and per-layer quotas. Asynchronous SSD completion order therefore
cannot change what ends up cached, which makes runs reproducible and cache behaviour debuggable.

**Cross-turn residency.** `reset()` clears sequence state but keeps experts. Returning to a previously seen task after
two unrelated ones saved 31 GiB of SSD reads in the validation benchmark.

**Allocator hygiene.** With eight workers calling `mx.array()` concurrently, MLX's free-buffer cache grew to ~10 GiB
and produced hundreds of 100–380 ms allocation stalls on warm turns. Capping it at 2 GiB removed the tail without
changing hit rates or SSD traffic. This is the kind of finding the `benchmarks/` instrumentation exists to make.

**Exactness.** Every kernel has a NumPy reference (`fp8_ref.py`, `fp4.py`) and the routing, sampling
(Gumbel-max equivalent of the official exponential-race sampler), prompt encoding and completion parsing use the
checkpoint's own code paths.

## Benchmarks

```bash
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
3. **Cache policy.** Frequency-aware admission, per-layer budgets from measured skew, optional static hot set,
   RAM byte tier above the MLX budget. Parallel fetch of a layer's decode misses.
4. **Batched prefill.** Windowed causal attention, batched router/HC/shared expert, FP4 dequant + GEMM for
   many-token expert application. Removes the per-token Python overhead that costs ~90 s on a 512-token cold prefill.
5. **DSpark / MTP speculative decoding.** Amortizes expert loads across drafted tokens; the standard answer for
   bandwidth-bound decode.
6. **Native kernels** only where profiling shows Python or dispatch overhead dominates after the above.

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
