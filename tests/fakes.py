"""Test doubles for the expert storage layer. No checkpoint, no SSD."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cachalot.storage.index import ExpertEntry, TensorRange
from cachalot.storage.reader import ReadChunk

# Small but shape-faithful fake expert: 3 weights + 3 scales, contiguous
# scales first then weights, mirroring the real checkpoint layout.
SCALE_BYTES = 64
WEIGHT_BYTES = 1024
EXPERT_BYTES = 3 * (SCALE_BYTES + WEIGHT_BYTES)


def make_entry(layer: int, expert: int, shard: Path = Path("/fake/shard.safetensors")) -> ExpertEntry:
    base = (layer * 1000 + expert) * EXPERT_BYTES
    tensors = []
    off = base
    for w in ("w1", "w2", "w3"):
        tensors.append(TensorRange(f"layers.{layer}.ffn.experts.{expert}.{w}.scale", shard, off, off + SCALE_BYTES))
        off += SCALE_BYTES
    for w in ("w1", "w2", "w3"):
        tensors.append(TensorRange(f"layers.{layer}.ffn.experts.{expert}.{w}.weight", shard, off, off + WEIGHT_BYTES))
        off += WEIGHT_BYTES
    return ExpertEntry(layer=layer, expert=expert, tensors=tuple(tensors))


def make_index(n_layers: int, n_experts: int) -> dict[tuple[int, int], ExpertEntry]:
    return {(layer, e): make_entry(layer, e) for layer in range(n_layers) for e in range(n_experts)}


class FakeReader:
    """Deterministic in-memory ExpertReader with optional artificial latency."""

    def __init__(self, latency_s: float = 0.0):
        self.latency_s = latency_s
        self.reads = 0
        self.bytes = 0

    def read_expert(self, entry: ExpertEntry) -> tuple[ReadChunk, ...]:
        from cachalot.storage.index import merge_contiguous_ranges

        if self.latency_s:
            time.sleep(self.latency_s)
        chunks = []
        for rr in merge_contiguous_ranges(entry):
            rng = np.random.default_rng(rr.start)
            data = rng.integers(0, 256, rr.size, dtype=np.uint8).tobytes()
            chunks.append(ReadChunk(shard=rr.shard, start=rr.start, data=data))
            self.reads += 1
            self.bytes += rr.size
        return tuple(chunks)

    def close(self) -> None:
        pass
