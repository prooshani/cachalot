"""
Prefix cache for multi-turn generation.

The runtime is stateful: after a prompt is prefilled and a reply decoded,
its attention windows, compressed KV, indexer/compressor state and Engram
history describe exactly the token sequence it has consumed. When the next
request's token sequence begins with a sequence we have already processed,
we restore the matching snapshot and prefill only the new suffix instead of
the whole conversation.

Snapshots are small (tens of MB) because V4.1 Flash keeps KV compressed:
window rings are 128 x 512 per layer and compressed caches are
max_seq_len / ratio x 512 for four source layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class SequenceSnapshot:
    """Opaque copy of TextDecodeRuntime sequence state at `position`."""

    tokens: tuple[int, ...]
    position: int
    logits: mx.array | None
    windows: dict[int, mx.array]
    compressed_caches: dict[int, mx.array]
    compressor_kv: dict[int, mx.array]
    compressor_score: dict[int, mx.array]
    indexer_k: dict[int, mx.array]
    engram_history: list[int]
    shared_compress_kv: mx.array | None
    shared_index_k_layer: int | None
    shared_topk_idxs: mx.array | None
    shared_candidates: mx.array | None

    @property
    def nbytes(self) -> int:
        total = 0
        for group in (
            self.windows,
            self.compressed_caches,
            self.compressor_kv,
            self.compressor_score,
            self.indexer_k,
        ):
            total += sum(a.nbytes for a in group.values())
        if self.logits is not None:
            total += self.logits.nbytes
        return total


def common_prefix_len(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class PrefixCache:
    """
    Keeps the most recent snapshots and finds the longest one that is a
    prefix of a requested token sequence.
    """

    # Each conversation keeps two snapshots (after prompt, after reply); a
    # snapshot is ~30-60 MB, so 16 entries cover 8 interleaved conversations
    # for well under 1 GiB.
    max_entries: int = 16
    _entries: list[SequenceSnapshot] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    reused_tokens: int = 0

    def add(self, snapshot: SequenceSnapshot) -> None:
        # Drop snapshots for the same position (a re-run of the same prefix).
        self._entries = [s for s in self._entries if s.tokens != snapshot.tokens]
        self._entries.append(snapshot)
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def find(self, tokens: tuple[int, ...]) -> SequenceSnapshot | None:
        """
        Longest snapshot whose full token sequence is a prefix of `tokens`.
        A snapshot equal to `tokens` also qualifies (nothing left to prefill).
        """
        best: SequenceSnapshot | None = None
        for snap in self._entries:
            if len(snap.tokens) > len(tokens):
                continue
            if tokens[: len(snap.tokens)] != snap.tokens:
                continue
            if best is None or len(snap.tokens) > len(best.tokens):
                best = snap
        if best is None:
            self.misses += 1
        else:
            self.hits += 1
            self.reused_tokens += len(best.tokens)
        return best

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self._entries)
