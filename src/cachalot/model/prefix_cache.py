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
from typing import Callable

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
    # (start, length, digest) per image span consumed; see
    # TextDecodeRuntime.image_spans.
    image_spans: tuple[tuple[int, int, str], ...] = ()
    # When the published compress_kv is one of compressed_caches, the layer
    # it came from (restore re-points at the padded copy); else None and
    # shared_compress_kv holds the array itself.
    shared_compress_kv_layer: int | None = None

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

    # Each request adds a snapshot after its prompt, one after its reply, and
    # one per 4096-token prefill chunk boundary. Snapshots keep only the
    # written cache rows, ~15 MB at 3k tokens and ~33 MB at 9k (HANDOFF
    # section 15.3), so 16 entries stay well under 1 GiB.
    max_entries: int = 16
    _entries: list[SequenceSnapshot] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    reused_tokens: int = 0
    # Called with the snapshots taken at a prompt boundary (where a system
    # prompt ends), which outlive the conversation; the server sets it to
    # write them to disk (snapshot_store). None keeps everything in memory.
    persist: Callable[[SequenceSnapshot], None] | None = None
    # Boundary snapshots (an agent's system block) are evicted only after
    # every per-turn snapshot: one long agent session adds two snapshots per
    # request and used to push the system block out of a 16-entry LRU, so the
    # next new session paid the whole cold prefill again (HANDOFF section
    # 15.5). At most max_pinned are kept this way; the least recently used
    # one beyond that becomes an ordinary entry.
    max_pinned: int = 8
    _pinned: set = field(default_factory=set)

    def add(self, snapshot: SequenceSnapshot, *, boundary: bool = False) -> None:
        if boundary and self.persist is not None:
            try:
                self.persist(snapshot)
            except Exception as exc:  # a full disk must not fail the request
                print(f"[prefix-cache] could not persist snapshot: {exc}", flush=True)
        # Drop snapshots for the same position (a re-run of the same prefix).
        self._entries = [s for s in self._entries if s.tokens != snapshot.tokens]
        self._entries.append(snapshot)
        if boundary:
            self._pinned.add(snapshot.tokens)
        pinned = [s for s in self._entries if s.tokens in self._pinned]
        for s in pinned[: max(0, len(pinned) - self.max_pinned)]:
            self._pinned.discard(s.tokens)  # oldest first: _entries is in LRU order
        while len(self._entries) > self.max_entries:
            victim = next(
                (s for s in self._entries if s.tokens not in self._pinned), self._entries[0]
            )
            self._entries.remove(victim)
            self._pinned.discard(victim.tokens)

    def find(
        self,
        tokens: tuple[int, ...],
        image_spans: tuple[tuple[int, int, str], ...] = (),
    ) -> SequenceSnapshot | None:
        """
        Longest snapshot whose full token sequence is a prefix of `tokens`.
        A snapshot equal to `tokens` also qualifies (nothing left to prefill).

        Every image position carries the same token id, so token equality
        alone would match two different images; a snapshot also has to
        hold exactly the requested image spans that fall inside it.
        """
        best: SequenceSnapshot | None = None
        for snap in self._entries:
            if len(snap.tokens) > len(tokens):
                continue
            if tokens[: len(snap.tokens)] != snap.tokens:
                continue
            inside = tuple(s for s in image_spans if s[0] < len(snap.tokens))
            if inside != snap.image_spans:
                continue
            if best is None or len(snap.tokens) > len(best.tokens):
                best = snap
        if best is None:
            self.misses += 1
        else:
            self.hits += 1
            self.reused_tokens += len(best.tokens)
            # least-recently-used eviction: a hit moves to the back
            self._entries.remove(best)
            self._entries.append(best)
        return best

    def clear(self) -> None:
        self._entries.clear()
        self._pinned.clear()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self._entries)
