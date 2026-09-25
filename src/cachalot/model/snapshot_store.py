"""
Prefix-cache snapshots on disk, so a server restart keeps an agent's system prompt.

An agent harness opens every conversation with the same long system prompt and
tool schemas (Hermes Agent: 13.5k tokens, 160-285 s of cold prefill on this
machine). The in-memory prefix cache makes every conversation after the first
reuse it, but a restarted server starts empty and pays the whole prefill again
on its first request. This module writes the snapshot taken where a system
prompt ends to one safetensors file and loads it back at startup (HANDOFF
section 15.4).

A snapshot is only valid for the exact runtime state that produced it: the
same checkpoint, the same expert bank, the same runtime code and the same
max_seq_len (the cache sizes depend on it). All four go into an identity
string stored in every file; a file whose identity does not match is ignored
(and ages out under the `keep` cap), so switching banks back and forth keeps
each bank's snapshot. Identity uses file sizes and modification times, not contents
hashes, so it costs microseconds on a 150 GB bank.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from cachalot.model.prefix_cache import SequenceSnapshot

FORMAT = 1

# What a snapshot's numbers depend on in the code, as a version. It used to be the package version, so every
# release, docs-only ones included, threw away an agent's saved system prompt and cost one cold re-prefill
# (~6-7 minutes at Hermes Desktop's 22k tokens). Bump this, and only this, with any change that can move a
# prefill's KV bits: kernels, quantization, attention, Engram, the tokenizer path, chunk sizes. A scheduling
# change (the Darwin role, MLX_METAL_FAST_SYNCH) is bit-identical and does not bump it. HANDOFF section 15.9.
NUMERICS_VERSION = "0.13.0"


def _stat_key(path: Path) -> str:
    try:
        st = path.stat()
    except OSError:
        return f"{path}:missing"
    return f"{path}:{st.st_size}:{st.st_mtime_ns}"


def runtime_identity(model_path, bank_path, max_seq_len: int, version: str) -> str:
    """Everything a snapshot's numbers depend on, as one string."""
    model_path = Path(model_path)
    bank_path = Path(bank_path)
    parts = [
        f"format={FORMAT}",
        f"version={version}",
        f"max_seq_len={max_seq_len}",
        _stat_key(model_path / "config.json"),
        _stat_key(model_path / "model.safetensors.index.json"),
        _stat_key(bank_path / "config.json"),
    ]
    parts += sorted(_stat_key(p) for p in bank_path.glob("*.safetensors"))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _file_name(snap: SequenceSnapshot) -> str:
    digest = hashlib.sha256(np.asarray(snap.tokens, dtype=np.int32).tobytes())
    return f"prefix-{len(snap.tokens)}-{digest.hexdigest()[:16]}.safetensors"


def _write(snap: SequenceSnapshot, path: Path, identity: str) -> None:
    arrays: dict[str, mx.array] = {"tokens": mx.array(snap.tokens, dtype=mx.int32)}
    for prefix, group in (
        ("windows", snap.windows),
        ("compressed", snap.compressed_caches),
        ("compressor_kv", snap.compressor_kv),
        ("compressor_score", snap.compressor_score),
        ("indexer_k", snap.indexer_k),
    ):
        for k, v in group.items():
            arrays[f"{prefix}.{k}"] = v
    arrays["engram_history"] = mx.array(snap.engram_history, dtype=mx.int64)
    for name in ("logits", "shared_compress_kv", "shared_topk_idxs", "shared_candidates"):
        value = getattr(snap, name)
        if value is not None:
            arrays[name] = value
    meta = {
        "identity": identity,
        "position": str(snap.position),
        "shared_index_k_layer": json.dumps(snap.shared_index_k_layer),
        "shared_compress_kv_layer": json.dumps(snap.shared_compress_kv_layer),
        "image_spans": json.dumps([list(s) for s in snap.image_spans]),
    }
    # write then rename, so a crash never leaves a truncated file behind
    tmp = path.with_name(path.stem + ".tmp.safetensors")
    mx.save_safetensors(str(tmp), arrays, metadata=meta)
    os.replace(tmp, path)


def save(snap: SequenceSnapshot, directory, identity: str, keep: int = 8) -> Path:
    """Write `snap` to `directory` and keep only the `keep` newest files.

    Hermes puts its working directory into the system prompt, so each project
    an agent works in has its own system block and its own file (HANDOFF
    section 15.5); eight keep a handful of projects and harnesses warm.
    The server uses SnapshotStore, which prunes by reuse instead.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _file_name(snap)
    _write(snap, path, identity)
    files = sorted(directory.glob("prefix-*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        old.unlink(missing_ok=True)
    return path


def read_tokens(path) -> tuple[int, ...] | None:
    """A snapshot file's token ids, read from its safetensors header without loading the arrays."""
    try:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
            info = header["tokens"]
            start, end = info["data_offsets"]
            f.seek(8 + n + start)
            data = f.read(end - start)
        return tuple(np.frombuffer(data, dtype=np.int32).tolist())
    except (OSError, ValueError, KeyError, struct.error):
        return None


class SnapshotStore:
    """The server's snapshot directory: system blocks kept across restarts, loaded when a request needs them.

    Every system block the server sees is written. Hermes's `delegate_task`
    subagents each get a block of their own (their task's context is inside
    it) that no later conversation will match, and under 0.15.0's rule (all
    files loaded at startup, the 8 newest kept) one batch of seven subagents
    pushed the main agent's 22k-token block off the disk, so the next restart
    paid its cold prefill (HANDOFF sections 15.10, 15.12).

    Now the directory keeps up to `keep` files (about 65 MB each at 20k
    tokens), least recently *used* pruned first, and a restart loads only the
    `preload` most recently used into memory. The rest stay on disk with their
    token ids indexed; when a request starts with one of them and memory holds
    nothing as long, `fetch` loads it (~0.1 s). Use times live in `index.json`
    next to the files; a file missing from it counts as used at its mtime.
    """

    INDEX = "index.json"

    def __init__(self, directory, identity: str, keep: int = 32, preload: int = 4, clock=time.time):
        self.directory = Path(directory)
        self.identity = identity
        self.keep = keep
        self.preload = preload
        self.clock = clock
        self.fetched = 0
        self.index: dict[str, dict] = self._read_index()
        self.tokens: dict[str, tuple[int, ...]] = {}

    # -- file system; the replay benchmark overrides these ------------------------------------------------
    def _read_index(self) -> dict[str, dict]:
        try:
            return json.loads((self.directory / self.INDEX).read_text())
        except (OSError, ValueError):
            return {}

    def _write_index(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / (self.INDEX + ".tmp")
        tmp.write_text(json.dumps(self.index, indent=1, sort_keys=True))
        os.replace(tmp, self.directory / self.INDEX)

    def _files(self) -> dict[str, float]:
        """Snapshot file name to mtime."""
        if not self.directory.is_dir():
            return {}
        return {
            p.name: p.stat().st_mtime
            for p in self.directory.glob("prefix-*.safetensors")
            if not p.name.endswith(".tmp.safetensors")
        }

    def _read_file_tokens(self, name: str) -> tuple[int, ...] | None:
        return read_tokens(self.directory / name)

    def _load_file(self, name: str) -> SequenceSnapshot | None:
        return load(self.directory / name, self.identity)

    def _write_file(self, snap: SequenceSnapshot, name: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        _write(snap, self.directory / name, self.identity)

    def _remove_file(self, name: str) -> None:
        (self.directory / name).unlink(missing_ok=True)

    # -- policy ---------------------------------------------------------------------------------------------
    def _used(self, name: str, files: dict[str, float]) -> float:
        e = self.index.get(name)
        return e["used"] if e else files.get(name, 0.0)

    def load_all(self) -> list[SequenceSnapshot]:
        """Index every file's tokens; return the `preload` most recently used snapshots, oldest first."""
        for tmp in self.directory.glob("*.tmp.safetensors") if self.directory.is_dir() else ():
            tmp.unlink(missing_ok=True)
        files = self._files()
        order = sorted(files, key=lambda n: -self._used(n, files))
        out = []
        for name in order:
            if len(out) < self.preload:
                snap = self._load_file(name)
                if snap is None:  # another runtime's file: never matches, ages out
                    continue
                self.tokens[name] = snap.tokens
                out.append(snap)
            else:
                toks = self._read_file_tokens(name)
                if toks is not None:
                    self.tokens[name] = toks
        return out[::-1]

    def persist(self, snap: SequenceSnapshot) -> None:
        """PrefixCache.persist: a system block was snapshotted."""
        name = _file_name(snap)
        self.index[name] = {"used": self.clock()}
        self.tokens[name] = snap.tokens
        files = self._files()
        if name not in files:
            self._write_file(snap, name)
            files[name] = self.clock()
        self._prune(files)

    def on_find(self, tokens: tuple[int, ...], blocks: list[SequenceSnapshot]) -> None:
        """PrefixCache.on_find: `blocks` are the cached system blocks that `tokens` starts with."""
        touched = False
        for block in blocks:
            name = _file_name(block)
            if name in self.tokens:
                self.index[name] = {"used": self.clock()}
                touched = True
        if touched:
            self._write_index()

    def fetch(self, tokens: tuple[int, ...], longer_than: int) -> SequenceSnapshot | None:
        """PrefixCache.fetch: the longest block on disk that `tokens` starts with, if longer than `longer_than`."""
        best = None
        for name, toks in self.tokens.items():
            n = len(toks)
            if n > longer_than and n <= len(tokens) and tokens[:n] == toks and (best is None or n > len(best[1])):
                best = (name, toks)
        if best is None:
            return None
        snap = self._load_file(best[0])
        if snap is None or snap.tokens != best[1]:
            self.tokens.pop(best[0], None)
            return None
        self.fetched += 1
        self.index[best[0]] = {"used": self.clock()}
        self._write_index()
        return snap

    def _prune(self, files: dict[str, float]) -> None:
        keep = set(sorted(files, key=lambda n: -self._used(n, files))[: self.keep])
        for name in files:
            if name not in keep:
                self._remove_file(name)
                self.tokens.pop(name, None)
        self.index = {k: v for k, v in self.index.items() if k in keep}
        self._write_index()


def _group(arrays: dict[str, mx.array], prefix: str) -> dict[int, mx.array]:
    return {
        int(name.split(".", 1)[1]): a
        for name, a in arrays.items()
        if name.startswith(prefix + ".")
    }


def load(path, identity: str) -> SequenceSnapshot | None:
    """The snapshot in `path`, or None if it belongs to another runtime state."""
    try:
        arrays, meta = mx.load(str(path), return_metadata=True)
    except Exception:
        return None
    if meta.get("identity") != identity:
        return None
    mx.eval(*arrays.values())
    return SequenceSnapshot(
        tokens=tuple(arrays["tokens"].tolist()),
        position=int(meta["position"]),
        logits=arrays.get("logits"),
        windows=_group(arrays, "windows"),
        compressed_caches=_group(arrays, "compressed"),
        compressor_kv=_group(arrays, "compressor_kv"),
        compressor_score=_group(arrays, "compressor_score"),
        indexer_k=_group(arrays, "indexer_k"),
        engram_history=[int(x) for x in arrays["engram_history"].tolist()],
        shared_compress_kv=arrays.get("shared_compress_kv"),
        shared_index_k_layer=json.loads(meta["shared_index_k_layer"]),
        shared_topk_idxs=arrays.get("shared_topk_idxs"),
        shared_candidates=arrays.get("shared_candidates"),
        image_spans=tuple(tuple(s) for s in json.loads(meta["image_spans"])),
        shared_compress_kv_layer=json.loads(meta["shared_compress_kv_layer"]),
    )


def load_all(directory, identity: str) -> list[SequenceSnapshot]:
    """Every snapshot in `directory` that matches `identity`, oldest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    out = []
    files = sorted(directory.glob("prefix-*.safetensors"), key=lambda p: p.stat().st_mtime)
    for path in files:
        if path.name.endswith(".tmp.safetensors"):
            path.unlink(missing_ok=True)
            continue
        snap = load(path, identity)
        if snap is not None:
            out.append(snap)
    return out
