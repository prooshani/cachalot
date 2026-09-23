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
from pathlib import Path

import mlx.core as mx
import numpy as np

from cachalot.model.prefix_cache import SequenceSnapshot

FORMAT = 1


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


def save(snap: SequenceSnapshot, directory, identity: str, keep: int = 8) -> Path:
    """Write `snap` to `directory` and keep only the `keep` newest files.

    Hermes puts its working directory into the system prompt, so each project
    an agent works in has its own system block and its own file (HANDOFF
    section 15.5); eight keep a handful of projects and harnesses warm.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
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
    path = directory / _file_name(snap)
    # write then rename, so a crash never leaves a truncated file behind
    tmp = path.with_name(path.stem + ".tmp.safetensors")
    mx.save_safetensors(str(tmp), arrays, metadata=meta)
    os.replace(tmp, path)
    files = sorted(directory.glob("prefix-*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        old.unlink(missing_ok=True)
    return path


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
