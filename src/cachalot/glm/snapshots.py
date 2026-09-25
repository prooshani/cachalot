"""
GLM-5.3-Flash prefix snapshots on disk, so a restarted `serve-glm.sh` keeps an agent's system prompt.

The DeepSeek server has done this since 0.11.0 (HANDOFF sections 15.4, 15.12); GLM kept its prefix cache in
memory only, so every server start prefilled Hermes's ~20k-token system block again. This module reuses
`snapshot_store.SnapshotStore`'s policy (32 files kept by last use, 4 preloaded, the rest fetched on demand)
and swaps its file format for one that holds GLM's cache objects.

A GLM cache is a list with one object per layer: `ArraysCache` (the Kimi-Delta layers' conv and recurrent
state), `CacheList` of `KVCache`s plus a `PoolingCache` (the MLA layers' latent cache and indexer), and
`GlmModel`'s `_NoProjectedCache`. `encode_cache` walks those objects' attributes, stores every array under its
path and everything else (ints, None, the object types) in a JSON skeleton; `decode_cache` rebuilds the same
objects. A `KVCache` is saved up to its offset, not its step-allocated buffer.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import mlx.core as mx

from cachalot.model.snapshot_store import SnapshotStore, runtime_identity

# What a GLM snapshot's numbers depend on in the code. Bump with any change that can move a GLM prefill's
# cache bits (the vendored model code, quantization, the prefill chunk size is part of the identity already).
# Scheduling changes (the scan-resistant expert path, 0.18.0) are bit-identical and do not bump it.
GLM_NUMERICS_VERSION = "glm-0.17.0"

# Only these modules' classes are rebuilt from a file.
_ALLOWED_MODULES = ("cachalot.third_party.mlx_vlm.models.cache", "cachalot.glm.model")


def _encode(obj, path: str, arrays: dict[str, mx.array]):
    if isinstance(obj, mx.array):
        arrays[path] = obj
        return {"a": path}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"v": obj}
    if isinstance(obj, list):
        return {"l": [_encode(v, f"{path}.{i}", arrays) for i, v in enumerate(obj)]}
    if isinstance(obj, tuple):
        return {"t": [_encode(v, f"{path}.{i}", arrays) for i, v in enumerate(obj)]}
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError(f"{path}: dict keys must be strings")
        return {"d": {k: _encode(v, f"{path}.{k}", arrays) for k, v in obj.items()}}
    cls = type(obj)
    if cls.__module__ not in _ALLOWED_MODULES or not hasattr(obj, "__dict__"):
        raise TypeError(f"{path}: cannot save a {cls.__module__}.{cls.__qualname__}")
    fields = dict(obj.__dict__)
    keys = fields.get("keys")
    if isinstance(keys, mx.array) and isinstance(fields.get("offset"), int) and fields.get("values") is not None:
        # a KVCache buffer grows in steps of 256 tokens; only the first `offset` are data
        n = fields["offset"]
        fields["keys"] = keys[..., :n, :]
        fields["values"] = fields["values"][..., :n, :]
    return {
        "o": [cls.__module__, cls.__qualname__],
        "f": {k: _encode(v, f"{path}.{k}", arrays) for k, v in fields.items()},
    }


def _decode(spec, arrays: dict[str, mx.array]):
    if "a" in spec:
        return arrays[spec["a"]]
    if "v" in spec:
        return spec["v"]
    if "l" in spec:
        return [_decode(v, arrays) for v in spec["l"]]
    if "t" in spec:
        return tuple(_decode(v, arrays) for v in spec["t"])
    if "d" in spec:
        return {k: _decode(v, arrays) for k, v in spec["d"].items()}
    module, name = spec["o"]
    if module not in _ALLOWED_MODULES:
        raise ValueError(f"refusing to rebuild {module}.{name}")
    cls = getattr(importlib.import_module(module), name)
    obj = cls.__new__(cls)
    obj.__dict__.update({k: _decode(v, arrays) for k, v in spec["f"].items()})
    return obj


def encode_cache(cache: list) -> tuple[dict[str, mx.array], str]:
    """(arrays by path, JSON skeleton) of a GLM cache list."""
    arrays: dict[str, mx.array] = {}
    skeleton = _encode(list(cache), "c", arrays)
    return arrays, json.dumps(skeleton)


def decode_cache(arrays: dict[str, mx.array], skeleton: str) -> list:
    return _decode(json.loads(skeleton), arrays)


def write_snapshot(snap, path: Path, identity: str) -> None:
    arrays, skeleton = encode_cache(snap.cache)
    arrays["tokens"] = mx.array(snap.tokens, dtype=mx.int32)
    if snap.logits is not None:
        arrays["logits"] = snap.logits
    tmp = path.with_name(path.stem + ".tmp.safetensors")
    mx.save_safetensors(str(tmp), arrays, metadata={"identity": identity, "glm_skeleton": skeleton})
    os.replace(tmp, path)


def read_snapshot(path: Path, identity: str):
    """The snapshot in `path`, or None if it belongs to another runtime state or is not a GLM file."""
    from cachalot.glm.model import Snapshot

    try:
        arrays, meta = mx.load(str(path), return_metadata=True)
    except Exception:
        return None
    if meta.get("identity") != identity or "glm_skeleton" not in meta:
        return None
    mx.eval(*arrays.values())
    tokens = tuple(arrays.pop("tokens").tolist())
    logits = arrays.pop("logits", None)
    try:
        cache = decode_cache(arrays, meta["glm_skeleton"])
    except (KeyError, ValueError, AttributeError, TypeError):
        return None
    return Snapshot(tokens, cache, logits)


class GlmSnapshotStore(SnapshotStore):
    """SnapshotStore's policy over GLM snapshot files."""

    def _load_file(self, name: str):
        return read_snapshot(self.directory / name, self.identity)

    def _write_file(self, snap, name: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        write_snapshot(snap, self.directory / name, self.identity)


def glm_identity(model_path, max_seq_len: int, prefill_chunk: int, numerics: str = "") -> str:
    # the checkpoint is its own "bank": its shards' sizes and mtimes go into the identity; `numerics` is a
    # model's own numerics tag (MiniMax's fused RMSNorm since 0.21.0, HANDOFF 18.2)
    return runtime_identity(model_path, model_path, max_seq_len,
                            f"{GLM_NUMERICS_VERSION}-chunk{prefill_chunk}{numerics}")
