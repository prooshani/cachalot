"""
GLM-5.3-Flash on Cachalot: mlx-vlm's model code (vendored, unmodified) for everything but the
routed experts, which stream from SSD through Cachalot's wired expert store.

Only the non-expert weights are loaded into memory (~10 GB for the 4-bit build: attention, the
34 KDA and 11 MLA layers, the dense and shared MLPs, hyper-connections, embeddings, head). The
12,096 routed experts (42 MoE layers x 288, 13.5 MiB each at 4-bit) are read on demand into a
fixed pool of wired slots sized by the expert budget, least recently used evicted first.

Text only for now: the vision tower and the MTP layer are not loaded.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.glm.experts import StreamingSwitchGLU, build_glm_expert_index, tensor_sizes
from cachalot.storage.index import read_safetensors_header
from cachalot.storage.reader import ExpertReader
from cachalot.third_party.mlx_vlm.models.glm5_next.config import TextConfig
from cachalot.third_party.mlx_vlm.models.cache import CacheList, KVCache
from cachalot.third_party.mlx_vlm.models.glm5_next.language import Glm5NextMoE, LanguageModel

# the resident expert set survives a restart (HANDOFF 18.2); CACHALOT_WARM_SET=0 turns it off
WARM_SET = os.environ.get("CACHALOT_WARM_SET", "1") != "0"

_NP_DTYPE = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "U32": np.uint32, "I32": np.int32,
             "U8": np.uint8, "I64": np.int64}


def _is_routed_expert(name: str) -> bool:
    return ".mlp.experts." in name


def _wanted(name: str) -> bool:
    """Non-expert text weights; the vision tower and the MTP block stay on disk."""
    if name.startswith("model.visual.") or ".mtp." in name or name.startswith("mtp."):
        return False
    return not _is_routed_expert(name)


def load_non_expert_weights(model_path: Path, wanted=_wanted) -> dict[str, mx.array]:
    """Read every wanted tensor by its byte range (no pass over the expert bytes)."""
    out: dict[str, mx.array] = {}
    for shard in sorted(model_path.glob("model-*.safetensors")):
        header, data_start = read_safetensors_header(shard)
        names = [n for n in header if n != "__metadata__" and wanted(n)]
        if not names:
            continue
        with open(shard, "rb", buffering=0) as f:
            for name in names:
                meta = header[name]
                a, b = meta["data_offsets"]
                f.seek(data_start + a)
                buf = f.read(b - a)
                arr = np.frombuffer(buf, dtype=_NP_DTYPE[meta["dtype"]]).reshape(meta["shape"])
                value = mx.array(arr)
                if meta["dtype"] == "BF16":
                    value = value.view(mx.bfloat16)
                out[name] = value
    return out


def _remap(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """mlx-vlm Model.sanitize's key renames for the text side."""
    out = {}
    for key, value in weights.items():
        if key.startswith("model.language_model."):
            key = "language_model.model." + key[len("model.language_model."):]
        elif key.startswith("lm_head."):
            key = "language_model." + key
        out[key] = value
    return out


class _NoProjectedCache(KVCache):
    """Stands in for the MLA layers' projected prefill cache, which mlx-vlm keeps per head:
    ~720 KB per token over the 11 MLA layers (64 heads x 256 x K and V), ~14 GB at Hermes's
    20k-token prompts, on top of the expert cache. Reporting size -1 makes every prefill chunk
    project from the compact latent cache instead (language.py, Glm5NextAttention._attend)."""

    def size(self):
        return -1


def _clone(obj):
    """A snapshot of a cache tree that later in-place updates cannot reach.

    MLX slice assignment updates an array object in place, so a snapshot that shares array
    objects with the live cache would change under it; every array is re-wrapped instead
    (mx.array(x) is a new object over the same data, no copy until one side is updated)."""
    if isinstance(obj, mx.array):
        return mx.array(obj)
    if isinstance(obj, list):
        return [_clone(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and type(obj).__module__.endswith("models.cache") or isinstance(obj, _NoProjectedCache):
        new = copy.copy(obj)
        new.__dict__ = {k: _clone(v) for k, v in obj.__dict__.items()}
        return new
    return obj


@dataclass
class Snapshot:
    tokens: tuple[int, ...]
    cache: list
    logits: mx.array | None

    @property
    def nbytes(self) -> int:
        total = 0

        def walk(o):
            nonlocal total
            if isinstance(o, mx.array):
                total += o.nbytes
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v)
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif hasattr(o, "__dict__"):
                for v in o.__dict__.values():
                    walk(v)

        walk(self.cache)
        return total


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    prefill_seconds: float = 0.0
    completion_tokens: int = 0
    decode_seconds: float = 0.0
    misses: int = 0
    hits: int = 0
    read_bytes: int = 0

    @property
    def decode_tok_s(self) -> float:
        return self.completion_tokens / self.decode_seconds if self.decode_seconds else 0.0


class GlmModel:
    """GLM-5.3-Flash with routed experts streamed from SSD."""

    PREFILL_CHUNK = int(os.environ.get("CACHALOT_GLM_PREFILL_CHUNK", "2048"))

    def __init__(
        self,
        model_path,
        *,
        expert_budget_gib: float = 52.0,
        wired_limit_gib: float | None = None,
        load_workers: int = 8,
        heartbeat_seconds: float = 0.5,
        verbose: bool = True,
    ) -> None:
        t0 = time.perf_counter()
        self.model_path = Path(model_path)
        config = json.loads((self.model_path / "config.json").read_text())
        self.config = TextConfig.from_dict(config["text_config"])
        quant = config.get("quantization") or config.get("quantization_config") or {}

        self.expert_format, index = build_glm_expert_index(self.model_path)
        # layer 45 is the MTP block's MoE, not loaded
        self.expert_index = {k: v for k, v in index.items() if k[0] < self.config.num_hidden_layers}
        sizes = tensor_sizes(self.expert_format)
        expert_bytes = sum(sizes.values())
        budget = int(expert_budget_gib * 1024**3)

        # MLX keeps freed buffers for reuse; unbounded, a long prefill's activations pile up past the wired
        # set and macOS swaps them (HANDOFF 17.1). DeepSeek's runtime caps it at 2 GiB (config.py) as well.
        mx.set_cache_limit(int(float(os.environ.get("CACHALOT_GLM_MLX_CACHE_GIB", "2")) * 1024**3))
        if wired_limit_gib is None:
            wired_limit_gib = float(os.environ.get("CACHALOT_MLX_WIRED_LIMIT_GIB", "80"))
        if wired_limit_gib > 0:
            from cachalot.config import device_memory

            _, recommended = device_memory()  # Metal refuses a limit above its working set
            mx.set_wired_limit(int(min(wired_limit_gib * 1024**3, recommended)))

        self.store = ResidentExpertStore(
            budget,
            ExpertReader(),
            tensor_sizes=sizes,
            # one prefill layer's misses plus the next layer read early (experts.PREFILL_SCAN)
            transient_slots=2 * self.config.n_routed_experts + 16,
            load_workers=load_workers,
            verbose=verbose,
        )
        self.store.format = self.expert_format

        self.model = LanguageModel(self.config)
        n_moe = 0
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer.mlp, Glm5NextMoE):
                activation = layer.mlp.switch_mlp.activation
                # replaced before anything evaluates: the stacked expert parameters are never allocated
                layer.mlp.switch_mlp = StreamingSwitchGLU(
                    i, self.store, self.expert_index, self.expert_format, activation
                )
                n_moe += 1

        weights = self.model.sanitize(_remap(load_non_expert_weights(self.model_path)))
        weights = {k[len("language_model."):]: v for k, v in weights.items() if k.startswith("language_model.")}

        def quantized(path, module):
            return hasattr(module, "to_quantized") and f"{path}.scales" in weights

        nn.quantize(
            self.model,
            group_size=int(quant.get("group_size", 64)),
            bits=int(quant.get("bits", 4)),
            mode=quant.get("mode", "affine"),
            class_predicate=quantized,
        )
        params = dict(nn.utils.tree_flatten(self.model.parameters()))
        missing = sorted(set(params) - set(weights))
        unexpected = sorted(set(weights) - set(params))
        if missing:
            raise ValueError(f"{len(missing)} model parameters have no weight, e.g. {missing[:5]}")
        self.model.load_weights(list((k, v) for k, v in weights.items() if k in params))
        mx.eval(self.model.parameters())
        self.model.eval()
        self.unused_weights = unexpected
        self.trunk_bytes = sum(v.nbytes for v in params.values())

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.eos_ids = set(config["text_config"].get("eos_token_id") or [])

        self.prefix: list[Snapshot] = []
        self.disk = None  # snapshots.GlmSnapshotStore, attach_snapshot_store()
        self.max_seq_len = int(os.environ.get("CACHALOT_GLM_MAX_SEQ_LEN", "131072"))
        self._lock = threading.RLock()
        self._busy = False
        self._idle_since = time.perf_counter()
        self._stop = threading.Event()
        if heartbeat_seconds > 0:
            threading.Thread(target=self._heartbeat, args=(heartbeat_seconds,), daemon=True, name="glm-heartbeat").start()

        if verbose:
            print(
                f"GLM-5.3-Flash: {len(self.model.layers)} layers ({n_moe} MoE), trunk {self.trunk_bytes / 1024**3:.1f} GiB, "
                f"{len(self.expert_index)} experts x {expert_bytes / 2**20:.2f} MiB, "
                f"{self.store.capacity} resident slots ({self.store.budget_bytes / 1024**3:.1f} GiB), "
                f"loaded in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    # -- keep the wired set wired while idle (macOS un-wires an idle Metal queue) -------------------------
    def _heartbeat(self, period: float) -> None:
        probe = mx.zeros((1,))
        while not self._stop.wait(period):
            if self._busy or time.perf_counter() - self._idle_since < period:
                continue
            if self._lock.acquire(blocking=False):
                try:
                    if not self._busy:
                        mx.eval(probe + 1)
                finally:
                    self._lock.release()

    def close(self) -> None:
        self._stop.set()
        self.store.close()

    # -- text ---------------------------------------------------------------------------------------------
    # -- the model family's chat format (GlmEngine and `cachalot chat` go through these) -------------------
    @property
    def splitter_cls(self):
        from cachalot.glm.engine import _GlmSplitter

        return _GlmSplitter

    def render_chat(self, messages, *, tools=None, thinking=False, effort=None, add_generation_prompt=True) -> str:
        """The prompt text. GLM's template always opens `<think>`; thinking off closes it at once."""
        from cachalot.glm.engine import THINK_END, _effort

        kwargs = {}
        effort = _effort(effort)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False, **kwargs
        )
        if add_generation_prompt and not thinking:
            text += THINK_END
        return text

    def parse_tool_calls(self, text: str, tools=None):
        from cachalot.glm.engine import parse_tool_calls

        return parse_tool_calls(text, tools)

    def encode_chat(self, messages, *, tools=None, reasoning_effort=None, add_generation_prompt=True) -> list[int]:
        kwargs = {}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False, **kwargs
        )
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def new_cache(self):
        caches = self.model.make_cache()
        out = []
        for layer, cache in zip(self.model.layers, caches):
            if isinstance(cache, CacheList):
                parts = list(cache.caches)
                # the projected cache is the last KVCache of an MLA layer's list (language.py make_cache)
                parts[-1] = _NoProjectedCache()
                cache = CacheList(*parts)
            out.append(cache)
        return out

    @staticmethod
    def snapshot(tokens, cache, logits=None) -> Snapshot:
        return Snapshot(tuple(tokens), _clone(cache), logits)

    @staticmethod
    def restore(snap: Snapshot) -> list:
        return _clone(snap.cache)

    def _forward(self, tokens: list[int], cache) -> mx.array:
        out = self.model(mx.array(tokens, dtype=mx.int32)[None], cache=cache)
        return out.logits[:, -1, :]

    def prefill(self, tokens: list[int], cache) -> mx.array:
        logits = None
        try:
            for start in range(0, len(tokens), self.PREFILL_CHUNK):
                logits = self._forward(tokens[start:start + self.PREFILL_CHUNK], cache)
                mx.eval(logits)
        finally:
            self.store.release_prefill()
        return logits

    @staticmethod
    def _sample(logits: mx.array, temperature: float, top_p: float) -> int:
        if temperature <= 0:
            return int(mx.argmax(logits, axis=-1).item())
        probs = mx.softmax(logits.astype(mx.float32) / temperature, axis=-1)[0]
        p = np.array(probs, dtype=np.float64)
        if top_p < 1.0:
            order = np.argsort(-p)
            keep = np.cumsum(p[order]) <= top_p
            keep[0] = True
            mask = np.zeros_like(p, dtype=bool)
            mask[order[keep]] = True
            p = np.where(mask, p, 0.0)
        p /= p.sum()
        return int(np.random.choice(len(p), p=p))

    # -- prefix cache ---------------------------------------------------------------------------------------
    PREFIX_BYTES = int(float(os.environ.get("CACHALOT_GLM_PREFIX_GIB", "3")) * 1024**3)

    def attach_snapshot_store(self, directory) -> str:
        """Keep system-block snapshots in `directory` across restarts (HANDOFF 17.1); returns a status line."""
        from cachalot.glm.snapshots import GlmSnapshotStore, glm_identity

        t0 = time.perf_counter()
        # files kept on disk; MiniMax's full-attention cache is ~120 KB per token (~2.4 GB at 20k), GLM's ~12 KB
        keep = int(os.environ.get("CACHALOT_SNAPSHOT_KEEP", "32"))
        store = GlmSnapshotStore(directory, glm_identity(self.model_path, self.max_seq_len, self.PREFILL_CHUNK,
                                                          self.NUMERICS_TAG), keep=keep)
        loaded = store.load_all()
        for snap in loaded:
            self._add_prefix(snap)
        self.disk = store
        warm = self.start_warm_set(Path(directory) / "resident-set.json") if WARM_SET else "warm set off"
        return (f"prefix snapshots: {len(loaded)} loaded from {directory} "
                f"({', '.join(str(len(s.tokens)) for s in loaded) or 'none'} tokens), "
                f"{len(store.tokens) - len(loaded)} more on disk, in {time.perf_counter() - t0:.2f}s; {warm}")

    # -- warm restart (HANDOFF 18.2) ---------------------------------------------------------------------
    # The resident expert set is written after every request and read back into the cache at startup, in the
    # background, so the first turn after a restart decodes at the last session's hit rate, not a cold one.
    NUMERICS_TAG = ""

    def _warm_identity(self) -> dict:
        return {"model": str(self.model_path), "experts": len(self.expert_index), "format": repr(self.expert_format)}

    def start_warm_set(self, path) -> str:
        self._warm_path = Path(path)
        try:
            data = json.loads(self._warm_path.read_text())
        except (OSError, ValueError):
            return f"warm set: none at {path}"
        if data.get("identity") != self._warm_identity():
            return f"warm set: {path} is another model's, ignored"
        keys = [tuple(k) for k in data.get("keys", []) if tuple(k) in self.expert_index]
        # oldest first, so the preload keeps their recency order; only the newest that fit
        keys = keys[-self.store.capacity:]
        entries = [self.expert_index[k] for k in keys]

        def run():
            t0 = time.perf_counter()
            n = self.store.preload(entries, reserve_fraction=0.0)
            print(f"warm set: {n} experts ({n * self.store.expert_bytes / 2**30:.1f} GiB) "
                  f"read back in {time.perf_counter() - t0:.1f}s", flush=True)

        self._warm_thread = threading.Thread(target=run, daemon=True, name="warm-set")
        self._warm_thread.start()
        return f"warm set: reading {len(entries)} experts back in the background"

    def _wait_warm_set(self) -> None:
        thread = getattr(self, "_warm_thread", None)
        if thread is not None:
            thread.join()
            self._warm_thread = None

    def _save_warm_set(self) -> None:
        path = getattr(self, "_warm_path", None)
        if path is None:
            return
        try:
            data = {"identity": self._warm_identity(), "keys": [list(k) for k in self.store.resident_keys()]}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
        except OSError as exc:
            print(f"warm set not saved: {exc}", flush=True)

    def _persist(self, snap: Snapshot) -> None:
        if self.disk is None:
            return
        try:
            self.disk.persist(snap)
        except Exception as exc:  # a full disk must not fail the request
            print(f"prefix snapshot not saved: {exc}", flush=True)

    def _find_prefix(self, tokens: tuple[int, ...]) -> Snapshot | None:
        best = None
        for snap in self.prefix:
            n = len(snap.tokens)
            if n > len(tokens) or tokens[:n] != snap.tokens:
                continue
            if n == len(tokens) and snap.logits is None:
                continue
            if best is None or n > len(best.tokens):
                best = snap
        if self.disk is not None:
            try:
                fetched = self.disk.fetch(tokens, len(best.tokens) if best is not None else 0)
                if fetched is not None and len(fetched.tokens) < len(tokens):
                    self._add_prefix(fetched)
                    best = fetched
                self.disk.on_find(tokens, [p for p in self.prefix if tokens[:len(p.tokens)] == p.tokens])
            except Exception as exc:
                print(f"prefix snapshot lookup failed: {exc}", flush=True)
        if best is not None:  # most recently used last
            self.prefix.remove(best)
            self.prefix.append(best)
        return best

    def _add_prefix(self, snap: Snapshot) -> None:
        self.prefix = [p for p in self.prefix if p.tokens != snap.tokens]
        self.prefix.append(snap)
        while len(self.prefix) > 1 and sum(p.nbytes for p in self.prefix) > self.PREFIX_BYTES:
            self.prefix.pop(0)

    def stream(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        cancel: threading.Event | None = None,
        boundary: int = 0,
    ):
        """Yields ("prefill", reused, seconds), ("token", id) ..., ("done", finish, decode_seconds)."""
        with self._lock:
            self._busy = True
            try:
                self._wait_warm_set()
                tokens = tuple(prompt_tokens)
                t0 = time.perf_counter()
                snap = self._find_prefix(tokens)
                if snap is not None:
                    cache, reused, logits = self.restore(snap), len(snap.tokens), snap.logits
                else:
                    cache, reused, logits = self.new_cache(), 0, None
                pos = reused
                cuts = sorted({c for c in (boundary,) if reused < c < len(tokens)} | {len(tokens)})
                for cut in cuts:
                    while pos < cut:
                        if cancel is not None and cancel.is_set():
                            yield ("done", "cancel", 0.0)
                            return
                        end = min(pos + self.PREFILL_CHUNK, cut)
                        logits = self._forward(list(tokens[pos:end]), cache)
                        mx.eval(logits)
                        pos = end
                    if cut < len(tokens):
                        block = self.snapshot(tokens[:cut], cache, None)
                        self._add_prefix(block)
                        if cut == boundary:
                            self._persist(block)
                self.store.release_prefill()
                if reused < len(tokens):
                    self._add_prefix(self.snapshot(tokens, cache, logits))
                yield ("prefill", reused, time.perf_counter() - t0)
                t1 = time.perf_counter()
                out: list[int] = []
                finish = "length"
                for _ in range(max_new_tokens):
                    if cancel is not None and cancel.is_set():
                        finish = "cancel"
                        break
                    token = self._sample(logits, temperature, top_p)
                    out.append(token)
                    if token in self.eos_ids:
                        finish = "stop"
                        yield ("token", token)
                        break
                    yield ("token", token)
                    logits = self._forward([token], cache)
                    mx.eval(logits)
                if out and finish != "cancel":
                    # prompt + reply without the final token, which was never fed: the next turn's
                    # prompt repeats the reply and continues from it
                    fed = tokens + tuple(out[:-1]) if finish == "stop" else tokens + tuple(out)
                    self._add_prefix(self.snapshot(fed, cache, None if finish == "stop" else logits))
                yield ("done", finish, time.perf_counter() - t1)
            finally:
                self.store.release_prefill()
                self._save_warm_set()
                self._idle_since = time.perf_counter()
                self._busy = False

    def generate(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        on_token=None,
        stats: GenerationStats | None = None,
    ) -> list[int]:
        stats = stats if stats is not None else GenerationStats()
        with self._lock:
            self._busy = True
            try:
                s0 = self.store.stats()
                cache = self.new_cache()
                t0 = time.perf_counter()
                logits = self.prefill(prompt_tokens, cache)
                stats.prompt_tokens = len(prompt_tokens)
                stats.prefill_seconds = time.perf_counter() - t0
                out: list[int] = []
                t1 = time.perf_counter()
                for _ in range(max_new_tokens):
                    token = self._sample(logits, temperature, top_p)
                    out.append(token)
                    if on_token is not None:
                        on_token(token)
                    if token in self.eos_ids:
                        break
                    logits = self._forward([token], cache)
                    mx.eval(logits)
                stats.completion_tokens = len(out)
                stats.decode_seconds = time.perf_counter() - t1
                s1 = self.store.stats()
                stats.hits = s1.cache_hits - s0.cache_hits
                stats.misses = s1.cache_misses - s0.cache_misses
                stats.read_bytes = s1.ssd_bytes_read - s0.ssd_bytes_read
                return out
            finally:
                self._idle_since = time.perf_counter()
                self._busy = False
