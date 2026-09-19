from __future__ import annotations

import json
import os
from dataclasses import dataclass
from dataclasses import replace as _replace
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter

import mlx.core as mx
from transformers import AutoTokenizer

from cachalot.cache.resident_store import (
    ResidentExpertStore,
    tensor_sizes_from_entry,
)
from cachalot.config import (
    DEFAULT_CONFIG,
    load_config,
    resolve_expert_budget,
    resolve_wired_limit,
)
from cachalot.io.resident_prefetch import (
    ResidentExpertPrefetcher,
)
from cachalot.metrics.routing_trace import (
    RoutingTracer,
)
from cachalot.model.block_compressed_index_source import (
    compressed_index_source_block_decode,
)
from cachalot.model.block_compressed_index_source_prefill import (
    compressed_index_source_block_prefill,
)
from cachalot.model.block_compressed_reuse import (
    compressed_reuse_block_decode,
)
from cachalot.model.block_compressed_reuse_prefill import (
    compressed_reuse_block_prefill,
)
from cachalot.model.block_compressed_source import (
    compressed_source_block_decode,
)
from cachalot.model.block_compressed_source_prefill import (
    compressed_source_block_prefill,
)
from cachalot.model.block_layer0 import (
    layer0_block_decode,
)
from cachalot.model.block_layer0_prefill import (
    layer0_block_prefill,
)
from cachalot.model.block_sliding_window import (
    sliding_window_block_decode,
)
from cachalot.model.block_sliding_window_prefill import (
    sliding_window_block_prefill,
)
from cachalot.model.compressor_mlx import (
    CompressorState,
)
from cachalot.model.engram_hash import (
    EngramHashConfig,
    EngramHashState,
    build_compressed_token_map,
)
from cachalot.model.engram_mlx import (
    engram_forward_decode,
)
from cachalot.model.engram_rows import (
    load_engram_rows,
)
from cachalot.model.indexer_mlx import (
    IndexerState,
)
from cachalot.model.model_boundary_mlx import (
    embed_token_decode,
    final_logits_decode,
    make_identity_pre_mix_decode,
)
from cachalot.model.norm_rope_mlx import (
    precompute_freqs,
)
from cachalot.model.prefix_cache import (
    PrefixCache,
    SequenceSnapshot,
)
from cachalot.model.resident_layer import (
    ResidentLayer,
)
from cachalot.model.resident_trunk import (
    ResidentTrunk,
    load_resident_trunk,
)
from cachalot.model.router_mlx import (
    RouterResult,
)
from cachalot.model.shared_attention import (
    SharedAttentionRuntime,
)
from cachalot.model.wo_a_dequant import (
    dequantize_wo_a,
)
from cachalot.storage.engram_index import (
    build_engram_table_layout,
)
from cachalot.storage.engram_reader import (
    EngramRowReader,
)
from cachalot.storage.index import (
    detect_expert_bank,
)
from cachalot.storage.tensor_index import (
    build_tensor_index,
)

DIM = 5120
HC_MULT = 4

N_LAYERS = 40

WINDOW_SIZE = 128
HEAD_DIM = 512
ROPE_HEAD_DIM = 64

N_HEADS = 64
N_GROUPS = 8
O_LORA_RANK = 1024

NORM_EPS = 1e-20
HC_EPS = 1e-6
HC_SINKHORN_ITERS = 20

INDEX_TOPK = 512

SLIDING_ROPE_THETA = 10000.0

COMPRESSED_ROPE_THETA = 160000.0
ORIGINAL_SEQ_LEN = 65536
ROPE_FACTOR = 16.0
BETA_FAST = 32.0
BETA_SLOW = 1.0

ENGRAM_LAYER_IDS = (1, 14)

ENGRAM_NUM_EMBEDDINGS = (
    384006168,
    384016682,
)

ENGRAM_MAX_NGRAM_SIZE = 4
ENGRAM_VOCAB_SIZE = 16000000
ENGRAM_N_HEADS = 8
ENGRAM_PAD_ID = 2

EXPECTED_COMPRESSED_VOCAB_SIZE = 99092

# DSpark's draft head reads the *input* of these layers, meaned over the
# hyper-connection copies, and concatenates them into one [DIM * 3] vector. The
# ids come from text_config.dspark_target_layer_ids in the checkpoint config.
# Nothing in the decode path uses this unless capture_main_hidden is set.
MTP_TARGET_LAYERS = (
    37,
    38,
    39,
)

SOURCE_LAYERS = {
    2: 2,
    8: 2,
    14: 2,
    20: 1,
}

INDEX_ONLY_SOURCE_LAYERS = {
    24,
    28,
    32,
    36,
}

COMPRESS_RATIO = {
    layer_id: (
        0
        if layer_id < 2
        else 2
        if layer_id < 20
        else 1
    )
    for layer_id in range(N_LAYERS)
}


@dataclass(frozen=True)
class DecodeResult:
    logits: mx.array
    hidden: mx.array
    routes: tuple[object, ...]
    position: int
    # Concatenated inputs of the DSpark target layers, [.., DIM * len(MTP_TARGET_LAYERS)],
    # captured only when capture_main_hidden is set. None on every ordinary forward.
    main_hidden: mx.array | None = None


class TextDecodeRuntime:
    """
    Stateful single-token decode runtime for the official
    DeepSeek-V4.1-Flash checkpoint.

    Text path only:
        embedding
        -> layers 0..39
        -> final HC collapse
        -> RMSNorm
        -> FP32 ParallelHead logits

    Vision and MTP are intentionally excluded.

    This first integrated runtime favors semantic correctness.
    Expert prefetch / scheduling optimization remains separate.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_seq_len: int = DEFAULT_CONFIG.max_seq_len,
        expert_cache_budget_bytes: int = DEFAULT_CONFIG.expert_cache_budget_bytes,
        mlx_cache_limit_bytes: int = DEFAULT_CONFIG.mlx_cache_limit_bytes,
        mlx_wired_limit_bytes: int = DEFAULT_CONFIG.mlx_wired_limit_bytes,
        io_workers: int = DEFAULT_CONFIG.io_workers,
        head_chunk_size: int = 4096,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.verbose = bool(verbose)

        # Background hotlist read, joined before the first prompt.
        self._hotlist_future = None
        self._hotlist_pool = None

        # Set to capture the DSpark draft head's input alongside every forward.
        # Off by default: when it is off the decode and prefill paths build no
        # extra graph at all, so an unused draft head costs nothing.
        self.capture_main_hidden = False

        if mlx_cache_limit_bytes < 0:
            raise ValueError(
                "mlx_cache_limit_bytes must be "
                "non-negative"
            )

        self.mlx_cache_limit_bytes = int(
            mlx_cache_limit_bytes
        )

        mx.set_cache_limit(
            self.mlx_cache_limit_bytes
        )

        if mlx_wired_limit_bytes < 0:
            raise ValueError(
                "mlx_wired_limit_bytes must be "
                "non-negative"
            )

        if expert_cache_budget_bytes < 0:
            raise ValueError(
                "expert_cache_budget_bytes must be "
                "non-negative (0 = auto)"
            )

        # Resolve auto (0) budgets against this machine's memory.
        # Environment overrides (CACHALOT_*) apply to auto-sizing here too:
        # an explicit constructor value wins, 0 (auto) defers to the
        # environment (CACHALOT_EXPERT_CACHE_BUDGET_GIB, CACHALOT_MLX_WIRED_LIMIT_GIB)
        # and only then to the machine-sized formula.
        env_cfg = load_config()
        resolved_cfg = _replace(
            env_cfg,
            expert_cache_budget_bytes=(
                int(expert_cache_budget_bytes)
                if expert_cache_budget_bytes > 0
                else env_cfg.expert_cache_budget_bytes
            ),
            mlx_cache_limit_bytes=self.mlx_cache_limit_bytes,
            mlx_wired_limit_bytes=(
                int(mlx_wired_limit_bytes)
                if mlx_wired_limit_bytes > 0
                else env_cfg.mlx_wired_limit_bytes
            ),
        )
        expert_cache_budget_bytes = resolve_expert_budget(resolved_cfg)
        self.expert_cache_budget_bytes = expert_cache_budget_bytes

        # Keep trunk + resident experts wired so the OS cannot compress
        # them under memory pressure (GPU access to a compressed page
        # costs a decompression fault; measured as 3-10x slower decode).
        self.mlx_wired_limit_bytes = resolve_wired_limit(
            resolved_cfg,
            expert_cache_budget_bytes + 128 * 18_800_640,
        )

        if self.mlx_wired_limit_bytes > 0:
            mx.set_wired_limit(
                self.mlx_wired_limit_bytes
            )

        if max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be positive"
            )

        # Ratio-2 compressed caches use the official
        # max_seq_len // ratio allocation.
        if max_seq_len % 2:
            raise ValueError(
                "max_seq_len must be divisible by 2"
            )

        self.max_seq_len = max_seq_len
        self.head_chunk_size = head_chunk_size

        if io_workers <= 0:
            raise ValueError(
                "io_workers must be positive"
            )

        self.io_workers = int(
            io_workers
        )

        if self.verbose:
            print(
                "Building checkpoint tensor index..."
            )

        self.tensor_index = build_tensor_index(
            self.model_path
        )

        if self.verbose:
            print(
                "Loading resident text trunk..."
            )

        self.trunk = load_resident_trunk(
            self.tensor_index
        )

        if self.verbose:
            print(
                "Building zero-copy ResidentLayer views..."
            )

        self.layers = self._build_layer_views(
            self.trunk
        )

        if self.verbose:
            print(
                "Building routed-expert index..."
            )

        # CACHALOT_EXPERT_BANK: serve routed experts from another directory
        # (e.g. an oMLX-converted 3-bit bank) while trunk, Engram and head
        # stay with the shipped checkpoint. storage.index detects the layout.
        bank = os.environ.get("CACHALOT_EXPERT_BANK")
        self.expert_bank_path = Path(bank) if bank else self.model_path
        self.expert_format, self.expert_index = detect_expert_bank(
            self.expert_bank_path
        )
        if not self.expert_index:
            raise FileNotFoundError(
                f"no routed experts found under {self.expert_bank_path}"
            )
        if self.verbose or bank:
            n_bytes = sum(t.size for t in next(iter(self.expert_index.values())).tensors)
            print(
                f"expert bank: {self.expert_bank_path} ({self.expert_format.kind}, "
                f"{self.expert_format.bits}-bit, {n_bytes / 2**20:.2f} MiB/expert, "
                f"{len(self.expert_index)} experts)",
                flush=True,
            )

        if self.verbose:
            print(
                "Allocating resident expert slots..."
            )

        self.expert_store = ResidentExpertStore(
            budget_bytes=(
                expert_cache_budget_bytes
            ),
            tensor_sizes=tensor_sizes_from_entry(
                next(iter(self.expert_index.values()))
            ),
            load_workers=self.io_workers,
            verbose=self.verbose,
        )

        self.expert_store.format = self.expert_format

        self._preload_hotlist()

        self.expert_prefetcher = (
            ResidentExpertPrefetcher(
                self.expert_store,
                workers=self.io_workers,
            )
        )

        if self.verbose:
            print(
                "Loading tokenizer..."
            )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        )

        if self.verbose:
            print(
                "Building official compressed Engram "
                "token map..."
            )

        (
            token_map,
            compressed_vocab_size,
        ) = build_compressed_token_map(
            self.tokenizer
        )

        if (
            compressed_vocab_size
            != EXPECTED_COMPRESSED_VOCAB_SIZE
        ):
            raise ValueError(
                "Unexpected compressed Engram vocab "
                f"size: {compressed_vocab_size}; "
                "expected "
                f"{EXPECTED_COMPRESSED_VOCAB_SIZE}"
            )

        self.engram_hash = EngramHashState(
            EngramHashConfig(
                layer_ids=ENGRAM_LAYER_IDS,
                num_embeddings=(
                    ENGRAM_NUM_EMBEDDINGS
                ),
                max_ngram_size=(
                    ENGRAM_MAX_NGRAM_SIZE
                ),
                vocab_size=ENGRAM_VOCAB_SIZE,
                n_heads=ENGRAM_N_HEADS,
                compressed_vocab_size=(
                    compressed_vocab_size
                ),
                pad_token_id=ENGRAM_PAD_ID,
            ),
            token_map,
        )

        self.engram_reader = EngramRowReader()
        self._engram_pool = None
        self._engram_prefetch: dict = {}

        # These two table locations were established from the
        # actual checkpoint inventory:
        #
        #   layer 1  -> shard 47
        #   layer 14 -> shard 48
        #
        self.engram_layouts = {
            1: build_engram_table_layout(
                self.model_path
                / "model-00047-of-00048.safetensors",
                1,
            ),
            14: build_engram_table_layout(
                self.model_path
                / "model-00048-of-00048.safetensors",
                14,
            ),
        }

        if self.verbose:
            print(
                "Precomputing sliding-window RoPE..."
            )

        # Official semantics for compress_ratio == 0:
        #
        #   original_seq_len = 0
        #   rope_theta = 10000
        #
        self.sliding_rope_cos, (
            self.sliding_rope_sin
        ) = precompute_freqs(
            dim=ROPE_HEAD_DIM,
            seqlen=self.max_seq_len,
            original_seq_len=0,
            base=SLIDING_ROPE_THETA,
            factor=ROPE_FACTOR,
            beta_fast=BETA_FAST,
            beta_slow=BETA_SLOW,
        )

        if self.verbose:
            print(
                "Precomputing compressed-attention RoPE..."
            )

        self.compressed_rope_cos, (
            self.compressed_rope_sin
        ) = precompute_freqs(
            dim=ROPE_HEAD_DIM,
            seqlen=self.max_seq_len,
            original_seq_len=ORIGINAL_SEQ_LEN,
            base=COMPRESSED_ROPE_THETA,
            factor=ROPE_FACTOR,
            beta_fast=BETA_FAST,
            beta_slow=BETA_SLOW,
        )

        mx.eval(
            self.sliding_rope_cos,
            self.sliding_rope_sin,
            self.compressed_rope_cos,
            self.compressed_rope_sin,
        )

        # wo_a is dequantized lazily on first use per layer.
        # This avoids one large initialization-time burst and
        # preserves the validated BF16 representation.
        self._wo_a: dict[
            int,
            mx.array,
        ] = {}

        self.shared_attn = (
            SharedAttentionRuntime()
        )

        self.windows: dict[
            int,
            mx.array,
        ] = {}

        self.compressed_caches: dict[
            int,
            mx.array,
        ] = {}

        self.compressor_states: dict[
            int,
            CompressorState,
        ] = {}

        self.indexer_states: dict[
            int,
            IndexerState,
        ] = {}

        # wo_a is dequantized once here (~4 s) instead of lazily during the
        # first prompt, so prefill timing is stable from the first request.
        if self.verbose:
            print("Dequantizing attention wo_a for all layers...")

        for layer_id in range(N_LAYERS):
            self._get_wo_a(layer_id)

        # Router weights by layer for one-layer-early routing prediction
        # in decode (moe_layer_metal.PREDICT_TOPK).
        self.expert_store.decode_gates = {
            layer_id: (self._t(layer_id, "ffn.gate.weight"), self._t(layer_id, "ffn.gate.bias"))
            for layer_id in range(N_LAYERS)
        }

        # Idle heartbeat: keep the Metal residency set wired while no
        # forward pass runs (see RuntimeConfig.idle_heartbeat_seconds).
        self._gpu_busy = False
        self._gpu_idle_since = perf_counter()
        # Serialises the heartbeat's eval with forward passes: MLX evaluations
        # from two threads at once (heartbeat vs. a typing-time prefill that
        # starts while the probe is in flight) can stall the Metal queue.
        self._gpu_lock = Lock()
        self._heartbeat_stop = Event()
        self._heartbeat_thread: Thread | None = None
        self.heartbeats = 0
        period = float(resolved_cfg.idle_heartbeat_seconds)
        if period > 0:
            self._heartbeat_thread = Thread(
                target=self._heartbeat_loop, args=(period,), name="idle-heartbeat", daemon=True
            )
            self._heartbeat_thread.start()

        self.position = 0

        # Token ids consumed by the current sequence (prompt + decode inputs).
        self.tokens: list[int] = []

        self.tracer: RoutingTracer | None = None

        # Snapshots of completed prompts/replies for multi-turn prefix reuse.
        self.prefix_cache = PrefixCache(
            max_entries=resolved_cfg.prefix_cache_entries
        )

        self.reset()

    @staticmethod
    def _build_layer_views(
        trunk: ResidentTrunk,
    ) -> dict[int, ResidentLayer]:
        layers = {}

        for layer_id in range(N_LAYERS):
            prefix = f"layers.{layer_id}."

            tensors = {
                name: tensor
                for name, tensor
                in trunk.tensors.items()
                if name.startswith(prefix)
            }

            if not tensors:
                raise KeyError(
                    "No resident tensors found for "
                    f"layer {layer_id}"
                )

            layers[layer_id] = (
                ResidentLayer(
                    layer_id=layer_id,
                    tensors=tensors,
                )
            )

        return layers

    def _global(
        self,
        name: str,
    ) -> mx.array:
        try:
            return self.trunk.get(name).data
        except KeyError as exc:
            raise KeyError(
                f"Missing global tensor {name!r}"
            ) from exc

    def _t(
        self,
        layer_id: int,
        suffix: str,
    ) -> mx.array:
        name = (
            f"layers.{layer_id}.{suffix}"
        )

        try:
            return (
                self.layers[
                    layer_id
                ].tensors[name].data
            )
        except KeyError as exc:
            raise KeyError(
                f"Missing tensor {name!r}"
            ) from exc

    def _get_wo_a(
        self,
        layer_id: int,
    ) -> mx.array:
        value = self._wo_a.get(
            layer_id
        )

        if value is None:
            value = dequantize_wo_a(
                self._t(
                    layer_id,
                    "attn.wo_a.weight",
                ),
                self._t(
                    layer_id,
                    "attn.wo_a.scale",
                ),
            )

            mx.eval(value)

            self._wo_a[
                layer_id
            ] = value

        return value

    def _common_block_kwargs(
        self,
        layer_id: int,
        *,
        compressed: bool,
    ) -> dict:
        if compressed:
            rope_cos = (
                self.compressed_rope_cos
            )
            rope_sin = (
                self.compressed_rope_sin
            )
        else:
            rope_cos = (
                self.sliding_rope_cos
            )
            rope_sin = (
                self.sliding_rope_sin
            )

        return {
            "hc_attn_fn": self._t(
                layer_id,
                "hc_attn_fn",
            ),
            "hc_attn_scale": self._t(
                layer_id,
                "hc_attn_scale",
            ),
            "hc_attn_base": self._t(
                layer_id,
                "hc_attn_base",
            ),
            "hc_ffn_fn": self._t(
                layer_id,
                "hc_ffn_fn",
            ),
            "hc_ffn_scale": self._t(
                layer_id,
                "hc_ffn_scale",
            ),
            "hc_ffn_base": self._t(
                layer_id,
                "hc_ffn_base",
            ),
            "attn_norm_weight": self._t(
                layer_id,
                "attn_norm.weight",
            ),
            "ffn_norm_weight": self._t(
                layer_id,
                "ffn_norm.weight",
            ),
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "attn_sink": self._t(
                layer_id,
                "attn.attn_sink",
            ),
            "q_norm_weight": self._t(
                layer_id,
                "attn.q_norm.weight",
            ),
            "kv_norm_weight": self._t(
                layer_id,
                "attn.kv_norm.weight",
            ),
            "wq_a": self._t(
                layer_id,
                "attn.wq_a.weight",
            ),
            "wq_a_scales": self._t(
                layer_id,
                "attn.wq_a.scale",
            ),
            "wq_b": self._t(
                layer_id,
                "attn.wq_b.weight",
            ),
            "wq_b_scales": self._t(
                layer_id,
                "attn.wq_b.scale",
            ),
            "wkv": self._t(
                layer_id,
                "attn.wkv.weight",
            ),
            "wkv_scales": self._t(
                layer_id,
                "attn.wkv.scale",
            ),
            "wo_a_bf16": self._get_wo_a(
                layer_id
            ),
            "wo_b": self._t(
                layer_id,
                "attn.wo_b.weight",
            ),
            "wo_b_scales": self._t(
                layer_id,
                "attn.wo_b.scale",
            ),
            "gate_weight": self._t(
                layer_id,
                "ffn.gate.weight",
            ),
            "gate_bias": self._t(
                layer_id,
                "ffn.gate.bias",
            ),
            "expert_index": (
                self.expert_index
            ),
            "expert_store": (
                self.expert_store
            ),
            "shared_w1": self._t(
                layer_id,
                (
                    "ffn.shared_experts."
                    "w1.weight"
                ),
            ),
            "shared_w1_scales": self._t(
                layer_id,
                (
                    "ffn.shared_experts."
                    "w1.scale"
                ),
            ),
            "shared_w2": self._t(
                layer_id,
                (
                    "ffn.shared_experts."
                    "w2.weight"
                ),
            ),
            "shared_w2_scales": self._t(
                layer_id,
                (
                    "ffn.shared_experts."
                    "w2.scale"
                ),
            ),
            "shared_w3": self._t(
                layer_id,
                (
                    "ffn.shared_experts."
                    "w3.weight"
                ),
            ),
            "shared_w3_scales": self._t(
                layer_id,
                (
                    "ffn.shared_experts."
                    "w3.scale"
                ),
            ),
            "hc_mult": HC_MULT,
            "hc_sinkhorn_iters": (
                HC_SINKHORN_ITERS
            ),
            "hc_eps": HC_EPS,
            "norm_eps": NORM_EPS,
            "window_size": WINDOW_SIZE,
            "n_heads": N_HEADS,
            "head_dim": HEAD_DIM,
            "rope_head_dim": (
                ROPE_HEAD_DIM
            ),
            "n_groups": N_GROUPS,
            "o_lora_rank": (
                O_LORA_RANK
            ),
        }

    def _preload_hotlist(self) -> None:
        """
        Admit a recorded hot set before the first prompt, if one is configured.

        CACHALOT_HOTLIST names a JSON file written by benchmarks/build_hotlist.py:
        {"experts": [[layer, expert], ...]} in descending order of how often
        past sessions routed to them. CACHALOT_HOTLIST_GIB caps what is read;
        the default of 8 GiB is 5.6 % of the 2-bit bank and covered about 30 %
        of an unseen prompt's requests in benchmarks/hotlist_coverage.py.

        Off unless the variable is set. A hot set that does not match the bank
        being served is skipped with a warning rather than failing the run:
        expert ids are bank-independent, but a stale file naming layers or
        experts this bank does not have is a configuration mistake, not a
        reason to refuse to start.
        """
        path = os.environ.get("CACHALOT_HOTLIST")
        if not path:
            return

        try:
            payload = json.loads(Path(path).read_text())
            wanted = [
                (int(layer), int(expert))
                for layer, expert in payload["experts"]
            ]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"hotlist: ignoring {path}: {exc}", flush=True)
            return

        gib = float(os.environ.get("CACHALOT_HOTLIST_GIB", "8"))
        expert_bytes = sum(
            t.size for t in next(iter(self.expert_index.values())).tensors
        )
        max_experts = max(0, int(gib * (1024**3)) // max(1, expert_bytes))

        entries = []
        missing = 0
        for key in wanted[:max_experts]:
            entry = self.expert_index.get(key)
            if entry is None:
                missing += 1
                continue
            entries.append(entry)

        if not entries:
            print(f"hotlist: {path} names no expert this bank holds", flush=True)
            return

        # Read it on a thread and join before the first prompt. Measured with
        # the read in the foreground, a hotlist was a net loss: it bought
        # 4.0 points of first-turn hit rate and 0.7 s of prefill, and spent
        # 1.3 s of startup doing it. The same bytes read during prefill hide
        # under prefill's own compute; read at startup they hide under nothing
        # unless they are overlapped with the rest of becoming ready, which is
        # RoPE precompute, the Engram reader and the prefetcher.
        #
        # Nothing here touches MLX, only positional reads into slot buffers on
        # the store's own load pool, so the thread-local-stream pitfall does
        # not apply.
        note = f", {missing} not in this bank" if missing else ""
        print(
            f"hotlist: reading {len(entries)} experts in the background{note}",
            flush=True,
        )
        from concurrent.futures import ThreadPoolExecutor

        self._hotlist_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hotlist"
        )
        self._hotlist_future = self._hotlist_pool.submit(
            self.expert_store.preload, entries
        )

    def _await_hotlist(self) -> None:
        """Join the background hotlist read. Cheap and idempotent after the first call."""
        future = self._hotlist_future
        if future is None:
            return
        self._hotlist_future = None
        admitted = future.result()
        self._hotlist_pool.shutdown(wait=True)
        self._hotlist_pool = None
        print(
            f"hotlist: {admitted} experts preloaded "
            f"({self.expert_store.preload_bytes / 2**30:.1f} GiB in "
            f"{self.expert_store.preload_seconds:.1f} s of reading)",
            flush=True,
        )

    def warmup(self) -> None:
        """
        Compile every prefill and decode kernel once (a two-token prefill and
        one decode step, then reset). Without this the first decoded token of
        a process costs ~1 s of Metal kernel compilation instead of ~0.35 s.
        """
        try:
            ids = list(self.tokenizer.encode("Hello there"))[:2] or [1]
        except Exception:
            ids = [1]
        self.reset()
        result = self.prefill_tokens(ids)
        mx.eval(result.logits)
        result = self.decode_token(int(result.logits.argmax().item()))
        mx.eval(result.logits)
        self.reset()

    def reset(self) -> None:
        """
        Begin a new independent text sequence.

        Resident checkpoint tensors and resident expert LRU
        remain loaded; sequence-dependent attention/Engram
        state is cleared.
        """
        self.position = 0
        self.tokens = []

        self.engram_hash.history.clear()

        self.shared_attn.reset()

        self.windows = {
            layer_id: mx.zeros(
                (
                    WINDOW_SIZE,
                    HEAD_DIM,
                ),
                dtype=mx.bfloat16,
            )
            for layer_id in range(
                N_LAYERS
            )
        }

        self.compressed_caches = {
            2: mx.zeros(
                (
                    self.max_seq_len // 2,
                    HEAD_DIM,
                ),
                dtype=mx.bfloat16,
            ),
            8: mx.zeros(
                (
                    self.max_seq_len // 2,
                    HEAD_DIM,
                ),
                dtype=mx.bfloat16,
            ),
            14: mx.zeros(
                (
                    self.max_seq_len // 2,
                    HEAD_DIM,
                ),
                dtype=mx.bfloat16,
            ),
            20: mx.zeros(
                (
                    self.max_seq_len,
                    HEAD_DIM,
                ),
                dtype=mx.bfloat16,
            ),
        }

        self.compressor_states = {
            layer_id: (
                CompressorState.create(
                    max_batch_size=1,
                    compress_ratio=2,
                    head_dim=HEAD_DIM,
                )
            )
            for layer_id in (
                2,
                8,
                14,
            )
        }

        self.indexer_states = {
            2: IndexerState.create(
                max_seq_len=(
                    self.max_seq_len
                ),
                compress_ratio=2,
            ),
            8: IndexerState.create(
                max_seq_len=(
                    self.max_seq_len
                ),
                compress_ratio=2,
            ),
            14: IndexerState.create(
                max_seq_len=(
                    self.max_seq_len
                ),
                compress_ratio=2,
            ),
            20: IndexerState.create(
                max_seq_len=(
                    self.max_seq_len
                ),
                compress_ratio=1,
            ),
        }

    # ------------------------------------------------------------
    # Sequence snapshots (prefix cache support)
    # ------------------------------------------------------------

    def snapshot(
        self,
        logits: mx.array | None = None,
    ) -> SequenceSnapshot:
        """
        Copy the complete sequence-dependent state.

        Window/compressed caches are replaced functionally by the block
        implementations, so holding references is safe. Compressor and
        indexer states are mutated in place and must be copied.
        """
        def copy(a: mx.array) -> mx.array:
            out = mx.array(a)
            return out

        snap = SequenceSnapshot(
            tokens=tuple(self.tokens),
            position=self.position,
            logits=logits,
            windows=dict(self.windows),
            compressed_caches=dict(self.compressed_caches),
            compressor_kv={
                k: copy(v.kv_state)
                for k, v in self.compressor_states.items()
            },
            compressor_score={
                k: copy(v.score_state)
                for k, v in self.compressor_states.items()
            },
            indexer_k={
                k: copy(v.k_cache)
                for k, v in self.indexer_states.items()
            },
            engram_history=list(self.engram_hash.history),
            shared_compress_kv=self.shared_attn.compress_kv,
            # index_k aliases an in-place-mutated IndexerState cache; remember
            # which layer published it and re-point at the restored copy.
            shared_index_k_layer=next(
                (
                    k
                    for k, v in self.indexer_states.items()
                    if v.k_cache is self.shared_attn.index_k
                ),
                None,
            ),
            shared_topk_idxs=self.shared_attn.topk_idxs,
            shared_candidates=self.shared_attn.candidates,
        )

        mx.eval(
            *snap.compressor_kv.values(),
            *snap.compressor_score.values(),
            *snap.indexer_k.values(),
        )

        return snap

    def restore(
        self,
        snap: SequenceSnapshot,
    ) -> None:
        """Rewind the sequence state to a snapshot taken by snapshot()."""
        self.position = snap.position
        self.tokens = list(snap.tokens)

        self.windows = dict(snap.windows)
        self.compressed_caches = dict(snap.compressed_caches)

        for k, state in self.compressor_states.items():
            state.kv_state = mx.array(snap.compressor_kv[k])
            state.score_state = mx.array(snap.compressor_score[k])

        for k, state in self.indexer_states.items():
            state.k_cache = mx.array(snap.indexer_k[k])

        self.engram_hash.history = list(snap.engram_history)

        self.shared_attn.compress_kv = snap.shared_compress_kv
        self.shared_attn.index_k = (
            self.indexer_states[snap.shared_index_k_layer].k_cache
            if snap.shared_index_k_layer is not None
            else None
        )
        self.shared_attn.topk_idxs = snap.shared_topk_idxs
        self.shared_attn.candidates = snap.shared_candidates

    def _apply_engram(
        self,
        x: mx.array,
        layer_id: int,
        hash_rows,
    ) -> mx.array:
        layer_hash_index = (
            ENGRAM_LAYER_IDS.index(
                layer_id
            )
        )

        row_ids = hash_rows[
            layer_hash_index
        ]

        embed_rows = load_engram_rows(
            self.engram_reader,
            self.engram_layouts[
                layer_id
            ],
            row_ids,
        )

        return engram_forward_decode(
            x,
            embed_rows,
            self.layers[layer_id],
        )

    def _decode_layer0(
        self,
        x: mx.array,
        pre_mix: mx.array,
        start_pos: int,
    ):
        return layer0_block_decode(
            x,
            start_pos=start_pos,
            pre_mix=pre_mix,
            window_cache=(
                self.windows[0]
            ),
            **self._common_block_kwargs(
                0,
                compressed=False,
            ),
        )

    def _decode_sliding(
        self,
        layer_id: int,
        x: mx.array,
        pre_mix: mx.array,
        start_pos: int,
    ):
        return sliding_window_block_decode(
            layer_id,
            x,
            start_pos=start_pos,
            pre_mix=pre_mix,
            window_cache=(
                self.windows[
                    layer_id
                ]
            ),
            **self._common_block_kwargs(
                layer_id,
                compressed=False,
            ),
        )

    def _decode_source(
        self,
        layer_id: int,
        x: mx.array,
        pre_mix: mx.array,
        start_pos: int,
    ):
        ratio = SOURCE_LAYERS[
            layer_id
        ]

        common = (
            self._common_block_kwargs(
                layer_id,
                compressed=True,
            )
        )

        if ratio == 1:
            compressor_state = None
            compressor_wgate = None
        else:
            compressor_state = (
                self.compressor_states[
                    layer_id
                ]
            )
            compressor_wgate = self._t(
                layer_id,
                (
                    "attn.compressor."
                    "wgate.weight"
                ),
            )

        return compressed_source_block_decode(
            layer_id,
            x,
            start_pos=start_pos,
            pre_mix=pre_mix,
            compress_ratio=ratio,
            window_cache=(
                self.windows[
                    layer_id
                ]
            ),
            compressed_cache=(
                self.compressed_caches[
                    layer_id
                ]
            ),
            compressor_state=(
                compressor_state
            ),
            indexer_state=(
                self.indexer_states[
                    layer_id
                ]
            ),
            shared_attn=(
                self.shared_attn
            ),
            compressor_norm_weight=(
                self._t(
                    layer_id,
                    (
                        "attn.compressor."
                        "norm.weight"
                    ),
                )
            ),
            compressor_wkv_weight=(
                self._t(
                    layer_id,
                    (
                        "attn.compressor."
                        "wkv.weight"
                    ),
                )
            ),
            compressor_wgate_weight=(
                compressor_wgate
            ),
            indexer_weights_proj_weight=(
                self._t(
                    layer_id,
                    (
                        "attn.indexer."
                        "weights_proj.weight"
                    ),
                )
            ),
            indexer_wq_b_weight=(
                self._t(
                    layer_id,
                    (
                        "attn.indexer."
                        "wq_b.weight"
                    ),
                )
            ),
            indexer_wq_b_scales=(
                self._t(
                    layer_id,
                    (
                        "attn.indexer."
                        "wq_b.scale"
                    ),
                )
            ),
            indexer_wk_weight=(
                self._t(
                    layer_id,
                    (
                        "attn.indexer."
                        "wk.weight"
                    ),
                )
            ),
            indexer_k_norm_weight=(
                self._t(
                    layer_id,
                    (
                        "attn.indexer."
                        "k_norm.weight"
                    ),
                )
            ),
            index_topk=INDEX_TOPK,
            **common,
        )

    def _decode_reuse(
        self,
        layer_id: int,
        x: mx.array,
        pre_mix: mx.array,
        start_pos: int,
    ):
        return compressed_reuse_block_decode(
            layer_id,
            x,
            start_pos=start_pos,
            pre_mix=pre_mix,
            compress_ratio=(
                COMPRESS_RATIO[
                    layer_id
                ]
            ),
            window_cache=(
                self.windows[
                    layer_id
                ]
            ),
            shared_attn=(
                self.shared_attn
            ),
            **self._common_block_kwargs(
                layer_id,
                compressed=True,
            ),
        )

    def _decode_index_source(
        self,
        layer_id: int,
        x: mx.array,
        pre_mix: mx.array,
        start_pos: int,
    ):
        return (
            compressed_index_source_block_decode(
                layer_id,
                x,
                start_pos=start_pos,
                pre_mix=pre_mix,
                compress_ratio=1,
                window_cache=(
                    self.windows[
                        layer_id
                    ]
                ),
                shared_attn=(
                    self.shared_attn
                ),
                indexer_weights_proj_weight=(
                    self._t(
                        layer_id,
                        (
                            "attn.indexer."
                            "weights_proj.weight"
                        ),
                    )
                ),
                indexer_wq_b_weight=(
                    self._t(
                        layer_id,
                        (
                            "attn.indexer."
                            "wq_b.weight"
                        ),
                    )
                ),
                indexer_wq_b_scales=(
                    self._t(
                        layer_id,
                        (
                            "attn.indexer."
                            "wq_b.scale"
                        ),
                    )
                ),
                index_topk=INDEX_TOPK,
                **self._common_block_kwargs(
                    layer_id,
                    compressed=True,
                ),
            )
        )

    @staticmethod
    def _last_token_route(
        route: RouterResult,
    ) -> RouterResult:
        """
        Convert a batched prefill RouterResult back to the
        single-token RouterResult contract exposed by decode_token().
        """
        if route.indices.ndim != 2:
            raise ValueError(
                "Expected batched route indices "
                f"[tokens, topk], got {route.indices.shape}"
            )

        if route.weights.ndim != 2:
            raise ValueError(
                "Expected batched route weights "
                f"[tokens, topk], got {route.weights.shape}"
            )

        if route.scores.ndim != 2:
            raise ValueError(
                "Expected batched route scores "
                f"[tokens, experts], got {route.scores.shape}"
            )

        return RouterResult(
            indices=route.indices[-1],
            weights=route.weights[-1],
            scores=route.scores[-1],
        )

    def set_tracer(
        self,
        tracer: RoutingTracer | None,
    ) -> None:
        """Install (or remove) a routing tracer for offline analysis."""
        self.tracer = tracer

    def _trace_prefill_route(
        self,
        layer_id: int,
        start_pos: int,
        route: RouterResult,
    ) -> RouterResult:
        if self.tracer is not None:
            self.tracer.record(
                "prefill",
                layer_id,
                start_pos,
                route.indices,
            )

        return self._last_token_route(route)

    def _trace_decode_route(
        self,
        layer_id: int,
        start_pos: int,
        route: RouterResult,
    ) -> RouterResult:
        if self.tracer is not None:
            self.tracer.record(
                "decode",
                layer_id,
                start_pos,
                route.indices,
            )

        return route

    def _engram_row_ids(self, layer_id: int, hash_rows_by_token):
        import numpy as np

        layer_hash_index = ENGRAM_LAYER_IDS.index(layer_id)
        ids = np.stack(
            [np.asarray(rows[layer_hash_index], dtype=np.int64) for rows in hash_rows_by_token]
        )  # [tokens, n_hash_cols]
        unique, inverse = np.unique(ids.reshape(-1), return_inverse=True)
        return ids, unique, inverse

    def _start_engram_prefetch(self, hash_rows_by_token) -> None:
        """Read both Engram layers' rows for this prompt chunk in the background."""
        import os

        self._engram_prefetch = {}
        if os.environ.get("CACHALOT_PREFILL_BATCHED_ENGRAM", "1") == "0" or not hash_rows_by_token:
            return
        if os.environ.get("CACHALOT_ENGRAM_PREFETCH", "1") == "0":
            return
        if self._engram_pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._engram_pool = ThreadPoolExecutor(len(ENGRAM_LAYER_IDS), thread_name_prefix="engram-prefetch")
        for layer_id in ENGRAM_LAYER_IDS:
            ids, unique, _ = self._engram_row_ids(layer_id, hash_rows_by_token)
            future = self._engram_pool.submit(self.engram_reader.read_rows, self.engram_layouts[layer_id], unique)
            self._engram_prefetch[layer_id] = (ids, future)

    def _prefill_apply_engram(
        self,
        x: mx.array,
        layer_id: int,
        hash_rows_by_token: tuple[
            object,
            ...,
        ],
    ) -> mx.array:
        """
        Apply Engram to a prompt chunk.

        Batched (default): read every requested table row once, gather
        per token, and run the gating math with a leading token axis.
        Sequential fallback reproduces the single-token path exactly.
        """
        import os

        import numpy as np

        if os.environ.get("CACHALOT_PREFILL_BATCHED_ENGRAM", "1") != "0":
            from cachalot.model.engram_mlx import (
                engram_forward_batched,
            )
            from cachalot.model.engram_rows import (
                load_engram_rows,
            )

            ids, unique, inverse = self._engram_row_ids(layer_id, hash_rows_by_token)
            prefetched = self._engram_prefetch.pop(layer_id, None)
            if prefetched is not None and np.array_equal(prefetched[0], ids):
                from cachalot.model.engram_rows import engram_rows_to_array

                values = engram_rows_to_array(prefetched[1].result(), self.engram_layouts[layer_id])
            else:
                values = load_engram_rows(
                    self.engram_reader,
                    self.engram_layouts[layer_id],
                    unique,
                )  # [unique, head_dim] fp32
            gathered = values[mx.array(inverse.astype(np.int32))].reshape(
                ids.shape[0],
                ids.shape[1],
                -1,
            )
            return engram_forward_batched(
                x,
                gathered,
                self.layers[layer_id],
            )

        outputs = []

        for token_offset in range(
            x.shape[0]
        ):
            outputs.append(
                self._apply_engram(
                    x[token_offset],
                    layer_id,
                    hash_rows_by_token[
                        token_offset
                    ],
                )
            )

        return mx.stack(
            outputs,
            axis=0,
        )

    def _heartbeat_loop(self, period: float) -> None:
        probe = mx.zeros((1,))
        while not self._heartbeat_stop.wait(period):
            if self._gpu_busy or perf_counter() - self._gpu_idle_since < period:
                continue
            if not self._gpu_lock.acquire(blocking=False):
                continue
            try:
                if not self._gpu_busy:
                    mx.eval(probe + 1)
                    self.heartbeats += 1
            finally:
                self._gpu_lock.release()

    def prefill_tokens(
        self,
        token_ids,
    ) -> DecodeResult:
        """Layer-major prompt prefill (see _prefill_tokens_impl); marks the GPU busy for the idle heartbeat."""
        self._await_hotlist()
        with self._gpu_lock:
            self._gpu_busy = True
            try:
                return self._prefill_tokens_impl(token_ids)
            finally:
                self._gpu_idle_since = perf_counter()
                self._gpu_busy = False

    def decode_token(
        self,
        token_id: int,
    ) -> DecodeResult:
        """Decode one token (see _decode_token_impl); marks the GPU busy for the idle heartbeat."""
        self._await_hotlist()
        with self._gpu_lock:
            self._gpu_busy = True
            try:
                return self._decode_token_impl(token_id)
            finally:
                self._gpu_idle_since = perf_counter()
                self._gpu_busy = False

    def _prefill_tokens_impl(
        self,
        token_ids,
    ) -> DecodeResult:
        """
        Layer-major prompt prefill.

        The prompt is processed a layer at a time so routed MoE work
        can be grouped expert-major across prompt tokens while
        preserving the exact token-sequential state evolution required
        by attention, compressor/indexer state, Engram, and shared
        attention publications.

        Persistent sequence state after this call is equivalent to
        calling decode_token() for the same tokens in order.

        The returned DecodeResult corresponds to the LAST prompt token,
        whose logits predict the first generated token.
        """
        token_ids = tuple(
            int(token_id)
            for token_id in token_ids
        )

        n_tokens = len(
            token_ids
        )

        if n_tokens == 0:
            raise ValueError(
                "prefill_tokens requires at least one token"
            )

        start_pos = self.position

        if (
            start_pos
            + n_tokens
            > self.max_seq_len
        ):
            raise IndexError(
                "Prefill exceeds "
                f"max_seq_len={self.max_seq_len}: "
                f"start_pos={start_pos}, "
                f"tokens={n_tokens}"
            )

        if self.verbose:
            print(
                "[prefill] "
                f"start_pos={start_pos} "
                f"tokens={n_tokens}"
            )

        # --------------------------------------------------------
        # Engram hash history must advance in token order exactly
        # as decode_token() would.
        # --------------------------------------------------------

        hash_rows_by_token = []

        for token_id in token_ids:
            hash_rows_by_token.append(
                self.engram_hash.push(
                    token_id
                )
            )

        hash_rows_by_token = tuple(
            hash_rows_by_token
        )

        # Engram row ids depend on the token ids only, so both Engram
        # layers' table rows can be read in the background from here on,
        # instead of on the critical path at layers 1 and 14 (12k random
        # 5 KB reads that otherwise queue behind the expert loads).
        self._start_engram_prefetch(hash_rows_by_token)

        # --------------------------------------------------------
        # Exact input-boundary semantics.
        #
        # Reuse embed_token_decode() rather than introducing a new
        # batched embedding numerical path.
        # --------------------------------------------------------

        x = mx.stack(
            [
                embed_token_decode(
                    token_id,
                    self._global(
                        "embed.weight"
                    ),
                    hc_mult=HC_MULT,
                )
                for token_id in token_ids
            ],
            axis=0,
        )

        identity_pre_mix = (
            make_identity_pre_mix_decode(
                hc_mult=HC_MULT
            )
        )

        pre_mix = mx.stack(
            [
                identity_pre_mix
                for _ in range(
                    n_tokens
                )
            ],
            axis=0,
        )

        routes = []

        # --------------------------------------------------------
        # Layer 0
        # --------------------------------------------------------

        if self.verbose:
            print(
                "[prefill] layer 0"
            )

        (
            x,
            pre_mix,
            self.windows[0],
            route,
        ) = layer0_block_prefill(
            x,
            start_pos=start_pos,
            pre_mix=pre_mix,
            window_cache=(
                self.windows[0]
            ),
            expert_prefetcher=(
                self.expert_prefetcher
            ),
            **self._common_block_kwargs(
                0,
                compressed=False,
            ),
        )

        routes.append(
            self._trace_prefill_route(
                0,
                start_pos,
                route,
            )
        )

        # --------------------------------------------------------
        # Layer 1: Engram -> sliding block.
        # --------------------------------------------------------

        if self.verbose:
            print(
                "[prefill] layer 1 Engram"
            )

        x = self._prefill_apply_engram(
            x,
            1,
            hash_rows_by_token,
        )

        if self.verbose:
            print(
                "[prefill] layer 1"
            )

        (
            x,
            pre_mix,
            self.windows[1],
            route,
        ) = sliding_window_block_prefill(
            1,
            x,
            start_pos=start_pos,
            pre_mix=pre_mix,
            window_cache=(
                self.windows[1]
            ),
            expert_prefetcher=(
                self.expert_prefetcher
            ),
            **self._common_block_kwargs(
                1,
                compressed=False,
            ),
        )

        routes.append(
            self._trace_prefill_route(
                1,
                start_pos,
                route,
            )
        )

        # --------------------------------------------------------
        # Layers 2..39.
        #
        # Position-dependent publications:
        #
        #   source layer:
        #       top-k snapshots are replayed by subsequent reuse
        #       layers.
        #
        #   layer 20:
        #       additionally publishes persistent candidate snapshots
        #       consumed by 24/28/32/36.
        # --------------------------------------------------------

        shared_topk_by_token = None
        layer20_candidates_by_token = None

        main_hiddens = [] if self.capture_main_hidden else None

        for layer_id in range(
            2,
            N_LAYERS,
        ):
            if self.verbose:
                print(
                    f"[prefill] layer {layer_id}"
                )

            if (
                main_hiddens is not None
                and layer_id
                in MTP_TARGET_LAYERS
            ):
                main_hiddens.append(
                    mx.mean(
                        x,
                        axis=1,
                    )
                )

            if layer_id == 14:
                if self.verbose:
                    print(
                        "[prefill] layer 14 Engram"
                    )

                x = self._prefill_apply_engram(
                    x,
                    14,
                    hash_rows_by_token,
                )

            if layer_id in SOURCE_LAYERS:
                ratio = SOURCE_LAYERS[
                    layer_id
                ]

                common = (
                    self._common_block_kwargs(
                        layer_id,
                        compressed=True,
                    )
                )

                if ratio == 1:
                    compressor_state = None
                    compressor_wgate = None
                else:
                    compressor_state = (
                        self.compressor_states[
                            layer_id
                        ]
                    )

                    compressor_wgate = self._t(
                        layer_id,
                        (
                            "attn.compressor."
                            "wgate.weight"
                        ),
                    )

                (
                    x,
                    pre_mix,
                    self.windows[
                        layer_id
                    ],
                    self.compressed_caches[
                        layer_id
                    ],
                    route,
                    index_results,
                ) = compressed_source_block_prefill(
                    layer_id,
                    x,
                    start_pos=start_pos,
                    pre_mix=pre_mix,
                    compress_ratio=ratio,
                    window_cache=(
                        self.windows[
                            layer_id
                        ]
                    ),
                    compressed_cache=(
                        self.compressed_caches[
                            layer_id
                        ]
                    ),
                    compressor_state=(
                        compressor_state
                    ),
                    indexer_state=(
                        self.indexer_states[
                            layer_id
                        ]
                    ),
                    shared_attn=(
                        self.shared_attn
                    ),
                    compressor_norm_weight=(
                        self._t(
                            layer_id,
                            (
                                "attn.compressor."
                                "norm.weight"
                            ),
                        )
                    ),
                    compressor_wkv_weight=(
                        self._t(
                            layer_id,
                            (
                                "attn.compressor."
                                "wkv.weight"
                            ),
                        )
                    ),
                    compressor_wgate_weight=(
                        compressor_wgate
                    ),
                    indexer_weights_proj_weight=(
                        self._t(
                            layer_id,
                            (
                                "attn.indexer."
                                "weights_proj.weight"
                            ),
                        )
                    ),
                    indexer_wq_b_weight=(
                        self._t(
                            layer_id,
                            (
                                "attn.indexer."
                                "wq_b.weight"
                            ),
                        )
                    ),
                    indexer_wq_b_scales=(
                        self._t(
                            layer_id,
                            (
                                "attn.indexer."
                                "wq_b.scale"
                            ),
                        )
                    ),
                    indexer_wk_weight=(
                        self._t(
                            layer_id,
                            (
                                "attn.indexer."
                                "wk.weight"
                            ),
                        )
                    ),
                    indexer_k_norm_weight=(
                        self._t(
                            layer_id,
                            (
                                "attn.indexer."
                                "k_norm.weight"
                            ),
                        )
                    ),
                    index_topk=INDEX_TOPK,
                    expert_prefetcher=(
                        self.expert_prefetcher
                    ),
                    **common,
                )

                shared_topk_by_token = tuple(
                    result.topk_idxs
                    for result in index_results
                )

                if layer_id == 20:
                    layer20_candidates_by_token = tuple(
                        result.candidates
                        for result in index_results
                    )

                    if any(
                        candidate is None
                        for candidate
                        in layer20_candidates_by_token
                    ):
                        raise RuntimeError(
                            "Layer 20 prefill did not publish "
                            "candidate snapshots"
                        )

            elif (
                layer_id
                in INDEX_ONLY_SOURCE_LAYERS
            ):
                if (
                    layer20_candidates_by_token
                    is None
                ):
                    raise RuntimeError(
                        "Index-only source reached before "
                        "layer 20 candidate publication"
                    )

                (
                    x,
                    pre_mix,
                    self.windows[
                        layer_id
                    ],
                    route,
                    index_results,
                ) = compressed_index_source_block_prefill(
                    layer_id,
                    x,
                    start_pos=start_pos,
                    pre_mix=pre_mix,
                    compress_ratio=1,
                    window_cache=(
                        self.windows[
                            layer_id
                        ]
                    ),
                    shared_attn=(
                        self.shared_attn
                    ),
                    shared_candidates_by_token=(
                        layer20_candidates_by_token
                    ),
                    indexer_weights_proj_weight=(
                        self._t(
                            layer_id,
                            (
                                "attn.indexer."
                                "weights_proj.weight"
                            ),
                        )
                    ),
                    indexer_wq_b_weight=(
                        self._t(
                            layer_id,
                            (
                                "attn.indexer."
                                "wq_b.weight"
                            ),
                        )
                    ),
                    indexer_wq_b_scales=(
                        self._t(
                            layer_id,
                            (
                                "attn.indexer."
                                "wq_b.scale"
                            ),
                        )
                    ),
                    index_topk=INDEX_TOPK,
                    expert_prefetcher=(
                        self.expert_prefetcher
                    ),
                    **self._common_block_kwargs(
                        layer_id,
                        compressed=True,
                    ),
                )

                shared_topk_by_token = tuple(
                    result.topk_idxs
                    for result in index_results
                )

            else:
                if shared_topk_by_token is None:
                    raise RuntimeError(
                        "Reuse layer reached before a source "
                        "published per-token top-k state"
                    )

                (
                    x,
                    pre_mix,
                    self.windows[
                        layer_id
                    ],
                    route,
                ) = compressed_reuse_block_prefill(
                    layer_id,
                    x,
                    start_pos=start_pos,
                    pre_mix=pre_mix,
                    compress_ratio=(
                        COMPRESS_RATIO[
                            layer_id
                        ]
                    ),
                    window_cache=(
                        self.windows[
                            layer_id
                        ]
                    ),
                    shared_attn=(
                        self.shared_attn
                    ),
                    shared_topk_by_token=(
                        shared_topk_by_token
                    ),
                    expert_prefetcher=(
                        self.expert_prefetcher
                    ),
                    **self._common_block_kwargs(
                        layer_id,
                        compressed=True,
                    ),
                )

            routes.append(
                self._trace_prefill_route(
                    layer_id,
                    start_pos,
                    route,
                )
            )

        # --------------------------------------------------------
        # Only the LAST prompt position needs final logits.
        #
        # This is exactly the result generation.py uses to predict
        # the first completion token.
        # --------------------------------------------------------

        if self.verbose:
            print(
                "[prefill] final HC/RMS/head"
            )

        hidden, logits = final_logits_decode(
            x[-1],
            pre_mix[-1],
            self._global(
                "norm.weight"
            ),
            self._global(
                "head.weight"
            ),
            norm_eps=NORM_EPS,
            head_chunk_size=(
                self.head_chunk_size
            ),
        )

        main_hidden = (
            mx.concatenate(
                main_hiddens,
                axis=-1,
            )
            if main_hiddens
            else None
        )

        mx.eval(
            hidden,
            logits,
        )

        if main_hidden is not None:
            mx.eval(main_hidden)

        final_position = (
            start_pos
            + n_tokens
            - 1
        )

        self.position = (
            start_pos
            + n_tokens
        )

        self.tokens.extend(token_ids)

        return DecodeResult(
            logits=logits,
            hidden=hidden,
            routes=tuple(routes),
            position=final_position,
            main_hidden=main_hidden,
        )

    def _decode_token_impl(
        self,
        token_id: int,
    ) -> DecodeResult:
        """
        Decode exactly one token at the current position.

        The input token becomes part of all persistent caches.
        Returned logits predict the next token.
        """
        start_pos = self.position

        if start_pos >= self.max_seq_len:
            raise IndexError(
                "Decode position exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        hash_rows = (
            self.engram_hash.push(
                token_id
            )
        )

        x = embed_token_decode(
            token_id,
            self._global(
                "embed.weight"
            ),
            hc_mult=HC_MULT,
        )

        pre_mix = (
            make_identity_pre_mix_decode(
                hc_mult=HC_MULT
            )
        )

        routes = []

        if self.verbose:
            print(
                f"[decode] position={start_pos} token_id={token_id}"
            )

        # ----------------------------------------------------
        # Layer 0: pure sliding-window attention.
        # ----------------------------------------------------
        if self.verbose:
            print("[decode] layer 0")

        (
            x,
            pre_mix,
            self.windows[0],
            route,
        ) = self._decode_layer0(
            x,
            pre_mix,
            start_pos,
        )

        routes.append(
            self._trace_decode_route(
                0,
                start_pos,
                route,
            )
        )

        # ----------------------------------------------------
        # Layer 1: Engram first, then sliding-window block.
        # ----------------------------------------------------
        if self.verbose:
            print("[decode] layer 1 Engram")

        x = self._apply_engram(
            x,
            1,
            hash_rows,
        )

        if self.verbose:
            print("[decode] layer 1")

        (
            x,
            pre_mix,
            self.windows[1],
            route,
        ) = self._decode_sliding(
            1,
            x,
            pre_mix,
            start_pos,
        )

        routes.append(
            self._trace_decode_route(
                1,
                start_pos,
                route,
            )
        )

        # ----------------------------------------------------
        # Layers 2..39.
        #
        # Sources:
        #   2, 8, 14: ratio-2 KV + index
        #   20:       ratio-1 KV + index + candidates
        #
        # Index-only sources:
        #   24, 28, 32, 36
        #
        # Every other upper layer reuses the most recent
        # published shared attention state.
        # ----------------------------------------------------
        main_hiddens = [] if self.capture_main_hidden else None

        for layer_id in range(
            2,
            N_LAYERS,
        ):
            if self.verbose:
                print(
                    f"[decode] layer {layer_id}"
                )

            if (
                main_hiddens is not None
                and layer_id
                in MTP_TARGET_LAYERS
            ):
                main_hiddens.append(
                    mx.mean(
                        x,
                        axis=0,
                    )
                )

            if layer_id == 14:
                if self.verbose:
                    print(
                        "[decode] layer 14 Engram"
                    )
                x = self._apply_engram(
                    x,
                    14,
                    hash_rows,
                )

            if layer_id in SOURCE_LAYERS:
                (
                    x,
                    pre_mix,
                    self.windows[
                        layer_id
                    ],
                    self.compressed_caches[
                        layer_id
                    ],
                    route,
                    _index_result,
                ) = self._decode_source(
                    layer_id,
                    x,
                    pre_mix,
                    start_pos,
                )

            elif (
                layer_id
                in INDEX_ONLY_SOURCE_LAYERS
            ):
                (
                    x,
                    pre_mix,
                    self.windows[
                        layer_id
                    ],
                    route,
                    _index_result,
                ) = self._decode_index_source(
                    layer_id,
                    x,
                    pre_mix,
                    start_pos,
                )

            else:
                (
                    x,
                    pre_mix,
                    self.windows[
                        layer_id
                    ],
                    route,
                ) = self._decode_reuse(
                    layer_id,
                    x,
                    pre_mix,
                    start_pos,
                )

            routes.append(
                self._trace_decode_route(
                    layer_id,
                    start_pos,
                    route,
                )
            )

        if self.verbose:
            print("[decode] final HC/RMS/head")

        hidden, logits = final_logits_decode(
            x,
            pre_mix,
            self._global(
                "norm.weight"
            ),
            self._global(
                "head.weight"
            ),
            norm_eps=NORM_EPS,
            head_chunk_size=(
                self.head_chunk_size
            ),
        )

        main_hidden = (
            mx.concatenate(
                main_hiddens,
                axis=-1,
            )
            if main_hiddens
            else None
        )

        mx.eval(
            hidden,
            logits,
        )

        if main_hidden is not None:
            mx.eval(main_hidden)

        self.position += 1
        self.tokens.append(int(token_id))

        return DecodeResult(
            logits=logits,
            hidden=hidden,
            routes=tuple(routes),
            position=start_pos,
            main_hidden=main_hidden,
        )

    def close(self) -> None:
        self._await_hotlist()
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5.0)
            self._heartbeat_thread = None
        if self._engram_pool is not None:
            self._engram_pool.shutdown(wait=False, cancel_futures=True)
            self._engram_pool = None
        self.engram_reader.close()
        self.expert_prefetcher.close()
        self.expert_store.close()

    def __enter__(
        self,
    ) -> TextDecodeRuntime:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
