from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class SharedAttentionRuntime:
    """
    Cross-layer attention values published by source layers.

    Mirrors DeepSeek V4.1 Flash inference/model.py::
    SharedAttentionRuntime.

    Layers execute sequentially, so one published slot per
    value is sufficient:

        compress_kv
            Latest compressed-KV cache published by a
            kv_source layer.

        index_k
            Latest compressed index-K cache published by a
            kv/index source layer.

        topk_idxs
            Latest compressed-position selection published by
            an index_source layer.

            IMPORTANT:
            In this runtime these remain RELATIVE compressed
            cache positions. The consuming attention layer
            applies its current window-KV offset when building
            the concatenated [window | compressed] view.

        candidates
            Candidate block selection published by the
            candidate source layer (layer 20). Not wired yet.
    """

    compress_kv: mx.array | None = None
    index_k: mx.array | None = None
    topk_idxs: mx.array | None = None
    candidates: mx.array | None = None

    def reset(self) -> None:
        """
        Clear published cross-layer references before starting
        an independent sequence/model run.

        Per-layer CompressorState and IndexerState caches are
        reset separately by their owners.
        """
        self.compress_kv = None
        self.index_k = None
        self.topk_idxs = None
        self.candidates = None
