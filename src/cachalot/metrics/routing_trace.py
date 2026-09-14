from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PHASE_PREFILL = 0
PHASE_DECODE = 1

_PHASE_IDS = {"prefill": PHASE_PREFILL, "decode": PHASE_DECODE}


@dataclass
class RoutingTracer:
    """
    Records routed-expert selections for offline cache-policy analysis.

    One record per (token position, layer). Storage is columnar so a
    multi-thousand-token session costs a few megabytes.

    Enabled explicitly by the caller; the hot path pays one attribute check
    when no tracer is installed.
    """

    _phase: list[np.ndarray] = field(default_factory=list)
    _layer: list[np.ndarray] = field(default_factory=list)
    _position: list[np.ndarray] = field(default_factory=list)
    _experts: list[np.ndarray] = field(default_factory=list)
    _segments: list[dict] = field(default_factory=list)
    _count: int = 0

    def record(
        self,
        phase: str,
        layer: int,
        start_pos: int,
        indices,
    ) -> None:
        """
        indices: [n_tokens, topk] or [topk] array-like of expert ids for
        consecutive positions starting at start_pos.
        """
        arr = np.asarray(indices, dtype=np.int16)

        if arr.ndim == 1:
            arr = arr[None, :]

        n_tokens = arr.shape[0]

        self._phase.append(
            np.full(n_tokens, _PHASE_IDS[phase], dtype=np.int8)
        )
        self._layer.append(np.full(n_tokens, layer, dtype=np.int16))
        self._position.append(
            np.arange(start_pos, start_pos + n_tokens, dtype=np.int32)
        )
        self._experts.append(arr)
        self._count += n_tokens

    def mark(self, label: str, **meta) -> None:
        """Annotate a boundary (new turn, new prompt) at the current record count."""
        self._segments.append({"label": label, "at": self._count, **meta})

    @property
    def records(self) -> int:
        return self._count

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._experts:
            return {
                "phase": np.zeros(0, np.int8),
                "layer": np.zeros(0, np.int16),
                "position": np.zeros(0, np.int32),
                "experts": np.zeros((0, 0), np.int16),
            }

        return {
            "phase": np.concatenate(self._phase),
            "layer": np.concatenate(self._layer),
            "position": np.concatenate(self._position),
            "experts": np.concatenate(self._experts, axis=0),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            segments=np.array(json.dumps(self._segments)),
            **self.arrays(),
        )

        return path


def load_trace(path: str | Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    with np.load(path) as data:
        arrays = {
            key: data[key]
            for key in ("phase", "layer", "position", "experts")
        }
        segments = json.loads(str(data["segments"]))

    return arrays, segments
