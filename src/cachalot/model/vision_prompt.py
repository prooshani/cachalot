"""
Vision phase 1, piece 4: turn a chat prompt's image records into the token
sequence and embedding rows the prefill splice (piece 3) consumes.

Port of the official `inference/image_processor.prepare_vl_inputs` plus the
delimiter half of `inference/model.py`'s `merge_image_embeddings`:

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every position of that span carries `image_token_id` in the token ids. The
IMAGE slots take the aligner's rows in reading order; the three delimiter
kinds take the checkpoint's learned `image_start` / `image_newline` /
`image_end` vectors. The reference writes both into the embedded sequence;
here they are assembled into one `[span_len, dim]` block per image, so
`vision_mlx.merge_image_embeddings` (which consumes one row per
`image_token_id` position) receives exactly what the reference writes.

The vision tower (~0.9 GiB BF16) is loaded lazily on the first image, so a
text-only server never pays for it.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import mlx.core as mx

from cachalot.model.image_processor_mlx import (
    ImageProcessorConfig,
    load_image,
    load_image_bytes,
)
from cachalot.model.vision_mlx import (
    VisionConfig,
    VisionWeights,
    load_vision_weights,
    vision_embed,
)

IMAGE_TOKEN_ID = 129264

# token types, matching the reference's image_processor.py
TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)

DELIMITER_NAMES = ("image_start", "image_newline", "image_end")


@dataclass(frozen=True)
class ImageSpan:
    """One image's span in the expanded prompt."""

    start: int
    length: int
    digest: str
    rows: mx.array  # [length, dim], one row per image_token_id position

    @property
    def key(self) -> tuple[int, int, str]:
        """Identity for the prefix cache: where the span sits and what it holds."""
        return (self.start, self.length, self.digest)


@dataclass
class PromptImages:
    tokens: list[int]
    spans: list[ImageSpan] = field(default_factory=list)

    @property
    def keys(self) -> tuple[tuple[int, int, str], ...]:
        return tuple(s.key for s in self.spans)

    def rows_between(self, start: int, end: int) -> mx.array | None:
        """Image rows for a prefill call covering positions [start, end)."""
        rows = []
        for span in self.spans:
            s0, s1 = span.start, span.start + span.length
            if s1 <= start or s0 >= end:
                continue
            if s0 < start or s1 > end:
                raise ValueError(
                    f"prefill chunk [{start}, {end}) splits an image span [{s0}, {s1})"
                )
            rows.append(span.rows)
        return mx.concatenate(rows, axis=0) if rows else None

    def keys_between(self, start: int, end: int) -> tuple[tuple[int, int, str], ...]:
        return tuple(s.key for s in self.spans if start <= s.start < end)

    def rows_from(self, position: int) -> mx.array | None:
        """
        The image rows a prefill starting at `position` needs: every span
        at or after it, in order. A span straddling `position` would mean a
        snapshot boundary inside an image, which the prefix cache never
        produces (snapshots end at a prompt or a reply); refuse it loudly.
        """
        rows = []
        for span in self.spans:
            if span.start + span.length <= position:
                continue
            if span.start < position:
                raise ValueError(
                    f"prefill boundary {position} falls inside an image span "
                    f"[{span.start}, {span.start + span.length})"
                )
            rows.append(span.rows)
        if not rows:
            return None
        return mx.concatenate(rows, axis=0)


def image_span_rows(
    aligner_rows: mx.array,
    n_llm_h: int,
    n_llm_w: int,
    image_start: mx.array,
    image_newline: mx.array,
    image_end: mx.array,
) -> mx.array:
    """
    Lay the aligner's `[n_llm_h * n_llm_w, dim]` rows out in the span's order,
    with a newline row after each grid row and the start/end rows around it.
    """
    dim = aligner_rows.shape[-1]
    if aligner_rows.shape[0] != n_llm_h * n_llm_w:
        raise ValueError(
            f"aligner produced {aligner_rows.shape[0]} rows for a "
            f"{n_llm_h}x{n_llm_w} token grid"
        )
    dtype = aligner_rows.dtype
    grid = aligner_rows.reshape(n_llm_h, n_llm_w, dim)
    newline = mx.broadcast_to(image_newline.astype(dtype)[None, None, :], (n_llm_h, 1, dim))
    body = mx.concatenate([grid, newline], axis=1).reshape(n_llm_h * (n_llm_w + 1), dim)
    return mx.concatenate(
        [image_start.astype(dtype)[None, :], body, image_end.astype(dtype)[None, :]],
        axis=0,
    )


class VisionEncoder:
    """ViT + aligner + delimiter vectors, loaded from the checkpoint on first use."""

    def __init__(
        self,
        tensor_index,
        cfg: VisionConfig = VisionConfig(),
        image_cfg: ImageProcessorConfig = ImageProcessorConfig(),
    ) -> None:
        self._index = tensor_index
        self.cfg = cfg
        self.image_cfg = image_cfg
        self._weights: VisionWeights | None = None
        self._delims: dict[str, mx.array] | None = None
        self._load_lock = threading.Lock()
        # An agent resends every image in the history on every turn. The
        # span rows are a pure function of the image bytes, so they are kept
        # by digest (a few MB each) instead of re-running the tower.
        self._rows: OrderedDict[str, mx.array] = OrderedDict()
        self.max_cached = 16
        self.cache_hits = 0

    def _ensure_loaded(self) -> None:
        if self._weights is not None:
            return
        with self._load_lock:
            if self._weights is not None:
                return
            from cachalot.storage.tensor_loader import load_resident_tensor

            names = [
                n for n in self._index
                if n.startswith("vision.") or n.startswith("aligner.") or n in DELIMITER_NAMES
            ]
            tensors = {n: load_resident_tensor(self._index[n]) for n in names}
            missing = [n for n in DELIMITER_NAMES if n not in tensors]
            if missing:
                raise KeyError(f"checkpoint has no vision delimiter tensors: {missing}")
            weights = load_vision_weights(tensors, self.cfg)
            delims = {n: tensors[n].data for n in DELIMITER_NAMES}
            mx.eval(*delims.values())
            self._delims = delims
            self._weights = weights

    @property
    def loaded(self) -> bool:
        return self._weights is not None

    def encode(self, record: dict) -> tuple[mx.array, str]:
        """One image record -> (span rows [span_len, dim], content digest)."""
        raw = load_image_bytes(record)
        digest = hashlib.sha256(raw).hexdigest()[:32]
        cached = self._rows.get(digest)
        if cached is not None:
            self._rows.move_to_end(digest)
            self.cache_hits += 1
            return cached, digest
        self._ensure_loaded()
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image({"data": raw}, self.image_cfg)
        aligner_rows = vision_embed(patches, n_vit_h, n_vit_w, self._weights, self.cfg)
        rows = image_span_rows(
            aligner_rows,
            n_llm_h,
            n_llm_w,
            self._delims["image_start"],
            self._delims["image_newline"],
            self._delims["image_end"],
        )
        mx.eval(rows)
        self._rows[digest] = rows
        while len(self._rows) > self.max_cached:
            self._rows.popitem(last=False)
        return rows, digest


def expand_prompt_images(
    tokens: list[int],
    records: list[dict],
    encoder: VisionEncoder | None,
    *,
    image_token_id: int = IMAGE_TOKEN_ID,
) -> PromptImages:
    """
    `prepare_vl_inputs`: replace each placeholder token with its image span.

    `tokens` is the tokenized prompt, one `image_token_id` per image, in the
    order `records` lists them (the order the official encoding collected
    them in). Text-only prompts pass through untouched.
    """
    n_placeholders = sum(1 for t in tokens if t == image_token_id)
    if n_placeholders != len(records):
        raise ValueError(
            f"prompt has {n_placeholders} image placeholders but {len(records)} images"
        )
    if not records:
        return PromptImages(tokens=list(tokens))
    if encoder is None:
        raise ValueError("this server has no vision encoder loaded; images are not supported")

    out: list[int] = []
    spans: list[ImageSpan] = []
    it = iter(records)
    for tok in tokens:
        if tok != image_token_id:
            out.append(tok)
            continue
        rows, digest = encoder.encode(next(it))
        spans.append(ImageSpan(start=len(out), length=rows.shape[0], digest=digest, rows=rows))
        out.extend([image_token_id] * rows.shape[0])
    return PromptImages(tokens=out, spans=spans)
