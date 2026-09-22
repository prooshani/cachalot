"""
Port of the official DeepSeek-V4.1 image preprocessing
(`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/inference/image_processor.py`).

Feasibility-spike piece only (HANDOFF.md section 16, piece 2). The resize-ratio
solver and patchify are pure arithmetic/NumPy in the reference already; this
is a near-verbatim port with PIL replacing torch for image I/O and ml_dtypes
supplying the BF16 patches instead of `torch.Tensor.to(bfloat16)`.

`prepare_vl_inputs` (tokenizer-prompt expansion) is deliberately not ported
here — piece 3/4 territory, not this spike.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from urllib.request import urlopen

import ml_dtypes
import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageProcessorConfig:
    patch_size: int = 14
    downsample_ratio: int = 3
    max_n_token: int = 1024
    min_pixels: int = 544 * 544
    max_wh_ratio: float | None = None


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int) -> tuple[int, int]:
    return (
        math.ceil((best_height // patch_size) / downsample_ratio),
        math.ceil((best_width // patch_size) / downsample_ratio),
    )


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int]:
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, cfg: ImageProcessorConfig) -> tuple[int, int, int, int]:
    p = cfg.patch_size
    if cfg.max_wh_ratio is not None and width > height * cfg.max_wh_ratio:
        width = height * cfg.max_wh_ratio
    if 0 < width * height < cfg.min_pixels:
        ratio = (cfg.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, cfg.downsample_ratio, cfg.max_n_token)


def load_image_bytes(record: dict) -> bytes:
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)

    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]})

    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as response:
                return response.read()
        with open(url, "rb") as file:
            return file.read()

    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def load_image(
    record: dict,
    cfg: ImageProcessorConfig = ImageProcessorConfig(),
) -> tuple[mx.array, int, int, int, int]:
    """Load and transform one image record into ViT patches: [n_vit_h * n_vit_w, 3, p, p] BF16."""
    p = cfg.patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")

    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, cfg)
    n_vit_h, n_vit_w = best_height // p, best_width // p

    if cfg.max_wh_ratio is not None and image.width >= cfg.max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))

    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0  # [3, H, W]
    x = (x - 0.5) / 0.5
    x = x.astype(ml_dtypes.bfloat16)

    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .transpose(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )

    return mx.array(patches), n_vit_h, n_vit_w, n_llm_h, n_llm_w
