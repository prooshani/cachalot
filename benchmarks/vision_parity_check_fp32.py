"""
FP32 variant of vision_parity_check.py's Part B, to separate BF16 rounding
noise (expected, accumulates over 32 layers) from an actual port defect.
Same weights, same patches, both paths upcast to FP32 before the forward.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cachalot.model.image_processor_mlx import ImageProcessorConfig  # noqa: E402
from cachalot.model.vision_mlx import VisionConfig, load_vision_weights, vision_embed  # noqa: E402
from cachalot.storage.index import read_safetensors_header  # noqa: E402
from cachalot.storage.tensor_index import CheckpointTensor  # noqa: E402
from cachalot.storage.tensor_loader import load_resident_tensor  # noqa: E402

CHECKPOINT_DIR = Path("/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash")
INFERENCE_DIR = CHECKPOINT_DIR / "inference"
SHARD = CHECKPOINT_DIR / "model-00001-of-00048.safetensors"
IMAGE_PATH = Path(sys.argv[1])

CFG = VisionConfig()
IMG_CFG = ImageProcessorConfig()


def load_reference_module(name: str) -> types.ModuleType:
    path = INFERENCE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"deepseek_v41_official_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        dim=5120,
        vision_dim=CFG.dim,
        vision_n_heads=CFG.n_heads,
        vision_n_layers=CFG.n_layers,
        vision_inter_dim=CFG.inter_dim,
        vision_patch_size=CFG.patch_size,
        vision_rope_theta=CFG.rope_theta,
        vision_downsample_ratio=CFG.downsample_ratio,
        vision_max_n_token=IMG_CFG.max_n_token,
        vision_min_pixels=IMG_CFG.min_pixels,
        vision_max_wh_ratio=IMG_CFG.max_wh_ratio,
    )


def load_mlx_vision_tensors() -> dict:
    header, data_start = read_safetensors_header(SHARD)
    out = {}
    for name, meta in header.items():
        if name == "__metadata__" or not (name.startswith("vision.") or name.startswith("aligner.")):
            continue
        rs, re = meta["data_offsets"]
        ct = CheckpointTensor(
            name=name, shard=SHARD, dtype=meta["dtype"],
            shape=tuple(int(x) for x in meta["shape"]),
            start=data_start + rs, end=data_start + re,
        )
        out[name] = load_resident_tensor(ct)
    return out


def load_torch_state_dict(prefix: str) -> dict:
    state = {}
    with safe_open(str(SHARD), framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = f.get_tensor(key).float()
    return state


def main() -> None:
    image_processor_ref = load_reference_module("image_processor")
    vision_ref = load_reference_module("vision")
    args = reference_args()

    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_processor_ref.load_image({"url": str(IMAGE_PATH)}, args)
    patches_f32 = patches.float()

    vit = vision_ref.ViT(args)
    vit.load_state_dict(load_torch_state_dict("vision."))
    vit = vit.float().eval()
    aligner = vision_ref.Aligner(args)
    aligner.load_state_dict(load_torch_state_dict("aligner."))
    aligner = aligner.float().eval()

    with torch.no_grad():
        ref_out = aligner(vit(patches_f32, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    tensors = load_mlx_vision_tensors()
    tensors_f32 = {k: type(v)(v.name, v.checkpoint_dtype, v.shape, v.data.astype(mx.float32)) for k, v in tensors.items()}
    weights = load_vision_weights(tensors_f32, CFG)
    patches_mx = mx.array(patches_f32.numpy())
    mlx_out = vision_embed(patches_mx, n_vit_h, n_vit_w, weights, CFG)
    mx.eval(mlx_out)

    ref_np = ref_out.detach().numpy()
    diff = (mlx_out - mx.array(ref_np)).abs()
    print(f"FP32 check: shape MLX {mlx_out.shape} ref {tuple(ref_out.shape)}")
    print(f"max|diff|  {float(diff.max()):.6e}   ref max|value| {float(abs(ref_np).max()):.4f}")
    print(f"mean|diff| {float(diff.mean()):.6e}")


if __name__ == "__main__":
    main()
