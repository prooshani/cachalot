"""
Vision port feasibility spike (HANDOFF.md section 16, piece 1+2). Not wired
into TextDecodeRuntime — this only proves the MLX ViT+Aligner port and the
image_processor port against the official PyTorch reference, on one real
image, before either touches the text model.

Two checks:
  A. image_processor_mlx.load_image vs the reference image_processor.load_image
     on the same image file — same grid shape, same patch values.
  B. vision_mlx.vision_embed vs the reference ViT+Aligner forward, fed the
     SAME (bit-identical) patches, loading the SAME checkpoint weights — isolates
     the port's numerics from any preprocessing discrepancy already caught by A.

Needs torch + pillow in the venv (dev-only, not a runtime dependency of
cachalot itself): `~/venvs/deepseek-v41/bin/pip install torch pillow`.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import ml_dtypes
import mlx.core as mx
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cachalot.model.image_processor_mlx import ImageProcessorConfig  # noqa: E402
from cachalot.model.image_processor_mlx import load_image as load_image_mlx  # noqa: E402
from cachalot.model.vision_mlx import VisionConfig, load_vision_weights, vision_embed  # noqa: E402
from cachalot.storage.index import read_safetensors_header  # noqa: E402
from cachalot.storage.tensor_index import CheckpointTensor  # noqa: E402
from cachalot.storage.tensor_loader import load_resident_tensor  # noqa: E402

CHECKPOINT_DIR = Path("/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash")
INFERENCE_DIR = CHECKPOINT_DIR / "inference"
SHARD = CHECKPOINT_DIR / "model-00001-of-00048.safetensors"
IMAGE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else CHECKPOINT_DIR / "assets" / "dsv41_kv_cache.png"

CFG = VisionConfig()
IMG_CFG = ImageProcessorConfig()


def load_reference_module(name: str) -> types.ModuleType:
    path = INFERENCE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"deepseek_v41_official_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_args(vision_module: types.ModuleType) -> types.SimpleNamespace:
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


def load_mlx_vision_tensors() -> dict[str, "object"]:
    header, data_start = read_safetensors_header(SHARD)
    out = {}
    for name, meta in header.items():
        if name == "__metadata__" or not (name.startswith("vision.") or name.startswith("aligner.")):
            continue
        rs, re = meta["data_offsets"]
        ct = CheckpointTensor(
            name=name,
            shard=SHARD,
            dtype=meta["dtype"],
            shape=tuple(int(x) for x in meta["shape"]),
            start=data_start + rs,
            end=data_start + re,
        )
        out[name] = load_resident_tensor(ct)
    return out


def load_torch_state_dict(prefix: str) -> dict[str, torch.Tensor]:
    state = {}
    with safe_open(str(SHARD), framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = f.get_tensor(key)
    return state


def bf16_torch_to_mx(t: torch.Tensor) -> mx.array:
    u16 = t.view(torch.uint16).numpy()
    return mx.array(u16.view(ml_dtypes.bfloat16))


def main() -> None:
    print(f"image:  {IMAGE_PATH}")
    print(f"shard:  {SHARD}")

    image_processor_ref = load_reference_module("image_processor")
    vision_ref = load_reference_module("vision")
    args = reference_args(vision_ref)

    # --- Part A: image_processor port vs reference, same file -------------
    mx_patches, mvh, mvw, mlh, mlw = load_image_mlx({"url": str(IMAGE_PATH)}, IMG_CFG)
    ref_patches, rvh, rvw, rlh, rlw = image_processor_ref.load_image({"url": str(IMAGE_PATH)}, args)

    print("\n[A] image_processor: MLX vs reference")
    print(f"    grid  MLX n_vit=({mvh},{mvw}) n_llm=({mlh},{mlw})")
    print(f"    grid  ref n_vit=({rvh},{rvw}) n_llm=({rlh},{rlw})")
    assert (mvh, mvw, mlh, mlw) == (rvh, rvw, rlh, rlw), "grid shape mismatch"

    mx_patches_f32 = mx_patches.astype(mx.float32)
    ref_patches_f32 = ref_patches.float().numpy()
    diff_a = (mx_patches_f32 - mx.array(ref_patches_f32)).abs()
    print(f"    max|diff| patches: {float(diff_a.max()):.6e}  (both BF16-quantized fp32 pixels)")

    # --- Part B: ViT+Aligner port vs reference, SAME bit-identical patches -
    vit = vision_ref.ViT(args)
    vit.load_state_dict(load_torch_state_dict("vision."))
    vit = vit.to(torch.bfloat16).eval()

    aligner = vision_ref.Aligner(args)
    aligner.load_state_dict(load_torch_state_dict("aligner."))
    aligner = aligner.to(torch.bfloat16).eval()

    with torch.no_grad():
        ref_out = aligner(vit(ref_patches, rvh, rvw), rvh, rvw)

    shared_patches_mx = bf16_torch_to_mx(ref_patches)
    weights = load_vision_weights(load_mlx_vision_tensors(), CFG)
    mlx_out = vision_embed(shared_patches_mx, rvh, rvw, weights, CFG)
    mx.eval(mlx_out)

    ref_out_f32 = ref_out.float().numpy()
    mlx_out_f32 = mlx_out.astype(mx.float32)
    diff_b = (mlx_out_f32 - mx.array(ref_out_f32)).abs()
    ref_scale = float(abs(ref_out_f32).max())

    print("\n[B] ViT+Aligner: MLX vs reference, same patches, same weights")
    print(f"    output shape MLX {mlx_out.shape}  ref {tuple(ref_out.shape)}")
    print(f"    max|diff|  {float(diff_b.max()):.6e}   ref max|value| {ref_scale:.4f}")
    print(f"    mean|diff| {float(diff_b.mean()):.6e}")


if __name__ == "__main__":
    main()
