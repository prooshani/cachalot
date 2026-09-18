"""
Build a smaller routed-expert bank from the shipped FP4 checkpoint.

Decode on this machine is bytes-bound: at a 36 GiB budget a token needs about
988 MiB of experts, which is 155 ms of drive time against 134 ms of compute, so
the only lever that moves the floor is reading fewer bytes per expert
(docs/HANDOFF-2026-09-17.md, section 15.3). This tool writes a bank in the same
stacked affine layout the oQ3e download uses, so the runtime reads it with no
code change at all - `CACHALOT_EXPERT_BANK=<out>` is the whole switch - but with
fewer bits per weight:

    3-bit group 64                  15,482,880 B/expert   221.5 GiB of bank
    3-bit group 128                 14,376,960 B/expert   205.6 GiB
    2-bit group 64                  11,059,200 B/expert   158.2 GiB
    2-bit group 128                  9,953,280 B/expert   142.4 GiB

Weights come from the FP4 checkpoint, which is the highest precision this
machine holds, are dequantized to fp32 and re-quantized per group. The fit is
the searched one in quant_affine.py, which beats mx.quantize's max-abs fit at
every width and costs no extra bytes: about 17 % of output error at 2 bits and
26 % at 3, where it also beats the calibrated oQ3e download. benchmarks/
quant_fit_screen.py ranks fits at a fixed format in minutes, benchmarks/
expert_requant_error.py ranks formats, and benchmarks/nll_expert_precision.py
--experts requant is the cheap quality gate; above 2 bits the gate that decides
adoption is benchmarks/code_validity.py on matched generations, because top-1
and NLL both called the 3-bit mx.quantize bank a win and a compiler did not.

Each layer becomes one shard holding nine stacked tensors, written row by row so
the builder never holds more than one expert in memory. An interrupted build is
resumed by re-running the same command: a shard that is already complete and the
right size is skipped.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh \
      --budget-gib 8 --max-seconds 14400 --tag bank2g64 -- env PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/build_affine_bank.py \
      --out /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g64 --bits 2 --group 64
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import struct
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from expert_requant_error import read_fp4_expert  # noqa: E402
from quant_affine import (  # noqa: E402
    fit_minmax, fit_search, fit_search_lsq, fit_search_wide_lsq, quantize_affine,
)
from cachalot.storage.index import build_expert_index, detect_expert_bank  # noqa: E402

HIDDEN, INTER, N_EXPERTS, N_LAYERS = 5120, 2304, 384, 40
PROJ_ROWS = {"w1": (INTER, HIDDEN), "w3": (INTER, HIDDEN), "w2": (HIDDEN, INTER)}
FITS = {"mlx": None, "minmax": fit_minmax, "search": fit_search,
        "search-lsq": fit_search_lsq, "wide-lsq": fit_search_wide_lsq}
FIELD_DTYPE = {"weight": "U32", "scales": "BF16", "biases": "BF16"}
ITEM_BYTES = {"U32": 4, "BF16": 2}


def tensor_name(layer: int, proj: str, field: str) -> str:
    return f"language_model.layers.{layer}.ffn.experts.{proj}.{field}"


def tensor_shape(proj: str, field: str, bits: int, group: int) -> tuple[int, int, int]:
    rows, cols = PROJ_ROWS[proj]
    if field == "weight":
        return (N_EXPERTS, rows, cols * bits // 32)
    return (N_EXPERTS, rows, cols // group)


def expert_bytes(bits: int, group: int) -> int:
    return sum(
        shape[1] * shape[2] * ITEM_BYTES[FIELD_DTYPE[field]]
        for proj in ("w1", "w2", "w3")
        for field in ("weight", "scales", "biases")
        for shape in (tensor_shape(proj, field, bits, group),)
    )


def shard_name(layer: int) -> str:
    return f"model-{layer + 1:05d}-of-{N_LAYERS:05d}.safetensors"


def shard_plan(layer: int, bits: int, group: int) -> tuple[dict, int, dict[str, int]]:
    """(safetensors header, data start, per-tensor data offset) for one layer's shard."""
    header: dict[str, dict] = {}
    starts: dict[str, int] = {}
    offset = 0
    for proj in ("w1", "w2", "w3"):
        for field in ("weight", "scales", "biases"):
            shape = tensor_shape(proj, field, bits, group)
            dtype = FIELD_DTYPE[field]
            size = shape[0] * shape[1] * shape[2] * ITEM_BYTES[dtype]
            name = tensor_name(layer, proj, field)
            header[name] = {"dtype": dtype, "shape": list(shape),
                            "data_offsets": [offset, offset + size]}
            starts[name] = offset
            offset += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(blob)) % 8
    blob += b" " * pad
    return header, 8 + len(blob), starts


def write_header(path: Path, layer: int, bits: int, group: int) -> tuple[int, dict[str, int], int]:
    header, data_start, starts = shard_plan(layer, bits, group)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    total = data_start + sum(
        h["data_offsets"][1] - h["data_offsets"][0] for h in header.values()
    )
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.truncate(total)
    return data_start, starts, total


def shard_is_complete(path: Path, layer: int, bits: int, group: int) -> bool:
    header, _, _ = shard_plan(layer, bits, group)
    _, data_start, _ = shard_plan(layer, bits, group)
    expected_size = data_start + sum(
        h["data_offsets"][1] - h["data_offsets"][0] for h in header.values()
    )
    if not path.exists() or path.stat().st_size != expected_size:
        return False
    try:
        with path.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            got = json.loads(fh.read(n))
    except (OSError, json.JSONDecodeError, struct.error):
        return False
    return all(
        name in got and got[name]["shape"] == meta["shape"] and got[name]["dtype"] == meta["dtype"]
        for name, meta in header.items()
    )


MX_DTYPE = {"U32": mx.uint32, "BF16": mx.bfloat16}


def quantize_expert(dense: dict[str, mx.array], bits: int, group: int, fit: str,
                    ) -> dict[tuple[str, str], mx.array]:
    """One expert in the bank's on-disk dtypes.

    The cast is not cosmetic. mx.quantize returns scales and biases in the
    dtype of its input, and the dense weights arrive as fp32, so the mlx fit
    produced fp32 scales while the shard header declares BF16 and reserves two
    bytes per element. quantize_affine casts to bf16 itself, which is
    why every searched-fit bank was correct and the first --fit mlx bank was not:
    its scales were written at double stride, over the top of the tensors that
    followed. bf16 is also what the oQ3e 3-bit bank stores, and fp32 scales
    measured as a null against bf16 at 2 bits.
    """
    out: dict[tuple[str, str], mx.array] = {}
    for proj in ("w1", "w2", "w3"):
        if fit == "mlx":
            q, scales, biases = mx.quantize(dense[proj], group_size=group, bits=bits)
        else:
            q, scales, biases = quantize_affine(dense[proj], group_size=group, bits=bits,
                                                fit=FITS[fit])
        out[(proj, "weight")] = q.astype(MX_DTYPE[FIELD_DTYPE["weight"]])
        out[(proj, "scales")] = scales.astype(MX_DTYPE[FIELD_DTYPE["scales"]])
        out[(proj, "biases")] = biases.astype(MX_DTYPE[FIELD_DTYPE["biases"]])
    mx.eval(*out.values())
    return out


def write_config(out: Path, bits: int, group: int, fit: str, source: str) -> None:
    modules = {
        f"language_model.layers.{layer}.ffn.experts.{proj}": {
            "bits": bits, "group_size": group, "mode": "affine", "quantize_input": True,
        }
        for layer in range(N_LAYERS)
        for proj in ("w1", "w2", "w3")
    }
    (out / "config.json").write_text(json.dumps({
        "omlx_deepseek_v41": {"version": 1, "quantized_modules": modules},
        "cachalot_bank": {
            "built_from": source,
            "bits": bits,
            "group_size": group,
            "fit": fit,
            "bytes_per_expert": expert_bytes(bits, group),
            "note": "routed experts only; trunk, Engram, head and tokenizer stay with the "
                    "shipped checkpoint (CACHALOT_MODEL_PATH)",
        },
    }, indent=1))

    weight_map = {
        tensor_name(layer, proj, field): shard_name(layer)
        for layer in range(N_LAYERS)
        for proj in ("w1", "w2", "w3")
        for field in ("weight", "scales", "biases")
    }
    (out / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": expert_bytes(bits, group) * N_EXPERTS * N_LAYERS},
        "weight_map": weight_map,
    }, indent=1))


def verify(out: Path, index, bits: int, group: int, fit: str, samples: int, seed: int) -> None:
    """Read sampled experts back through the real index and compare with a fresh quantization."""
    fmt, bank_index = detect_expert_bank(out)
    if fmt.bits != bits or fmt.group_size != group:
        raise SystemExit(f"verify: bank reports {fmt.bits}-bit/{fmt.group_size}, expected {bits}/{group}")
    if len(bank_index) != N_EXPERTS * N_LAYERS:
        raise SystemExit(f"verify: bank holds {len(bank_index)} experts, expected {N_EXPERTS * N_LAYERS}")

    rng = random.Random(seed)
    source_fds: dict[Path, int] = {}
    bank_fds: dict[Path, int] = {}
    for _ in range(samples):
        layer, expert = rng.randrange(N_LAYERS), rng.randrange(N_EXPERTS)
        want = quantize_expert(read_fp4_expert(index[(layer, expert)], source_fds), bits, group, fit)
        ranges = {".".join(t.name.rsplit(".", 2)[-2:]): t for t in bank_index[(layer, expert)].tensors}
        for (proj, field), arr in want.items():
            short = f"{proj}.{field}"
            t = ranges.get(short)
            if t is None:
                raise SystemExit(f"verify: layer {layer} expert {expert} has no {short} in the bank")
            fd = bank_fds.get(t.shard)
            if fd is None:
                fd = bank_fds[t.shard] = os.open(t.shard, os.O_RDONLY)
            got = os.pread(fd, t.end - t.start, t.start)
            if len(got) != arr.nbytes or got != bytes(memoryview(arr)):
                raise SystemExit(f"verify: layer {layer} expert {expert} {short} does not match "
                                 f"the source ({len(got)} B in bank vs {arr.nbytes} B expected)")
    for fd in (*source_fds.values(), *bank_fds.values()):
        os.close(fd)
    print(f"verify: {samples} sampled experts match the source byte for byte")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=MODEL_PATH, help="FP4 source checkpoint")
    ap.add_argument("--out", required=True, help="bank directory to write")
    ap.add_argument("--bits", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument("--group", type=int, default=64, choices=[32, 64, 128])
    ap.add_argument("--fit", choices=list(FITS), default="search")
    ap.add_argument("--layers", default="all", help="'all' or a comma-separated list")
    ap.add_argument("--verify", type=int, default=8, help="experts to read back and check, 0 to skip")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--free-margin-gib", type=float, default=8.0)
    args = ap.parse_args()

    layers = (list(range(N_LAYERS)) if args.layers == "all"
              else [int(v) for v in args.layers.split(",")])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_expert = expert_bytes(args.bits, args.group)
    todo = [layer for layer in layers
            if not shard_is_complete(out / shard_name(layer), layer, args.bits, args.group)]
    needed = per_expert * N_EXPERTS * len(todo)
    free = shutil.disk_usage(out).free
    print(f"bank {out}: {args.bits}-bit group {args.group} ({args.fit} fit), "
          f"{per_expert:,} B/expert, {per_expert * N_EXPERTS * len(layers) / 2**30:.1f} GiB total")
    print(f"{len(layers) - len(todo)}/{len(layers)} layers already complete; "
          f"{needed / 2**30:.1f} GiB to write, {free / 2**30:.1f} GiB free")
    if needed + args.free_margin_gib * 2**30 > free:
        raise SystemExit(f"refusing to start: {needed / 2**30:.1f} GiB to write plus a "
                         f"{args.free_margin_gib:.0f} GiB margin exceeds {free / 2**30:.1f} GiB free")

    index = build_expert_index(args.model_path)
    if not index:
        raise SystemExit(f"no FP4 experts under {args.model_path}")

    fds: dict[Path, int] = {}
    written = 0
    t_start = perf_counter()
    for n, layer in enumerate(todo, 1):
        path = out / shard_name(layer)
        partial = path.with_suffix(".safetensors.partial")
        data_start, starts, total = write_header(partial, layer, args.bits, args.group)
        fd = os.open(partial, os.O_WRONLY)
        t0 = perf_counter()
        for expert in range(N_EXPERTS):
            entry = index.get((layer, expert))
            if entry is None:
                os.close(fd)
                partial.unlink(missing_ok=True)
                raise SystemExit(f"source is missing layer {layer} expert {expert}")
            packed = quantize_expert(read_fp4_expert(entry, fds), args.bits, args.group, args.fit)
            for (proj, field), arr in packed.items():
                name = tensor_name(layer, proj, field)
                # Stride from the *plan*, never from the array. Taking it from
                # the array is what let an unexpected dtype write every expert
                # at double stride, silently over the following tensors; the
                # shard header is the contract and a mismatch is a bug, not
                # something to accommodate.
                shape = tensor_shape(proj, field, args.bits, args.group)
                row = shape[1] * shape[2] * ITEM_BYTES[FIELD_DTYPE[field]]
                if arr.nbytes != row:
                    raise ValueError(
                        f"{name}: quantizer produced {arr.nbytes} B per expert "
                        f"but the shard header reserves {row} B "
                        f"({FIELD_DTYPE[field]}); refusing to write a corrupt bank"
                    )
                os.pwrite(fd, memoryview(arr), data_start + starts[name] + expert * row)
        os.fsync(fd)
        os.close(fd)
        partial.replace(path)
        written += total
        dt = perf_counter() - t0
        elapsed = perf_counter() - t_start
        print(f"layer {layer:2d} ({n}/{len(todo)}): {total / 2**30:.2f} GiB in {dt:.0f} s "
              f"({N_EXPERTS / dt:.1f} experts/s), {written / 2**30:.1f} GiB written, "
              f"{elapsed / 60:.0f} min elapsed, "
              f"eta {(len(todo) - n) * elapsed / n / 60:.0f} min", flush=True)

    for fd in fds.values():
        os.close(fd)
    write_config(out, args.bits, args.group, args.fit, str(args.model_path))
    print(f"wrote {out} in {(perf_counter() - t_start) / 60:.0f} min")

    if args.verify and args.layers == "all":
        verify(out, index, args.bits, args.group, args.fit, args.verify, args.seed)
    print("\nuse it with:\n"
          f"  CACHALOT_EXPERT_BANK={out} CACHALOT_MODEL_PATH={args.model_path}")


if __name__ == "__main__":
    main()
