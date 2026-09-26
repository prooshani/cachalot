"""
A MiniMax-M3 routed-expert bank without the biases (HANDOFF 18.4).

MLX's affine quantization stores, per group of 64 weights, a bf16 scale and a bf16 bias, and the bias is the
group's edge value, which the quantizer makes a whole multiple of the scale. For this checkpoint every one of
its 6.46 billion expert groups satisfies

    bias == bf16_round_to_nearest_even(float32(k) * float32(scale)),   k in {-7, -6, -5, -4, -3}

(benchmarks/minimax_coded_bank.py --scan checks it). So a bias is two bits (k + 6 for k in -6..-3) once the
scale is known, and an expert shrinks from 23.62 MiB (nine pieces in the stacked safetensors) to 22.16 MiB,
stored as one contiguous record per expert:

    head    w1.scales | w3.scales | w2.scales | codes w1 | codes w3 | codes w2     (1.90 MiB, padded to 16 KiB)
    weights w1.weight | w3.weight | w2.weight                                      (20.25 MiB)

The 3.6 % of experts with any k = -7 group keep their biases (a "raw" record: the six scale/bias pieces as the
head, then the weights). A read issues the three weight preads first, reads the head into the slot's scale
views and a code buffer, and rebuilds the biases straight into the slot's bias views through a 256K-entry table
(scale bits, code) -> bias bits while the weights are still arriving. The slot then holds exactly the bytes
the stacked checkpoint would have put there, so nothing downstream changes and outputs are bit-identical.

Layout on disk: `bank.json` (format, the records: layer, expert, file, offset, kind) and one `layer-NNN.bin`
per MoE layer. A bank may cover only some layers; the others are read from the checkpoint as before.
`CACHALOT_MINIMAX_BANK` selects the bank directory, `CACHALOT_MINIMAX_BANK_MIRROR` an identical copy on a
second drive (the tail `CACHALOT_MIRROR_FRACTION` of each weight piece is read from it, like the stacked
checkpoint's mirror). ENABLED (a module constant) may be flipped between reads: both paths fill a slot with
the same bytes, which is what makes a token-by-token A/B possible (TF_ALTERNATE).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cachalot.storage.index import ExpertEntry, ExpertFormat
from cachalot.storage.reader import ExpertReader

BANK_VERSION = 1
ALIGN = 16384
PROJS = ("w1", "w3", "w2")
K_BASE = -6  # code c in 0..3 means k = c - 6
ENABLED = int(os.environ.get("CACHALOT_MINIMAX_BANK_ENABLED", "1"))  # an int, so TF_ALTERNATE can flip it


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def bf16_bits_rne(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns, round to nearest even (what MLX's cast does)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _bias_table() -> np.ndarray:
    """table[(scale_bits << 2) | code] = bf16 bits of (code + K_BASE) * scale."""
    scale = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    table = np.empty(65536 * 4, np.uint16)
    with np.errstate(all="ignore"):
        for c in range(4):
            table[c::4] = bf16_bits_rne(scale * np.float32(c + K_BASE))
    return table


_TABLE: np.ndarray | None = None


def bias_table() -> np.ndarray:
    global _TABLE
    if _TABLE is None:
        _TABLE = _bias_table()
    return _TABLE


_SHIFTS = np.array([0, 2, 4, 6], np.uint8)


def encode_biases(scales: np.ndarray, biases: np.ndarray) -> np.ndarray | None:
    """Packed 2-bit codes (4 per byte, low bits first) for bf16 bit patterns, or None when any group's bias
    is not (k * scale) rounded for k in -6..-3."""
    s = (scales.astype(np.uint32) << 16).view(np.float32)
    b = (biases.astype(np.uint32) << 16).view(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.round(b / s)
    k = np.where(s == 0, K_BASE, k)
    if k.min() < K_BASE or k.max() > K_BASE + 3:
        return None
    codes = (k - K_BASE).astype(np.uint8)
    if not np.array_equal(decode_biases(scales, pack_codes(codes)), biases):
        return None
    return pack_codes(codes)


def pack_codes(codes: np.ndarray) -> np.ndarray:
    return (codes.reshape(-1, 4) << _SHIFTS).sum(axis=1, dtype=np.uint8)


def decode_biases(scales: np.ndarray, packed: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """bf16 bias bit patterns from bf16 scale bit patterns and packed codes (the read path's rebuild)."""
    codes = ((packed[:, None] >> _SHIFTS) & 3).reshape(-1)
    idx = (scales.astype(np.uint32) << 2) | codes
    if out is None:
        out = np.empty(idx.shape, np.uint16)
    np.take(bias_table(), idx, out=out)
    return out


@dataclass(frozen=True)
class BankLayout:
    """Byte sizes of one expert's pieces (from the checkpoint's ExpertFormat)."""

    weight: int  # per projection
    scales: int  # per projection (bf16)

    @property
    def codes(self) -> int:
        return self.scales // 2 // 4

    @property
    def coded_head(self) -> int:
        return _align(3 * (self.scales + self.codes))

    @property
    def raw_head(self) -> int:
        return _align(6 * self.scales)

    def record(self, kind: str) -> int:
        return (self.coded_head if kind == "coded" else self.raw_head) + 3 * self.weight


def layout_from_sizes(sizes: dict[str, int]) -> BankLayout:
    w = {sizes[f"{p}.weight"] for p in PROJS}
    s = {sizes[f"{p}.scales"] for p in PROJS} | {sizes[f"{p}.biases"] for p in PROJS}
    if len(w) != 1 or len(s) != 1:
        raise ValueError(f"projections differ in size: {sizes}")
    return BankLayout(weight=w.pop(), scales=s.pop())


class CodedBankReader(ExpertReader):
    """ExpertReader that serves the experts a coded bank holds from it, the rest from the checkpoint."""

    def __init__(self, bank_dir, mirror_dir=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bank_dir = Path(bank_dir)
        meta = json.loads((self.bank_dir / "bank.json").read_text())
        if meta.get("version") != BANK_VERSION:
            raise ValueError(f"{self.bank_dir}: bank version {meta.get('version')}, expected {BANK_VERSION}")
        self.layout = BankLayout(**meta["layout"])
        self.records: dict[tuple[int, int], tuple[str, int, str]] = {
            (r[0], r[1]): (r[2], r[3], r[4]) for r in meta["records"]
        }
        self.checkpoint = meta.get("checkpoint")
        self.bank_mirror = Path(mirror_dir) if mirror_dir else None
        if self.bank_mirror is not None and not (self.bank_mirror / "bank.json").is_file():
            print(f"[bank] mirror {self.bank_mirror} has no bank.json; bank mirror off", flush=True)
            self.bank_mirror = None
        self.coded_reads = 0
        bias_table()

    def covers(self, layer: int, expert: int) -> bool:
        return (layer, expert) in self.records

    def read_expert_into(self, entry: ExpertEntry, views) -> int:
        rec = self.records.get((entry.layer, entry.expert)) if ENABLED else None
        if rec is None:
            if not entry.tensors:
                # a trimmed checkpoint (index_from_bank) has no other copy of this expert
                raise LookupError(f"expert {entry.layer}/{entry.expert}: not in {self.bank_dir} and no checkpoint bytes")
            return super().read_expert_into(entry, views)
        return self._read_record(rec, views)

    def _read_record(self, rec, views) -> int:
        fname, offset, kind = rec
        lay = self.layout
        fd = self._fd(self.bank_dir / fname)
        head = lay.coded_head if kind == "coded" else lay.raw_head
        pool = self._piece_executor()
        mfd = None
        if self.bank_mirror is not None and self.mirror_fraction > 0:
            mfile = self.bank_mirror / fname
            if mfile.exists():
                mfd = self._fd(mfile)
        # the three weight pieces go first: they are the long pole
        futures = []
        pos = offset + head
        for p in PROJS:
            buf = memoryview(views[f"{p}.weight"]).cast("B")
            if mfd is not None:
                cut = int(lay.weight * (1.0 - self.mirror_fraction)) // 4096 * 4096
                futures.append((pool.submit(os.preadv, mfd, [buf[cut:]], pos + cut), lay.weight - cut, True, fd, buf[cut:], pos + cut))
                futures.append((pool.submit(os.preadv, fd, [buf[:cut]], pos), cut, False, fd, None, 0))
            else:
                futures.append((pool.submit(os.preadv, fd, [buf], pos), lay.weight, False, fd, None, 0))
            pos += lay.weight
        scales = [np.asarray(views[f"{p}.scales"]).view(np.uint16) for p in PROJS]
        biases = [np.asarray(views[f"{p}.biases"]).view(np.uint16) for p in PROJS]
        if kind == "coded":
            codes = bytearray(3 * lay.codes)
            bufs = [memoryview(views[f"{p}.scales"]).cast("B") for p in PROJS] + [memoryview(codes)]
            got = os.preadv(fd, bufs, offset)
            if got != 3 * (lay.scales + lay.codes):
                raise OSError(f"short head read from {fname}@{offset}: {got}")
            packed = np.frombuffer(codes, np.uint8)
            n = lay.codes
            rebuilds = [pool.submit(decode_biases, scales[i], packed[i * n:(i + 1) * n], biases[i]) for i in (1, 2)]
            decode_biases(scales[0], packed[:n], biases[0])
            for f in rebuilds:
                f.result()
            total = got
        else:
            bufs = [memoryview(views[f"{p}.scales"]).cast("B") for p in PROJS] + [
                memoryview(views[f"{p}.biases"]).cast("B") for p in PROJS
            ]
            got = os.preadv(fd, bufs, offset)
            if got != 6 * lay.scales:
                raise OSError(f"short head read from {fname}@{offset}: {got}")
            total = got
        for future, size, from_mirror, pfd, pbuf, poff in futures:
            try:
                got = future.result()
            except OSError as exc:
                if not from_mirror:
                    raise
                print(f"[bank] mirror read failed ({exc}); bank mirror off", flush=True)
                self.bank_mirror = None
                got = os.preadv(pfd, [pbuf], poff)
            if got != size:
                raise OSError(f"short weight read from {fname}@{offset}: {got} of {size}")
            total += got
        self.coded_reads += 1
        return total


def format_to_json(fmt: ExpertFormat) -> dict:
    return {"kind": fmt.kind, "bits": fmt.bits, "group_size": fmt.group_size,
            "tensor_names": list(fmt.tensor_names), "shapes": {k: list(v) for k, v in fmt.shapes.items()},
            "dtypes": dict(fmt.dtypes)}


def index_from_bank(bank_dir) -> tuple[ExpertFormat, dict[tuple[int, int], ExpertEntry]]:
    """The expert format and index from a complete bank alone, for a checkpoint trimmed to its non-expert
    weights: the entries carry no checkpoint byte ranges (every read is served by the bank)."""
    meta = json.loads((Path(bank_dir) / "bank.json").read_text())
    if "format" not in meta:
        raise ValueError(f"{bank_dir}/bank.json has no expert format")
    f = meta["format"]
    fmt = ExpertFormat(kind=f["kind"], bits=int(f["bits"]), group_size=int(f["group_size"]),
                       tensor_names=tuple(f["tensor_names"]), shapes={k: tuple(v) for k, v in f["shapes"].items()},
                       dtypes=dict(f["dtypes"]))
    index = {(r[0], r[1]): ExpertEntry(layer=r[0], expert=r[1], tensors=()) for r in meta["records"]}
    return fmt, index


def reader_from_env(**kwargs) -> ExpertReader:
    """The MiniMax store's reader: the coded bank when CACHALOT_MINIMAX_BANK names one, else the checkpoint's."""
    bank = os.environ.get("CACHALOT_MINIMAX_BANK")
    if not bank:
        return ExpertReader(**kwargs)
    return CodedBankReader(bank, os.environ.get("CACHALOT_MINIMAX_BANK_MIRROR") or None, **kwargs)
