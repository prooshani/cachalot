"""
Write, check or scan a bias-free MiniMax-M3 expert bank (cachalot.minimax.coded_bank, HANDOFF 18.4).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    # every expert group's bias against (k * scale); prints the k histogram (~40 s, reads 24 GiB)
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_coded_bank.py --scan MODEL
    # write the bank for MoE layers 3-42 (or all with no --layers), then check 64 random experts byte for byte
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_coded_bank.py --write MODEL OUT [--layers 3-42]
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_coded_bank.py --verify MODEL OUT [--n 64]

Records are written in expert order, one file per layer; a layer file is written to `.part` and renamed, and
`bank.json` is rewritten after every layer, so an interrupted run leaves a usable bank of the layers done.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cachalot.glm.experts import tensor_sizes
from cachalot.minimax.coded_bank import (
    BANK_VERSION,
    PROJS,
    CodedBankReader,
    encode_biases,
    format_to_json,
    layout_from_sizes,
)
from cachalot.minimax.experts import build_minimax_expert_index
from cachalot.storage.reader import ExpertReader


def read_pieces(reader: ExpertReader, entry, sizes) -> dict[str, np.ndarray]:
    views = {k: np.empty(v, np.uint8) for k, v in sizes.items()}
    reader.read_expert_into(entry, views)
    return views


def build_record(views, lay) -> tuple[str, bytes]:
    scales = [views[f"{p}.scales"].view(np.uint16) for p in PROJS]
    biases = [views[f"{p}.biases"].view(np.uint16) for p in PROJS]
    codes = [encode_biases(s, b) for s, b in zip(scales, biases, strict=True)]
    if all(c is not None for c in codes):
        head = b"".join(s.tobytes() for s in scales) + b"".join(c.tobytes() for c in codes)
        kind, head_size = "coded", lay.coded_head
    else:
        head = b"".join(s.tobytes() for s in scales) + b"".join(b.tobytes() for b in biases)
        kind, head_size = "raw", lay.raw_head
    head += b"\0" * (head_size - len(head))
    body = b"".join(views[f"{p}.weight"].tobytes() for p in PROJS)
    return kind, head + body


def write(model, out, layers):
    fmt, index = build_minimax_expert_index(model)
    sizes = tensor_sizes(fmt)
    lay = layout_from_sizes(sizes)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "bank.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "version": BANK_VERSION,
        "checkpoint": str(Path(model).resolve()),
        "layout": {"weight": lay.weight, "scales": lay.scales},
        "records": [],
    }
    meta["format"] = format_to_json(fmt)
    done = {r[0] for r in meta["records"]}
    all_layers = sorted({k[0] for k in index})
    todo = [L for L in all_layers if (layers is None or L in layers) and L not in done]
    reader = ExpertReader(bypass_page_cache=True)
    pool = ThreadPoolExecutor(8)
    n_exp = max(k[1] for k in index) + 1
    for L in todo:
        t0 = time.perf_counter()
        fname = f"layer-{L:03d}.bin"
        part = out / (fname + ".part")
        fd = os.open(part, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        offset, recs, raw = 0, [], 0

        def job(e, L=L):
            return build_record(read_pieces(reader, index[(L, e)], sizes), lay)

        for e, (kind, data) in enumerate(pool.map(job, range(n_exp))):
            os.pwrite(fd, data, offset)
            recs.append([L, e, fname, offset, kind])
            raw += kind == "raw"
            offset += len(data)
        os.fsync(fd)
        os.close(fd)
        os.replace(part, out / fname)
        meta["records"] += recs
        tmp = out / "bank.json.part"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, meta_path)
        dt = time.perf_counter() - t0
        print(f"layer {L}: {offset / 2**30:.2f} GiB, {raw} raw of {n_exp}, {dt:.1f} s", flush=True)
    reader.close()


def verify(model, out, n):
    fmt, index = build_minimax_expert_index(model)
    sizes = tensor_sizes(fmt)
    bank = CodedBankReader(out, bypass_page_cache=True)
    plain = ExpertReader(bypass_page_cache=True)
    keys = sorted(bank.records)
    random.seed(int(os.environ.get("SEED", "5")))
    sample = random.sample(keys, min(n, len(keys)))
    raw = [k for k in keys if bank.records[k][2] == "raw"]
    sample += random.sample(raw, min(4, len(raw)))
    bad = 0
    for key in sample:
        a = {k: np.full(v, 0xAB, np.uint8) for k, v in sizes.items()}
        b = {k: np.empty(v, np.uint8) for k, v in sizes.items()}
        bank.read_expert_into(index[key], a)
        plain.read_expert_into(index[key], b)
        same = all(np.array_equal(a[k], b[k]) for k in sizes)
        bad += not same
        if not same:
            print("MISMATCH", key, bank.records[key], [k for k in sizes if not np.array_equal(a[k], b[k])])
    print(f"VERIFY {len(sample)} experts ({sum(bank.records[k][2] == 'raw' for k in sample)} raw) mismatched {bad}; "
          f"bank covers {len(keys)} experts, {len(raw)} raw")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("model")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--layers", default=None, help="a-b, inclusive")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    layers = None
    if a.layers:
        lo, hi = (int(x) for x in a.layers.split("-"))
        layers = set(range(lo, hi + 1))
    if a.scan:
        scan(a.model)
    if a.write:
        write(a.model, a.out, layers)
    if a.verify or a.write:
        sys.exit(0 if verify(a.model, a.out, a.n) else 1)


def scan(model):
    """Every routed-expert group's bias against bf16(k * scale), k = round(bias / scale); the k histogram."""
    fmt, index = build_minimax_expert_index(model)
    reader = ExpertReader(bypass_page_cache=True)
    hist, bad, total, t0 = {}, 0, 0, time.perf_counter()
    for L in sorted({k[0] for k in index}):
        for key in [k for k in sorted(index) if k[0] == L]:
            for p in PROJS:
                ranges = {t.name.rsplit(".", 2)[-2] + "." + t.name.rsplit(".", 1)[-1]: t for t in index[key].tensors}
                s_t, b_t = ranges[f"{p}.scales"], ranges[f"{p}.biases"]
                fd = reader._fd(s_t.shard)
                s = np.frombuffer(os.pread(fd, s_t.size, s_t.start), np.uint16)
                b = np.frombuffer(os.pread(reader._fd(b_t.shard), b_t.size, b_t.start), np.uint16)
                sf, bf = (s.astype(np.uint32) << 16).view(np.float32), (b.astype(np.uint32) << 16).view(np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    k = np.where(sf == 0, 0, np.round(bf / sf))
                from cachalot.minimax.coded_bank import bf16_bits_rne

                ok = (bf16_bits_rne((k * sf).astype(np.float32)) == b) | ((sf == 0) & (bf == 0))
                bad += int((~ok).sum())
                total += b.size
                for v, c in zip(*np.unique(k, return_counts=True), strict=True):
                    hist[int(v)] = hist.get(int(v), 0) + int(c)
    print(f"SCAN groups {total} mismatched {bad} k {hist} {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
