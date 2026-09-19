"""Scattered Engram row reads: shipped FP8 table on the USB drive against the
affine 3-bit table inside the oQ3e bank on the internal SSD.

Pure I/O. No model is loaded, no GPU work is done and nothing is wired, so this
is safe to run outside the memory guardian.

A 512-token prefill asks each of the two Engram layers for n_hash_cols (24) rows
per token, deduplicated. The default row count here reproduces that shape: the
ids are unique and sorted, exactly as np.unique leaves them in the runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

FP4_PATH = Path("/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash")
OQ3E_PATH = Path("/Users/hamedprooshani/DeepSeek-V4.1-Flash-oQ3e-mtp")

FP4_SHARDS = {1: "model-00047-of-00048.safetensors", 14: "model-00048-of-00048.safetensors"}


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


@dataclass(frozen=True)
class Region:
    name: str
    start: int
    row_bytes: int


@dataclass(frozen=True)
class Table:
    label: str
    layer: int
    path: Path
    rows: int
    regions: tuple[Region, ...]

    @property
    def row_payload_bytes(self) -> int:
        return sum(region.row_bytes for region in self.regions)


def fp4_table(layer: int) -> Table:
    path = FP4_PATH / FP4_SHARDS[layer]
    header, data_start = read_header(path)
    weight = header[f"layers.{layer}.engram.embed.weight"]
    scale = header[f"layers.{layer}.engram.embed.scale"]
    rows, head_dim = (int(x) for x in weight["shape"])
    scale_count = int(scale["shape"][1])
    return Table(
        label="fp8-e4m3 / USB",
        layer=layer,
        path=path,
        rows=rows,
        regions=(
            Region("weight", data_start + int(weight["data_offsets"][0]), head_dim),
            Region("scale", data_start + int(scale["data_offsets"][0]), scale_count),
        ),
    )


def oq3e_table(layer: int) -> Table:
    config = json.loads((OQ3E_PATH / "config.json").read_text())
    entry = config["omlx_deepseek_v41"]["engram_tables"][f"language_model.layers.{layer}.engram.embed"]
    path = OQ3E_PATH / entry["weight_file"]
    header, data_start = read_header(path)
    weight = header[entry["weight_key"]]
    scales = header[entry["scale_key"]]
    biases = header[entry["bias_key"]]
    rows = int(weight["shape"][0])
    return Table(
        label="affine 3-bit / internal",
        layer=layer,
        path=path,
        rows=rows,
        regions=(
            Region("weight", data_start + int(weight["data_offsets"][0]), int(weight["shape"][1]) * 4),
            Region("scales", data_start + int(scales["data_offsets"][0]), int(scales["shape"][1]) * 2),
            Region("biases", data_start + int(biases["data_offsets"][0]), int(biases["shape"][1]) * 2),
        ),
    )


def read_rows(table: Table, ids: np.ndarray, workers: int, page_cache: bool) -> float:
    fd = os.open(table.path, os.O_RDONLY)
    if not page_cache:
        import fcntl

        fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
    try:
        buffers = [bytearray(ids.size * region.row_bytes) for region in table.regions]
        views = [memoryview(buffer) for buffer in buffers]

        def fetch(lo: int, hi: int) -> None:
            for i in range(lo, hi):
                row = int(ids[i])
                for region, view in zip(table.regions, views, strict=False):
                    width = region.row_bytes
                    os.preadv(fd, [view[i * width : (i + 1) * width]], region.start + row * width)

        start = perf_counter()
        n = int(ids.size)
        if workers > 1 and n >= 64:
            from concurrent.futures import ThreadPoolExecutor

            step = (n + workers - 1) // workers
            with ThreadPoolExecutor(workers, thread_name_prefix="engram-bench") as pool:
                list(pool.map(lambda lo: fetch(lo, min(lo + step, n)), range(0, n, step)))
        else:
            fetch(0, n)
        return perf_counter() - start
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=12288, help="unique rows per layer, as a 512-token prefill asks for")
    parser.add_argument("--workers", type=int, default=16, help="matches EngramRowReader.WORKERS")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 14])
    parser.add_argument("--no-page-cache", action="store_true", help="set F_NOCACHE, as the runtime does when CACHALOT_PAGE_CACHE is unset")
    parser.add_argument("--warm", action="store_true", help="reuse one set of ids across repeats, measuring the page cache rather than the drive")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    page_cache = not args.no_page_cache

    print(f"rows per layer {args.rows} | workers {args.workers} | repeats {args.repeats} | page cache {'on' if page_cache else 'off'}")

    totals: dict[str, list[float]] = {}

    for layer in args.layers:
        tables = [fp4_table(layer), oq3e_table(layer)]
        if len({table.rows for table in tables}) != 1:
            raise SystemExit(f"layer {layer}: row counts differ between tables")
        rows = tables[0].rows

        # Fresh ids per repeat unless --warm: a real prompt asks for rows this
        # process has not touched, so reusing one id set would time the page
        # cache instead of the drive.
        id_sets = [np.unique(rng.integers(0, rows, size=args.rows, dtype=np.int64)) for _ in range(1 if args.warm else args.repeats)]
        print(f"\nlayer {layer}: {rows:,} rows, sampling {id_sets[0].size:,} unique ids per repeat")

        for table in tables:
            times = []
            for repeat in range(args.repeats):
                ids = id_sets[0] if args.warm else id_sets[repeat]
                times.append(read_rows(table, ids, args.workers, page_cache))
            ids = id_sets[0]
            payload = ids.size * table.row_payload_bytes
            median = sorted(times)[len(times) // 2]
            totals.setdefault(table.label, []).append(median)
            per_repeat = " ".join(f"{t * 1e3:.1f}" for t in times)
            print(
                f"  {table.label:<24} {table.row_payload_bytes:>4} B/row  "
                f"{payload / 1e6:7.2f} MB  "
                f"median {median * 1e3:8.1f} ms  [{per_repeat}]  "
                f"{ids.size / median / 1e3:7.1f} krow/s  {payload / median / 1e6:7.1f} MB/s"
            )

    if len(totals) == 2:
        (label_a, times_a), (label_b, times_b) = totals.items()
        print(f"\nper-prefill total, median of {args.repeats}, both Engram layers:")
        print(f"  {label_a:<24} {sum(times_a) * 1e3:8.1f} ms")
        print(f"  {label_b:<24} {sum(times_b) * 1e3:8.1f} ms")
        print(f"  speedup {sum(times_a) / sum(times_b):.2f}x")


if __name__ == "__main__":
    main()
