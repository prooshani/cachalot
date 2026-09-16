"""How fast the expert reader actually delivers experts, without the model.

decode_anatomy.py measured 6.06 ms per miss inside the runtime against the
2.8 ms a single nine-piece expert read was measured to take. This probe reads
random experts through the real ExpertReader into plain bytearrays, so the only
things in the picture are the drive, the reader's piece pool and the loader
threads. Nothing is wired and no GPU work happens, so it is safe outside the
guardian.

The runtime runs `--loaders 8` (config.io_workers) against a piece pool of 16
threads shared by every load, so eight concurrent experts put 64 queued piece
reads behind 16 threads. Sweeping loaders and the piece-pool size shows whether
the per-expert latency in the runtime is the drive or that queue.
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from cachalot.storage.index import detect_expert_bank
from cachalot.storage.reader import ExpertReader

BANK = Path("/Users/hamedprooshani/DeepSeek-V4.1-Flash-oQ3e-mtp")


def make_views(fmt, entry) -> dict[str, bytearray]:
    views = {}
    for tensor in entry.tensors:
        short = ".".join(tensor.name.rsplit(".", 2)[-2:])
        views[short] = bytearray(tensor.size)
    return views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default=str(BANK))
    parser.add_argument("--loaders", default="1,2,4,8,16", help="concurrent expert loads, runtime uses io_workers=8")
    parser.add_argument("--piece-pool", type=int, default=0, help="override ExpertReader piece-pool size, 0 keeps the shipped 16")
    parser.add_argument("--experts", type=int, default=256, help="experts read per arm")
    parser.add_argument("--page-cache", action="store_true", help="leave F_NOCACHE off, as CACHALOT_PAGE_CACHE=1 does")
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    fmt, index = detect_expert_bank(args.bank)
    keys = sorted(index)
    expert_bytes = sum(t.size for t in index[keys[0]].tensors)
    pieces = len(index[keys[0]].tensors)
    print(
        f"bank {args.bank}\nformat {fmt.kind} {fmt.bits}-bit | {len(keys):,} experts | "
        f"{expert_bytes / 2**20:.2f} MiB and {pieces} tensors per expert | "
        f"page cache {'on' if args.page_cache else 'off (F_NOCACHE)'}"
    )
    print("\n| loaders | experts | wall s | ms/expert | GB/s |")
    print("|---:|---:|---:|---:|---:|")

    rng = random.Random(args.seed)

    for loaders in (int(x) for x in args.loaders.split(",")):
        reader = ExpertReader(bypass_page_cache=not args.page_cache)
        if args.piece_pool:
            reader._piece_pool = ThreadPoolExecutor(args.piece_pool, thread_name_prefix="expert-pieces")
        try:
            entries = [index[keys[rng.randrange(len(keys))]] for _ in range(args.experts)]
            buffers = [make_views(fmt, entry) for entry in entries[:loaders]]

            def read(i: int) -> int:
                # Reuse one buffer set per loader slot: allocation is not what
                # is being measured and the runtime reuses wired slots anyway.
                return reader.read_expert_into(entries[i], buffers[i % loaders])

            start = perf_counter()
            if loaders == 1:
                total = sum(read(i) for i in range(len(entries)))
            else:
                with ThreadPoolExecutor(loaders, thread_name_prefix="expert-load") as pool:
                    total = sum(pool.map(read, range(len(entries))))
            wall = perf_counter() - start
            print(
                f"| {loaders} | {len(entries)} | {wall:.2f} | "
                f"{wall / len(entries) * 1e3:.2f} | {total / wall / 1e9:.2f} |",
                flush=True,
            )
        finally:
            reader.close()


if __name__ == "__main__":
    main()
