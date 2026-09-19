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

--wire-gib is why this probe is no longer safe outside the guardian. Every
earlier measurement here ran on an idle machine with nothing wired, while the
runtime holds around 57 GiB wired at a 36 GiB expert budget. Wired pages can be
neither compressed nor evicted, so with the page cache enabled the kernel must
reclaim a page for every page it reads, and that cost simply cannot appear on
an empty machine. --wire-gib N allocates and wires N GiB of MLX buffers before
reading so the drive is measured under the memory conditions decode actually
runs in. Run it under benchmarks/guarded_run.sh whenever N is large.

Note that `ms/expert` in the table below is amortized wall clock per expert,
total time divided by expert count, not the latency of an individual read: at
8 concurrent loaders an individual read takes roughly eight times the figure
shown. Comparing it to a per-miss latency measured inside the runtime is a
units error.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from cachalot.cache.slots import ExpertSlotPool
from cachalot.storage.index import detect_expert_bank
from cachalot.storage.reader import ExpertReader

BANK = Path("/Users/hamedprooshani/DeepSeek-V4.1-Flash-oQ3e-mtp")


def wired_gib() -> float:
    """Wired pages as macOS reports them, which is what the page cache competes with."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = 4096
    wired = 0
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        elif line.startswith("Pages wired down"):
            wired = int(line.split(":")[1].strip().rstrip("."))
    return wired * page / 2**30


def allocate_ballast(gib: float):
    """
    Hold `gib` GiB of wired MLX buffers, reproducing the runtime's memory
    conditions so the page cache has to reclaim to serve a read.

    The buffers are returned and must stay referenced for the whole run; MLX
    wires them on evaluation up to the wired limit set here.

    A heartbeat thread is required, not optional. Within about six seconds of
    an idle Metal queue macOS un-wires the whole working set despite
    set_wired_limit (docs/HANDOFF-2026-09-16.md, commit 83015d4), and the first
    version of this ballast showed exactly that: the second arm of every wired
    run reported the system down to 3-29 GiB wired, so it measured less memory
    pressure than intended. TextDecodeRuntime keeps its own working set wired
    the same way.
    """
    import threading

    import mlx.core as mx

    before = wired_gib()
    mx.set_wired_limit(int((gib + 4) * 2**30))
    chunk_bytes = 2**30
    elements = chunk_bytes // 2  # bfloat16
    blocks = []
    for _ in range(int(gib)):
        block = mx.zeros((elements,), dtype=mx.bfloat16)
        mx.eval(block)
        blocks.append(block)

    beat = mx.zeros((1,), dtype=mx.float32)

    def heartbeat() -> None:
        while True:
            mx.eval(beat + 1)
            time.sleep(0.5)

    threading.Thread(target=heartbeat, daemon=True, name="ballast-heartbeat").start()

    after = wired_gib()
    print(
        f"ballast: asked for {gib:.0f} GiB, system wired {before:.1f} -> {after:.1f} GiB "
        f"(+{after - before:.1f}), heartbeat on",
        flush=True,
    )
    return blocks


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
    parser.add_argument(
        "--mlx-slots",
        action="store_true",
        help="read into wired MLX slot buffers, the runtime's destination, instead of bytearrays",
    )
    parser.add_argument("--wired-gib", type=float, default=2.0, help="wired limit for --mlx-slots, a few slots only")
    parser.add_argument(
        "--gpu-load",
        action="store_true",
        help="run MLX work on another thread while reading, as decode does, to expose GIL and CPU contention",
    )
    parser.add_argument(
        "--wire-gib",
        type=float,
        default=0.0,
        help="wire this many GiB of MLX buffers before reading, so the page cache competes "
             "for memory the way it does under the runtime (~57 GiB at a 36 GiB expert budget); "
             "run under guarded_run.sh when this is large",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--expert-offset",
        type=int,
        default=0,
        help="start this far into the shuffled expert order. Arms must read experts nothing "
             "has read yet: F_NOCACHE does NOT reliably keep them out of the page cache "
             "(measured 2026-09-17: a repeat read of the same experts goes 5.55 -> 9.07 GB/s "
             "with F_NOCACHE set, and 21.5 GB/s with the page cache on), so two arms sharing a "
             "seed measure RAM, not the drive. Each arm consumes --experts entries and the next "
             "arm continues after it, so give every process a disjoint offset.",
    )
    args = parser.parse_args()

    ballast = allocate_ballast(args.wire_gib) if args.wire_gib > 0 else None

    fmt, index = detect_expert_bank(args.bank)
    keys = sorted(index)
    expert_bytes = sum(t.size for t in index[keys[0]].tensors)
    pieces = len(index[keys[0]].tensors)
    print(
        f"bank {args.bank}\nformat {fmt.kind} {fmt.bits}-bit | {len(keys):,} experts | "
        f"{expert_bytes / 2**20:.2f} MiB and {pieces} tensors per expert | "
        f"page cache {'on' if args.page_cache else 'off (F_NOCACHE)'} | "
        f"experts from offset {args.expert_offset} of a seed-{args.seed} shuffle | "
        f"system wired {wired_gib():.1f} GiB"
    )
    print("\n| loaders | experts | wall s | ms/expert | GB/s | wired GiB during arm |")
    print("|---:|---:|---:|---:|---:|---:|")

    # one fixed shuffled order, sliced disjointly, so no arm re-reads another's experts
    order = list(keys)
    random.Random(args.seed).shuffle(order)
    cursor = args.expert_offset

    pool = None
    if args.mlx_slots:
        import mlx.core as mx

        mx.set_wired_limit(int(args.wired_gib * 2**30))
        sizes = {
            ".".join(t.name.rsplit(".", 2)[-2:]): t.size for t in index[keys[0]].tensors
        }
        pool = ExpertSlotPool(sizes, max(int(x) for x in args.loaders.split(",")))
        print(f"destination: {pool.capacity} wired MLX slots of {pool.slot_bytes / 2**20:.2f} MiB")
    else:
        print("destination: plain bytearrays")

    stop_gpu = False

    def gpu_worker() -> None:
        import mlx.core as mx

        a = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
        b = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
        mx.eval(a, b)
        while not stop_gpu:
            mx.eval((a @ b).sum())

    gpu_thread = None
    if args.gpu_load:
        import threading

        gpu_thread = threading.Thread(target=gpu_worker, daemon=True, name="gpu-load")
        gpu_thread.start()
        print("background MLX work: on")

    for loaders in (int(x) for x in args.loaders.split(",")):
        reader = ExpertReader(bypass_page_cache=not args.page_cache)
        if args.piece_pool:
            reader._piece_pool = ThreadPoolExecutor(args.piece_pool, thread_name_prefix="expert-pieces")
        try:
            if cursor + args.experts > len(order):
                raise SystemExit(
                    f"expert order exhausted: offset {cursor} + {args.experts} > {len(order)}; "
                    f"lower --experts or --expert-offset"
                )
            entries = [index[k] for k in order[cursor:cursor + args.experts]]
            cursor += args.experts
            if pool is not None:
                slots = [pool.slot(i) if hasattr(pool, "slot") else pool._slots[i] for i in range(loaders)]
                buffers = [slot.views for slot in slots]
            else:
                buffers = [make_views(fmt, entry) for entry in entries[:loaders]]

            def read(i: int, reader=reader, entries=entries,
                     buffers=buffers, loaders=loaders) -> int:
                # Reuse one buffer set per loader slot: allocation is not what
                # is being measured and the runtime reuses wired slots anyway.
                # Everything the closure reads is bound as a default, so an arm
                # cannot pick up the next arm's reader or buffers.
                return reader.read_expert_into(entries[i], buffers[i % loaders])

            wired_before = wired_gib()
            start = perf_counter()
            if loaders == 1:
                total = sum(read(i) for i in range(len(entries)))
            else:
                # not `pool`: that name holds the MLX slot pool for --mlx-slots,
                # and shadowing it here broke every arm after the second one
                with ThreadPoolExecutor(loaders, thread_name_prefix="expert-load") as loader_pool:
                    total = sum(loader_pool.map(read, range(len(entries))))
            wall = perf_counter() - start
            print(
                f"| {loaders} | {len(entries)} | {wall:.2f} | "
                f"{wall / len(entries) * 1e3:.2f} | {total / wall / 1e9:.2f} | "
                f"{wired_before:.1f} -> {wired_gib():.1f} |",
                flush=True,
            )
        finally:
            reader.close()

    stop_gpu = True
    if gpu_thread is not None:
        gpu_thread.join(timeout=5.0)
    if ballast is not None:
        # referenced here so nothing frees the wired set mid-run
        print(f"ballast held: {len(ballast)} blocks, system wired {wired_gib():.1f} GiB")


if __name__ == "__main__":
    main()
