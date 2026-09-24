"""
Sample the machine beside a running `serve.sh`, to catch the slow-decode window (HANDOFF sections 15.5, 15.7).

Decode alternates between ~8 and ~4 tok/s at the same misses per token. This records, every --every seconds,
what could make a miss cost twice as much: how much of the expert bank the page cache holds (mincore over
the shards, every --bank-every seconds), wired / file-backed / compressor
memory, swap, pageins, GPU utilization and display power, the busiest processes, and the server's own
counters from /v1/stats: expert reads, the share fast enough to have come from the page cache, and the mean
read time.

Standard library only; run it in a second terminal while the server is up:

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    /usr/bin/python3 benchmarks/slow_window_sampler.py

Ctrl-C stops it. Each sample is one CSV row in benchmarks/results/slow_window/ and one line on stdout.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import glob
import json
import mmap
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

GiB = 1024**3
REPO = Path(__file__).resolve().parent.parent
DEFAULT_BANK = "/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128"

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def bank_resident_gib(bank: str) -> float:
    """GiB of the bank's shards currently in the page cache. Mapping a file and
    asking mincore touches no page, so this does not warm anything."""
    page = os.sysconf("SC_PAGE_SIZE")
    resident = 0
    for path in sorted(glob.glob(os.path.join(bank, "*.safetensors"))):
        size = os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)
        try:
            addr = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
            if addr in (None, ctypes.c_void_p(-1).value):
                continue
            pages = (size + page - 1) // page
            vec = ctypes.create_string_buffer(pages)
            if _libc.mincore(addr, size, vec) == 0:
                # bit 0 is "resident"; the other flag bits are only ever set on
                # resident pages, so counting non-zero bytes is exact and runs in C
                raw = vec.raw
                resident += len(raw) - raw.count(0)
            _libc.munmap(addr, size)
        finally:
            os.close(fd)
    return resident * page / GiB


def vm() -> dict[str, float]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    n = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(.+?):\s+(\d+)\.", out, re.M)}
    return {
        "free_gib": n.get("Pages free", 0) * page / GiB,
        "wired_gib": n.get("Pages wired down", 0) * page / GiB,
        "file_gib": n.get("File-backed pages", 0) * page / GiB,
        "anon_gib": n.get("Anonymous pages", 0) * page / GiB,
        "compressor_gib": n.get("Pages occupied by compressor", 0) * page / GiB,
        "pageins": n.get("Pageins", 0),
        "decompressions": n.get("Decompressions", 0),
    }


def swap_used_mib() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)([MG])", out)
    if not m:
        return 0.0
    return float(m.group(1)) * (1024 if m.group(2) == "G" else 1)


def top_cpu(n: int = 4) -> str:
    out = subprocess.run(["ps", "-Ao", "pcpu=,comm="], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            rows.append((float(parts[0]), os.path.basename(parts[1])[:24]))
    rows.sort(reverse=True)
    return "; ".join(f"{name} {cpu:.0f}%" for cpu, name in rows[:n])


def gpu_util() -> int | None:
    """GPU 'Device Utilization %' from the IOAccelerator registry entry: with no runtime running, this is
    what window compositing and video wallpapers take (21-41 % on a three-display desk, section 15.7)."""
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
    m = re.search(r'"Device Utilization %"=(\d+)', out)
    return int(m.group(1)) if m else None


def display_on() -> bool | None:
    """Last display power event in the power log: True after 'turned on', False after 'turned off'."""
    out = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True).stdout
    events = re.findall(r"Display is turned (on|off)", out)
    return (events[-1] == "on") if events else None


def server_stats(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return json.load(r)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=float, default=10.0)
    ap.add_argument("--bank-every", type=float, default=60.0)
    ap.add_argument("--bank", default=os.environ.get("CACHALOT_EXPERT_BANK", DEFAULT_BANK))
    ap.add_argument("--stats-url", default="http://127.0.0.1:8011/v1/stats")
    ap.add_argument("--samples", type=int, default=0, help="stop after N samples (0 = until Ctrl-C)")
    args = ap.parse_args()

    out_dir = REPO / "benchmarks/results/slow_window"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"sampler_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    fields = ["time", "tokens", "tok_s", "reads", "fast_pct", "read_ms", "gpu_pct", "display", "bank_pc_gib",
              "free_gib", "wired_gib",
              "file_gib", "anon_gib", "compressor_gib", "swap_mib", "pageins", "decompressions", "top_cpu"]
    print(f"writing {out}", file=sys.stderr)
    prev_stats, prev_vm, prev_t = None, None, None
    bank_gib, bank_t = float("nan"), 0.0
    taken = 0
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        try:
            while True:
                t = time.time()
                if t - bank_t >= args.bank_every:
                    bank_gib, bank_t = bank_resident_gib(args.bank), t
                v, st = vm(), server_stats(args.stats_url)
                row = {"time": time.strftime("%H:%M:%S"), "bank_pc_gib": round(bank_gib, 2),
                       "swap_mib": round(swap_used_mib()), "top_cpu": top_cpu(), "gpu_pct": gpu_util(),
                       "display": {True: "on", False: "off"}.get(display_on(), "")}
                row.update({k: round(x, 2) for k, x in v.items() if k.endswith("_gib")})
                row["pageins"] = v["pageins"] - prev_vm["pageins"] if prev_vm else 0
                row["decompressions"] = v["decompressions"] - prev_vm["decompressions"] if prev_vm else 0
                if st and prev_stats:
                    dtok = st.get("tokens_generated", 0) - prev_stats.get("tokens_generated", 0)
                    dreads = st.get("expert_reads", 0) - prev_stats.get("expert_reads", 0)
                    dfast = st.get("expert_fast_reads", 0) - prev_stats.get("expert_fast_reads", 0)
                    dsec = st.get("expert_read_seconds", 0.0) - prev_stats.get("expert_read_seconds", 0.0)
                    row.update(tokens=dtok, tok_s=round(dtok / (t - prev_t), 2), reads=dreads,
                               fast_pct=round(100 * dfast / dreads, 1) if dreads else "",
                               read_ms=round(1000 * dsec / dreads, 2) if dreads else "")
                w.writerow(row)
                fh.flush()
                print(" ".join(f"{k}={row.get(k, '')}" for k in fields), flush=True)
                prev_stats, prev_vm, prev_t = st, v, t
                taken += 1
                if args.samples and taken >= args.samples:
                    break
                time.sleep(max(0.0, args.every - (time.time() - t)))
        except KeyboardInterrupt:
            pass
    print(f"csv: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
