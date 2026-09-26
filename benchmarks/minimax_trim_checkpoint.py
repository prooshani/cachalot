"""
Copy a MiniMax-M3 checkpoint without its stacked routed-expert tensors (HANDOFF 18.4), for serving every expert
from a complete bias-free bank (cachalot.minimax.coded_bank). Each shard that holds anything else is rewritten
under the same name with only those tensors, byte for byte; config, tokenizer and index files are copied.

    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_trim_checkpoint.py SRC DST
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

from cachalot.storage.index import read_safetensors_header


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=False)
    kept = total = 0
    for f in sorted(src.iterdir()):
        if f.is_dir():
            continue
        if not (f.name.startswith("model-") and f.suffix == ".safetensors"):
            shutil.copy2(f, dst / f.name)
            continue
        header, data_start = read_safetensors_header(f)
        names = [n for n in header if n != "__metadata__" and ".switch_mlp." not in n]
        if not names:
            continue
        out, offset = {}, 0
        if "__metadata__" in header:
            out["__metadata__"] = header["__metadata__"]
        for n in names:
            a, b = header[n]["data_offsets"]
            out[n] = {"dtype": header[n]["dtype"], "shape": header[n]["shape"], "data_offsets": [offset, offset + b - a]}
            offset += b - a
        blob = json.dumps(out, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % 8)
        with open(f, "rb") as fin, open(dst / f.name, "wb") as fout:
            fout.write(struct.pack("<Q", len(blob)) + blob)
            for n in names:
                a, b = header[n]["data_offsets"]
                fin.seek(data_start + a)
                fout.write(fin.read(b - a))
        kept += len(names)
        total += offset
    print(f"TRIM {kept} tensors, {total / 2**30:.2f} GiB into {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
