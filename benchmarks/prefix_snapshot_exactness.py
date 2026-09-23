"""Snapshot -> restore must reproduce the sequence exactly (HANDOFF section 15.3).

Snapshots keep only the written rows of the position-indexed caches and
restore() pads the zeros back. This proves that on the real bank: prefill a
prompt of real prose (chunked, as the server does), snapshot, teacher-force K
tokens and record every logit vector; restore the snapshot, force the same K
tokens again, and require bit-identical logits. Also reports snapshot size.

    <serve.sh's env> PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \\
        benchmarks/prefix_snapshot_exactness.py 3000 9000
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

from cachalot.model import generation
from cachalot.model import text_decode_runtime as tdr
from cachalot.model.api import V41Model

GiB = 1 << 30
ROOT = Path(__file__).resolve().parent.parent
K = 12


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [3000]
    model = V41Model.from_pretrained(
        str(tdr.DEFAULT_CONFIG.resolved_model_path), max_seq_len=65536, expert_cache_budget_bytes=52 * GiB
    )
    rt = model.runtime
    text = list(rt.tokenizer.encode((ROOT / "docs" / "HANDOFF.md").read_text()))
    ok_all = True
    for n in sizes:
        ids = (text * (1 + (n + K) // len(text)))[: n + K]
        prompt, cont = ids[:n], ids[n:]
        rt.reset()
        res, _ = generation.prepare_prompt(rt, prompt, use_prefix_cache=False)
        snap = rt.snapshot(logits=res.logits)
        first = []
        for t in cont:
            r = rt.decode_token(t)
            first.append(r.logits.astype(mx.float32))
        mx.eval(*first)
        rt.reset()  # scramble: the restore must not lean on live state
        rt.restore(snap)
        again = []
        for t in cont:
            r = rt.decode_token(t)
            again.append(r.logits.astype(mx.float32))
        mx.eval(*again)
        same = all(bool(mx.array_equal(a, b).item()) for a, b in zip(first, again, strict=True))
        worst = max(mx.max(mx.abs(a - b)).item() for a, b in zip(first, again, strict=True))
        ok_all &= same
        print(f"{n} tokens: snapshot {snap.nbytes / 2**20:.1f} MiB, "
              f"{K} decode steps after restore {'BIT-IDENTICAL' if same else 'DIFFER'} (max |d| {worst:.3g})",
              flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
