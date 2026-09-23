"""Prefill peak memory and time against prompt length, on the shipped config.

HANDOFF section 15.2. Hermes Agent's first request is a ~13.5k-token prompt
(24 tool schemas plus a 17k-character system prompt) and died in prefill with
`[METAL] Command buffer execution failed: Insufficient Memory`. This finds
where prefill's memory goes, per layer block, split into the part before the
routed MoE (attention, indexer, compressor, hyper-connections) and the MoE
itself, so a fix can target the piece that actually grows.

Run it the way serve.sh runs the server (same env), with nothing else on the GPU:

    CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \\
    CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 \\
    CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=80 \\
    CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json CACHALOT_HOTLIST_GIB=8 \\
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/prefill_memory_sweep.py 1024 4096 8192

The prompt is real prose (docs/HANDOFF.md), so routing looks like real text.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

GiB = 1 << 30
ROOT = Path(__file__).resolve().parent.parent

from cachalot.model import (  # noqa: E402
    block_compressed_index_source_prefill,
    block_compressed_reuse_prefill,
    block_compressed_source_prefill,
    block_layer0_prefill,
    block_sliding_window_prefill,
)
from cachalot.model import text_decode_runtime as tdr  # noqa: E402
from cachalot.model.api import V41Model  # noqa: E402

BLOCK_MODULES = {
    "layer0": (block_layer0_prefill, "layer0_block_prefill"),
    "sliding": (block_sliding_window_prefill, "sliding_window_block_prefill"),
    "c_source": (block_compressed_source_prefill, "compressed_source_block_prefill"),
    "c_index_source": (block_compressed_index_source_prefill, "compressed_index_source_block_prefill"),
    "c_reuse": (block_compressed_reuse_prefill, "compressed_reuse_block_prefill"),
}

records: list[dict] = []
_current: dict = {}


def _eval_tree(obj) -> None:
    arrays = []

    def walk(o):
        if isinstance(o, mx.array):
            arrays.append(o)
        elif isinstance(o, (tuple, list)):
            for v in o:
                walk(v)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)

    walk(obj)
    if arrays:
        mx.eval(*arrays)


def instrument() -> None:
    for kind, (module, fn_name) in BLOCK_MODULES.items():
        block = getattr(tdr, fn_name)
        moe = module.moe_prefill_grouped

        def moe_wrapper(*args, _moe=moe, **kwargs):
            _eval_tree(args)
            _eval_tree(kwargs)
            _current["pre_moe_peak"] = mx.get_peak_memory()
            mx.reset_peak_memory()
            out = _moe(*args, **kwargs)
            _eval_tree(out)
            _current["moe_peak"] = mx.get_peak_memory()
            return out

        def block_wrapper(*args, _block=block, _kind=kind, **kwargs):
            _current.clear()
            mx.reset_peak_memory()
            base = mx.get_active_memory()
            t0 = time.perf_counter()
            out = _block(*args, **kwargs)
            _eval_tree(out)
            records.append(
                {
                    "kind": _kind,
                    "base": base,
                    "pre_moe_peak": _current.get("pre_moe_peak", 0),
                    "moe_peak": _current.get("moe_peak", 0),
                    "seconds": time.perf_counter() - t0,
                }
            )
            return out

        module.moe_prefill_grouped = moe_wrapper
        setattr(tdr, fn_name, block_wrapper)


def prompt_tokens(tokenizer, n: int) -> list[int]:
    text = (ROOT / "docs" / "HANDOFF.md").read_text()
    ids = list(tokenizer.encode(text))
    while len(ids) < n:
        ids += ids
    return ids[:n]


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [1024, 4096]
    instrument()
    model = V41Model.from_pretrained(
        str(tdr.DEFAULT_CONFIG.resolved_model_path),
        max_seq_len=65536,
        expert_cache_budget_bytes=52 * GiB,
    )
    rt = model.runtime
    for n in sizes:
        ids = prompt_tokens(rt.tokenizer, n)
        rt.reset()
        records.clear()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        t0 = time.perf_counter()
        try:
            res = rt.prefill_tokens(ids)
            mx.eval(res.logits)
            status = "ok"
        except Exception as exc:  # the OOM surfaces as a RuntimeError
            status = f"FAILED: {str(exc)[:120]}"
        dt = time.perf_counter() - t0
        worst_pre = max(records, key=lambda r: r["pre_moe_peak"] - r["base"], default=None)
        worst_moe = max(records, key=lambda r: r["moe_peak"] - r["base"], default=None)
        print(f"\n=== {n} tokens: {status}  {dt:.1f}s  ({n / dt:.1f} tok/s)  "
              f"active before {base / GiB:.2f} GiB, layers done {len(records)}")
        if worst_pre:
            print(f"  worst pre-MoE (attention/indexer/HC) peak over base: "
                  f"{(worst_pre['pre_moe_peak'] - worst_pre['base']) / GiB:.2f} GiB in {worst_pre['kind']}")
        if worst_moe:
            print(f"  worst MoE peak over base: "
                  f"{(worst_moe['moe_peak'] - worst_moe['base']) / GiB:.2f} GiB in {worst_moe['kind']}")
        by_kind: dict[str, list[float]] = {}
        for r in records:
            by_kind.setdefault(r["kind"], []).append(r["seconds"])
        for kind, secs in by_kind.items():
            print(f"  {kind:15s} {len(secs):2d} layers  {sum(secs):7.1f}s total  {sum(secs) / len(secs):6.2f}s/layer")
        if status != "ok":
            break


if __name__ == "__main__":
    main()
