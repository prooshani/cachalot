"""How much of a prefill is spent waiting on Engram table rows.

Prefill starts a background read of both Engram layers' rows as soon as the
token ids are known, so the raw cost of those reads is not the cost that a
faster table would remove. What a faster table can remove is the time the
critical path still spends blocked on them, plus the dequantization that
follows.

This script instruments the runtime from the outside, changes no production
code, and reports, for one prefill:

  read      wall time of the background row read, per Engram layer
  blocked   how much of that read was still outstanding when the layer needed it
  dequant   time spent turning the raw rows into an fp32 array

Run it under benchmarks/guarded_run.sh like every other benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model import engram_rows as engram_rows_module  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from cachalot.storage.engram_reader import EngramRowReader  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402

reads: dict[int, tuple[float, float, int]] = {}
enters: dict[int, float] = {}
dequant_seconds = 0.0


def instrument(runtime: TextDecodeRuntime) -> None:
    original_read = EngramRowReader.read_rows
    original_apply = type(runtime)._prefill_apply_engram
    original_to_array = engram_rows_module.engram_rows_to_array

    def read_rows(self, layout, row_ids):
        start = perf_counter()
        rows = original_read(self, layout, row_ids)
        reads[layout.layer] = (start, perf_counter(), rows.count)
        return rows

    def apply_engram(self, x, layer_id, hash_rows_by_token):
        enters.setdefault(layer_id, perf_counter())
        return original_apply(self, x, layer_id, hash_rows_by_token)

    def to_array(rows, layout):
        global dequant_seconds
        start = perf_counter()
        values = original_to_array(rows, layout)
        dequant_seconds += perf_counter() - start
        return values

    EngramRowReader.read_rows = read_rows
    type(runtime)._prefill_apply_engram = apply_engram
    engram_rows_module.engram_rows_to_array = to_array

    # _prefill_apply_engram resolves engram_rows_to_array through the module at
    # call time for the prefetched path, but imports it by name for the fallback.
    import cachalot.model.text_decode_runtime as runtime_module

    runtime_module.engram_rows_to_array = to_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, default=512)
    args = parser.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as runtime:
        print(
            f"runtime ready (expert budget {runtime.expert_cache_budget_bytes / 2**30:.1f} GiB, "
            f"wired {runtime.mlx_wired_limit_bytes / 2**30:.1f} GiB)",
            flush=True,
        )
        encoding = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(runtime, encoding, text, args.prompt_tokens)
        runtime.reset()
        instrument(runtime)

        start = perf_counter()
        result = runtime.prefill_tokens(ids)
        mx.eval(result.logits)
        prefill_seconds = perf_counter() - start

        print(f"\nprefill {len(ids)} tokens: {prefill_seconds:.2f} s")
        blocked_total = 0.0
        for layer in sorted(reads):
            read_start, read_end, count = reads[layer]
            enter = enters.get(layer)
            if enter is None:
                print(f"  layer {layer:>2}: {count:,} unique rows | read {(read_end - read_start) * 1e3:7.1f} ms | never applied")
                continue
            blocked = max(0.0, read_end - enter)
            blocked_total += blocked
            print(
                f"  layer {layer:>2}: {count:,} unique rows | "
                f"read {(read_end - read_start) * 1e3:7.1f} ms | "
                f"started {(read_start - start) * 1e3:7.1f} ms into prefill | "
                f"needed {(enter - start) * 1e3:7.1f} ms | "
                f"blocked {blocked * 1e3:7.1f} ms"
            )
        print(
            f"  blocked total {blocked_total * 1e3:.1f} ms | dequant total {dequant_seconds * 1e3:.1f} ms | "
            f"removable share of prefill {(blocked_total + dequant_seconds) / prefill_seconds:.1%}"
        )


if __name__ == "__main__":
    main()
