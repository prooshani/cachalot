"""
What would verifying K speculative positions in one forward actually cost?

Lever 1 of docs/HANDOFF.md needs two numbers. `dspark_acceptance.py` supplies
the first -- how many drafted tokens survive. This supplies the second -- what a
K-position forward costs against a one-position decode -- and it needs no new
decode code at all, because the runtime's prefill path *is* a multi-position
forward. Prefilling K tokens at the current position computes exactly what a
verifier would compute.

The two numbers multiply:

    speed-up  =  K-position forward cost / (tokens accepted per forward)
                 ------------------------------------------------------
                            one-position decode cost

Both arms run from the same snapshot over the same tokens, so they route to the
same experts and differ only in how many positions share one forward. The arms
are interleaved and repeated, because decode timing's run-to-run spread reaches
7 % (operating rule 3).

The interesting quantity is not only time: it is misses per forward. A forward
over K positions asks for up to 6K experts per layer instead of 6, and adjacent
tokens overlap only about 29 %, so bytes per forward rise nearly linearly while
compute may not. Which of the two binds is what decides the lever.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh \
      --budget-gib 36 --max-seconds 2400 --tag verifycost -- env \
      CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 \
      CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
      benchmarks/verify_forward_cost.py --prompt-tokens 512 --warm-tokens 32 --repeats 4
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR, StoreSnapshot  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--warm-tokens", type=int, default=32,
                    help="tokens decoded before measuring, to reach a realistic cache state")
    ap.add_argument("--widths", default="1,2,3,4,6")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",")]
    max_width = max(widths)

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        print(
            f"runtime ready (expert budget {rt.expert_cache_budget_bytes / 2**30:.1f} GiB, "
            f"wired {rt.mlx_wired_limit_bytes / 2**30:.1f} GiB)",
            flush=True,
        )
        enc = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)

        rt.reset()
        result = rt.prefill_tokens(ids)
        token = int(result.logits.argmax().item())

        # Warm to a realistic cache state and record the continuation the
        # verification arms will replay.
        continuation = [token]
        for _ in range(args.warm_tokens + max_width):
            step = rt.decode_token(token)
            token = int(step.logits.argmax().item())
            continuation.append(token)

        # Rewind to just after the warm-up: every arm starts here.
        rt.reset()
        rt.prefill_tokens(ids)
        for tok in continuation[: args.warm_tokens]:
            rt.decode_token(tok)
        base = rt.snapshot()
        chunk_tokens = continuation[args.warm_tokens : args.warm_tokens + max_width]

        samples: dict[int, list[tuple[float, int]]] = {w: [] for w in widths}

        for repeat in range(args.repeats):
            order = widths if repeat % 2 == 0 else list(reversed(widths))
            for width in order:
                rt.restore(base)
                before = StoreSnapshot.take(rt)
                mx.synchronize()
                t0 = perf_counter()
                out = rt.prefill_tokens(chunk_tokens[:width])
                mx.eval(out.logits)
                mx.synchronize()
                seconds = perf_counter() - t0
                delta = before.delta(StoreSnapshot.take(rt))
                samples[width].append((seconds, delta.cache_misses))
            print(f"  repeat {repeat + 1}/{args.repeats} done", flush=True)

        rt.restore(base)

        print("\n| positions | ms/forward | ms/position | misses/forward | MiB/forward | vs 1 |")
        print("|---:|---:|---:|---:|---:|---:|")
        rows = []
        expert_bytes = rt.expert_store.format.expert_bytes if hasattr(
            rt.expert_store.format, "expert_bytes"
        ) else 9_953_280
        base_ms = None
        for width in widths:
            times = [s * 1e3 for s, _ in samples[width]]
            misses = [m for _, m in samples[width]]
            ms = statistics.median(times)
            if base_ms is None:
                base_ms = ms
            mib = statistics.median(misses) * expert_bytes / 2**20
            rows.append(
                {
                    "positions": width,
                    "ms_per_forward": ms,
                    "ms_per_position": ms / width,
                    "misses_per_forward": statistics.median(misses),
                    "mib_per_forward": mib,
                    "times_ms": times,
                    "misses": misses,
                }
            )
            print(
                f"| {width} | {ms:.1f} | {ms / width:.1f} | "
                f"{statistics.median(misses):.0f} | {mib:.0f} | {ms / base_ms:.2f}x |"
            )

        print(
            "\nRead this against the acceptance curve: a forward over K positions "
            "pays the K-position cost once and returns the accepted prefix, so "
            "the break-even is (K-position ms) / (tokens accepted) < (1-position ms)."
        )

        out_path = Path(args.out) if args.out else RESULTS_DIR / "verify_forward_cost.json"
        out_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
