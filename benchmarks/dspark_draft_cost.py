"""
What does one DSpark draft cost, and can it be hidden?

Acceptance says the draft is worth verifying; this says whether producing it is
cheap enough for that to matter. A decode token costs 182 ms at a 36 GiB budget,
so a draft that costs 150 ms cannot pay for itself no matter how well it is
accepted, while one that costs 20 ms is nearly free.

The draft needs no checkpoint reads once its 384 routed experts are resident --
6.72 GiB at FP4, of which four prompts touched 3.24 GiB -- so this runs without
the main runtime and measures the draft head alone: total time per block, and
the split between the three stages' attention, their MoE, the shared head and
the sequential Markov correction.

The Markov loop is worth watching. It is five full-vocabulary matmuls that
cannot be batched, because position i's correction depends on the token sampled
at position i-1; if it dominates, the fix is a smaller candidate set rather than
a faster kernel.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/dspark_draft_cost.py --blocks 40
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
from _common import MODEL_PATH, RESULTS_DIR  # noqa: E402
from cachalot.model.dspark_draft import (  # noqa: E402
    BLOCK_SIZE,
    DIM,
    N_MTP_LAYERS,
    ROPE_HEAD_DIM,
    DSparkDraft,
)
from cachalot.model.norm_rope_mlx import precompute_freqs  # noqa: E402
from cachalot.storage.tensor_index import build_tensor_index  # noqa: E402
from cachalot.storage.tensor_loader import load_resident_tensor  # noqa: E402
from quant_affine import quantize_2bit  # noqa: E402


def timed(fn, repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        mx.synchronize()
        t0 = perf_counter()
        fn()
        mx.synchronize()
        out.append((perf_counter() - t0) * 1e3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=40, help="draft blocks to time")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--positions", type=int, default=160,
                    help="main-model positions to seed the window with")
    ap.add_argument("--draft-expert-bits", type=int, default=0,
                    help="0 serves the draft's experts from FP4 as shipped; 2 or 3 "
                         "quantizes them once at load, which removes the per-matmul "
                         "FP4 repack and shrinks what stays resident")
    ap.add_argument("--draft-expert-group", type=int, default=128)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    index = build_tensor_index(MODEL_PATH)
    embed = load_resident_tensor(index["embed.weight"]).data
    head = load_resident_tensor(index["head.weight"]).data
    mx.eval(embed, head)

    cos, sin = precompute_freqs(
        dim=ROPE_HEAD_DIM, seqlen=4096, original_seq_len=0, base=10000.0,
        factor=16.0, beta_fast=32.0, beta_slow=1.0,
    )
    mx.eval(cos, sin)

    if args.draft_expert_bits and args.draft_expert_bits != 2:
        raise SystemExit("only --draft-expert-bits 2 has a packer here")
    draft = DSparkDraft.load(
        MODEL_PATH, embed_weight=embed, head_weight=head,
        rope_cos=cos, rope_sin=sin, verbose=True,
        expert_quantizer=quantize_2bit if args.draft_expert_bits else None,
        expert_bits=args.draft_expert_bits or 2,
        expert_group=args.draft_expert_group,
    )

    # Synthetic hidden states are fine here: the cost of a draft depends on its
    # shapes and on how many distinct experts the router picks, not on whether
    # the tokens it produces are any good.
    mx.random.seed(20260917)
    main_hidden = (
        mx.random.normal((args.positions, DIM * N_MTP_LAYERS), dtype=mx.float32) * 0.02
    ).astype(mx.bfloat16)
    draft.observe(main_hidden, 0)

    anchor = 1000
    start = args.positions - 1

    for _ in range(args.warmup):
        draft.draft(main_hidden[-1], anchor, start)

    print(f"experts resident after warm-up: {draft.resident_expert_bytes() / 2**30:.2f} GiB")

    totals = timed(lambda: draft.draft(main_hidden[-1], anchor, start), args.blocks)

    # Stage split, measured the same way the block runs them.
    x = mx.take(
        embed,
        mx.array([anchor] + [128799] * (BLOCK_SIZE - 1), dtype=mx.int32),
        axis=0,
    )
    x = mx.broadcast_to(x[:, None, :], (BLOCK_SIZE, 4, DIM)).astype(embed.dtype)
    pre_mix = mx.concatenate(
        [mx.ones((BLOCK_SIZE, 1), dtype=mx.float32),
         mx.zeros((BLOCK_SIZE, 3), dtype=mx.float32)],
        axis=-1,
    )

    def one_stage():
        mx.eval(*draft._block(0, x, pre_mix, start_pos=start))

    stage = timed(one_stage, max(8, args.blocks // 4))

    def markov_only():
        for token in (anchor, anchor + 1, anchor + 2, anchor + 3, anchor + 4):
            bias, _ = draft._markov_bias(token)
            mx.eval(bias)

    markov = timed(markov_only, max(8, args.blocks // 4))

    def head_only():
        # Exactly what DSparkDraft.draft does: a bf16 product cast afterwards,
        # not a cast of the [129280, 5120] head itself.
        mx.eval(
            mx.matmul(
                mx.zeros((BLOCK_SIZE, DIM), dtype=head.dtype), head.T
            ).astype(mx.float32)
        )

    head_ms = timed(head_only, max(8, args.blocks // 4))

    total = statistics.median(totals)
    print(f"\ndraft block: median {total:.1f} ms "
          f"(min {min(totals):.1f}, max {max(totals):.1f}, n={len(totals)})")
    print(f"  one of {N_MTP_LAYERS} stages : {statistics.median(stage):6.1f} ms "
          f"-> {statistics.median(stage) * N_MTP_LAYERS:.1f} ms for all stages")
    print(f"  shared head            : {statistics.median(head_ms):6.1f} ms")
    print(f"  Markov correction x{BLOCK_SIZE}  : {statistics.median(markov):6.1f} ms")
    print(f"\nper drafted token: {total / BLOCK_SIZE:.1f} ms")
    print("A decode token costs 182 ms at a 36 GiB budget (handoff section 7.1); "
          "the draft has to stay far under that for speculation to pay.")

    payload = {
        "args": vars(args),
        "total_ms": totals,
        "stage_ms": stage,
        "markov_ms": markov,
        "head_ms": head_ms,
        "resident_expert_gib": draft.resident_expert_bytes() / 2**30,
    }
    out_path = Path(args.out) if args.out else RESULTS_DIR / "dspark_draft_cost.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
