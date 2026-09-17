"""
Does DSpark's draft head agree with the model it drafts for?

Lever 1 of docs/HANDOFF.md turns entirely on one number, and this measures it
before a single line of the decode path is changed. The draft produces five
tokens per main forward (see cachalot.model.dspark_draft). Verification is only
worth building if enough of them survive, so the question is the per-depth
acceptance rate: how often does drafted position k equal the token the main
model itself went on to produce?

Method:

  1. Prefill a prompt with the production runtime, capturing the DSpark target
     layers' hidden states.
  2. Greedily decode N tokens, capturing the same per position. This is the
     ground truth the draft is scored against.
  3. Replay the positions in order through the draft head, which rebuilds its own
     sliding-window cache exactly as an integrated runtime would, and compare.

The headline outputs are the per-depth acceptance curve and the mean accepted
block length under the usual rule -- accept the longest matching prefix, stop at
the first mismatch -- which is the speed-up ceiling per main forward.

Greedy is the right setting for the ceiling: any sampling temperature can only
lower agreement between two different distributions.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh \
      --budget-gib 16 --max-seconds 3600 --tag dspark -- env \
      CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 \
      CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
      benchmarks/dspark_acceptance.py --prompt-tokens 64 --decode-tokens 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR  # noqa: E402
from cachalot.model.dspark_draft import BLOCK_SIZE, DSparkDraft  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--prompts", type=int, default=1, help="distinct prompts to average over")
    ap.add_argument("--draft-temperature", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sources = prompt_sources()[: max(1, args.prompts)]

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        print(
            f"runtime ready (expert budget {rt.expert_cache_budget_bytes / 2**30:.1f} GiB, "
            f"wired {rt.mlx_wired_limit_bytes / 2**30:.1f} GiB)",
            flush=True,
        )
        enc = load_official_encoding(MODEL_PATH)

        t0 = perf_counter()
        draft = DSparkDraft.load(
            MODEL_PATH,
            embed_weight=rt._global("embed.weight"),
            head_weight=rt._global("head.weight"),
            rope_cos=rt.sliding_rope_cos,
            rope_sin=rt.sliding_rope_sin,
            verbose=True,
        )
        print(f"draft head loaded in {perf_counter() - t0:.1f} s", flush=True)

        matches = [0] * BLOCK_SIZE
        trials = [0] * BLOCK_SIZE
        accepted_lengths: list[int] = []
        # Per block: the confidence the head gave each drafted position, and
        # whether that position matched. This is what a confidence-scheduled
        # verifier would decide on, and it costs nothing to record.
        blocks: list[dict] = []
        confidence_accepted: list[float] = []
        confidence_rejected: list[float] = []
        per_prompt = []

        for label, text in sources:
            rt.capture_main_hidden = True
            rt.reset()
            ids = build_prompt(rt, enc, text, args.prompt_tokens)
            rt.reset()

            result = rt.prefill_tokens(ids)
            prompt_hidden = result.main_hidden
            n_prompt = len(ids)
            token = int(result.logits.argmax().item())

            # Ground truth: the main model's own greedy continuation, and the
            # target-layer hidden state at every position that produced it.
            hidden_by_position = {n_prompt - 1: prompt_hidden[-1]}
            produced = [token]  # produced[i] is the token at position n_prompt + i
            for _ in range(args.decode_tokens):
                step = rt.decode_token(token)
                hidden_by_position[step.position] = step.main_hidden
                token = int(step.logits.argmax().item())
                produced.append(token)

            print(
                f"[{label}] decoded {len(produced)} tokens: "
                f"{rt.tokenizer.decode(produced[:24])!r}...",
                flush=True,
            )

            # Replay through the draft head, in position order, so its window
            # cache fills exactly as an integrated runtime's would.
            draft.reset()
            draft.observe(prompt_hidden, 0)

            prompt_matches = [0] * BLOCK_SIZE
            prompt_trials = [0] * BLOCK_SIZE
            for i in range(len(produced)):
                start_pos = n_prompt - 1 + i
                hidden = hidden_by_position.get(start_pos)
                if hidden is None:
                    break
                anchor = produced[i]
                # Actual tokens at the drafted positions, if they were decoded.
                truth = produced[i + 1 : i + 1 + BLOCK_SIZE]
                if not truth:
                    break

                out = draft.draft(
                    hidden,
                    anchor,
                    start_pos,
                    temperature=args.draft_temperature,
                )
                confidence = out.confidence.tolist()

                hits: list[bool] = []
                run = True
                for k, actual in enumerate(truth):
                    hit = out.tokens[k] == actual
                    hits.append(hit)
                    prompt_trials[k] += 1
                    trials[k] += 1
                    if hit:
                        prompt_matches[k] += 1
                        matches[k] += 1
                    (confidence_accepted if hit else confidence_rejected).append(
                        float(confidence[k])
                    )
                    if run and not hit:
                        run = False
                        accepted_lengths.append(k)
                if run:
                    accepted_lengths.append(len(truth))
                blocks.append(
                    {
                        "prompt": label,
                        "confidence": [float(c) for c in confidence[: len(truth)]],
                        "hits": hits,
                    }
                )

            per_prompt.append(
                {
                    "prompt": label,
                    "acceptance": [
                        m / t if t else None
                        for m, t in zip(prompt_matches, prompt_trials, strict=True)
                    ],
                }
            )
            print(
                f"[{label}] acceptance "
                + "  ".join(
                    f"k{k + 1} {m / t:.1%}" if t else f"k{k + 1} -"
                    for k, (m, t) in enumerate(zip(prompt_matches, prompt_trials, strict=True))
                ),
                flush=True,
            )
            rt.capture_main_hidden = False

        mean_accepted = (
            sum(accepted_lengths) / len(accepted_lengths) if accepted_lengths else 0.0
        )
        print("\n| depth | drafted | accepted | rate |")
        print("|---:|---:|---:|---:|")
        for k, (m, t) in enumerate(zip(matches, trials, strict=True)):
            rate = f"{m / t:.1%}" if t else "-"
            print(f"| {k + 1} | {t} | {m} | {rate} |")

        print(
            f"\nmean accepted draft length {mean_accepted:.2f} of {BLOCK_SIZE}"
            f"  ->  {1 + mean_accepted:.2f} tokens per main forward"
        )

        # Verifying fewer positions costs fewer expert reads, so the depth that
        # pays is not necessarily the deepest one. E[L_d] is what a verifier
        # limited to d drafted positions would accept per main forward.
        histogram = [accepted_lengths.count(k) for k in range(BLOCK_SIZE + 1)]
        expected_by_depth = []
        for depth in range(1, BLOCK_SIZE + 1):
            value = (
                sum(min(length, depth) for length in accepted_lengths)
                / len(accepted_lengths)
                if accepted_lengths
                else 0.0
            )
            expected_by_depth.append(value)
        print("\n| verified depth | tokens per main forward |")
        print("|---:|---:|")
        for depth, value in enumerate(expected_by_depth, 1):
            print(f"| {depth} | {1 + value:.3f} |")
        print(f"accepted-length histogram (0..{BLOCK_SIZE}): {histogram}")
        if confidence_accepted and confidence_rejected:
            acc = sum(confidence_accepted) / len(confidence_accepted)
            rej = sum(confidence_rejected) / len(confidence_rejected)
            print(
                f"confidence: accepted mean {acc:.3f} ({len(confidence_accepted)}), "
                f"rejected mean {rej:.3f} ({len(confidence_rejected)})"
            )
        print(
            f"draft experts touched: {draft.resident_expert_bytes() / 2**30:.2f} GiB "
            f"of {3 * 128 * 18_800_640 / 2**30:.2f} GiB"
        )

        payload = {
            "args": vars(args),
            "depth_trials": trials,
            "depth_matches": matches,
            "mean_accepted": mean_accepted,
            "tokens_per_forward": 1 + mean_accepted,
            "accepted_length_histogram": histogram,
            "tokens_per_forward_by_depth": [1 + v for v in expected_by_depth],
            "confidence_accepted_mean": (
                sum(confidence_accepted) / len(confidence_accepted)
                if confidence_accepted
                else None
            ),
            "confidence_rejected_mean": (
                sum(confidence_rejected) / len(confidence_rejected)
                if confidence_rejected
                else None
            ),
            "per_prompt": per_prompt,
            "blocks": blocks,
        }
        out_path = Path(args.out) if args.out else RESULTS_DIR / "dspark_acceptance.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
