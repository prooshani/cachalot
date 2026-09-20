"""
What does the model actually rank at the moment it emits a malformed include?

**Why this exists.** token_rank_probe.py teacher-forces correct text and finds
the token after `#include` at rank 1, 16 of 16, mean logprob -0.014 -- the model
knows `' <'` belongs there with about 98.6 % confidence. The greedy corpus run
emits `#include>` anyway, at temperature 0 with the frequency penalty off, where
the emitted token is by construction the argmax. Both of those cannot be true of
the same forward pass, so something about the real generation setting differs
from the probe's: the chat template, the length of the prefill, or the state the
reply itself builds up.

This replays a real corpus prompt through the real generation path and records,
at every step, what the model ranked and what came out. At an include site that
is the whole question in one line: if the argmax is `' <'` and something else is
emitted, the fault is after the logits; if the argmax is already `'>'`, the
forward pass is wrong under these conditions and the next question is which of
them.

It samples greedily by default so that "emitted" and "argmax" are the same thing
unless something is broken between them.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh \
      --budget-gib 24 --max-seconds 3600 --tag emit-trace -- env \
      CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts \
      CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
      benchmarks/emit_trace.py --task cpp-lru-cache --max-new-tokens 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "coding_tasks.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cpp-lru-cache",
                    help="task id from coding_tasks.json")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--raw-prompt", action="store_true",
                    help="skip the chat template and feed the task prompt as plain "
                         "text, to separate the template from the length")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    corpus = json.loads(CORPUS.read_text())
    tasks = {t["id"]: t for t in corpus["tasks"]}
    if args.task not in tasks:
        raise SystemExit(f"unknown task {args.task}; have {sorted(tasks)}")
    task = tasks[args.task]

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=args.max_seq_len) as rt:
        bank = Path(rt.expert_bank_path).name
        print(f"runtime ready (bank {bank}, budget "
              f"{rt.expert_cache_budget_bytes / 2**30:.1f} GiB)", flush=True)

        if args.raw_prompt:
            prompt = task["prompt"]
        else:
            encoding = load_official_encoding(MODEL_PATH)
            prompt = encoding.encode_messages(
                [{"role": "user", "content": task["prompt"]}],
                thinking_mode="chat",
                reasoning_effort=None,
            )
        ids = list(rt.tokenizer.encode(prompt))
        print(f"task {args.task}, prompt {len(ids)} tokens "
              f"({'raw' if args.raw_prompt else 'chat template'})", flush=True)

        rt.reset()
        result = rt.prefill_tokens(ids)

        eos = rt.tokenizer.eos_token_id
        produced: list[int] = []
        steps = []
        for _ in range(args.max_new_tokens):
            logits = result.logits.astype(mx.float32)
            logp = logits - mx.logsumexp(logits)
            order = mx.argsort(-logits)[: args.top_k].tolist()
            chosen = int(order[0])          # greedy: emitted is the argmax
            steps.append({
                "index": len(produced),
                "chosen_id": chosen,
                "chosen": rt.tokenizer.decode([chosen]),
                "top": [
                    {"id": int(t), "token": rt.tokenizer.decode([int(t)]),
                     "logprob": float(logp[int(t)].item())}
                    for t in order
                ],
            })
            if chosen == eos:
                break
            produced.append(chosen)
            result = rt.decode_token(chosen)

    text = rt.tokenizer.decode(produced)

    # The include sites: every step whose *previous* emitted token was
    # '#include'. Under a correct path each one should choose ' <' (id 818).
    include_sites = []
    for step in steps:
        i = step["index"]
        if i == 0:
            continue
        prev = steps[i - 1]["chosen"]
        if prev.strip().endswith("#include"):
            include_sites.append(step)

    print(f"\ngenerated {len(produced)} tokens\n")
    print("=== what the model chose at each '#include' site ===")
    if not include_sites:
        print("  no include site in this many tokens; raise --max-new-tokens "
              "or pick a task whose reply starts with a header block")
    for step in include_sites:
        competitors = "  ".join(
            f"{t['token']!r} {t['logprob']:+.3f}" for t in step["top"]
        )
        flag = "OK " if step["chosen_id"] == 818 else "BAD"
        print(f"  [{flag}] step {step['index']:4d} chose {step['chosen']!r}")
        print(f"         top: {competitors}")

    good = sum(1 for s in include_sites if s["chosen_id"] == 818)
    print(f"\n{good}/{len(include_sites)} include sites chose ' <' (id 818)")

    print("\n=== the reply ===")
    print(text[:2000])

    if args.out:
        Path(args.out).write_text(json.dumps({
            "task": args.task,
            "raw_prompt": args.raw_prompt,
            "prompt_tokens": len(ids),
            "generated": len(produced),
            "include_sites": include_sites,
            "steps": steps,
            "text": text,
        }, indent=2))
        print(f"\nwrote {args.out}")
    else:
        print(f"\n(pass --out to save the full step trace; {RESULTS_DIR} is the usual place)")


if __name__ == "__main__":
    main()
