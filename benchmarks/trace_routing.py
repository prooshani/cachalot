"""
Phase-0 routing trace benchmark.

Runs several genuinely different chat prompts (code review, prose summary,
a second code file, an engineering document), each prefilled to a fixed token
count and followed by greedy decode, then returns to the first prompt to
measure cross-turn expert reuse.

Outputs (benchmarks/results/):
    trace_routing.trace.npz     per-(token, layer) expert selections
    trace_routing.json          per-phase wall/hit/SSD/memory report

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/trace_routing.py \
        --prompt-tokens 512 --decode-tokens 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    MODEL_PATH,
    RESULTS_DIR,
    StoreSnapshot,
    Timer,
    phase_report,
    write_json,
)
from cachalot.metrics.routing_trace import RoutingTracer  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def prompt_sources() -> list[tuple[str, str]]:
    code_a = (REPO / "src/cachalot/cache/resident_store.py").read_text()
    code_b = (Path(MODEL_PATH) / "encoding/encoding.py").read_text()
    prose = (Path(MODEL_PATH) / "README.md").read_text()
    doc_path = Path.home() / "Downloads/DEEPSEEK_V41_MAC_RUNTIME_ENGINEERING_HANDOFF.md"
    doc = doc_path.read_text() if doc_path.exists() else prose[::-1]

    return [
        ("A_code_review", "Review this Python module for concurrency bugs and explain the cache admission policy.\n\n```python\n" + code_a + "\n```"),
        ("B_prose_summary", "Summarize the following model card in five bullet points for an executive audience.\n\n" + prose),
        ("C_code_explain", "Explain what this file does and list every public function with a one-line description.\n\n```python\n" + code_b + "\n```"),
        ("D_engineering_doc", "You are a staff engineer. Read this handoff document and propose the three highest-value next steps.\n\n" + doc),
    ]


def build_prompt(runtime, encoding, text: str, prompt_tokens: int) -> list[int]:
    # Neutralize DeepSeek special-token delimiters that may appear in source text.
    text = text.replace("<\uff5c", "<|").replace("\uff5c>", "|>")
    # Encode through the official chat protocol, then trim the *user content*
    # until the full prompt fits the token budget exactly.
    lo, hi = 1, len(text)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        prompt = encoding.encode_messages(
            [{"role": "user", "content": text[:mid]}],
            thinking_mode="chat",
            reasoning_effort=None,
        )
        ids = list(runtime.tokenizer.encode(prompt))
        if len(ids) <= prompt_tokens:
            best = ids
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise RuntimeError("prompt budget too small")
    return best


def run_turn(runtime, tracer, name, ids, decode_tokens, reports):
    runtime.reset()
    tracer.mark(f"{name}:prefill", tokens=len(ids))

    before = StoreSnapshot.take(runtime)
    with Timer() as t:
        result = runtime.prefill_tokens(ids)
    reports.append(
        phase_report(f"{name}:prefill", t.seconds, len(ids), before.delta(StoreSnapshot.take(runtime)), runtime)
    )

    tracer.mark(f"{name}:decode", tokens=decode_tokens)
    token = int(result.logits.argmax().item())
    generated = []
    before = StoreSnapshot.take(runtime)
    with Timer() as t:
        for _ in range(decode_tokens):
            generated.append(token)
            result = runtime.decode_token(token)
            token = int(result.logits.argmax().item())
    rep = phase_report(f"{name}:decode", t.seconds, decode_tokens, before.delta(StoreSnapshot.take(runtime)), runtime)
    rep["generated_ids"] = generated
    rep["generated_text"] = runtime.tokenizer.decode(generated, skip_special_tokens=False)
    reports.append(rep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--expert-budget-gib", type=float, default=0.0, help="0 = auto")
    ap.add_argument("--io-workers", type=int, default=8)
    ap.add_argument("--out", default="trace_routing")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracer = RoutingTracer()
    reports: list[dict] = []

    with TextDecodeRuntime(
        args.model,
        max_seq_len=4096,
        expert_cache_budget_bytes=int(args.expert_budget_gib * 1024**3),
        io_workers=args.io_workers,
        verbose=False,
    ) as runtime:
        runtime.set_tracer(tracer)
        encoding = load_official_encoding(args.model)

        prompts = [
            (name, build_prompt(runtime, encoding, text, args.prompt_tokens))
            for name, text in prompt_sources()
        ]
        for name, ids in prompts:
            print(f"prompt {name}: {len(ids)} tokens", flush=True)

        for name, ids in prompts:
            run_turn(runtime, tracer, name, ids, args.decode_tokens, reports)

        # Return to the first task: cross-turn reuse.
        name, ids = prompts[0]
        run_turn(runtime, tracer, name + "_return", ids, args.decode_tokens, reports)

        trace_path = tracer.save(RESULTS_DIR / f"{args.out}.trace.npz")
        write_json(
            RESULTS_DIR / f"{args.out}.json",
            {
                "config": vars(args),
                "prompt_tokens": {name: len(ids) for name, ids in prompts},
                "phases": reports,
                "trace": str(trace_path),
                "trace_records": tracer.records,
            },
        )
        print("saved", trace_path, flush=True)


if __name__ == "__main__":
    main()
