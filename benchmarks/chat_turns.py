"""
Replay a short multi-turn chat exactly as `cachalot chat` does (chat template,
prefix cache, greedy decode) and account for each turn: prefill seconds for
the new tokens, expert misses and SSD bytes during prefill and during decode,
read concurrency, and decode tok/s.

Motivation (2026-09-16 terminal chat, 28 GiB budget): turn 2 prefilled 13 new
tokens in 10.5 s and turn 3 prefilled 20 in 10.8 s, while a 16-token prefill
after a 512-token context takes 4 s in benchmarks/short_prefill.py.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    benchmarks/guarded_run.sh --budget-gib 28 --tag chat_turns -- env PYTHONPATH=src \
        ~/venvs/deepseek-v41/bin/python benchmarks/chat_turns.py
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from time import perf_counter, sleep

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, StoreSnapshot  # noqa: E402
import mlx.core as mx  # noqa: E402

from cachalot.model.generation import SamplingParams, load_official_encoding, stream_tokens  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

TURNS = [
    "Hi",
    "answer only in english, now, Hi!",
    "write a 100 word story of a small fish living in a greek.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--verbose", action="store_true", help="runtime verbose=True, as `cachalot chat --verbose`")
    ap.add_argument("--idle-seconds", type=float, default=0.0, help="sleep before each turn after the first (human typing/reading time)")
    ap.add_argument("--heartbeat", type=float, default=0.0, help="during idle, evaluate a tiny MLX op every N seconds to keep the Metal queue active")
    args = ap.parse_args()
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=args.max_seq_len, verbose=args.verbose) as rt:
        print(f"runtime ready (expert budget {rt.expert_cache_budget_bytes / 2**30:.1f} GiB, "
              f"wired {rt.mlx_wired_limit_bytes / 2**30:.1f} GiB)", flush=True)
        enc = load_official_encoding(MODEL_PATH)
        params = SamplingParams(max_new_tokens=args.max_new_tokens, temperature=0.0)
        messages: list[dict] = []
        for i, user_text in enumerate(TURNS, 1):
            if i > 1 and args.idle_seconds > 0:
                print(f"idle {args.idle_seconds:.0f} s before turn {i}" + (f" (heartbeat every {args.heartbeat} s)" if args.heartbeat else ""), flush=True)
                if args.heartbeat > 0:
                    stop = threading.Event()
                    beats = [0]

                    def beat():
                        a = mx.zeros((1,))
                        while not stop.is_set():
                            mx.eval(a + 1)
                            beats[0] += 1
                            stop.wait(args.heartbeat)

                    th = threading.Thread(target=beat, daemon=True)
                    th.start()
                    sleep(args.idle_seconds)
                    stop.set()
                    th.join()
                    print(f"heartbeats {beats[0]}", flush=True)
                else:
                    sleep(args.idle_seconds)
            messages.append({"role": "user", "content": user_text})
            prompt = enc.encode_messages(messages, thinking_mode="chat", reasoning_effort=None)
            ids = list(rt.tokenizer.encode(prompt))
            before = StoreSnapshot.take(rt)
            st = rt.expert_store
            pred0 = (st.predicted_loads, st.predicted_used)
            out, prefill_ev, done_ev = [], None, None
            t0 = perf_counter()
            for ev in stream_tokens(rt, ids, params):
                if ev.kind == "prefill":
                    prefill_ev = ev
                    mid = StoreSnapshot.take(rt)
                elif ev.kind == "token":
                    out.append(ev.token)
                elif ev.kind == "done":
                    done_ev = ev
            total = perf_counter() - t0
            after = StoreSnapshot.take(rt)
            p = before.delta(mid)
            d = mid.delta(after)
            new = prefill_ev.prompt_tokens - prefill_ev.reused_prefix_tokens
            per_read = p.ssd_read_seconds / max(1, p.cache_misses) * 1e3
            conc = p.ssd_read_seconds / max(1e-9, prefill_ev.prefill_seconds)
            print(f"turn {i}: prompt {prefill_ev.prompt_tokens} tokens, reused {prefill_ev.reused_prefix_tokens}, "
                  f"new {new} | prefill {prefill_ev.prefill_seconds:.1f} s = {prefill_ev.prefill_seconds / max(1, new) * 1e3:.0f} ms/new token | "
                  f"prefill misses {p.cache_misses} ({p.cache_misses / max(1, new):.0f}/token, floor {p.cache_misses * 3.4e-3:.1f} s) "
                  f"hits {p.cache_hits} | {p.ssd_bytes_read / 1e9:.1f} GB, {per_read:.1f} ms/read, concurrency {conc:.1f}")
            n = len(out)
            print(f"        decode {n} tokens in {done_ev.decode_seconds:.1f} s = {n / max(1e-9, done_ev.decode_seconds):.2f} tok/s | "
                  f"misses {d.cache_misses} ({d.cache_misses / max(1, n):.0f}/token) hits {d.cache_hits} | "
                  f"predicted loads {st.predicted_loads - pred0[0]} used {st.predicted_used - pred0[1]} | "
                  f"resident {len(st)} | runtime heartbeats {getattr(rt, 'heartbeats', 0)} | total {total:.1f} s", flush=True)
            text = rt.tokenizer.decode(out)
            print(f"        text: {text[:120]!r}")
            messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()
