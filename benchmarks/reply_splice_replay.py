"""
Replay a server request dump offline and count prefill tokens with and without the reply splice
(HANDOFF section 15.5). No model is loaded: only the tokenizer, the official encoding and the server's own
splice code.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/reply_splice_replay.py /tmp/cachalot-requests.jsonl

The dump must come from a server that did *not* splice (CACHALOT_SERVER_DUMP on a pre-0.12.0 server, or
one that never had a reply to splice): a spliced prompt no longer equals the rendered request body, so the
replay stops matching replies to requests at the first splice. For each request with a recorded reply it
prints the prompt length and the tokens that a prefix cache holding every earlier prompt and prompt+reply
would leave to prefill, without and with the splice; the baseline reproduces the live `reused` numbers.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from transformers import AutoTokenizer  # noqa: E402
from transformers.utils import logging  # noqa: E402

from cachalot.model.generation import SamplingParams, load_official_encoding  # noqa: E402
from cachalot.server.app import _normalize_effort  # noqa: E402
from cachalot.server.engine import ChatRequest, Engine  # noqa: E402

MODEL_PATH = "/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash"


def chat_request(body: dict) -> ChatRequest:
    """The subset of the server's request mapping that decides the rendered prompt."""
    effort, off = _normalize_effort(body.get("reasoning_effort"))
    thinking = bool(body.get("thinking")) or (effort is not None and not off)
    return ChatRequest(
        messages=body["messages"],
        params=SamplingParams(),
        thinking_mode="thinking" if thinking else "chat",
        reasoning_effort=effort if thinking else None,
        tools=body.get("tools"),
        response_format=body.get("response_format"),
    )


def longest_prefix(snapshots: list[list[int]], tokens: list[int]) -> int:
    best = 0
    for s in snapshots:
        if len(s) <= len(tokens) and tokens[: len(s)] == s:
            best = max(best, len(s))
    return best


def main() -> None:
    logging.set_verbosity_error()
    engine = Engine.__new__(Engine)  # tokenizer and encoding only
    engine.encoding = load_official_encoding(MODEL_PATH)
    engine.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    engine._own_replies = deque(maxlen=64)
    engine.replies_spliced = 0

    requests = []
    for line in open(sys.argv[1]):
        row = json.loads(line)
        if "body" in row:
            req = chat_request(row["body"])
            requests.append([req, engine.encode_chat(req), None])
        elif "reply" in row:
            for r in requests:
                if r[2] is None and r[1] == row["reply"]["prompt_tokens"]:
                    r[2] = row["reply"]["reply_tokens"]
                    break

    base_snaps: list[list[int]] = []
    splice_snaps: list[list[int]] = []
    total_base = total_splice = 0
    for req, tokens, reply in requests:
        if reply is None:
            continue
        spliced = engine._splice_own_replies(list(tokens), req.thinking_mode)
        base = len(tokens) - longest_prefix(base_snaps, tokens)
        new = len(spliced) - longest_prefix(splice_snaps, spliced)
        total_base += base
        total_splice += new
        print(f"prompt={len(tokens):6d}  prefill without={base:6d}  with={new:6d}")
        base_snaps += [list(tokens), list(tokens) + list(reply)]
        splice_snaps += [spliced, spliced + list(reply)]
        text = engine.tokenizer.decode(reply, skip_special_tokens=False)
        engine._remember_reply(spliced, reply, text, req.thinking_mode)
    print(f"total prefill tokens: without {total_base}, with {total_splice}")


if __name__ == "__main__":
    main()
