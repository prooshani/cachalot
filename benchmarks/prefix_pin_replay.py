"""
Replay a server request dump through the real prefix cache, with and without pinning the chunk snapshots
inside a system block (HANDOFF section 15.7). No model is loaded: only the tokenizer, the official encoding,
the server's prompt rendering, `prepare_prompt` and `PrefixCache`, driving a runtime that records token ids.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/prefix_pin_replay.py /tmp/cachalot-requests.jsonl

Each request is rendered as the server rendered it (reply splice included), prefilled through
`prepare_prompt` with the system boundary the server would cut at, and followed by the reply the dump
recorded, so the cache holds what the live one held. Image-bearing requests are rendered without their image
rows and get no boundary, as on the server, which makes their token counts approximate. Prints one line per
request for each arm and the total tokens each arm prefilled; `--entries`, `--pinned` and `--gib` size the
cache. The recorded snapshots hold no arrays, so each one is charged what a real one of its length weighs
(~5.3 MB plus 3,050 bytes per token, from the files in the snapshot directory) against `--gib`.

HANDOFF section 15.10 replays Hamed's subagent session with it: 479,899 tokens prefilled under 0.14.0's
eviction (20 entries, 12 pins, pins above leaves), 256,597 under 0.15.0's (1.5 GiB, tiered).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoTokenizer  # noqa: E402
from transformers.utils import logging  # noqa: E402

from cachalot.model import generation  # noqa: E402
from cachalot.model.generation import load_official_encoding, prepare_prompt  # noqa: E402
from cachalot.model.prefix_cache import PrefixCache, SequenceSnapshot  # noqa: E402
from cachalot.server.engine import Engine  # noqa: E402
from reply_splice_replay import MODEL_PATH, chat_request  # noqa: E402


class TokenRuntime:
    """Just enough of TextDecodeRuntime for prepare_prompt: it tracks token ids."""

    def __init__(self, cache: PrefixCache):
        self.prefix_cache = cache
        self.tokens: list[int] = []
        self.position = 0

    def reset(self):
        self.tokens, self.position = [], 0

    def restore(self, snap):
        self.tokens, self.position = list(snap.tokens), snap.position

    def prefill_tokens(self, ids, **_):
        self.tokens.extend(ids)
        self.position += len(ids)
        return generation.DecodeResult(logits=None, hidden=None, routes=(), position=self.position - 1)

    def snapshot(self, logits=None):
        return SequenceSnapshot(tuple(self.tokens), self.position, logits, {}, {}, {}, {}, {}, [],
                                None, None, None, None)


# What a real snapshot of this length weighs on disk and in memory.
SequenceSnapshot.nbytes = property(lambda self: int(5.3e6 + 3050 * len(self.tokens)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--entries", type=int, default=64)
    ap.add_argument("--pinned", type=int, default=64)
    ap.add_argument("--gib", type=float, default=1.5, help="byte budget; 0 counts entries only")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    logging.set_verbosity_error()
    engine = Engine.__new__(Engine)
    engine.encoding = load_official_encoding(MODEL_PATH)
    engine.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    rows = [json.loads(line) for line in open(args.dump)]
    replies = [r["reply"] for r in rows if "reply" in r]
    bodies = [r for r in rows if "body" in r]

    totals = {}
    for arm, pin in (("unpinned", False), ("pinned", True)):
        engine._own_replies = deque(maxlen=64)
        engine.replies_spliced = 0
        cache = PrefixCache(max_entries=args.entries, max_pinned=args.pinned,
                            max_bytes=int(args.gib * 1024**3) if args.gib else None)
        if not pin:  # the 0.12.x behaviour: only the block itself is pinned
            add = cache.add
            cache.add = lambda s, boundary=False, pin=False, _add=add: _add(s, boundary=boundary)
        rt = TokenRuntime(cache)
        unused = list(replies)
        total = 0
        for row in bodies:
            req = chat_request(row["body"])
            tokens, images = engine._encode_chat(req)
            tokens = engine._splice_own_replies(tokens, req.thinking_mode)
            boundary = 0 if images else engine.system_prefix_len(req, tokens)
            _, reused = prepare_prompt(rt, tokens, boundaries=(boundary,) if boundary else ())
            total += len(tokens) - reused
            reply = next((r for r in unused if r["prompt_tokens"] == tokens), None)
            if reply is not None:
                unused.remove(reply)
                rt.tokens.extend(reply["reply_tokens"])
                rt.position += len(reply["reply_tokens"])
                cache.add(rt.snapshot())
                text = engine.tokenizer.decode(reply["reply_tokens"], skip_special_tokens=False)
                engine._remember_reply(tokens, reply["reply_tokens"], text, req.thinking_mode)
            if not args.quiet:
                print(f"{arm:9s} prompt={len(tokens):6d} system={boundary:6d} reused={reused:6d} "
                      f"prefilled={len(tokens) - reused:6d} images={len(images)}")
        totals[arm] = total
    print(f"total prefilled tokens: unpinned {totals['unpinned']}, pinned {totals['pinned']} "
          f"({totals['unpinned'] - totals['pinned']} fewer)")


if __name__ == "__main__":
    main()
