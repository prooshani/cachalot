"""
Replay a server request dump through the real prefix cache and the real disk snapshot store, with server
restarts, and compare the store's pruning rules (HANDOFF section 15.12). No model is loaded; the store writes
to an in-memory "disk" (file name to snapshot), so nothing touches ~/.cache.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/snapshot_store_replay.py /tmp/cachalot-requests.jsonl

Arms:
  newest8  0.15.0: every system block is written, the 8 newest files are kept and all load at startup;
  used8    SnapshotStore with 8 files, all preloaded: pruned by last use instead of by creation;
  used32   SnapshotStore as shipped (`--keep`, `--preload`): 32 files by last use, 4 preloaded, the rest
           fetched from disk when a request starts with one.

The server is restarted wherever two requests are more than `--gap-min` minutes apart (the dump spans several
sessions; the server was restarted between them), and the memory cache and reply memory are lost there. Two
numbers per arm:
  total      tokens prefilled over the whole dump under those restarts;
  penalty    for every request, the extra tokens it would prefill had the server been restarted just before
             it (fresh cache holding only the disk set at that moment), summed: what a restart at a random
             point costs, times the number of requests. Uses the request's tokens as the live run rendered
             them (reply splice included), so it slightly understates a real restart.
Also prints the disk set at the end: the blocks a restart tomorrow would load. `--then-subagents N` adds
Job 1's case after the dump: N more subagent first turns (the dump's subagent requests, each with its
system message made unique), a restart, then the main agent's last long turn again.
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

from cachalot.model import snapshot_store  # noqa: E402
from cachalot.model.generation import load_official_encoding, prepare_prompt  # noqa: E402
from cachalot.model.prefix_cache import PrefixCache  # noqa: E402
from cachalot.server.engine import Engine  # noqa: E402
from prefix_pin_replay import TokenRuntime  # noqa: E402  (also sets SequenceSnapshot.nbytes)
from reply_splice_replay import MODEL_PATH, chat_request  # noqa: E402


class MemoryStore(snapshot_store.SnapshotStore):
    """SnapshotStore on a dict instead of a directory."""

    def __init__(self, disk: dict, index: dict, clock, keep: int, preload: int):
        self.disk, self._index_box = disk, index
        super().__init__("/nonexistent", "replay", keep=keep, preload=preload, clock=clock)

    def _read_index(self):
        return dict(self._index_box)

    def _write_index(self):
        self._index_box.clear()
        self._index_box.update(self.index)

    def _files(self):
        return {name: mtime for name, (mtime, _) in self.disk.items()}

    def _read_file_tokens(self, name):
        return self.disk[name][1].tokens

    def _load_file(self, name):
        return self.disk[name][1] if name in self.disk else None

    def _write_file(self, snap, name):
        self.disk[name] = (self.clock(), snap)

    def _remove_file(self, name):
        self.disk.pop(name, None)


class Newest8:
    """0.15.0's rule, snapshot_store.save's pruning, on a dict."""

    def __init__(self, disk: dict, index: dict, clock, keep: int, preload: int):
        self.disk, self.clock, self.keep = disk, clock, keep

    def persist(self, snap):
        self.disk[snapshot_store._file_name(snap)] = (self.clock(), snap)
        for name in sorted(self.disk, key=lambda n: -self.disk[n][0])[self.keep:]:
            del self.disk[name]

    def on_find(self, tokens, blocks):
        pass

    def load_all(self):
        return [snap for _, (_, snap) in sorted(self.disk.items(), key=lambda kv: kv[1][0])]


def new_cache(args) -> PrefixCache:
    return PrefixCache(max_entries=args.entries, max_pinned=args.pinned, max_bytes=int(args.gib * 1024**3))


def start_server(args, arm, disk, index, clock):
    """A fresh process: empty memory cache, the disk set (or its preload) loaded as boundaries, store attached."""
    store_cls, keep, preload = arm
    cache = new_cache(args)
    store = store_cls(disk, index, clock, keep, preload)
    for snap in store.load_all():
        cache.add(snap, boundary=True)
    cache.persist = store.persist
    cache.on_find = store.on_find
    if hasattr(store, "fetch"):
        cache.fetch = store.fetch
    return cache, store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--gap-min", type=float, default=30.0)
    ap.add_argument("--keep", type=int, default=32)
    ap.add_argument("--preload", type=int, default=4)
    ap.add_argument("--entries", type=int, default=64)
    ap.add_argument("--pinned", type=int, default=64)
    ap.add_argument("--gib", type=float, default=1.5)
    ap.add_argument("--no-penalty", action="store_true", help="skip the per-request restart counterfactual")
    ap.add_argument("--then-subagents", type=int, default=0,
                    help="after the dump, run N new subagent first turns, restart, and replay the main agent's turn")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.set_verbosity_error()
    engine = Engine.__new__(Engine)
    engine.encoding = load_official_encoding(MODEL_PATH)
    engine.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    rows = [json.loads(line) for line in open(args.dump)]
    replies = [r["reply"] for r in rows if "reply" in r]
    bodies = [r for r in rows if "body" in r]

    arms = {
        "newest8": (Newest8, 8, None),
        "used8": (MemoryStore, 8, 8),
        f"used{args.keep}": (MemoryStore, args.keep, args.preload),
    }
    for arm, spec in arms.items():
        now = [bodies[0]["t"]]
        clock = lambda: now[0]  # noqa: E731
        disk: dict = {}
        index: dict = {}
        engine._own_replies = deque(maxlen=64)
        engine.replies_spliced = 0
        cache, store = start_server(args, spec, disk, index, clock)
        rt = TokenRuntime(cache)
        unused = list(replies)
        total = penalty = restarts = cold_after_restart = 0
        last_t = bodies[0]["t"]
        for row in bodies:
            now[0] = row["t"]
            if row["t"] - last_t > args.gap_min * 60:
                restarts += 1
                engine._own_replies = deque(maxlen=64)
                cache, store = start_server(args, spec, disk, index, clock)
                rt = TokenRuntime(cache)
                just_restarted = True
            else:
                just_restarted = False
            last_t = row["t"]
            req = chat_request(row["body"])
            tokens, images = engine._encode_chat(req)
            tokens = engine._splice_own_replies(tokens, req.thinking_mode)
            boundary = 0 if images else engine.system_prefix_len(req, tokens)

            if not args.no_penalty:
                # the same request on a server restarted just before it; copies, so the real run is untouched
                d2, i2 = dict(disk), json.loads(json.dumps(index))
                c2, _ = start_server(args, spec, d2, i2, clock)
                _, r2 = prepare_prompt(TokenRuntime(c2), tokens, boundaries=(boundary,) if boundary else ())

            _, reused = prepare_prompt(rt, tokens, boundaries=(boundary,) if boundary else ())
            total += len(tokens) - reused
            if just_restarted:
                cold_after_restart += len(tokens) - reused
            if not args.no_penalty:
                penalty += max(0, reused - r2)
            reply = next((r for r in unused if r["prompt_tokens"] == tokens), None)
            if reply is not None:
                unused.remove(reply)
                rt.tokens.extend(reply["reply_tokens"])
                rt.position += len(reply["reply_tokens"])
                cache.add(rt.snapshot())
                text = engine.tokenizer.decode(reply["reply_tokens"], skip_special_tokens=False)
                engine._remember_reply(tokens, reply["reply_tokens"], text, req.thinking_mode)
            if args.verbose:
                print(f"{arm:8s} prompt={len(tokens):6d} system={boundary:6d} reused={reused:6d}"
                      f"{' RESTART' if just_restarted else ''} disk={sorted(len(s.tokens) for _, s in disk.values())}")
        tail = ""
        if args.then_subagents:
            # Job 1's case: another batch of subagents after the session, a restart, then the main agent's turn
            session = [r for r in bodies if r["t"] >= max(
                b["t"] for a, b in zip(bodies, bodies[1:]) if b["t"] - a["t"] > args.gap_min * 60)]
            main_row = max(session, key=lambda r: engine.system_prefix_len(
                chat_request(r["body"]), engine._encode_chat(chat_request(r["body"]))[0]))
            sub_rows = [r for r in bodies if 19000 < len(json.dumps(r["body"]["messages"][0])) and
                        "Keep your final summary tight" in str(r["body"]["messages"][0]["content"])][:6]
            for k in range(args.then_subagents):
                now[0] += 60
                body = json.loads(json.dumps(sub_rows[k % len(sub_rows)]["body"]))
                body["messages"][0]["content"] += f"\n\nBatch {k}."
                req = chat_request(body)
                tokens, _ = engine._encode_chat(req)
                prepare_prompt(rt, tokens, boundaries=(engine.system_prefix_len(req, tokens),))
            now[0] += 3600
            cache, store = start_server(args, spec, disk, index, clock)
            engine._own_replies = deque(maxlen=64)
            req = chat_request(main_row["body"])
            tokens, _ = engine._encode_chat(req)
            _, reused = prepare_prompt(TokenRuntime(cache), tokens, boundaries=(engine.system_prefix_len(req, tokens),))
            tail = (f"; after {args.then_subagents} more subagents and a restart, the main turn "
                    f"({len(tokens)} tokens) reused {reused}")
        final = sorted(len(s.tokens) for _, s in disk.values())
        pen = "" if args.no_penalty else f", restart-anywhere penalty {penalty}"
        print(f"{arm:8s} total prefilled {total}, {restarts} restarts, first request after them {cold_after_restart}"
              f"{pen}{tail}; disk at the end: {final}")


if __name__ == "__main__":
    main()
