"""
Drive `cachalot chat` through a pseudo-terminal with human-paced typing and
report the follow-up turn's prefill line, to measure typing-time prefill
(cachalot.chat_input) end to end.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    benchmarks/guarded_run.sh --budget-gib 28 --tag pty_typing -- env PYTHONPATH=src \
        ~/venvs/deepseek-v41/bin/python benchmarks/chat_pty_typing.py [--no-typing-prefill]
"""
from __future__ import annotations

import argparse
import os
import pty
import re
import select
import sys
import time

PY = os.path.expanduser("~/venvs/deepseek-v41/bin/python")
STORY = "write a 100 word story of a small fish living in a greek."


class Chat:
    def __init__(self, extra_args):
        self.log = b""
        self.pos = 0  # end of the last match; expect() never looks behind it
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            env = dict(os.environ, PYTHONPATH="src", CACHALOT_PAGE_CACHE="1", CACHALOT_MLX_WIRED_LIMIT_GIB="52", PYTHONFAULTHANDLER="1")
            os.execvpe(PY, [PY, "-m", "cachalot.cli", "chat", "--expert-budget-gib", "28", "--max-seq-len", "8192",
                            "--max-new-tokens", "48", "--temperature", "0", *extra_args], env)

    def expect(self, pattern: str, timeout: float) -> re.Match:
        deadline = time.monotonic() + timeout
        while True:
            m = re.search(pattern.encode(), self.log[self.pos:])
            if m:
                self.pos += m.end()
                return m
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"waiting for {pattern!r}; tail: {self.log[-400:]!r}")
            r, _, _ = select.select([self.fd], [], [], min(remaining, 0.2))
            if r:
                try:
                    chunk = os.read(self.fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    raise EOFError(f"chat exited; tail: {self.log[-400:]!r}")
                self.log += chunk

    def type(self, text: str, per_char: float = 0.0, word_pause: float = 0.0) -> None:
        for ch in text:
            os.write(self.fd, ch.encode())
            # keep reading while typing so the chat never blocks on a full pty output buffer
            self.drain(word_pause if ch == " " else per_char)

    def drain(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            r, _, _ = select.select([self.fd], [], [], 0.1)
            if r:
                try:
                    self.log += os.read(self.fd, 65536)
                except OSError:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-typing-prefill", action="store_true")
    ap.add_argument("--chat-verbose", action="store_true", help="pass --verbose to cachalot chat")
    ap.add_argument("--per-char", type=float, default=0.08, help="seconds per typed character")
    ap.add_argument("--word-pause", type=float, default=0.7, help="seconds after each space")
    args = ap.parse_args()
    extra = (["--no-typing-prefill"] if args.no_typing_prefill else []) + (["--verbose"] if args.chat_verbose else [])
    chat = Chat(extra)
    try:
        run(chat, args)
    except Exception:
        import signal

        # PYTHONFAULTHANDLER=1: SIGABRT makes the chat dump every thread's stack into the pty log
        try:
            os.kill(chat.pid, signal.SIGABRT)
        except ProcessLookupError:
            pass
        chat.drain(3.0)
        raise
    finally:
        import signal

        log_path = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp",
                                "chat_pty_typing_off.log" if args.no_typing_prefill else "chat_pty_typing_on.log")
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "wb") as fh:
                fh.write(chat.log)
            print(f"pty log: {log_path}", file=sys.stderr)
        except OSError:
            pass

        try:
            os.kill(chat.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(chat.pid, 0)


def run(chat, args):
    STORY_LEN = len(STORY)
    chat.expect(r">>> ", timeout=300)
    chat.type("Hi\r")
    chat.expect(r"tok/s decode", timeout=120)
    chat.expect(r">>> ", timeout=30)
    t0 = time.monotonic()
    chat.type(STORY, per_char=args.per_char, word_pause=args.word_pause)
    typing_s = time.monotonic() - t0
    chat.type("\r")
    m = chat.expect(r"\[prefill (\d+) tokens, reused (\d+), ([\d.]+)s\]", timeout=90)
    n, reused, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    chat.expect(r"tok/s decode", timeout=120)
    chat.expect(r">>> ", timeout=30)
    chat.type("/stats\r")
    chat.expect(r"mlx_peak_bytes", timeout=30)
    chat.drain(1.0)
    stats = {k: v for k, v in re.findall(r'"(typing_prefill_\w+)": ([\d.]+)', chat.log.decode(errors="ignore"))}
    chat.type("/exit\r")
    chat.drain(5.0)
    mode = "typing prefill OFF" if args.no_typing_prefill else "typing prefill ON"
    print(f"{mode}: typed {STORY_LEN} chars in {typing_s:.1f} s | turn 2 prompt {n} tokens, reused {reused}, "
          f"new {n - reused}, prefill after Enter {s:.2f} s | {stats}")
    for line in re.findall(rb"\[prefill [^\]]*\]", chat.log):
        print("   ", line.decode())


if __name__ == "__main__":
    main()
