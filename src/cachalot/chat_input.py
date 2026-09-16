"""
Typing-time prefill for the terminal chat.

A follow-up turn adds ~20 prompt tokens: two template tokens, the user's
message, and a two-token template tail. Prefilling them costs 0.2-0.3 s per
token at the SSD floor, so the visible pause after Enter is 3-5 s while the
machine idled the whole time the user was typing.

RawLineReader reads the terminal one keystroke at a time and, whenever the
user pauses for `idle_seconds`, hands the partial message to a
SpeculativePrefiller. The prefiller tokenizes `template head + partial text`
and runs prepare_prompt on it (on the chat thread; MLX streams are
thread-local), which restores the longest cached prefix, prefills only the
new words and adds a snapshot to the prefix cache. When the user presses Enter the real prompt shares that prefix and
stream_tokens reuses it; only the last word and the template tail remain.

Partial text is cut at a single space between two non-space characters. The
DeepSeek tokenizer starts word tokens with a leading space, so tokens of
`head + "write a story"` are a prefix of tokens of `head + "write a story of"`
(40/40 checked positions); runs of spaces and newlines merge into one token
and are never used as cut points. A mismatch is harmless: the prefix cache
falls back to the longest matching earlier snapshot.
"""
from __future__ import annotations

import codecs
import os
import select
import sys
import time
from time import perf_counter

_SENTINEL = "@@cachalot-user-content-7f3a@@"


def split_prompt_template(encoding, messages: list[dict], *, thinking_mode: str, reasoning_effort) -> tuple[str, str]:
    """(head, tail) such that the prompt for a new user message `t` is head + t + tail."""
    probe = encoding.encode_messages(
        messages + [{"role": "user", "content": _SENTINEL}],
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
    )
    head, sep, tail = probe.partition(_SENTINEL)
    if not sep:
        raise ValueError("chat template did not reproduce the user content verbatim")
    return head, tail


def safe_partial(text: str) -> str | None:
    """
    Longest prefix of `text` that ends right before a single space between two
    non-space characters; "" when no such boundary exists yet; None when the
    text is a slash command and must not be prefilled.
    """
    if text.startswith("/"):
        return None
    cut = -1
    for i in range(1, len(text) - 1):
        if text[i] == " " and not text[i - 1].isspace() and not text[i + 1].isspace():
            cut = i
    return text[:cut] if cut > 0 else ""


class SpeculativePrefiller:
    """
    Prefills template head + partial user text while the user types.

    Runs on the calling (chat) thread, from RawLineReader's idle callback:
    MLX streams are thread-local in the Metal backend, so the runtime's graphs
    cannot be evaluated from a worker thread. A speculative prefill therefore
    blocks keystroke echo for the duration of one or two words (0.25-0.5 s);
    keys typed meanwhile stay in the terminal buffer and are echoed after.
    """

    def __init__(self, runtime, tokenizer, *, verbose: bool = False) -> None:
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.verbose = verbose
        self._head: str | None = None
        self._done: str | None = None
        self.runs = 0
        self.prefilled_tokens = 0
        self.seconds = 0.0

    def set_context(self, head: str) -> None:
        """New turn: remember the template head; the first idle callback prefills it."""
        self._head = head
        self._done = None

    def update(self, text: str) -> None:
        partial = safe_partial(text)
        if partial is None or self._head is None or partial == self._done:
            return
        from cachalot.model.generation import prepare_prompt

        ids = list(self.tokenizer.encode(self._head + partial))
        try:
            t0 = perf_counter()
            _, reused = prepare_prompt(self.runtime, ids, reset=True, use_prefix_cache=True)
            dt = perf_counter() - t0
        except Exception as exc:  # never let a speculative failure kill the chat
            print(f"\r\n[typing prefill failed: {exc!r}]", file=sys.stderr, flush=True)
            self._head = None
            return
        self._done = partial
        self.runs += 1
        self.prefilled_tokens += len(ids) - reused
        self.seconds += dt
        if self.verbose:
            print(f"\r\n[typing prefill: {len(ids) - reused} new tokens, {dt:.2f}s]", file=sys.stderr, flush=True)

    def wait_idle(self) -> None:
        return

    def close(self) -> None:
        return


class RawLineReader:
    """
    Line editor on a raw terminal: printable characters, backspace, Enter,
    Ctrl-D (EOF on an empty line); escape sequences (arrow keys) are ignored.
    Calls `on_idle(text)` when no key arrived for `idle_seconds` and the text
    changed since the last call. Ctrl-C raises KeyboardInterrupt as usual.
    """

    def __init__(self, on_idle=None, *, idle_seconds: float = 0.4) -> None:
        self.on_idle = on_idle
        self.idle_seconds = idle_seconds

    def readline(self, prompt: str) -> str:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        buf: list[str] = []
        notified: str | None = None
        last_key = time.monotonic()
        # TCSANOW, not tty.setcbreak's default TCSAFLUSH: keystrokes typed
        # while the model was still generating must survive the mode switch.
        tty.setcbreak(fd, termios.TCSANOW)
        try:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            while True:
                ready, _, _ = select.select([fd], [], [], 0.05)
                if ready:
                    data = decoder.decode(os.read(fd, 4096))
                    for ch in data:
                        if ch in "\r\n":
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                            return "".join(buf)
                        if ch == "\x04":
                            if not buf:
                                sys.stdout.write("\n")
                                raise EOFError
                            continue
                        if ch in "\x7f\x08":
                            if buf:
                                buf.pop()
                                sys.stdout.write("\b \b")
                            continue
                        if ch == "\x1b":
                            break  # drop the rest of this chunk (cursor keys etc.)
                        if ch.isprintable() or ch == "\t":
                            buf.append(ch)
                            sys.stdout.write(ch)
                    sys.stdout.flush()
                    last_key = time.monotonic()
                elif self.on_idle is not None and time.monotonic() - last_key >= self.idle_seconds:
                    text = "".join(buf)
                    if text != notified:
                        notified = text
                        self.on_idle(text)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
