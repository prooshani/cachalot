"""
Ask macOS to schedule this process like the focused application.

With the display on, window compositing competes with decode for the GPU. On Hamed's three-display Mac this
doubled the non-read part of a streaming token (HANDOFF section 15.7): decode fell from ~7.2 to 4.3-4.5 tok/s
at identical expert reads. Setting the process's Darwin role to UI_FOCAL (the role macOS gives the frontmost
app) won every display-on pair measured, by 6-40 %, and changed nothing with the display off. It changes
scheduling only, never numerics.

`serve` and `chat` apply it at startup. CACHALOT_DARWIN_ROLE picks another role (0 = leave the default).
"""

from __future__ import annotations

import ctypes
import os
import sys

PRIO_DARWIN_ROLE = 6
ROLE_UI_FOCAL = 1


def apply_darwin_role(default: int = ROLE_UI_FOCAL) -> int | None:
    """Set this process's Darwin role; returns the role now in effect, or None
    when it was left alone (not macOS, role 0, or the call failed)."""
    if sys.platform != "darwin":
        return None
    try:
        role = int(os.environ.get("CACHALOT_DARWIN_ROLE", default))
    except ValueError:
        role = default
    if role <= 0:
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.setpriority(PRIO_DARWIN_ROLE, 0, role) != 0:
            return None
        return int(libc.getpriority(PRIO_DARWIN_ROLE, 0))
    except (OSError, AttributeError):
        return None
