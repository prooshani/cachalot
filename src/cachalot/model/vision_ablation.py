"""
Debug switches that undo one of section 16.4's vision fixes at a time, for the ablation of HANDOFF section 15.7.

CACHALOT_VISION_ABLATE is a comma-separated list of:

  delims       image-span delimiter positions get zero vectors instead of the learned
               image_start / image_newline / image_end rows
  engram_mask  image positions take part in Engram like text (hash history alive, gate open)
  bias_vl      image rows route with the text gate bias instead of bias_vl

Empty (the default) changes nothing. Image identity in the prefix cache is a correctness guard, not a
candidate, and has no switch.
"""

from __future__ import annotations

import os
import sys

KNOWN = ("delims", "engram_mask", "bias_vl")


def ablated(name: str) -> bool:
    return name in _active()


def _active() -> frozenset[str]:
    raw = os.environ.get("CACHALOT_VISION_ABLATE", "")
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def announce() -> None:
    active = _active()
    unknown = active - set(KNOWN)
    if unknown:
        raise ValueError(f"CACHALOT_VISION_ABLATE: unknown {sorted(unknown)}; known {KNOWN}")
    if active:
        print(f"[vision] ABLATION ACTIVE, answers are not the shipped model's: {sorted(active)}",
              file=sys.stderr, flush=True)
