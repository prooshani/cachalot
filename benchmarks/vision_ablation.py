"""
Score image answers from a running server, for the ablation of section 16.4's vision fixes (HANDOFF section
15.7). Run it once per arm against a server started with CACHALOT_VISION_ABLATE set (see vision_ablation.sh).

Each case is an image with content that can be checked mechanically: synthetic pictures drawn here with PIL,
whose shapes, colours and labels are known, and the checkpoint's own KV-cache chart, whose numbers are
printed on it. Every question asks for a short answer, decoded greedily, and is scored by the fraction of
expected facts (regular expressions) found in the answer.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/vision_ablation.py --arm none
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks/results/vision_ablation"
CHART = Path("/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/assets/dsv41_kv_cache.png")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _font(size):
    return ImageFont.truetype(FONT, size)


def shapes_image() -> bytes:
    im = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((40, 60, 240, 260), fill=(30, 80, 220))
    d.text((60, 300), "ALPHA 17", fill="black", font=_font(40))
    d.polygon([(420, 60), (330, 260), (510, 260)], fill=(30, 170, 60))
    d.text((340, 300), "BRAVO 58", fill="black", font=_font(40))
    return _png(im)


def count_image() -> bytes:
    im = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(im)
    for x in (60, 200, 340):
        d.ellipse((x, 50, x + 90, 140), fill=(220, 30, 30))
    for x in (120, 330):
        d.rectangle((x, 220, x + 90, 310), fill=(30, 60, 220))
    return _png(im)


def invoice_image() -> bytes:
    im = Image.new("RGB", (720, 360), "white")
    d = ImageDraw.Draw(im)
    lines = ["INVOICE No. 7342", "Customer: Lindqvist Marine AB", "Date: 2026-03-14", "Total: 1,289.50 EUR"]
    for i, line in enumerate(lines):
        d.text((40, 40 + 70 * i), line, fill="black", font=_font(36))
    return _png(im)


def grid_image() -> bytes:
    """A 3x3 grid of letters: row and column structure is what the span's newline delimiters mark."""
    im = Image.new("RGB", (600, 600), "white")
    d = ImageDraw.Draw(im)
    for r, row in enumerate(("KQZ", "MWD", "PXH")):
        for c, ch in enumerate(row):
            d.rectangle((20 + 190 * c, 20 + 190 * r, 190 + 190 * c, 190 + 190 * r), outline="black", width=4)
            d.text((75 + 190 * c, 55 + 190 * r), ch, fill="black", font=_font(90))
    return _png(im)


GRID6 = ("BHKQTZ", "CJMRVX", "DFLNPW", "GSUYAE", "OIQKBM", "RTZHCL")


def grid6_image() -> bytes:
    """A 6x6 letter grid: six newline delimiters' worth of rows, dense enough to lose a row or a column."""
    im = Image.new("RGB", (720, 720), "white")
    d = ImageDraw.Draw(im)
    for r, row in enumerate(GRID6):
        for c, ch in enumerate(row):
            d.rectangle((12 + 116 * c, 12 + 116 * r, 124 + 116 * c, 124 + 116 * r), outline="black", width=3)
            d.text((45 + 116 * c, 35 + 116 * r), ch, fill="black", font=_font(60))
    return _png(im)


TABLE = [("Oslo", "412"), ("Bergen", "96"), ("Tromso", "1,305"), ("Stavanger", "58"), ("Bodo", "774"),
         ("Alesund", "2,019"), ("Narvik", "133"), ("Molde", "640"), ("Hamar", "27"), ("Skien", "3,881")]


def table_image() -> bytes:
    """Ten rows of a two-column table: many rows, each a (name, number) pair to keep aligned."""
    im = Image.new("RGB", (640, 720), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 20), "City", fill="black", font=_font(32))
    d.text((400, 20), "Orders", fill="black", font=_font(32))
    d.line((30, 64, 610, 64), fill="black", width=3)
    for i, (city, n) in enumerate(TABLE):
        y = 80 + 62 * i
        d.text((40, y), city, fill="black", font=_font(30))
        d.text((400, y), n, fill="black", font=_font(30))
        d.line((30, y + 54, 610, y + 54), fill=(170, 170, 170), width=1)
    return _png(im)


SMALL = ["Serial KX-4471-B", "Batch 20260311", "Voltage 48 V", "Weight 3.25 kg", "Made in Tampere"]


def small_text_image() -> bytes:
    """Five lines at 15 px on a 900x500 canvas: text a few patches tall."""
    im = Image.new("RGB", (900, 500), "white")
    d = ImageDraw.Draw(im)
    for i, line in enumerate(SMALL):
        d.text((30 + 150 * (i % 2), 40 + 85 * i), line, fill="black", font=_font(15))
    return _png(im)


def _png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


CASES = [
    ("shapes", shapes_image, "Describe each shape: its colour, its kind and the label under it. Be brief.",
     [r"blue", r"square|rectangle", r"ALPHA\s*17", r"green", r"triangle", r"BRAVO\s*58"]),
    ("count", count_image, "How many red circles and how many blue squares are there? Answer with two numbers.",
     [r"\b3\b|three", r"\b2\b|two"]),
    ("invoice", invoice_image, "Transcribe the invoice number, customer, date and total exactly.",
     [r"7342", r"Lindqvist", r"Marine", r"2026-03-14", r"1,?289\.50", r"EUR"]),
    ("grid", grid_image, "Read the 3x3 grid row by row, top to bottom, as three strings of three letters.",
     [r"KQZ|K,?\s*Q,?\s*Z", r"MWD|M,?\s*W,?\s*D", r"PXH|P,?\s*X,?\s*H"]),
    ("chart", lambda: CHART.read_bytes(), "What is the chart's title, and which four byte values does it show?",
     [r"KV\s*Cache", r"389,?120", r"48,?068", r"3,?514", r"\b890\b"]),
    # Harder cases (v42 Job 4): dense layout, many rows, small text.
    ("grid6", grid6_image, "Read the 6x6 grid row by row, top to bottom, as six strings of six letters.",
     [r",?\s*".join(row) for row in GRID6]),
    ("table", table_image, "List every row of the table as city: orders.",
     [rf"{city}\W+{re.escape(n)}\b" for city, n in TABLE]),
    ("small", small_text_image, "Transcribe every line of text exactly.",
     [re.escape(line).replace(r"\ ", r"\s*") for line in SMALL]),
]


def ask(url: str, image: bytes, question: str, max_tokens: int) -> tuple[str, float]:
    uri = "data:image/png;base64," + base64.b64encode(image).decode()
    body = {"model": "deepseek-v4.1-flash", "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": question}]}]}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        reply = json.load(r)
    return reply["choices"][0]["message"]["content"] or "", time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8011/v1/chat/completions")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--cases", default="", help="comma-separated case names; empty runs all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0.0
    cases = [c for c in CASES if not args.cases or c[0] in args.cases.split(",")]
    for name, make, question, facts in cases:
        answer, secs = ask(args.url, make(), question, args.max_tokens)
        hits = [bool(re.search(f, answer, re.I)) for f in facts]
        score = sum(hits) / len(hits)
        total += score
        row = {"arm": args.arm, "case": name, "score": round(score, 3), "missed": [f for f, h in zip(facts, hits)
               if not h], "seconds": round(secs, 1), "answer": answer}
        with open(OUT / "scores.jsonl", "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{args.arm:12s} {name:8s} {score:.2f} missed={row['missed']} ({secs:.0f} s)", flush=True)
    print(f"{args.arm:12s} mean score {total / len(cases):.3f}", flush=True)


if __name__ == "__main__":
    main()
