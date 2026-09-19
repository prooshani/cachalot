"""
Run the coding corpus against an OpenAI-compatible endpoint. No harness.

This is **run A**, the diagnostic. It answers "is the model itself clean?" and
nothing else, which is why it deliberately has no system prompt, no tools, no
retries on content and no scaffolding of any kind. A harness is a confound
here: if a harness-mediated run comes back clean, the harness and the runtime
are indistinguishable as the cause. Run the harness separately as run B.

Only the standard library is used, so it runs anywhere with a Python 3 and a
network connection.

## The two things that silently ruin this run

**Provider routing.** OpenRouter load-balances across providers, and different
providers serve different quantizations of the same model name. Forty requests
can land on three stacks and the arm then measures nothing. `--provider` pins
one and disables fallback; the provider that actually served each request is
recorded per case, and the summary refuses to be quiet if more than one appears.

**Silent dropping.** A case that errors must still be counted. Every case is
written, network failures are retried, and a case that never succeeds is
recorded as failed rather than omitted. `cases_planned` and `cases_completed`
both go in `notes.json`.

## Content is never retried

A network error is retried; a reply that looks broken is not. Retrying bad
content is what run B is for, and doing it here would quietly turn the
diagnostic into a best-of-N and overstate the endpoint.

    export OPENROUTER_API_KEY=sk-or-...
    python3 benchmarks/run_reference.py \\
      --pack ~/cachalot-corpus-pack \\
      --out ~/cachalot-reference-openrouter \\
      --model deepseek/deepseek-v4.1-flash \\
      --provider <provider-slug>
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

# `datetime.UTC` is 3.11+; this script has to run on whatever python3 the
# operator's machine has, so the portable spelling stays. UP017 is exempted for
# this file in pyproject.toml for the same reason.
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"


def call(url: str, key: str, body: dict, timeout: int, attempts: int) -> tuple[dict | None, str]:
    """POST once, retrying transport failures only. Returns (json, error)."""
    data = json.dumps(body).encode()
    last = ""
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/prooshani/cachalot",
                "X-Title": "cachalot-coding-corpus",
            },
            method="POST",
        )
        try:
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
                return json.loads(response.read().decode()), ""
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300]
            last = f"HTTP {exc.code}: {detail}"
            # 4xx other than rate limiting will not fix themselves.
            if exc.code not in (408, 409, 429) and exc.code < 500:
                return None, last
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            wait = min(60, 2 ** attempt)
            print(f"      transport failure, retrying in {wait}s ({last[:80]})", flush=True)
            time.sleep(wait)
    return None, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="directory holding prompts.jsonl and settings.json")
    ap.add_argument("--out", required=True, help="directory to write replies and notes.json into")
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--provider", default="",
                    help="pin one provider slug and disable fallback. Strongly recommended: "
                         "different providers serve different quantizations of the same model")
    ap.add_argument("--temperature", type=float, default=None, help="default: settings.json")
    ap.add_argument("--seeds", default="", help="comma-separated; default: settings.json")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=1.0, help="pause between calls")
    ap.add_argument("--only", default="", help="comma-separated task ids, for a smoke run")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY")

    pack = Path(args.pack)
    settings = json.loads((pack / "settings.json").read_text())
    tasks = [json.loads(line) for line in (pack / "prompts.jsonl").read_text().splitlines() if line.strip()]
    if args.only:
        wanted = {t.strip() for t in args.only.split(",") if t.strip()}
        unknown = wanted - {t["id"] for t in tasks}
        if unknown:
            raise SystemExit(f"unknown task ids: {sorted(unknown)}")
        tasks = [t for t in tasks if t["id"] in wanted]

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else settings["seeds"])
    temperature = args.temperature if args.temperature is not None else settings["temperature"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    planned = len(tasks) * len(seeds)

    notes = {
        "endpoint": args.url,
        "model_requested": args.model,
        "provider_pinned": args.provider or None,
        "harness": "none (bare API calls) -- this is run A, the diagnostic",
        "corpus": settings.get("corpus"),
        "settings_requested": {
            "temperature": temperature,
            "top_p": settings.get("top_p", 1.0),
            "max_tokens": settings["max_new_tokens"],
            "frequency_penalty": settings["frequency_penalty"],
            "seeds": seeds,
        },
        "unsupported_settings": [
            "penalty_window: Cachalot applies the frequency penalty over a 128-token "
            "sliding window; the OpenAI API has no window parameter and applies it to "
            "the whole context"
        ],
        "retries_or_tooling": "transport failures retried; content never retried",
        "cases_planned": planned,
        "cases_completed": 0,
        "cases_failed": 0,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "cases": [],
    }
    (out / "notes.json").write_text(json.dumps(notes, indent=2))

    print(f"{len(tasks)} tasks x {len(seeds)} seeds = {planned} cases -> {out}")
    print(f"model {args.model}"
          + (f", provider pinned to {args.provider}" if args.provider else
             "  *** NO PROVIDER PINNED: routing may mix quantizations ***"))
    print()

    finish = Counter()
    providers = Counter()
    served = Counter()

    for seed in seeds:
        for task in tasks:
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": task["prompt"]}],
                "temperature": temperature,
                "top_p": settings.get("top_p", 1.0),
                "max_tokens": settings["max_new_tokens"],
                "frequency_penalty": settings["frequency_penalty"],
                "seed": seed,
            }
            if args.provider:
                body["provider"] = {"order": [args.provider], "allow_fallbacks": False}

            payload, error = call(args.url, key, body, args.timeout, args.attempts)
            name = f"{task['id']}_seed{seed}.txt"
            case = {"task": task["id"], "seed": seed, "file": name}

            if payload is None:
                case.update({"ok": False, "error": error})
                notes["cases_failed"] += 1
                # Write the file anyway so the case cannot vanish from the count.
                (out / name).write_text("")
                print(f"  {task['id']:28s} seed {seed} | FAILED | {error[:70]}", flush=True)
            else:
                choice = (payload.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                reason = choice.get("finish_reason") or "?"
                provider = payload.get("provider") or ""
                model_served = payload.get("model") or ""
                usage = payload.get("usage") or {}
                (out / name).write_text(text)
                case.update({
                    "ok": True,
                    "finish_reason": reason,
                    "provider": provider,
                    "model_served": model_served,
                    "completion_tokens": usage.get("completion_tokens"),
                    "chars": len(text),
                })
                notes["cases_completed"] += 1
                finish[reason] += 1
                if provider:
                    providers[provider] += 1
                if model_served:
                    served[model_served] += 1
                print(f"  {task['id']:28s} seed {seed} | {reason:8s} | "
                      f"{usage.get('completion_tokens', '?')} tok | {provider or '?'}",
                      flush=True)

            notes["cases"].append(case)
            (out / "notes.json").write_text(json.dumps(notes, indent=2))
            time.sleep(args.sleep)

    notes["finish_reasons"] = dict(finish)
    notes["providers_seen"] = dict(providers)
    notes["models_served"] = dict(served)
    notes["finished_utc"] = datetime.now(timezone.utc).isoformat()
    notes["complete"] = notes["cases_completed"] + notes["cases_failed"] == planned
    (out / "notes.json").write_text(json.dumps(notes, indent=2))

    print()
    print(f"completed {notes['cases_completed']}, failed {notes['cases_failed']}, "
          f"planned {planned}")
    print(f"finish reasons: {dict(finish)}")
    print(f"providers seen: {dict(providers) or 'not reported by the endpoint'}")
    print(f"models served:  {dict(served) or 'not reported by the endpoint'}")

    if len(providers) > 1:
        print("\n*** MORE THAN ONE PROVIDER SERVED THIS RUN. ***")
        print("Different providers serve different quantizations, so these cases are")
        print("not one arm. Re-run with --provider pinned to a single slug.")
        sys.exit(2)
    if notes["cases_failed"]:
        print(f"\n{notes['cases_failed']} cases failed and were written empty. They stay in the")
        print("denominator; say so when handing the directory over.")
    print(f"\nwrote {out}")
    print("Hand back the whole directory, including notes.json.")


if __name__ == "__main__":
    main()
