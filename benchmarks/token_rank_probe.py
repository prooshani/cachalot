"""
Is a specific token mis-ranked, or is everything a bit blurry?

**Why this exists.** The coding corpus showed FP4 dropping the `<` after
`#include` at a high rate -- `#include escaping>`, `#include stdio.h>`,
`#includequeue>`, `#include>` -- 26 of 70 include lines in the first six C++
replies, against essentially clean Python `import` lines. HANDOFF section 7.4.4.

Two explanations fit that, and they call for completely different work:

**Quantization blur.** FP4 is a lossy expert bank and every token's logit moves
a little. `#include <` happens to be a place where one dropped character is
fatal, and it is frequent in C++, so it dominates what a reader notices. If
this is the story, the artefact rate is a bank property, the corpus is the
right instrument, and the performance ranking in section 9 stands.

**A systematic fault.** Something in this runtime -- not the quantization --
ranks one token badly. Blur should degrade everything roughly evenly; a 37 %
failure rate on one highly predictable character is not even.

**This separates them in one forward pass, with no sampling.** Teacher-force
the model over text that contains the construction spelled correctly, and at
each position where the *correct* next token is the one under suspicion, read
where the model actually ranked it. A collapse, a repetition loop and a
compiler are all out of the picture; if `<` sits at rank 1 with high
probability wherever it belongs, generation is not failing because the model
does not know it belongs there.

**Run it on two arms.** `--experts fp4` substitutes dense FP4 expert arithmetic
through `nll_expert_precision.py`'s own patching, which is as close to a clean
reference as this project has without the official implementation. The
production path is the default. If the two agree, the runtime is not adding the
fault and the artefact is the bank's. If they disagree, that is a runtime bug
and it outranks every performance lever in section 9.

This answers "does the model know", not "does the model emit". A token ranked 1
that still comes out wrong under sampling is a different bug, in the sampler.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh \
      --budget-gib 24 --max-seconds 3600 --tag rank-runtime -- env \
      CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts \
      CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
      benchmarks/token_rank_probe.py --tokens 600
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

#: Correctly spelled C++ that is dense in the construction under suspicion, plus
#: Python imports as the within-run control: if `<` is mis-ranked and `import`
#: is not, that is a fact about one token rather than about the whole model.
PROBE_TEXT = """\
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <optional>

import os
import sys
import json
import csv
import argparse
import sqlite3
import threading
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string field;
    while (std::getline(ss, field, delim)) {
        out.push_back(field);
    }
    return out;
}

std::map<std::string, std::vector<int>> group(const std::vector<int>& xs) {
    std::map<std::string, std::vector<int>> out;
    for (const auto& x : xs) {
        out[std::to_string(x % 3)].push_back(x);
    }
    return out;
}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="",
                    help="text file to teacher-force; default: the built-in probe text")
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--prefill", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=5,
                    help="how many competing tokens to print at each interesting position")
    ap.add_argument("--report-worst", type=int, default=15)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    text = Path(args.probe).read_text() if args.probe else PROBE_TEXT

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        bank = Path(rt.expert_bank_path).name
        print(f"runtime ready (bank {bank}, budget "
              f"{rt.expert_cache_budget_bytes / 2**30:.1f} GiB)", flush=True)

        ids = list(rt.tokenizer.encode(text))[: args.prefill + args.tokens + 1]
        if len(ids) < args.prefill + 8:
            raise SystemExit("probe text is too short for the requested prefill")

        rt.reset()
        logits = rt.prefill_tokens(ids[: args.prefill]).logits

        rows = []
        t0 = perf_counter()
        for i in range(args.prefill, min(args.prefill + args.tokens, len(ids) - 1)):
            target = ids[i]
            logp = (logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32)))
            # Rank of the correct token: how many tokens the model preferred.
            rank = int((logits > logits[target]).sum().item())
            order = mx.argsort(-logits)[: args.top_k].tolist()
            rows.append({
                "position": i,
                "target_id": int(target),
                "target": rt.tokenizer.decode([target]),
                "rank": rank,
                "logprob": float(logp[target].item()),
                "top": [
                    {"token": rt.tokenizer.decode([t]), "logprob": float(logp[t].item())}
                    for t in order
                ],
                "prev": rt.tokenizer.decode(ids[max(0, i - 6): i]),
            })
            logits = rt.decode_token(target).logits
        dt = perf_counter() - t0

    print(f"teacher-forced {len(rows)} positions in {dt:.1f} s\n")

    # The construction under suspicion: the token that should follow "#include".
    after_include = [r for r in rows if r["prev"].rstrip().endswith("#include")]
    after_import = [
        r for r in rows
        if r["prev"].rstrip().endswith(("import", "from")) and "#include" not in r["prev"]
    ]

    def summarise(name: str, group: list[dict]) -> dict:
        if not group:
            print(f"{name}: no positions matched")
            return {}
        top1 = sum(1 for r in group if r["rank"] == 0)
        stats = {
            "positions": len(group),
            "rank1": top1,
            "rank1_rate": top1 / len(group),
            "mean_logprob": sum(r["logprob"] for r in group) / len(group),
            "worst_rank": max(r["rank"] for r in group),
        }
        print(f"{name}: {top1}/{len(group)} at rank 1 "
              f"({stats['rank1_rate']:.0%}), mean logprob {stats['mean_logprob']:+.3f}, "
              f"worst rank {stats['worst_rank']}")
        return stats

    print("=== the token that should follow '#include' ===")
    inc = summarise("  after #include", after_include)
    print("=== control: the token that should follow 'import' / 'from' ===")
    imp = summarise("  after import/from", after_import)
    print("=== every position ===")
    allp = summarise("  all", rows)

    print(f"\n=== the {args.report_worst} worst-ranked correct tokens ===")
    for row in sorted(rows, key=lambda r: -r["rank"])[: args.report_worst]:
        competitors = " ".join(f"{t['token']!r}" for t in row["top"][:3])
        print(f"  rank {row['rank']:5d}  want {row['target']!r:14s} "
              f"logprob {row['logprob']:+7.3f}  after ...{row['prev'][-28:]!r}")
        print(f"         model preferred: {competitors}")

    print("\nHow to read this. If the token after '#include' sits at rank 1 with a")
    print("high logprob, the model knows it belongs there and generation is failing")
    print("somewhere after the logits -- look at the sampler, not the bank. If it is")
    print("ranked far down while the 'import' control is fine, one token is mis-ranked")
    print("and that is a systematic fault worth chasing before any performance lever.")
    print("If both are mediocre, it is ordinary quantization blur and the corpus in")
    print("HANDOFF section 7.4.2 is the right instrument. Run both arms before deciding.")

    out = Path(args.out) if args.out else RESULTS_DIR / f"token_rank_probe_{bank}.json"
    out.write_text(json.dumps(
        {"bank": bank, "after_include": inc, "after_import": imp, "all": allp, "rows": rows},
        indent=2,
    ))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
