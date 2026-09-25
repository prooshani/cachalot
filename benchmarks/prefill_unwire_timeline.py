"""
Timeline of one chunked prefill against the machine's wired memory (HANDOFF section 15.11).

Prefills N filler tokens (decode_vs_context.py's prompt) without the prefix cache, polls vm_stat every 0.2 s
from a side thread, prints every jump in wired / compressor memory next to the chunk boundaries, and ends
with one RESULT line: per-chunk seconds, minimum wired GiB, prefill keep-alive evals, how many chunk layers
took Engram rows read during the previous chunk (section 15.12), and the final logits' argmax and sum
(bit-identity across arms). Separate processes per arm, alternated (the section 15.12 A/B swaps
CACHALOT_PREFILL_KEEPALIVE for CACHALOT_ENGRAM_LOOKAHEAD=0/1):

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    for k in 0 0.5 0.5 0; do env CACHALOT_PREFILL_KEEPALIVE=$k CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=80 \
      CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json CACHALOT_HOTLIST_GIB=8 PYTHONPATH=src:benchmarks \
      ~/venvs/deepseek-v41/bin/python benchmarks/prefill_unwire_timeline.py 12288 2>&1 | grep RESULT; done
"""
import sys, time, threading, subprocess, re
sys.path.insert(0, "benchmarks")
import decode_vs_context as dvc
from cachalot.model.api import V41Model
from cachalot.model.generation import load_official_encoding, prepare_prompt
from _common import MODEL_PATH

T0 = time.perf_counter()
print(f"EPOCH T0={time.time():.3f}", flush=True)  # aligns EV times with an outside trace (iostat, section 15.13)
def now(): return time.perf_counter() - T0
events = []
def ev(s): events.append((now(), s)); print(f"{now():8.2f} EV {s}", flush=True)

stop = False
last_a = [0.0]
minw = [999.0]
def poll():
    last = None
    while not stop:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        n = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(.+?):\s+(\d+)\.", out, re.M)}
        w = n["Pages wired down"] * 16384 / 2**30; c = n["Pages occupied by compressor"] * 16384 / 2**30
        import mlx.core as mx
        a = mx.get_active_memory() / 2**30; k = mx.get_cache_memory() / 2**30; pk = mx.get_peak_memory() / 2**30
        if last is None or abs(w - last) > 3 or abs(a - last_a[0]) > 2:
            print(f"{now():8.2f} VM wired={w:.1f} comp={c:.1f} active={a:.1f} cache={k:.1f} peak={pk:.1f}", flush=True); last = w; last_a[0] = a
        minw[0] = min(minw[0], w) if now() > 30 else minw[0]
        time.sleep(0.2)

threading.Thread(target=poll, daemon=True).start()
model = V41Model.from_pretrained(MODEL_PATH, max_seq_len=65536, expert_cache_budget_bytes=int(52 * 2**30))
rt = model.runtime
ev("loaded")
orig_pf = rt.prefill_tokens
def pf(ids, **kw):
    ev(f"chunk start n={len(ids)} pos={rt.position}")
    r = orig_pf(ids, **kw)
    ev("chunk end")
    return r
rt.prefill_tokens = pf
orig_eng = rt._start_engram_prefetch
def eng(rows):
    ev("engram prefetch start"); r = orig_eng(rows); ev("engram prefetch issued"); return r
rt._start_engram_prefetch = eng
enc = load_official_encoding(MODEL_PATH)
# FILLER_FILE / FILLER_OFFSET: take the filler from another file, starting at a character offset, so each arm
# of an A/B reads Engram rows (and experts) no earlier run left in the page cache (section 15.12).
import os
if os.environ.get("FILLER_FILE"):
    _text = open(os.environ["FILLER_FILE"]).read()[int(os.environ.get("FILLER_OFFSET", "0")):]
    _orig_read = dvc.Path.read_text
    dvc.Path.read_text = lambda self, *a, **k: _text if self.name == "HANDOFF.md" else _orig_read(self, *a, **k)
prompt = dvc.build(rt.tokenizer, enc, int(sys.argv[1]))
ev(f"prompt {len(prompt)}")
res, _ = prepare_prompt(rt, prompt, use_prefix_cache=False)
import mlx.core as mx
ev("prefill done")
starts = [t for t, e in events if e.startswith("chunk start")]
ends = [t for t, e in events if e == "chunk end"]
lg = res.logits.astype(mx.float32)
import os
print("RESULT filler=%s@%s keepalive=%s lookahead=%s chunks=%s total=%.1f min_wired=%.1f keepalives=%d lookahead_hits=%d "
      "argmax=%d logit_sum=%.6f" % (
    os.path.basename(os.environ.get("FILLER_FILE", "HANDOFF.md")), os.environ.get("FILLER_OFFSET", "0"),
    os.environ.get("CACHALOT_PREFILL_KEEPALIVE", "0.5"), os.environ.get("CACHALOT_ENGRAM_LOOKAHEAD", "0"),
    [round(b - a, 1) for a, b in zip(starts, ends)], ends[-1] - starts[0], minw[0], rt.prefill_keepalives,
    getattr(rt, "engram_lookahead_hits", 0), int(lg.argmax().item()), float(lg.sum().item())), flush=True)
time.sleep(3)
stop = True
