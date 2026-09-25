"""
At a position where the model fails to copy, where does the attention go?

**Why this exists.** By 2026-09-20 the decode path had been read line by line
against the implementation the checkpoint ships and matched at every one of
about forty points, while the runtime still fails to copy a word it emitted ten
tokens earlier (HANDOFF section 7.4.7). Reading further is not the way to
separate what is left, because the two remaining candidates -- a weight loaded
into the wrong slot, and accumulated precision -- are both invisible to a
comparison of formulas.

This measures the thing itself. The sliding window is 128 and every layer
attends over the whole ring, so a word fewer than 128 tokens back is guaranteed
to be in the attended set. Teacher-force up to a known failing position, record
the attention distribution each layer produces, and map the window slots back to
absolute token positions.

  * mass lands on the source token and the answer is still wrong -> retrieval
    works; the fault is in the value path or in the weights behind it
  * mass is diffuse or lands elsewhere -> retrieval itself fails even though
    every formula matches, which points at the query or key content

It patches `cachalot.model.sparse_attn_mlx.sparse_attention`, so it must run
with CACHALOT_FUSED_DECODE=0; the fused kernel computes the same thing but does
not expose the distribution. Both were measured to give identical corpus
numbers, so the unfused path is the right instrument.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh \
      --budget-gib 24 --max-seconds 5400 --tag attnmass -- env \
      CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_EXPERT_BANK=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_PAGE_CACHE=1 CACHALOT_FUSED_DECODE=0 PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/attention_mass_probe.py \
      --probe <probe.txt> --position 333
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
import cachalot.model.sparse_attn_mlx as sam  # noqa: E402

WINDOW = 128

#: filled by the patched sparse_attention while the target step runs
CAPTURED: list[dict] = []
RECORDING = False


def _patch() -> None:
    original = sam.sparse_attention

    def recording_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale):
        out = original(q, kv, attn_sink, topk_idxs, softmax_scale)
        if RECORDING:
            # recompute the distribution the same way the reference does, so the
            # numbers reported are the ones the attention actually used
            idxs = topk_idxs[0]
            valid = idxs >= 0
            safe = mx.where(valid, idxs, mx.zeros_like(idxs))
            sel = mx.take(kv, safe, axis=0).astype(mx.float32)
            scores = mx.matmul(q[0].astype(mx.float32), sel.T) * softmax_scale
            scores = mx.where(valid[None, :], scores, mx.full(scores.shape, -mx.inf))
            sink = attn_sink.astype(mx.float32)[:, None]
            m = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
            w = mx.exp(scores - m)
            denom = mx.sum(w, axis=-1, keepdims=True) + mx.exp(sink - m)
            probs = w / denom
            # Mean over heads is a weak statistic: induction behaviour often lives
            # in a handful of heads and averaging over 64 buries it. Keep the max
            # too, and the head it came from.
            # Does the retrieved value actually reach the output? In this
            # architecture the same kv vector is both key and value, so a head
            # that puts p on one slot should emit roughly p * that vector. The
            # cosine between the head's output and the vector it attended to
            # says whether the value path delivered what selection chose.
            best_slot = mx.argmax(probs, axis=-1)
            deliver = []
            for h in range(probs.shape[0]):
                slot = int(best_slot[h].item())
                idx = int(idxs[slot].item())
                if idx < 0:
                    deliver.append(None)
                    continue
                v = kv[idx].astype(mx.float32)
                oh = out[0, h].astype(mx.float32)
                denom = float((mx.linalg.norm(v) * mx.linalg.norm(oh)).item())
                cos = float(mx.sum(v * oh).item()) / denom if denom else 0.0
                deliver.append({
                    "head": h,
                    "slot": slot,
                    "idx": idx,
                    "prob": float(probs[h, slot].item()),
                    "cos_out_vs_attended_value": cos,
                    "out_norm": float(mx.linalg.norm(oh).item()),
                    "value_norm": float(mx.linalg.norm(v).item()),
                })
            CAPTURED.append({
                "delivery": deliver,
                "idxs": [int(v) for v in idxs.tolist()],
                "probs_mean_over_heads": [float(v) for v in mx.mean(probs, axis=0).tolist()],
                "probs_max_over_heads": [float(v) for v in mx.max(probs, axis=0).tolist()],
                "argmax_head": [int(v) for v in mx.argmax(probs, axis=0).tolist()],
                "sink_mass": float(mx.mean(mx.exp(sink - m) / denom).item()),
            })
        return out

    sam.sparse_attention = recording_sparse_attention


def slot_to_position(slot: int, start_pos: int, window: int = WINDOW) -> int:
    """Ring slot -> absolute token position, for the newest occupant of that slot."""
    if slot < 0:
        return -1
    pos = slot + window * ((start_pos - slot) // window)
    return pos if 0 <= pos <= start_pos else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--position", type=int, required=True,
                    help="teacher-forced position whose attention to capture")
    ap.add_argument("--prefill", type=int, default=16)
    ap.add_argument("--top", type=int, default=8, help="attended positions to print per layer")
    ap.add_argument("--layers", default="", help="comma-separated layer indices; default: all captured")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    global RECORDING
    _patch()
    text = Path(args.probe).read_text()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        print(f"runtime ready (bank {Path(rt.expert_bank_path).name})", flush=True)
        ids = list(rt.tokenizer.encode(text))[: args.position + 1]
        if len(ids) <= args.position:
            raise SystemExit(f"probe has {len(ids)} tokens, need > {args.position}")

        # The step that PREDICTS ids[position] is the one that consumes
        # ids[position - 1], so recording has to happen on that step and not on
        # the one after it.
        rt.reset()
        rt.prefill_tokens(ids[: args.prefill])
        for i in range(args.prefill, args.position - 1):
            rt.decode_token(ids[i])

        RECORDING = True
        result = rt.decode_token(ids[args.position - 1])
        RECORDING = False

        logits = result.logits.astype(mx.float32)
        logp = logits - mx.logsumexp(logits)
        target = ids[args.position]
        rank = int((logits > logits[target]).sum().item())
        context = rt.tokenizer.decode(ids[max(0, args.position - 10): args.position])
        decode = rt.tokenizer.decode

    print(f"\nposition {args.position}, context ...{context!r}")
    print(f"correct next token {decode([target])!r}: rank {rank}, logprob {float(logp[target].item()):+.3f}")
    print(f"captured {len(CAPTURED)} attention calls (one per attention layer)\n")

    wanted = ({int(v) for v in args.layers.split(",") if v.strip()}
              if args.layers else set(range(len(CAPTURED))))

    rows = []
    for layer, cap in enumerate(CAPTURED):
        if layer not in wanted:
            continue
        pairs = []
        stat = cap.get("probs_max_over_heads") or cap["probs_mean_over_heads"]
        for slot_index, (idx, prob) in enumerate(zip(cap["idxs"], stat)):
            # the first WINDOW entries are ring slots; anything beyond is compressed
            pos = slot_to_position(idx, args.position - 1) if slot_index < WINDOW else -1
            pairs.append((prob, idx, pos, slot_index < WINDOW))
        pairs.sort(reverse=True)
        top = pairs[: args.top]
        # Window mass must come from the mean: it is a share of one token's
        # probability, and summing per-slot maxima over heads is not a
        # probability at all (it happily exceeds 1).
        window_mass = sum(
            prob for slot_index, prob in enumerate(cap["probs_mean_over_heads"])
            if slot_index < WINDOW
        )
        print(f"layer {layer:>2} | sink {cap['sink_mass']:.3f} | window mass {window_mass:.3f} (best-head probabilities)")
        shown = []
        for prob, idx, pos, is_w in top:
            tok = decode([ids[pos]]) if 0 <= pos < len(ids) else "?"
            kind = "win" if is_w else "cmp"
            shown.append(f"{prob:.3f}@{pos if is_w else idx}{'' if is_w else '(c)'}:{tok!r}")
        print("          " + "  ".join(shown))
        best = [d for d in (cap.get("delivery") or []) if d]
        if best:
            top_heads = sorted(best, key=lambda d: -d["prob"])[:3]
            print("          delivery: " + "  ".join(
                f"h{d['head']}@{slot_to_position(d['idx'], args.position - 1)}"
                f" p={d['prob']:.2f} cos={d['cos_out_vs_attended_value']:+.2f}"
                for d in top_heads))
        rows.append({"layer": layer, "sink": cap["sink_mass"],
                     "delivery": sorted(best, key=lambda d: -d["prob"])[:3],
                     "window_mass": window_mass,
                     "top": [{"prob": p, "idx": i, "pos": q, "window": w} for p, i, q, w in top]})

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"position": args.position, "rank": rank, "context": context, "layers": rows}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
