"""
How good can the one-layer-early routing predictor be?

The decode path prefetches layer L+1's experts by running L+1's router on layer
L's MoE input, because L+1's own input does not exist yet. Measured end to end
that predictor has 55 % precision and serves 41.8 of the 69.5 misses a token
takes early (section 6.1 of docs/HANDOFF.md). The 45 % it gets wrong is 613 MiB
of every token read and never used, on a drive that is busy 80.5 % of decode --
the largest open lever on the FP4 bank (section 9.12).

Before writing a better predictor, bound what a better predictor could win. The
question this answers is not "how good is the current one" but "how much of the
gap is the predictor's and how much is the model's": the input is one layer
stale by construction, so some of the error is not recoverable from that vector
at all, however it is used.

Everything here is offline. It needs no GPU and loads no experts: the recorded
MoE inputs come from benchmarks/capture_activations.py, whose per-layer samples
are token-aligned across layers because every layer sees every token, and the
gate tensors are pread straight out of the checkpoint's shard headers.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/predictor_recall.py \
      benchmarks/results/activations_moe_input_both.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR  # noqa: E402
from cachalot.storage.index import read_safetensors_header  # noqa: E402

N_LAYERS = 40
TOPK = 6


def load_gates(root: Path) -> dict[int, tuple[mx.array, mx.array]]:
    """{layer: (weight [E, H] fp32, bias [E] fp32)}, read tensor by tensor."""
    want = {}
    for layer in range(N_LAYERS):
        want[f"layers.{layer}.ffn.gate.weight"] = (layer, "weight")
        want[f"layers.{layer}.ffn.gate.bias"] = (layer, "bias")

    found: dict[int, dict[str, mx.array]] = {}
    for shard in sorted(root.glob("model-*.safetensors")):
        header, data_start = read_safetensors_header(shard)
        hits = [n for n in header if n in want]
        if not hits:
            continue
        fd = os.open(shard, os.O_RDONLY)
        try:
            for name in hits:
                meta = header[name]
                start, end = meta["data_offsets"]
                raw = os.pread(fd, end - start, data_start + start)
                arr = mx.array(np.frombuffer(raw, dtype=np.uint8))
                arr = arr.view(_DTYPES[meta["dtype"]]).reshape(meta["shape"])
                layer, field = want[name]
                found.setdefault(layer, {})[field] = arr.astype(mx.float32)
        finally:
            os.close(fd)

    gates = {}
    for layer, parts in found.items():
        if "weight" in parts and "bias" in parts:
            gates[layer] = (parts["weight"], parts["bias"])
    return gates


_DTYPES = {"F32": mx.float32, "F16": mx.float16, "BF16": mx.bfloat16}


def gate_scores(x: mx.array, weight: mx.array, bias: mx.array) -> np.ndarray:
    """The selection score the runtime's router ranks on, for all 384 experts."""
    # softplus via logaddexp, which is what the runtime's router uses and does
    # not overflow on the large logits a 5,120-wide dot product produces.
    scores = mx.sqrt(mx.logaddexp(mx.matmul(weight, x), mx.zeros((1,)))) + bias
    return np.array(scores)


def route(x: mx.array, weight: mx.array, bias: mx.array, k: int) -> np.ndarray:
    """The top-k the runtime's router would pick, strongest first."""
    return np.argsort(-gate_scores(x, weight, bias))[:k]


def route_margin(scores: np.ndarray, k: int, margin: float, cap: int) -> np.ndarray:
    """
    Top-k, plus every expert whose score is within `margin` of the k-th.

    The point of this is that a fixed width pays everywhere for a boundary that
    is only ambiguous somewhere. A stale router still knows where it is unsure:
    recall by rank falls from 95.5 % at rank 1 to 40.5 % at rank 6, and the
    experts it loses are the ones sitting near its own cut. `margin` is in units
    of the spread between the strongest and the k-th score, so it does not
    depend on the layer's score scale.
    """
    order = np.argsort(-scores)
    top = order[:k]
    cut = scores[order[k - 1]]
    spread = scores[order[0]] - cut
    if spread <= 0:
        return top
    extra = [e for e in order[k:cap] if (cut - scores[e]) <= margin * spread]
    return np.concatenate([top, np.array(extra, dtype=order.dtype)]) if extra else top


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("activations", help="npz from capture_activations.py")
    ap.add_argument("--model-path", default=str(MODEL_PATH))
    ap.add_argument("--widths", default="6,8,10,12,16,24")
    ap.add_argument("--out", default=str(RESULTS_DIR / "predictor_recall.json"))
    args = ap.parse_args()

    data = np.load(args.activations)
    samples = {}
    for layer in range(N_LAYERS):
        key = f"samples_{layer}"
        if key in data:
            samples[layer] = data[key]
    if len(samples) < N_LAYERS:
        raise SystemExit(f"{args.activations} has samples for {len(samples)} of {N_LAYERS} layers")

    n = min(v.shape[0] for v in samples.values())
    print(f"{n} token-aligned samples per layer from {args.activations}")

    gates = load_gates(Path(args.model_path))
    if len(gates) < N_LAYERS:
        raise SystemExit(f"found gate tensors for {len(gates)} of {N_LAYERS} layers")
    print(f"gate tensors for {len(gates)} layers read from {args.model_path}")

    widths = [int(w) for w in args.widths.split(",")]
    rows = []

    for k in widths:
        stale_hits = ahead2_hits = persist_hits = total = 0
        for layer in range(N_LAYERS - 1):
            w, b = gates[layer + 1]
            for i in range(n):
                truth = set(route(mx.array(samples[layer + 1][i]), w, b, TOPK).tolist())
                # what the runtime predicts: L+1's gate on layer L's input
                pred = route(mx.array(samples[layer][i]), w, b, k)
                stale_hits += len(truth & set(pred.tolist()))
                # two layers early, which is what PREDICT_AHEAD=2 adds
                if layer >= 1:
                    pred2 = route(mx.array(samples[layer - 1][i]), w, b, k)
                    ahead2_hits += len(truth & set(pred2.tolist()))
                # the free baseline: what this layer used for the previous token
                if i >= 1:
                    prev = route(mx.array(samples[layer + 1][i - 1]), w, b, k)
                    persist_hits += len(truth & set(prev.tolist()))
                total += TOPK

        n_ahead2 = TOPK * (N_LAYERS - 2) * n
        n_persist = TOPK * (N_LAYERS - 1) * (n - 1)
        row = {
            "width": k,
            "stale_recall": stale_hits / total,
            "ahead2_recall": ahead2_hits / n_ahead2,
            "persist_recall": persist_hits / n_persist,
            "loads_per_layer": k,
        }
        rows.append(row)
        print(f"  top-{k:<3d} one layer early {row['stale_recall']:6.1%} | "
              f"two layers early {row['ahead2_recall']:6.1%} | "
              f"previous token {row['persist_recall']:6.1%}", flush=True)

    # Which of the true six does the predictor miss? If the misses sit at the
    # weak end of the router's own ordering they are the experts that
    # contribute least to the output, and a wider net is buying the cheapest
    # experts at full price. `route` returns strongest first, so rank 0 is the
    # expert the router wants most.
    by_rank = [0] * TOPK
    seen_rank = [0] * TOPK
    for layer in range(N_LAYERS - 1):
        w, b = gates[layer + 1]
        for i in range(n):
            truth = route(mx.array(samples[layer + 1][i]), w, b, TOPK).tolist()
            pred = set(route(mx.array(samples[layer][i]), w, b, TOPK).tolist())
            for rank, expert in enumerate(truth):
                seen_rank[rank] += 1
                if expert in pred:
                    by_rank[rank] += 1

    # Adaptive width: spend the extra predictions only where the stale router is
    # unsure. Compared against fixed width on the same samples, the honest axis
    # is recall against mean experts predicted per layer -- bytes, not width.
    print("\n  adaptive width, extra predictions only near the stale router's own cut:")
    print("  | margin | mean predicted | recall | fixed width at the same cost |")
    print("  |---:|---:|---:|---:|")
    fixed = {r["loads_per_layer"]: r["stale_recall"] for r in rows}
    adaptive_rows = []
    for margin in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50):
        hits = total_truth = predicted = layers_seen = 0
        for layer in range(N_LAYERS - 1):
            w, b = gates[layer + 1]
            for i in range(n):
                truth = set(route(mx.array(samples[layer + 1][i]), w, b, TOPK).tolist())
                sc = gate_scores(mx.array(samples[layer][i]), w, b)
                pred = route_margin(sc, TOPK, margin, cap=24)
                hits += len(truth & set(pred.tolist()))
                total_truth += TOPK
                predicted += len(pred)
                layers_seen += 1
        mean_pred = predicted / layers_seen
        recall = hits / total_truth
        near = min(fixed, key=lambda k: abs(k - mean_pred)) if fixed else None
        ref = f"top-{near} at {fixed[near]:.1%}" if near else "-"
        adaptive_rows.append({"margin": margin, "mean_predicted": mean_pred, "recall": recall})
        print(f"  | {margin:.2f} | {mean_pred:.2f} | {recall:.1%} | {ref} |")

    print("\n  recall of the true top-6 by the router's own ranking, one layer early, top-6:")
    ranks = []
    for rank in range(TOPK):
        r = by_rank[rank] / seen_rank[rank]
        ranks.append(r)
        print(f"    rank {rank + 1} (of 6) {r:6.1%}")

    Path(args.out).write_text(json.dumps(
        {"args": vars(args), "samples": n, "rows": rows,
         "adaptive": adaptive_rows, "recall_by_rank": ranks}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
