"""
Record what the routed experts are actually multiplied by, per layer.

Every affine fit this project has built minimizes the *unweighted* squared
error of a group of weights. That is the right thing to minimize only if the
model is equally sensitive to every weight in the group, and it is not: the
error in column j of w1 or w3 reaches the output multiplied by x_j, and the
error in column j of w2 reaches it multiplied by the SwiGLU hidden h_j. A
column the model barely excites can be fitted badly for free; a column it leans
on cannot.

That is the part of the "dynamic quantization" idea worth keeping (section
9.0.1 of docs/HANDOFF.md): activation-weighted fitting, without the mixed
precision that `mx.quantized_matmul` and `storage/index.py` refuse anyway. It
needs no format change and no runtime change -- only better scales and biases.

This records the inputs. It wraps `moe_layer_forward` in every block module, so
it sees exactly the vector the decode path hands the routed experts, and saves
per layer:

  sumsq   [HIDDEN]  sum of x_j^2 over every decoded token
  tokens            how many tokens that sum covers
  samples [n, HIDDEN]  raw x vectors, evenly spaced through the decode

The sums are the weighting; the samples are what benchmarks/quant_fit_screen.py
evaluates output error on, in place of the random unit vectors it used before.
Random inputs make every column equally important by construction, which is
precisely the assumption being tested.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh \
      --budget-gib 24 --max-seconds 1800 --tag acts -- env \
      CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash \
      CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-fp4-experts \
      CACHALOT_PAGE_CACHE=1 PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
      benchmarks/capture_activations.py --tokens 128 --prefill 128 --samples 64

One text is one sample of what the model does. Recordings from several add up,
and --merge sums them without the model, which is how a weighting gets to be
about the model rather than about one document:

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python benchmarks/capture_activations.py \
      --merge benchmarks/results/activations_moe_input.npz,\
benchmarks/results/activations_moe_input_code.npz \
      --out benchmarks/results/activations_moe_input_both.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from nll_expert_precision import BLOCK_MODULES, HIDDEN  # noqa: E402


class Recorder:
    """Per-layer second moment of the MoE input, plus a sample of raw vectors.

    The accumulator is evaluated on every call rather than left in the graph.
    It is a 5,120-element reduction, so the cost is small, and the alternative
    is worse: nothing downstream needs it, so an unevaluated sum would keep
    every intermediate of every layer of every decoded token alive at once.
    This is a statistics run, not a timing run, and the extra evaluations
    change the timing without changing a single recorded number.
    """

    def __init__(self, layers: int, samples: int, every: int):
        self.sumsq: list[mx.array | None] = [None] * layers
        self.tokens = [0] * layers
        self.samples: list[list[np.ndarray]] = [[] for _ in range(layers)]
        self.ids: list[list[int]] = [[] for _ in range(layers)]
        self.max_samples, self.every = samples, every

    def record(self, layer: int, x: mx.array, route) -> None:
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        acc = mx.sum(flat * flat, axis=0)
        acc = acc if self.sumsq[layer] is None else self.sumsq[layer] + acc
        mx.eval(acc)
        self.sumsq[layer] = acc
        seen = self.tokens[layer]
        self.tokens[layer] = seen + flat.shape[0]
        if len(self.samples[layer]) < self.max_samples and seen % self.every == 0:
            first = flat[0]
            mx.eval(first)
            self.samples[layer].append(np.array(first, copy=True))
            top = -1
            if route is not None and getattr(route, "indices", None) is not None:
                mx.eval(route.indices)
                top = int(route.indices.reshape(-1)[0].item())
            self.ids[layer].append(top)

    def save(self, path: Path, meta: dict) -> None:
        out: dict[str, np.ndarray] = {}
        for layer, acc in enumerate(self.sumsq):
            if acc is None:
                continue
            mx.eval(acc)
            out[f"sumsq_{layer}"] = np.array(acc, copy=True)
            out[f"tokens_{layer}"] = np.array(self.tokens[layer])
            if self.samples[layer]:
                out[f"samples_{layer}"] = np.stack(self.samples[layer], axis=0)
                out[f"sample_expert_{layer}"] = np.array(self.ids[layer], dtype=np.int32)
        for k, v in meta.items():
            out[f"meta_{k}"] = np.array(v)
        np.savez(path, **out)


def install_recorder(recorder: Recorder) -> None:
    """Wrap, do not replace: the real MoE still does the work and the forward stays exact."""
    for mod_name in BLOCK_MODULES:
        mod = __import__(f"cachalot.model.{mod_name}", fromlist=["moe_layer_forward"])
        original = mod.moe_layer_forward

        def wrapped(x, *, layer_id, _original=original, _recorder=recorder, **kw):
            out, route = _original(x, layer_id=layer_id, **kw)
            _recorder.record(layer_id, x, route)
            return out, route

        mod.moe_layer_forward = wrapped


def merge(paths: list[Path], out: Path, layers: int) -> None:
    """Add recordings together: sum the second moments, pool the sample vectors.

    The second moment is a sum over tokens and the token counts are stored
    beside it, so adding two recordings is exactly recording one longer run --
    no reweighting, and a text that contributed more tokens contributes more.
    The samples are pooled rather than averaged, because they are used as probes
    and a probe set spanning two texts is the point.
    """
    acc: dict[int, list] = {}
    for path in paths:
        data = np.load(path)
        for layer in range(layers):
            if f"sumsq_{layer}" not in data:
                continue
            entry = acc.setdefault(layer, [np.zeros(1), 0, []])
            entry[0] = entry[0] + data[f"sumsq_{layer}"].astype(np.float64)
            entry[1] += int(data[f"tokens_{layer}"])
            if f"samples_{layer}" in data:
                entry[2].append(data[f"samples_{layer}"])
    if not acc:
        raise SystemExit(f"no per-layer recordings in {', '.join(str(p) for p in paths)}")

    written: dict[str, np.ndarray] = {}
    for layer, (sumsq, tokens, samples) in sorted(acc.items()):
        written[f"sumsq_{layer}"] = sumsq.astype(np.float32)
        written[f"tokens_{layer}"] = np.array(tokens)
        if samples:
            written[f"samples_{layer}"] = np.concatenate(samples, axis=0)
    written["meta_merged_from"] = np.array([str(p) for p in paths])
    np.savez(out, **written)
    probes = next(iter(written[k] for k in written if k.startswith("samples_")))
    print(f"merged {len(paths)} recordings over {len(acc)} layers "
          f"({probes.shape[0]} probes per layer) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", default="",
                    help="comma-separated npz files to add together instead of recording; "
                         "no model is loaded")
    ap.add_argument("--tokens", type=int, default=128, help="tokens decoded, teacher-forced")
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--samples", type=int, default=64, help="raw x vectors kept per layer")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--source", default=None, help="text file; default: the model README")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.merge:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = (Path(args.out) if args.out
               else RESULTS_DIR / "activations_moe_input_merged.npz")
        merge([Path(p) for p in args.merge.split(",") if p], out, args.layers)
        return

    text = Path(args.source).read_text() if args.source else (Path(MODEL_PATH) / "README.md").read_text()
    start = text.find("We introduce")
    text = text[start:] if start > 0 else text

    every = max(1, args.tokens // max(1, args.samples))
    recorder = Recorder(args.layers, args.samples, every)

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        print(f"runtime ready (bank {rt.expert_bank_path.name}, {rt.expert_format.bits}-bit, "
              f"budget {rt.expert_cache_budget_bytes / 2**30:.1f} GiB)", flush=True)
        install_recorder(recorder)

        ids = list(rt.tokenizer.encode(text))[: args.prefill + args.tokens + 1]
        rt.reset()
        rt.prefill_tokens(ids[: args.prefill])
        t0 = perf_counter()
        for i in range(args.prefill, args.prefill + args.tokens):
            rt.decode_token(ids[i])
        dt = perf_counter() - t0
        print(f"decoded {args.tokens} tokens in {dt:.0f} s "
              f"({args.tokens / dt:.2f} tok/s), recorded "
              f"{sum(1 for s in recorder.sumsq if s is not None)} layers", flush=True)
        bank = rt.expert_bank_path.name

    RESULTS_DIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / "activations_moe_input.npz"
    recorder.save(out, {"tokens": args.tokens, "prefill": args.prefill, "hidden": HIDDEN})
    print(f"wrote {out}")

    # The one number that decides whether any of this can pay: how unequal the
    # columns are *inside a group*, since scale and bias are fitted per group.
    # A group whose columns are all equally excited cannot be helped by
    # weighting them, however unequal the layer is as a whole.
    for group in (64, 128):
        ratios, spreads = [], []
        for layer in range(args.layers):
            acc = recorder.sumsq[layer]
            if acc is None:
                continue
            a = np.array(acc) / max(1, recorder.tokens[layer])
            g = a.reshape(-1, group)
            spreads.append(float(np.mean(g.max(axis=1) / np.maximum(g.min(axis=1), 1e-30))))
            ratios.append(float(a.max() / max(a.min(), 1e-30)))
        if spreads:
            print(f"group {group}: mean within-group max/min of E[x^2] = {np.mean(spreads):.1f} "
                  f"(whole-layer max/min {np.mean(ratios):.1f}); bank {bank}")


if __name__ == "__main__":
    main()
