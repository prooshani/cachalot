"""
Quality gate for a lower-precision routed-expert bank.

Teacher-forced NLL through the real runtime where every routed expert in
decode is evaluated with the SAME dense fp32 reference math but weights from
one of two sources:

  --experts fp4      the shipped FP4 experts (from the resident store), dequantized
  --experts oq3e     oMLX's calibrated 3-bit affine experts (group 64) read straight
                     from the converted checkpoint's shards and mx.dequantize'd
  --experts requant  the shipped FP4 experts re-quantized on the fly to a coarser
                     affine format, which is the quality gate for a smaller bank
                     built the same way (--requant-bits/-group/-fit)

Attention, shared experts, Engram and the head are untouched, so the two runs
differ only in routed-expert precision. Prefix (--prefill) tokens go through
the unpatched prefill path in both modes. Per-position argmax and NLL are
saved to benchmarks/results/nll_experts_<mode>.json so the two runs can be
compared token by token (greedy agreement).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    for m in fp4 oq3e; do
      benchmarks/guarded_run.sh --budget-gib 28 --max-seconds 1800 --tag nll_$m -- env PYTHONPATH=src \
        ~/venvs/deepseek-v41/bin/python benchmarks/nll_expert_precision.py --experts $m --tokens 160
    done
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, RESULTS_DIR  # noqa: E402
from cachalot.model.fp4_mlx import dequantize_fp4_weight  # noqa: E402
from cachalot.model.router_fused_metal import route_topk_fused  # noqa: E402
from cachalot.model.shared_expert_metal import shared_expert_forward  # noqa: E402
from quant_affine import fit_minmax, fit_search, quantize_2bit  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

HIDDEN, INTER, N_EXPERTS = 5120, 2304, 384
OQ3E_DEFAULT = "/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-oQ3e-mtp"
BLOCK_MODULES = ("block_sliding_window", "block_layer0", "block_compressed_source",
                 "block_compressed_reuse", "block_compressed_index_source")
NP_DTYPES = {"U32": np.uint32, "F16": np.float16, "F32": np.float32, "BF16": np.uint16}


def dense_expert(x: mx.array, w1: mx.array, w2: mx.array, w3: mx.array, weight: float, limit: float) -> mx.array:
    """Official routed-expert math with dense fp32 weights (bf16 linear outputs, fp32 gating)."""
    xb = x.astype(mx.bfloat16).astype(mx.float32)
    gate = (w1 @ xb).astype(mx.bfloat16).astype(mx.float32)
    up = (w3 @ xb).astype(mx.bfloat16).astype(mx.float32)
    if limit > 0:
        up = mx.clip(up, -limit, limit)
        gate = mx.minimum(gate, mx.array(limit, dtype=mx.float32))
    hidden = (nn.silu(gate) * up * weight).astype(mx.bfloat16).astype(mx.float32)
    return (w2 @ hidden).astype(mx.bfloat16).astype(mx.float32)


class FP4Dense:
    """Dense fp32 weights of the shipped FP4 experts, via the resident store."""

    def __init__(self, store, index):
        self.store, self.index = store, index

    def weights(self, layer: int, ids: list[int]) -> list[tuple[mx.array, mx.array, mx.array]]:
        entries = [self.index[(layer, e)] for e in ids]
        out = []
        for res in self.store.get_many(entries):
            a = res.as_model_dict()
            w1 = dequantize_fp4_weight(a["w1.weight"].reshape(INTER, HIDDEN // 2), a["w1.scale"].reshape(INTER, HIDDEN // 32))
            w3 = dequantize_fp4_weight(a["w3.weight"].reshape(INTER, HIDDEN // 2), a["w3.scale"].reshape(INTER, HIDDEN // 32))
            w2 = dequantize_fp4_weight(a["w2.weight"].reshape(HIDDEN, INTER // 2), a["w2.scale"].reshape(HIDDEN, INTER // 32))
            out.append((w1, w2, w3))
        return out


class OQ3EDense:
    """Dense fp32 weights of oMLX's 3-bit affine experts, read per expert from the stacked shard tensors."""

    def __init__(self, path: str, cache_bytes: int = 12 * 1024**3):
        self.path = Path(path)
        cfg = json.loads((self.path / "config.json").read_text())
        self.spec = cfg["omlx_deepseek_v41"]["quantized_modules"]
        self.map = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}
        self._fds: dict[str, int] = {}
        self.cache: OrderedDict[tuple[int, int], tuple] = OrderedDict()
        self.cache_bytes, self.cache_limit = 0, cache_bytes
        self.reads = 0
        self.read_bytes = 0
        self.read_seconds = 0.0

    def available_layers(self) -> list[int]:
        ok = []
        for layer in range(40):
            names = [f"language_model.layers.{layer}.ffn.experts.{p}.{f}" for p in ("w1", "w2", "w3") for f in ("weight", "scales", "biases")]
            if all(n in self.map and (self.path / self.map[n]).exists() for n in names):
                ok.append(layer)
        return ok

    def _header(self, filename: str) -> tuple[dict, int]:
        if filename not in self._headers:
            with open(self.path / filename, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._headers[filename] = (json.loads(fh.read(n)), 8 + n)
            self._fds[filename] = os.open(self.path / filename, os.O_RDONLY)
        return self._headers[filename]

    def _slab(self, name: str, expert: int) -> mx.array:
        filename = self.map[name]
        header, start = self._header(filename)
        ent = header[name]
        shape = ent["shape"]
        if shape[0] != N_EXPERTS:
            raise ValueError(f"{name}: expected stacked [{N_EXPERTS}, ...], got {shape}")
        row = int(np.prod(shape[1:])) * np.dtype(NP_DTYPES[ent["dtype"]]).itemsize
        off = start + ent["data_offsets"][0] + expert * row
        t0 = perf_counter()
        raw = os.pread(self._fds[filename], row, off)
        self.read_seconds += perf_counter() - t0
        self.reads += 1
        self.read_bytes += row
        arr = np.frombuffer(raw, dtype=NP_DTYPES[ent["dtype"]]).reshape(shape[1:])
        a = mx.array(arr)
        return a.view(mx.bfloat16) if ent["dtype"] == "BF16" else a

    def _quantized(self, layer: int, expert: int) -> dict[str, tuple[mx.array, mx.array, mx.array, dict]]:
        """Packed (weight, scales, biases, spec) per projection; 15.5 MB per expert, cached."""
        out = {}
        for proj in ("w1", "w2", "w3"):
            name = f"language_model.layers.{layer}.ffn.experts.{proj}"
            spec = self.spec[name]
            if spec["mode"] != "affine":
                raise ValueError(f"{name}: unsupported mode {spec['mode']}")
            out[proj] = (self._slab(name + ".weight", expert), self._slab(name + ".scales", expert),
                         self._slab(name + ".biases", expert), spec)
        mx.eval(*(a for v in out.values() for a in v[:3]))
        return out

    def weights(self, layer: int, ids: list[int]) -> list[tuple[mx.array, mx.array, mx.array]]:
        out = []
        for e in ids:
            key = (layer, e)
            hit = self.cache.get(key)
            if hit is None:
                hit = self._quantized(layer, e)
                nbytes = sum(a.nbytes for v in hit.values() for a in v[:3])
                while self.cache and self.cache_bytes + nbytes > self.cache_limit:
                    _, old = self.cache.popitem(last=False)
                    self.cache_bytes -= sum(a.nbytes for v in old.values() for a in v[:3])
                self.cache[key] = hit
                self.cache_bytes += nbytes
            else:
                self.cache.move_to_end(key)
            dense = []
            for proj in ("w1", "w2", "w3"):
                w, sc, b, spec = hit[proj]
                dense.append(mx.dequantize(w, sc, b, group_size=spec["group_size"], bits=spec["bits"]).astype(mx.float32))
            out.append(tuple(dense))
        return out


class RequantDense:
    """Dense fp32 weights of the FP4 experts re-quantized to a coarser affine format.

    The re-quantized triple (packed weights, scales, biases) is what a bank built
    this way would store, so it is what gets cached; the dense weights the
    reference math needs are dequantized per use. The cache holds the packed
    form, about 11 MB per expert at 2 bits and group 64.
    """

    FITS = {"mlx": None, "minmax": fit_minmax, "search": fit_search}

    def __init__(self, store, index, bits: int, group_size: int, fit: str,
                 cache_bytes: int = 8 * 1024**3):
        if fit not in self.FITS:
            raise ValueError(f"unknown fit {fit}")
        if fit != "mlx" and bits != 2:
            raise ValueError("the minmax and search fits are 2-bit only")
        self.fp4 = FP4Dense(store, index)
        self.bits, self.group_size, self.fit = bits, group_size, fit
        self.cache: OrderedDict[tuple[int, int], tuple] = OrderedDict()
        self.cache_bytes, self.cache_limit = 0, cache_bytes
        self.quantized = 0
        self.quantize_seconds = 0.0

    def label(self) -> str:
        return f"requant{self.bits}g{self.group_size}{self.fit}"

    def _pack(self, w: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        if self.fit == "mlx":
            return mx.quantize(w, group_size=self.group_size, bits=self.bits)
        return quantize_2bit(w, group_size=self.group_size, fit=self.FITS[self.fit])

    def _quantized(self, layer: int, expert: int) -> dict[str, tuple[mx.array, mx.array, mx.array]]:
        t0 = perf_counter()
        w1, w2, w3 = self.fp4.weights(layer, [expert])[0]
        out = {proj: self._pack(w) for proj, w in (("w1", w1), ("w2", w2), ("w3", w3))}
        mx.eval(*(a for v in out.values() for a in v))
        self.quantize_seconds += perf_counter() - t0
        self.quantized += 1
        return out

    def weights(self, layer: int, ids: list[int]) -> list[tuple[mx.array, mx.array, mx.array]]:
        out = []
        for e in ids:
            key = (layer, e)
            hit = self.cache.get(key)
            if hit is None:
                hit = self._quantized(layer, e)
                nbytes = sum(a.nbytes for v in hit.values() for a in v)
                while self.cache and self.cache_bytes + nbytes > self.cache_limit:
                    _, old = self.cache.popitem(last=False)
                    self.cache_bytes -= sum(a.nbytes for v in old.values() for a in v)
                self.cache[key] = hit
                self.cache_bytes += nbytes
            else:
                self.cache.move_to_end(key)
            out.append(tuple(
                mx.dequantize(*hit[proj], group_size=self.group_size, bits=self.bits).astype(mx.float32)
                for proj in ("w1", "w2", "w3")
            ))
        return out


def install_patch(source):
    def patched(x, *, layer_id, gate_weight, gate_bias, expert_index, expert_store,
                shared_w1, shared_w1_scales, shared_w2, shared_w2_scales, shared_w3, shared_w3_scales,
                topk=6, gate_temp=1.0, route_scale=1.5, norm_topk_prob=True, swiglu_limit=10.0, fused=None):
        route = route_topk_fused(x, gate_weight, gate_bias, topk=topk, gate_temp=gate_temp,
                                 route_scale=route_scale, norm_topk_prob=norm_topk_prob)
        mx.eval(route.indices, route.weights)
        ids = [int(i) for i in route.indices.tolist()]
        ws = route.weights.tolist()
        routed = mx.zeros(x.shape, dtype=mx.float32)
        for (w1, w2, w3), w in zip(source.weights(layer_id, ids), ws, strict=True):
            routed = routed + dense_expert(x, w1, w2, w3, float(w), swiglu_limit)
        shared = shared_expert_forward(x, w1=shared_w1, w1_scales=shared_w1_scales, w2=shared_w2,
                                       w2_scales=shared_w2_scales, w3=shared_w3, w3_scales=shared_w3_scales,
                                       swiglu_limit=swiglu_limit)
        return (routed + shared.astype(mx.float32)).astype(x.dtype), route

    for mod_name in BLOCK_MODULES:
        mod = __import__(f"cachalot.model.{mod_name}", fromlist=["moe_layer_forward"])
        mod.moe_layer_forward = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", choices=["fp4", "oq3e", "requant", "runtime"],
                    help="fp4/oq3e: dense reference math with that weight source; runtime: no patch, the production path (set CACHALOT_EXPERT_BANK to measure the affine bank as the runtime serves it)")
    ap.add_argument("--oq3e-path", default=OQ3E_DEFAULT)
    ap.add_argument("--requant-bits", type=int, default=2)
    ap.add_argument("--requant-group", type=int, default=64)
    ap.add_argument("--requant-fit", choices=["mlx", "minmax", "search"], default="search",
                    help="mlx: mx.quantize's max-abs fit; search: the lower-error 2-bit fit in quant_affine")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prefill", type=int, default=48)
    ap.add_argument("--source", default=None, help="text file; default: model README from 'We introduce'")
    ap.add_argument("--check-oq3e", action="store_true", help="only report which layers of the oQ3e download are complete")
    args = ap.parse_args()

    if not args.check_oq3e and args.experts is None:
        ap.error("--experts is required unless --check-oq3e is given")

    if args.check_oq3e:
        r = OQ3EDense(args.oq3e_path)
        layers = r.available_layers()
        print(f"oQ3e: {len(layers)}/40 layers complete: {layers}")
        return

    text = Path(args.source).read_text() if args.source else (Path(MODEL_PATH) / "README.md").read_text()
    start = text.find("We introduce")
    text = text[start:] if start > 0 else text

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        print(f"runtime ready (expert budget {rt.expert_cache_budget_bytes / 2**30:.1f} GiB, "
              f"wired {rt.mlx_wired_limit_bytes / 2**30:.1f} GiB)", flush=True)
        source = None
        if args.experts == "fp4":
            source = FP4Dense(rt.expert_store, rt.expert_index)
        elif args.experts == "requant":
            source = RequantDense(rt.expert_store, rt.expert_index, args.requant_bits,
                                  args.requant_group, args.requant_fit)
            print(f"requantizing FP4 experts to {args.requant_bits}-bit group {args.requant_group} "
                  f"({args.requant_fit} fit)", flush=True)
        elif args.experts == "oq3e":
            source = OQ3EDense(args.oq3e_path)
            missing = sorted(set(range(40)) - set(source.available_layers()))
            if missing:
                raise SystemExit(f"oQ3e checkpoint incomplete, layers missing: {missing}")
        if source is not None:
            install_patch(source)
        else:
            print(f"runtime path: expert bank {rt.expert_bank_path} ({rt.expert_format.kind}, {rt.expert_format.bits}-bit)", flush=True)

        ids = list(rt.tokenizer.encode(text))[: args.prefill + args.tokens + 1]
        rt.reset()
        res = rt.prefill_tokens(ids[: args.prefill])
        logits = res.logits
        nll, argmax = [], []
        t0 = perf_counter()
        for i in range(args.prefill, args.prefill + args.tokens):
            target = ids[i]
            logp = logits.astype(mx.float32)
            logp = logp - mx.logsumexp(logp)
            nll.append(-float(logp[target].item()))
            argmax.append(int(logits.argmax().item()))
            logits = rt.decode_token(target).logits
        dt = perf_counter() - t0
        mean = sum(nll) / len(nll)
        top1 = sum(int(a == t) for a, t in zip(argmax, ids[args.prefill:args.prefill + args.tokens]))
        mode = args.experts
        if isinstance(source, RequantDense):
            mode = source.label()
        elif args.experts == "runtime":
            # Name the result file after the bank being served, so two banks
            # measured on the production path do not overwrite each other.
            mode = f"runtime_{rt.expert_format.kind}{rt.expert_format.bits}g{rt.expert_format.group_size}"
        print(f"{mode}: tokens {len(nll)} | mean NLL {mean:.4f} nats | ppl {math.exp(mean):.3f} | "
              f"top-1 acc {top1 / len(nll):.1%} | worst {max(nll):.2f} at {nll.index(max(nll))} | {dt:.0f} s decode")
        if isinstance(source, RequantDense):
            print(f"requant: {source.quantized} experts quantized in {source.quantize_seconds:.0f} s "
                  f"({source.quantize_seconds / max(1, source.quantized) * 1e3:.0f} ms each), "
                  f"packed cache {source.cache_bytes / 2**30:.1f} GiB")
        if isinstance(source, OQ3EDense):
            print(f"oq3e reads: {source.reads} slabs, {source.read_bytes / 1e9:.1f} GB in {source.read_seconds:.0f} s "
                  f"({source.read_bytes / 1e9 / max(1e-9, source.read_seconds):.2f} GB/s), dense cache {source.cache_bytes / 2**30:.1f} GiB")
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"nll_experts_{mode}.json"
        out.write_text(json.dumps({"experts": mode, "prefill": args.prefill, "tokens": len(nll), "mean_nll": mean,
                                   "nll": nll, "argmax": argmax, "targets": ids[args.prefill:args.prefill + args.tokens]}))
        for other in sorted(RESULTS_DIR.glob("nll_experts_*.json")):
            if other == out:
                continue
            o = json.loads(other.read_text())
            if o["targets"] == ids[args.prefill:args.prefill + args.tokens]:
                agree = sum(int(a == b) for a, b in zip(argmax, o["argmax"]))
                print(f"vs {o['experts']}: mean NLL {o['mean_nll']:.4f} -> {mean:.4f} ({mean - o['mean_nll']:+.4f} nats), "
                      f"greedy agreement {agree}/{len(nll)} = {agree / len(nll):.1%}")


if __name__ == "__main__":
    main()
