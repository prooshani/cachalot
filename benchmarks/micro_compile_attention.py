"""
Screen: is `mx.compile` worth anything on decode attention, and does
`shapeless=True` change the answer?

Attention is the largest *named* GPU piece of a token (HANDOFF section 7.1.4:
14.3 ms across the thirty compressed-reuse layers alone) and the only piece
with an instrument nobody had tried. It is also the piece a plain trace should
be worst at, because its graph moves with the context: `attention_kv` is
`[window_size + (start_pos + 1) // compress_ratio, head_dim]`, so the shapes
grow every token or two, and `window_slot = start_pos % window_size` changes
the slice bounds of `updated_window_cache` on *every* token.

`shapeless=True` relaxes the shapes. It does not relax a Python int baked into
a slice, so the question this screen answers is whether what is left retraces
anyway.

Three arms, all on `compressed_attention_decode_reuse`, the function the thirty
reuse layers call:

  shipped          the runtime's path
  mx.compile       a plain trace
  mx.compile(shapeless=True)

Each arm is run over twelve consecutive decode positions -- a token's worth of
one layer, twelve times -- because a screen that holds `start_pos` fixed cannot
see a retrace. Construction is timed with no eval in the loop; the chained
figure runs thirty launches inside one eval, which is how a token pays for its
thirty reuse layers. Output is asserted bit-identical against the shipped arm
at every position.

    benchmarks/guarded_run.sh --budget-gib 40 ... python benchmarks/micro_compile_attention.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH  # noqa: E402
from cachalot.model import decode_fused_metal as dfm  # noqa: E402
from cachalot.model.attention_compressed import (  # noqa: E402
    compressed_attention_decode_reuse,
)
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402

POSITIONS = 12
REUSE_LAYERS = 30


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--layer", type=int, default=25,
                    help="25 is a ratio-1 reuse layer; 5 is a ratio-2 one")
    args = ap.parse_args()

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        _, text = prompt_sources()[0]
        ids = build_prompt(rt, enc, text, args.prompt_tokens)
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        rt.decode_token(tok)

        layer = args.layer
        ratio = 1 if layer >= 20 else 2
        kw = rt._common_block_kwargs(layer, compressed=True)
        attn_kw = {k: kw[k] for k in (
            "rope_cos", "rope_sin", "attn_sink", "q_norm_weight", "kv_norm_weight",
            "wq_a", "wq_a_scales", "wq_b", "wq_b_scales", "wkv", "wkv_scales",
            "wo_a_bf16", "wo_b", "wo_b_scales")}
        x = mx.random.normal((4, 5120)).astype(mx.bfloat16)
        pre_mix = mx.array([1.0, 0, 0, 0]).astype(mx.float32)
        mx.eval(x, pre_mix)
        xin = dfm.hc_pre_norm_1d(x, pre_mix, kw["attn_norm_weight"], eps=1e-20)
        mx.eval(xin)

        window = rt.windows[layer]
        shared = rt.shared_attn
        # Positions below the current one, so the published compressed cache is
        # long enough for every compress_len the loop asks for.
        last = rt.position - 1
        positions = list(range(last - POSITIONS + 1, last + 1))

        def shipped(start_pos):
            return compressed_attention_decode_reuse(
                xin, start_pos=start_pos, compress_ratio=ratio,
                window_cache=window, shared_attn=shared, **attn_kw)[0]

        # mx.compile sees the tensors as arguments; start_pos, the ratio and the
        # shared-attention object stay Python-level, exactly as the runtime
        # passes them.
        def traced_fn(xv, wc, ckv, tidx, start_pos):
            from cachalot.model.shared_attention import SharedAttentionRuntime
            sa = SharedAttentionRuntime(compress_kv=ckv, topk_idxs=tidx)
            return compressed_attention_decode_reuse(
                xv, start_pos=start_pos, compress_ratio=ratio,
                window_cache=wc, shared_attn=sa, **attn_kw)[0]

        # mx.compile cannot take a Python int that changes as a keyword and stay
        # traced, so each distinct start_pos gets its own compiled callable --
        # which is precisely the retrace this screen is measuring. One cache per
        # arm, keyed by start_pos, is what a runtime would end up with.
        caches: dict[tuple[bool, int], object] = {}

        def compiled_arm(shapeless):
            def arm(start_pos):
                key = (shapeless, start_pos)
                fn = caches.get(key)
                if fn is None:
                    fn = mx.compile(
                        (lambda sp: (lambda xv, wc, ckv, tidx: traced_fn(xv, wc, ckv, tidx, sp)))(start_pos),
                        shapeless=shapeless,
                    )
                    caches[key] = fn
                return fn(xin, window, shared.compress_kv, shared.topk_idxs)
            return arm

        arms = (
            ("shipped", shipped),
            ("mx.compile", compiled_arm(False)),
            ("mx.compile shapeless", compiled_arm(True)),
        )

        # ---------------- parity ----------------
        # An arm that cannot run at all is a result, not a crash: shapeless
        # tracing has to give every custom Metal kernel a symbolic output shape.
        live = [arms[0]]
        for name, arm in arms[1:]:
            same = 0
            worst = 0.0
            try:
                for start_pos in positions:
                    a, b = shipped(start_pos), arm(start_pos)
                    mx.eval(a, b)
                    if bool(mx.array_equal(a, b)):
                        same += 1
                    worst = max(worst, float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()))
            except Exception as exc:  # noqa: BLE001
                print(f"{name:22s} does not run: {type(exc).__name__}: {exc}", flush=True)
                continue
            live.append((name, arm))
            print(f"{name:22s} bit-identical at {same}/{len(positions)} positions, "
                  f"max abs delta {worst:.3e}", flush=True)
        arms = tuple(live)
        print()

        # ---------------- construction, CPU only ----------------
        def build_time(arm, repeats=5):
            times = []
            for _ in range(repeats):
                t0 = perf_counter()
                outs = [arm(p) for p in positions]
                t = (perf_counter() - t0) / len(positions) * 1e3
                mx.eval(outs)
                mx.synchronize()
                times.append(t)
            return min(times), statistics.median(times)

        print(f"graph construction per call, layer {layer} (ratio {ratio}), "
              f"{len(positions)} consecutive positions, steady state:")
        for name, arm in arms:
            lo, med = build_time(arm)
            print(f"  {name:22s} min {lo:7.4f} ms  median {med:7.4f} ms"
                  f"  -> {lo * REUSE_LAYERS:6.2f} ms/token over {REUSE_LAYERS} reuse layers", flush=True)

        # ---------------- first call at a fresh position ----------------
        print("\nfirst call at a position never seen before (trace cost included):")
        fresh = list(range(last - POSITIONS - 40, last - POSITIONS))
        for name, arm in arms:
            times = []
            for start_pos in fresh:
                t0 = perf_counter()
                out = arm(start_pos)
                mx.eval(out)
                mx.synchronize()
                times.append((perf_counter() - t0) * 1e3)
            print(f"  {name:22s} min {min(times):7.3f} ms  median {statistics.median(times):7.3f} ms"
                  f"  -> {statistics.median(times) * REUSE_LAYERS:6.1f} ms/token if every position retraces",
                  flush=True)

        # ---------------- chained, as a token pays ----------------
        def chained(arm, start_pos, n=REUSE_LAYERS, repeats=5):
            mx.eval(arm(start_pos))
            mx.synchronize()
            times = []
            for _ in range(repeats):
                t0 = perf_counter()
                outs = [arm(start_pos) for _ in range(n)]
                mx.eval(outs)
                mx.synchronize()
                times.append((perf_counter() - t0) / n * 1e3)
            return min(times), statistics.median(times)

        print(f"\nchained, {REUSE_LAYERS} launches in one eval (CPU + GPU), one position:")
        for name, arm in arms:
            lo, med = chained(arm, positions[-1])
            print(f"  {name:22s} min {lo:7.4f} ms  median {med:7.4f} ms"
                  f"  -> {lo * REUSE_LAYERS:6.2f} ms/token", flush=True)


if __name__ == "__main__":
    main()
