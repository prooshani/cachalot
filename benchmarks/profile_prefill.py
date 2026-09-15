"""Where does prefill wall time go? Wraps the main phases of one cold prefill and reports totals."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cachalot.model.moe_prefill_grouped as mpg  # noqa: E402
from _common import MODEL_PATH, StoreSnapshot  # noqa: E402
from cachalot.io import resident_prefetch as rp  # noqa: E402
from cachalot.model import (  # noqa: E402
    attention_prefill_batched as apb,
)
from cachalot.model import (
    block_compressed_index_source_prefill as bcis,
)
from cachalot.model import (
    block_compressed_reuse_prefill as bcr,
)
from cachalot.model import (
    block_compressed_source_prefill as bcs,
)
from cachalot.model import (
    block_layer0_prefill as bl0,
)
from cachalot.model import (
    block_sliding_window_prefill as bsw,
)
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt, prompt_sources  # noqa: E402

T = defaultdict(float)
N = defaultdict(int)


def timed(name, fn):
    def wrapper(*a, **k):
        t0 = perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name] += perf_counter() - t0
            N[name] += 1
    return wrapper


def main():
    n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    # loader wait
    rp.ResidentExpertPrefetcher.get = timed("prefetcher.get (wait for load)", rp.ResidentExpertPrefetcher.get)
    # batched attention (blocks import the name; patch in each module)
    for mod in (bl0, bsw, bcr):
        mod.attention_prefill_batched = timed("attention batched (32 layers)", apb.attention_prefill_batched)
    bcs.compressed_attention_decode_source = timed("attention sequential source (4 layers)", bcs.compressed_attention_decode_source)
    bcis.compressed_attention_decode_index_source = timed("attention sequential index-source (4 layers)", bcis.compressed_attention_decode_index_source)
    # whole MoE per layer and mx.eval inside it
    orig_eval = mx.eval

    def eval_timed(*a, **k):
        t0 = perf_counter()
        r = orig_eval(*a, **k)
        T["mx.eval (all)"] += perf_counter() - t0
        N["mx.eval (all)"] += 1
        return r

    mpg.mx.eval = eval_timed
    orig_moe = mpg.moe_prefill_grouped
    for mod in (bl0, bsw, bcr, bcs, bcis):
        mod.moe_prefill_grouped = timed("moe_prefill_grouped (whole)", orig_moe)
    mpg.routed_expert_forward_batched = timed("routed expert batched (issue only)", mpg.routed_expert_forward_batched)
    mpg.shared_expert_forward_batched = timed("shared expert batched (issue only)", mpg.shared_expert_forward_batched)
    from cachalot.model import hc_prefill_exact as hpe
    from cachalot.model import hyper_connection_mlx as hcm
    from cachalot.model import norm_rope_mlx as nrm
    for mod in (bl0, bsw, bcr, bcs, bcis):
        mod.hc_mixes_prefill_exact = timed("hc_mixes_prefill_exact (issue)", hpe.hc_mixes_prefill_exact)
        mod.hc_pre = timed("hc_pre (issue)", hcm.hc_pre)
        mod.hc_post = timed("hc_post (issue)", hcm.hc_post)
        mod.rms_norm = timed("rms_norm (issue)", nrm.rms_norm)
    mpg.route_topk_rows = timed("route_topk_rows (issue)", mpg.route_topk_rows)
    import cachalot.model.text_decode_runtime as tdr
    tdr.TextDecodeRuntime._prefill_apply_engram = timed("engram prefill", tdr.TextDecodeRuntime._prefill_apply_engram)
    from cachalot.cache import resident_store as rs
    rs.ResidentExpertStore.prepare_prefill_layer = timed("prepare_prefill_layer", rs.ResidentExpertStore.prepare_prefill_layer)
    orig_tf = rs.ResidentExpertStore.transient_free
    samples = []

    def tf(self):
        v = orig_tf(self)
        samples.append((v, self._transient_count, len(self._transients), self.pool.free_count))
        return v

    rs.ResidentExpertStore.transient_free = tf
    orig_rel = rs.ResidentExpertStore.release_transients

    def rel(self, keys):
        keys = list(keys)
        before = len(self._transients)
        orig_rel(self, keys)
        T["released transients"] += 0
        N["released transients"] += before - len(self._transients)
        N["release_transients calls"] += 1
        N["release keys requested"] += len(keys)
    rs.ResidentExpertStore.release_transients = rel

    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096) as rt:
        enc = load_official_encoding(MODEL_PATH)
        ids = build_prompt(rt, enc, prompt_sources()[0][1], n_tokens)
        rt.reset()
        before = StoreSnapshot.take(rt)
        t0 = perf_counter()
        rt.prefill_tokens(ids)
        wall = perf_counter() - t0
        d = before.delta(StoreSnapshot.take(rt))
        print(f"prefill {len(ids)} tokens: {wall:.1f}s wall, {d.cache_misses} misses ({d.ssd_bytes_read / 1e9:.0f} GB, "
              f"floor {d.ssd_bytes_read / 5.6e9:.0f}s at 5.6 GB/s)")
        for name in sorted(T, key=lambda k: -T[k]):
            print(f"  {name:48s} {T[name]:7.2f}s  ({N[name]} calls)")
        import numpy as np
        arr = np.array(samples)
        print(f"  transient_free samples: {len(arr)}; min free {arr[:,0].min()}, median {np.median(arr[:,0]):.0f}; "
              f"max _transient_count {arr[:,1].max()}, max len(_transients) {arr[:,2].max()}, min pool free {arr[:,3].min()}")
        print(f"  low-water hits (free < 10): {(arr[:,0] < 10).sum()}")


if __name__ == "__main__":
    main()
