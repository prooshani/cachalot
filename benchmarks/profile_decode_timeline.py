"""
Main-thread timeline of one decode step, cold vs all-resident.

Patches moe_layer_forward in every block module with a timed replica that
records, per layer: route eval wait, get_many wall, expert-op construction,
and the gap since the previous layer's MoE (attention + HC + Python).
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_PATH, StoreSnapshot  # noqa: E402
from cachalot.model import (  # noqa: E402
    block_compressed_index_source,
    block_compressed_reuse,
    block_compressed_source,
    block_layer0,
    block_sliding_window,
)
from cachalot.model.expert_metal import routed_expert_forward  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.router_mlx import route_topk  # noqa: E402
from cachalot.model.shared_expert_metal import shared_expert_forward  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402

TL = {"rows": [], "last_end": None}


def timed_moe_layer_forward(x, *, layer_id, gate_weight, gate_bias, expert_index, expert_store,
                            shared_w1, shared_w1_scales, shared_w2, shared_w2_scales, shared_w3, shared_w3_scales,
                            topk=6, gate_temp=1.0, route_scale=1.5, norm_topk_prob=True, swiglu_limit=10.0):
    t_enter = perf_counter()
    gap = 0.0 if TL["last_end"] is None else t_enter - TL["last_end"]
    route = route_topk(x, gate_weight, gate_bias, topk=topk, gate_temp=gate_temp, route_scale=route_scale, norm_topk_prob=norm_topk_prob)
    t0 = perf_counter()
    mx.eval(route.indices, route.weights)
    t_eval = perf_counter() - t0
    expert_ids = route.indices.tolist()
    router_weights = route.weights.tolist()
    entries = [expert_index[(layer_id, int(e))] for e in expert_ids]
    t0 = perf_counter()
    experts = expert_store.get_many(entries)
    t_get = perf_counter() - t0
    t0 = perf_counter()
    routed = mx.zeros(x.shape, dtype=mx.float32)
    for expert, w in zip(experts, router_weights, strict=True):
        m = expert.as_model_dict()
        y = routed_expert_forward(x, w1_packed=m["w1.weight"], w1_scales=m["w1.scale"], w2_packed=m["w2.weight"],
                                  w2_scales=m["w2.scale"], w3_packed=m["w3.weight"], w3_scales=m["w3.scale"],
                                  weight=float(w), swiglu_limit=swiglu_limit)
        routed = routed + y.astype(mx.float32)
    shared = shared_expert_forward(x, w1=shared_w1, w1_scales=shared_w1_scales, w2=shared_w2, w2_scales=shared_w2_scales,
                                   w3=shared_w3, w3_scales=shared_w3_scales, swiglu_limit=swiglu_limit)
    output = (routed + shared.astype(mx.float32)).astype(x.dtype)
    t_build = perf_counter() - t0
    TL["last_end"] = perf_counter()
    TL["rows"].append((layer_id, gap, t_eval, t_get, t_build))
    return output, route


for mod in (block_compressed_index_source, block_compressed_reuse, block_compressed_source, block_layer0, block_sliding_window):
    mod.moe_layer_forward = timed_moe_layer_forward


HEARTBEAT = {"on": False, "buf": None}


def install_heartbeat_get_many():
    """Patch get_many so future waits poll while issuing small GPU ops."""
    from concurrent.futures import wait as fut_wait

    from cachalot.cache import resident_store as rs

    def get_many(self, entries):
        results = [None] * len(entries)
        pending = []
        with self._lock:
            for i, entry in enumerate(entries):
                key = (entry.layer, entry.expert)
                cached = self._items.get(key)
                if cached is not None:
                    self._items.move_to_end(key)
                    self.cache_hits += 1
                    results[i] = cached
                else:
                    pending.append((i, entry))
        if not pending:
            return results
        unique = {}
        for _, entry in pending:
            unique.setdefault((entry.layer, entry.expert), entry)
        futures = {key: self._load_pool.submit(self._load_expert, entry) for key, entry in unique.items()}
        outstanding = set(futures.values())
        buf = HEARTBEAT["buf"]
        while outstanding:
            done, outstanding = fut_wait(outstanding, timeout=0.002)
            if outstanding and HEARTBEAT["on"]:
                mx.eval(buf.astype(mx.float32).sum())
        loaded = {key: fut.result() for key, fut in futures.items()}
        for i, entry in pending:
            key = (entry.layer, entry.expert)
            resident, payload_size, read_seconds, promotion_seconds = loaded[key]
            with self._lock:
                existing = self._items.get(key)
                if existing is not None:
                    self._items.move_to_end(key)
                    results[i] = existing
                    continue
                while self._items and self.current_bytes + resident.size > self.budget_bytes:
                    _, evicted = self._items.popitem(last=False)
                    self.current_bytes -= evicted.size
                self._items[key] = resident
                self.current_bytes += resident.size
                self.cache_misses += 1
                self.ssd_bytes_read += payload_size
                self.ssd_read_seconds += read_seconds
                self.promotion_seconds += promotion_seconds
            results[i] = resident
        return results

    rs.ResidentExpertStore.get_many = get_many


def run(rt, tok, label):
    TL["rows"].clear()
    TL["last_end"] = None
    before = StoreSnapshot.take(rt)
    t0 = perf_counter()
    res = rt.decode_token(tok)
    t_pre_head = perf_counter()
    mx.eval(res.logits)
    wall = perf_counter() - t0
    d = before.delta(StoreSnapshot.take(rt))
    rows = TL["rows"]
    s = [sum(r[i] for r in rows) for i in range(1, 5)]
    print(f"\n{label}: wall {wall:.2f}s misses {d.cache_misses} | attention/HC gaps {s[0]:.2f}s | route-eval waits {s[1]:.2f}s "
          f"| get_many {s[2]:.2f}s | expert-op build {s[3]:.2f}s | head+final {wall - (t_pre_head - t0):.2f}s", flush=True)
    print("  layer: gap  eval  get  build (ms)")
    for layer_id, gap, te, tg, tb in rows[:12] + rows[-3:]:
        print(f"  {layer_id:2d}: {gap * 1e3:6.1f} {te * 1e3:6.1f} {tg * 1e3:7.1f} {tb * 1e3:5.1f}")
    return res


def vm():
    import subprocess
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    keys = ("Pages free", "File-backed pages", "Pages occupied by compressor", "Decompressions", "Pageins")
    vals = {}
    for line in out.splitlines():
        for k in keys:
            if line.startswith(k):
                vals[k] = int(line.split(":")[1].strip().rstrip("."))
    return vals


def main():
    import os
    print("page cache bypass:", os.environ.get("CACHALOT_PAGE_CACHE", "0") != "1", flush=True)
    print("vm before load:", vm(), flush=True)
    wired_gib = float(os.environ.get("CACHALOT_WIRED_GIB", "56"))
    print("wired limit GiB:", wired_gib, flush=True)
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=4096, mlx_wired_limit_bytes=int(wired_gib * 2**30)) as rt:
        print("vm after load:", vm(), flush=True)
        enc = load_official_encoding(MODEL_PATH)
        ids = list(rt.tokenizer.encode(enc.encode_messages(
            [{"role": "user", "content": "List three facts about the deep ocean."}],
            thinking_mode="chat", reasoning_effort=None)))
        rt.reset()
        res = rt.prefill_tokens(ids)
        tok = int(res.logits.argmax().item())
        snap = rt.snapshot()
        HEARTBEAT["buf"] = mx.random.normal((2048, 5120)).astype(mx.bfloat16)
        mx.eval(HEARTBEAT["buf"])
        run(rt, tok, "cold #1 (no heartbeat)")
        rt.restore(snap)
        run(rt, tok, "warm (all resident)")
        r = run(rt, tok, "warm 2nd")
        tok2 = int(r.logits.argmax().item())
        run(rt, tok2, "cold #2 (no heartbeat)")
        # now evict everything this token loaded? simpler: use a third token with heartbeat on
        r = run(rt, tok2, "warm after cold #2")
        tok3 = int(r.logits.argmax().item())
        HEARTBEAT["on"] = True
        run(rt, tok3, "cold #3 WITH heartbeat")
        r = run(rt, tok3, "warm after cold #3")
        tok4 = int(r.logits.argmax().item())
        HEARTBEAT["on"] = False
        run(rt, tok4, "cold #4 (no heartbeat)")
        r = run(rt, tok4, "warm")
        tok5 = int(r.logits.argmax().item())
        HEARTBEAT["on"] = True
        run(rt, tok5, "cold #5 WITH heartbeat")
        print("vm after decode:", vm(), flush=True)


if __name__ == "__main__":
    main()
