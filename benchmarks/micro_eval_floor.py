"""
Screen: what does one `mx.eval` cost, and does anything make it cheaper?

`profile_decode_gpu.py`'s eval-drain arm measured 0.31 ms of round trip around a
router launch whose kernel is 0.02 ms. A token pays that 44 times -- forty
`mx.eval(route.indices, route.weights, ...)` calls at `moe_layer_metal.py:180`,
one per layer, plus the tail -- which is about 12 ms of a 76 ms all-resident
token and most of what section 7.1.4 could not account for.

This screen asks whether the 0.31 ms is a fixed Metal round trip or something
the call site chooses. It is model-free: the launch is a top-k over 384 random
logits, which is the shape of the thing the decode thread actually waits for.

Arms, in the order the decode thread would consider them:

  chained                 no eval at all -- the floor every chained row excludes
  eval                    mx.eval(idx)
  eval + tolist           what moe_layer_metal does
  async_eval + tolist     submit, then block in tolist
  np.array                numpy's own conversion, which evaluates implicitly
  item x1                 a single scalar off the GPU

and one question the call site cannot choose but the design can: how the cost
moves with the amount of work queued behind the eval, 1 to 40 launches.

    PYTHONPATH=src python benchmarks/micro_eval_floor.py
"""
from __future__ import annotations

import statistics
from time import perf_counter

import mlx.core as mx
import numpy as np

EXPERTS = 384
TOPK = 6
REPEATS = 400


def timed(fn, repeats=REPEATS):
    fn()
    mx.synchronize()
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        times.append((perf_counter() - t0) * 1e3)
    mx.synchronize()
    return min(times), statistics.median(times)


def _eval_then_list(idx):
    """What moe_layer_metal.py does: evaluate the routing, then read it out.

    The two steps have to act on the *same* array -- calling route() twice
    prices two launches and reads as a cost of `.tolist()` that it is not.
    """
    mx.eval(idx)
    return idx.tolist()


def main():
    mx.random.seed(0)
    logits = mx.random.normal((EXPERTS,)).astype(mx.float32)
    gate = mx.random.normal((EXPERTS, 512)).astype(mx.bfloat16)
    x = mx.random.normal((512,)).astype(mx.bfloat16)
    mx.eval(logits, gate, x)

    def route():
        scores = (gate @ x).astype(mx.float32)
        return mx.argpartition(-scores, TOPK)[:TOPK]

    print(f"one router-shaped launch, {EXPERTS} experts, top-{TOPK}, {REPEATS} repeats\n")

    arms = (
        ("chained, no eval", lambda: route()),
        ("mx.eval", lambda: mx.eval(route())),
        ("mx.eval + tolist", lambda: _eval_then_list(route())),
        ("tolist alone", lambda: route().tolist()),
        ("np.array", lambda: np.array(route())),
        ("item, one scalar", lambda: route()[0].item()),
    )
    base = None
    for name, fn in arms:
        lo, med = timed(fn)
        if base is None:
            base = lo
        print(f"  {name:22s} min {lo:7.4f} ms  median {med:7.4f} ms"
              f"  -> x40 layers {lo * 40:6.2f} ms/token   (+{(lo - base) * 40:5.2f} over chained)")

    # Does the round trip depend on how much is queued behind it? If it does,
    # the 0.31 ms is the GPU finishing work the eval is waiting for; if it does
    # not, it is the round trip itself and only the count of evals matters.
    print("\nqueue depth behind one eval:")
    for depth in (1, 2, 5, 10, 20, 40):
        def fn(d=depth):
            outs = [route() for _ in range(d)]
            mx.eval(outs)
        lo, med = timed(fn, repeats=max(40, REPEATS // depth))
        print(f"  {depth:3d} launches, one eval  min {lo:7.4f} ms  median {med:7.4f} ms"
              f"  -> {lo / depth:7.4f} ms per launch")

    # And the shape the decode thread actually evaluates: several arrays in one
    # call versus one call each.
    print("\none eval of k arrays versus k evals:")
    for k in (1, 2, 3):
        def one(kk=k):
            mx.eval([route() for _ in range(kk)])

        def many(kk=k):
            for _ in range(kk):
                mx.eval(route())
        lo_one, _ = timed(one)
        lo_many, _ = timed(many)
        print(f"  k={k}: one eval {lo_one:7.4f} ms   {k} evals {lo_many:7.4f} ms"
              f"   saving {(lo_many - lo_one) * 40:6.2f} ms/token at 40 layers")

    # Can the sync be hidden behind work that does not depend on it?
    #
    # A decode layer runs attention, then the router, then blocks on
    # mx.eval(route.indices) so the CPU can address the expert store, and only
    # then issues the MoE -- including the shared expert, which does not depend
    # on the routing at all. If the shared expert is launched before the sync,
    # the GPU has something to do while the CPU waits. This prices the upper
    # bound of that with a stand-in whose cost is the shared expert's 0.12 ms.
    work = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
    vec = mx.random.normal((4096,)).astype(mx.bfloat16)
    mx.eval(work, vec)

    def independent():
        return work @ vec

    def serial():
        idx = route()
        mx.eval(idx)
        idx.tolist()
        y = independent()
        mx.eval(y)

    def overlapped():
        y = independent()
        mx.async_eval(y)
        idx = route()
        mx.eval(idx)
        idx.tolist()
        mx.eval(y)

    lo_w, _ = timed(lambda: mx.eval(independent()))
    lo_s, med_s = timed(serial)
    lo_o, med_o = timed(overlapped)
    print("\nhiding the sync behind work that does not depend on it:")
    print(f"  the independent launch alone   min {lo_w:7.4f} ms")
    print(f"  sync first, then the work      min {lo_s:7.4f} ms  median {med_s:7.4f} ms")
    print(f"  work submitted first           min {lo_o:7.4f} ms  median {med_o:7.4f} ms"
          f"   -> {(lo_s - lo_o) * 40:6.2f} ms/token at 40 layers")


if __name__ == "__main__":
    main()
