"""
Price M13 (HANDOFF 18.5): slots that hold 2-bit bias codes instead of bf16 biases, the biases rebuilt on the GPU
per use. Reads real expert heads from the coded bank, checks the Metal rebuild against the bank's CPU table (bit for
bit), then times a decode-shaped chain: 57 layers x 4 experts x (w1, w3, w2) quantized matmuls on one row, with
and without one rebuild kernel per layer in front of them.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_bias_kernel_price.py ~/MiniMax-M3-coded-bank
"""
import json
import os
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

from cachalot.minimax.coded_bank import PROJS, BankLayout, decode_biases

BANK = sys.argv[1]
meta = json.loads(open(os.path.join(BANK, "bank.json")).read())
lay = BankLayout(**meta["layout"])
shapes = meta["format"]["shapes"]
coded = [r for r in meta["records"] if r[4] == "coded"]
rng = np.random.default_rng(0)
LAYERS, K = 57, 4
picks = [coded[i] for i in rng.choice(len(coded), LAYERS * K, replace=False)]

G = lay.scales // 2  # groups per projection
NC = lay.scales // 2 // 4  # code bytes per projection
# one head per expert: 3 scale arrays (bf16) then 3 code arrays, as the bank stores it
heads, cpu_biases = [], []
for fname, off, _ in [(r[2], r[3], r[4]) for r in picks]:
    with open(os.path.join(BANK, fname), "rb") as f:
        f.seek(off)
        raw = f.read(3 * (lay.scales + NC))
    heads.append(np.frombuffer(raw, np.uint8).copy())
    sc = np.frombuffer(raw[:3 * lay.scales], np.uint16).reshape(3, G)
    cd = np.frombuffer(raw[3 * lay.scales:], np.uint8).reshape(3, NC)
    cpu_biases.append(np.stack([decode_biases(sc[i], cd[i]) for i in range(3)]))

SRC = """
    // one thread per (expert, projection, code byte): four groups; expert e's head is input h<e>
    uint cb = thread_position_in_grid.x;
    uint p = thread_position_in_grid.y;
    uint e = thread_position_in_grid.z;
    if (cb >= G / 4) return;
    device const uint8_t* h = e == 0 ? h0 : e == 1 ? h1 : e == 2 ? h2 : h3;
    ushort4 sc = ((const device ushort4*)(h + p * G * 2))[cb];
    uint packed = h[3 * G * 2 + p * (G / 4) + cb];
    ushort4 out;
    for (int j = 0; j < 4; j++) {
        float f = float(int((packed >> (2 * j)) & 3) - 6) * as_type<float>(uint(sc[j]) << 16);
        uint u = as_type<uint>(f);
        out[j] = ushort((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    }
    uint q = e * 3 + p;
    device ushort4* o = (device ushort4*)(q == 0 ? o0 : q == 1 ? o1 : q == 2 ? o2 : q == 3 ? o3 : q == 4 ? o4 :
        q == 5 ? o5 : q == 6 ? o6 : q == 7 ? o7 : q == 8 ? o8 : q == 9 ? o9 : q == 10 ? o10 : o11);
    o[cb] = out;
"""
kern = mx.fast.metal_kernel(name="minimax_bias_rebuild", input_names=["h0", "h1", "h2", "h3"],
                            output_names=[f"o{i}" for i in range(12)], source=SRC)


def rebuild(hs):
    return kern(inputs=hs, template=[("G", G)], grid=(G // 4, 3, 4), threadgroup=(256, 1, 1),
                output_shapes=[(G,)] * 12, output_dtypes=[mx.uint16] * 12)


mh = [mx.array(h) for h in heads]
mx.eval(*mh)
bad = 0
for layer in range(LAYERS):
    outs = rebuild(mh[layer * K:(layer + 1) * K])
    mx.eval(*outs)
    for e in range(K):
        for p in range(3):
            bad += int((np.array(outs[e * 3 + p]) != cpu_biases[layer * K + e][p]).sum())
print(f"BITS experts={LAYERS * K} groups_checked={LAYERS * K * 3 * G} mismatches={bad}", flush=True)

# decode-shaped chain with random weights of the real shapes (the time does not depend on the values)
Wt = {p: mx.random.randint(0, 2**31, shapes[f"{p}.weight"], dtype=mx.uint32) for p in PROJS}
S = {p: mx.array(np.frombuffer(heads[0][i * lay.scales:(i + 1) * lay.scales], np.uint16)).view(mx.bfloat16)
     .reshape(shapes[f"{p}.scales"]) for i, p in enumerate(PROJS)}
B = {p: mx.array(cpu_biases[0][i]).view(mx.bfloat16).reshape(shapes[f"{p}.biases"]) for i, p in enumerate(PROJS)}
mx.eval(*Wt.values(), *S.values(), *B.values())
x0 = mx.random.normal((1, shapes["w1.weight"][1] * 32 // 3)).astype(mx.bfloat16)


def qmm(x, p, b):
    return mx.quantized_matmul(x, Wt[p], S[p], b, transpose=True, group_size=64, bits=3)


def token(with_kernel):
    x = x0
    for layer in range(LAYERS):
        if with_kernel:
            outs = rebuild(mh[layer * K:(layer + 1) * K])
            bs = [{p: outs[e * 3 + i].view(mx.bfloat16).reshape(shapes[f"{p}.biases"]) for i, p in enumerate(PROJS)}
                  for e in range(K)]
        else:
            bs = [B] * K
        y = 0
        for e in range(K):
            g = qmm(x, "w1", bs[e]["w1"])
            u = qmm(x, "w3", bs[e]["w3"])
            y = y + qmm(mx.sigmoid(g) * g * u, "w2", bs[e]["w2"])
        mx.eval(y)  # one sync per layer, as the decode does
        x = (y * 1e-3).astype(mx.bfloat16)
    return x


for arm in (0, 1, 0, 1, 0, 1):
    token(arm)
    ts = []
    for _ in range(20):
        t = time.perf_counter()
        token(arm)
        ts.append(time.perf_counter() - t)
    print(f"CHAIN kernel={arm} median_ms={1000 * statistics.median(ts):.2f} min_ms={1000 * min(ts):.2f}", flush=True)
