"""Shared-border geometric, Mt-KaHyPar and METIS partitions for a graph with
any number of mesh levels.

WARNING: the METIS call passes one weight per node and layer, but
pymetis.part_graph uses one weight per node (the first n entries), so the
METIS result is not a multi-constraint partition.

Usage: python run_any.py <exp_dir> <dir holding partition.py> <pymetis path>
       [k ...] [--seeds N]
Writes <exp_dir>/parts_any.npz with keys "<strategy>|k|seed|layer".
"""
import json
import sys
import time

import numpy as np

argv = sys.argv[1:]
seeds = 1
if "--seeds" in argv:
    i = argv.index("--seeds")
    seeds = int(argv[i + 1])
    del argv[i:i + 2]
exp, viewer_src, pymetis_path = argv[:3]
ks = [int(k) for k in argv[3:]] or [2, 4]
sys.path.insert(0, viewer_src)
sys.path.insert(0, pymetis_path)
import mtkahypar  # noqa: E402
import pymetis  # noqa: E402
from partition import STRATEGIES  # noqa: E402

NODE_UPDATE_COST = 0.75
d = np.load(f"{exp}/graph.npz")
meta = json.load(open(f"{exp}/graph_meta.json"))
L = len(meta["layers"])
sizes = [len(d[f"w_{i}"]) for i in range(L)]
offset = np.concatenate([[0], np.cumsum(sizes)])
n = int(offset[-1])
edge_sets = [(m, d[f"ei_{j}"]) for j, m in enumerate(meta["edge_sets"])]
w_mem = np.concatenate([d[f"w_{i}"] for i in range(L)])
results = {}


def store(name, k, seed, parts, secs):
    for i in range(L):
        results[f"{name}|{k}|{seed}|{i}"] = np.asarray(
            parts[offset[i]:offset[i + 1]], dtype=np.int64)
    results[f"{name}|{k}|{seed}|time"] = np.array([secs])
    print(f"{name} k={k} seed={seed}: {secs:.1f}s", flush=True)


# Shared-border geometric strategies, balanced by memory weight
xyz = np.concatenate([d[f"xyz_{i}"] for i in range(L)])
for name in ("lon_bands", "rcb"):
    for k in ks:
        t = time.time()
        parts = STRATEGIES[name](k).fit(xyz, w_mem).assign(xyz)
        store(name, k, 0, parts, time.time() - t)

# Mt-KaHyPar: one net per (sender, edge set) = sender + its receivers,
# weighted by exchanges, so km1 = weighted halo traffic
nets, net_w = [], []
for m, ei in edge_sets:
    src = ei[0] + offset[m["src"]]
    dst = ei[1] + offset[m["dst"]]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    pairs = np.unique(np.stack([src, dst]), axis=1)
    starts = np.flatnonzero(np.r_[True, pairs[0, 1:] != pairs[0, :-1]])
    for s_, rec in zip(pairs[0, starts], np.split(pairs[1], starts[1:])):
        nets.append([int(s_)] + rec.tolist())
        net_w.append(int(m["exchanges"]))
node_w = np.maximum(1, np.rint(w_mem)).astype(int).tolist()
print(f"hypergraph: {n} nodes, {len(nets)} nets", flush=True)
mtk = mtkahypar.initialize(8)
for k in ks:
    for seed in range(seeds):
        mtkahypar.set_seed(seed)
        ctx = mtk.context_from_preset(mtkahypar.PresetType.QUALITY)
        ctx.set_partitioning_parameters(k, 0.03, mtkahypar.Objective.KM1)
        ctx.logging = False
        hg = mtk.create_hypergraph(ctx, n, len(nets), nets, node_w, net_w)
        t = time.time()
        ph = hg.partition(ctx)
        parts = np.array([ph.block_id(v) for v in range(n)])
        store("mtkahypar_km1", k, seed, parts, time.time() - t)

# METIS: undirected union graph, edge weight = exchanges, one balance
# constraint per layer (the node's compute in its own layer)
a_l, b_l, w_l = [], [], []
work = np.zeros(n)
for m, ei in edge_sets:
    a = ei[0] + offset[m["src"]]
    b = ei[1] + offset[m["dst"]]
    keep = a != b
    a_l += [a[keep], b[keep]]
    b_l += [b[keep], a[keep]]
    w_l += [np.full(keep.sum(), m["exchanges"])] * 2
    deg = np.bincount(ei[1], minlength=sizes[m["dst"]])
    work[offset[m["dst"]]:offset[m["dst"] + 1]] += m["uses"] * (
        deg + NODE_UPDATE_COST)
a, b, w = np.concatenate(a_l), np.concatenate(b_l), np.concatenate(w_l)
key, inv = np.unique(a * n + b, return_inverse=True)
ew = np.bincount(inv.ravel(), weights=w).astype(np.int64)
src, dst = key // n, key % n
xadj = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))])
vw = np.zeros((n, L), dtype=np.int64)
for i in range(L):
    sl = slice(offset[i], offset[i + 1])
    vw[sl, i] = np.maximum(1, np.rint(4 * work[sl]))
for k in ks:
    for seed in range(seeds):
        opts = pymetis.Options()
        opts.ufactor = 10
        opts.seed = seed
        t = time.time()
        r = pymetis.part_graph(k, xadj=xadj.astype(np.int64),
                               adjncy=dst.astype(np.int64),
                               vweights=vw.ravel(), eweights=ew, options=opts)
        store("metis", k, seed, np.asarray(r.vertex_part), time.time() - t)
np.savez(f"{exp}/parts_any.npz", **results)
