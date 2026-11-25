"""Run candidate partitioners on the exported graph (scratch venv).

Usage: python run_partitioners.py <exp_dir> <repo graph_viewer dir>
Writes <exp_dir>/parts.npz with key "<strategy>|k|seed|layer" -> owners.
"""
import json, sys, time
from collections import defaultdict
import numpy as np
import kahip
import mtkahypar

exp, gv_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, gv_dir)
from partition import STRATEGIES  # numpy-only module of the viewer

d = np.load(f"{exp}/graph.npz")
meta = json.load(open(f"{exp}/graph_meta.json"))
L = len(meta["layers"])
sizes = [len(d[f"w_{i}"]) for i in range(L)]
offset = np.concatenate([[0], np.cumsum(sizes)])
n_total = int(offset[-1])
w_mem = np.concatenate([d[f"w_{i}"] for i in range(L)])
edge_sets = [(m, d[f"ei_{j}"]) for j, m in enumerate(meta["edge_sets"])]

def split(global_parts):
    return [global_parts[offset[i]:offset[i + 1]] for i in range(L)]

results = {}
def store(name, k, seed, parts, secs):
    for i, p in enumerate(parts):
        results[f"{name}|{k}|{seed}|{i}"] = np.asarray(p, dtype=np.int64)
    results[f"{name}|{k}|{seed}|time"] = np.array([secs])

# ---------------- hierarchical: split mesh 1, children follow parents
up = next(ei for m, ei in edge_sets if m["name"].startswith("up"))
m2g = next(ei for m, ei in edge_sets if m["name"] == "m2g")
parent0 = np.full(sizes[1], -1); parent0[up[0]] = up[1]   # mesh0 -> mesh1
assert (parent0 >= 0).all()
dots = np.einsum("ij,ij->i", d["xyz_1"][m2g[0]], d["xyz_0"][m2g[1]])
order = np.lexsort((-dots, m2g[1]))                        # nearest sender first
first = np.unique(m2g[1][order], return_index=True)[1]
parentg = np.full(sizes[0], -1); parentg[m2g[1][order][first]] = m2g[0][order][first]
assert (parentg >= 0).all()
subtree = d["w_2"].copy()
np.add.at(subtree, parent0, d["w_1"])
np.add.at(subtree, parent0[parentg], d["w_0"])
for base in ("rcb", "lon_bands"):
    for k in (2, 3, 4):
        t = time.time()
        top = STRATEGIES[base](k).fit(d["xyz_2"], subtree).assign(d["xyz_2"])
        p1 = top[parent0]; p0 = p1[parentg]
        store(f"hier_{base}", k, 0, [p0, p1, top], time.time() - t)

# ---------------- Mt-KaHyPar: exact weighted-halo (km1) objective
nets, net_w = [], []
for m, ei in edge_sets:
    src = ei[0] + offset[m["src"]]; dst = ei[1] + offset[m["dst"]]
    recv = defaultdict(set)
    for s_, r_ in zip(src.tolist(), dst.tolist()):
        if s_ != r_:
            recv[s_].add(r_)
    for s_, rs in recv.items():
        nets.append([s_] + sorted(rs)); net_w.append(int(m["exchanges"]))
node_w = np.maximum(1, np.rint(w_mem)).astype(int).tolist()
print(f"hypergraph: {n_total} nodes, {len(nets)} nets, {sum(map(len, nets))} pins")
mtk = mtkahypar.initialize(8)
for k in (2, 3, 4):
    for seed in (0, 1, 2):
        mtkahypar.set_seed(seed)
        ctx = mtk.context_from_preset(mtkahypar.PresetType.QUALITY)
        ctx.set_partitioning_parameters(k, 0.03, mtkahypar.Objective.KM1)
        ctx.logging = False
        hg = mtk.create_hypergraph(ctx, n_total, len(nets), nets, node_w, net_w)
        t = time.time()
        ph = hg.partition(ctx)
        parts = np.array([ph.block_id(v) for v in range(n_total)])
        store("mtkahypar_km1", k, seed, split(parts), time.time() - t)
        print(f"mtkahypar k={k} seed={seed} km1={ph.km1()} imbalance={ph.imbalance(ctx):.3f} {time.time()-t:.1f}s")

# ---------------- KaHIP: edge cut on the symmetrised union graph
adj = defaultdict(float)
for m, ei in edge_sets:
    src = ei[0] + offset[m["src"]]; dst = ei[1] + offset[m["dst"]]
    for a, b in zip(src.tolist(), dst.tolist()):
        if a != b:
            key = (a, b) if a < b else (b, a)
            adj[key] += m["exchanges"]
nbrs = defaultdict(list)
for (a, b), wt in adj.items():
    nbrs[a].append((b, wt)); nbrs[b].append((a, wt))
xadj, adjncy, adjw = [0], [], []
for v in range(n_total):
    for u, wt in sorted(nbrs[v]):
        adjncy.append(u); adjw.append(int(round(wt)))
    xadj.append(len(adjncy))
print(f"union graph: {n_total} nodes, {len(adjncy)//2} undirected edges")
for k in (2, 3, 4):
    for seed in (0, 1, 2):
        t = time.time()
        cut, blocks = kahip.kaffpa(node_w, xadj, adjw, adjncy, k, 0.03, True, seed, 2)  # 2 = STRONG
        store("kahip_cut", k, seed, split(np.array(blocks)), time.time() - t)
        print(f"kahip k={k} seed={seed} cut={cut} {time.time()-t:.1f}s")

np.savez(f"{exp}/parts.npz", **results)
print("saved", len(results))
