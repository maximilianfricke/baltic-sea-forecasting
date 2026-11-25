"""Multi-constraint METIS: one balance constraint per layer (scratch venv).

Usage: python run_metis.py <exp_dir> <pymetis site-packages>
"""
import json, sys, time
import numpy as np
sys.path.insert(0, sys.argv[2])
import pymetis

NODE_UPDATE_COST = 0.75
exp = sys.argv[1]
d = np.load(f"{exp}/graph.npz"); meta = json.load(open(f"{exp}/graph_meta.json"))
L = len(meta["layers"]); sizes = [len(d[f"w_{i}"]) for i in range(L)]
offset = np.concatenate([[0], np.cumsum(sizes)]); n = int(offset[-1])
edge_sets = [(m, d[f"ei_{j}"]) for j, m in enumerate(meta["edge_sets"])]

# Undirected union graph, edge weight = sum of exchanges over both directions
a_list, b_list, w_list = [], [], []
work = np.zeros(n)
for m, ei in edge_sets:
    a = ei[0] + offset[m["src"]]; b = ei[1] + offset[m["dst"]]
    keep = a != b
    a_list += [a[keep], b[keep]]; b_list += [b[keep], a[keep]]
    w_list += [np.full(keep.sum(), m["exchanges"])] * 2
    deg = np.bincount(ei[1], minlength=sizes[m["dst"]])
    work[offset[m["dst"]]:offset[m["dst"] + 1]] += m["uses"] * (deg + NODE_UPDATE_COST)
a = np.concatenate(a_list); b = np.concatenate(b_list); w = np.concatenate(w_list)
key, inv = np.unique(a * n + b, return_inverse=True)
ew = np.bincount(inv.ravel(), weights=w).astype(np.int64)
src, dst = key // n, key % n            # sorted by src, then dst
xadj = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))]).astype(np.int64)
print(f"union graph: {n} nodes, {len(key) // 2} undirected edges")

# One constraint per layer: the node's compute in its own layer
vw = np.zeros((n, L), dtype=np.int64)
for i in range(L):
    sl = slice(offset[i], offset[i + 1])
    vw[sl, i] = np.maximum(1, np.rint(4 * work[sl]))

results = {}
for k in (2, 3, 4):
    for obj_name, obj in (("cut", pymetis.ObjType.CUT),):
        for ufactor in (10, 1):
            for seed in (0, 1, 2):
                opts = pymetis.Options(); opts.objtype = obj; opts.ufactor = ufactor; opts.seed = seed
                t = time.time()
                r = pymetis.part_graph(k, xadj=xadj, adjncy=dst.astype(np.int64), vweights=vw.ravel(),
                                       eweights=ew, options=opts, recursive=True)
                secs = time.time() - t
                parts = np.asarray(r.vertex_part, dtype=np.int64)
                name = f"metis_rb_u{ufactor}"
                for i in range(L):
                    results[f"{name}|{k}|{seed}|{i}"] = parts[offset[i]:offset[i + 1]]
                results[f"{name}|{k}|{seed}|time"] = np.array([secs])
        print(f"k={k} {obj_name} done ({secs:.2f}s per run)", flush=True)
np.savez(f"{exp}/parts4.npz", **results)
