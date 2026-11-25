"""Write benchmark partition files (scratch venv).

Usage: python make_partition_files.py <exp_dir> <viewer_src_dir>
Chooses the Mt-KaHyPar seed with the lowest synchronised communication.
"""
import glob, json, sys
import numpy as np
exp, src = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from partition import partition_layers

d = np.load(f"{exp}/graph.npz"); meta = json.load(open(f"{exp}/graph_meta.json"))
L = len(meta["layers"])
sizes = [len(d[f"w_{i}"]) for i in range(L)]
edge_sets = [(m, d[f"ei_{j}"]) for j, m in enumerate(meta["edge_sets"])]

def c_sync(parts, k):
    total = 0
    for m, ei in edge_sets:
        sp, dp = parts[m["src"]][ei[0]], parts[m["dst"]][ei[1]]
        cut = sp != dp
        pairs = np.unique(np.stack([ei[0, cut], dp[cut]]), axis=1)
        total += m["exchanges"] * np.bincount(pairs[1], minlength=k).max()
    return total

def write(name, k, parts):
    for i, p in enumerate(parts):
        assert len(p) == sizes[i] and np.bincount(p, minlength=k).min() > 0
    arrays = {"world_size": np.array(k), "name": np.array(name), "grid": parts[0]}
    for level in range(1, L):
        arrays[f"mesh_{level - 1}"] = parts[level]
    path = f"{exp}/partitions/{name}_k{k}.npz"
    np.savez(path, **arrays)
    print(f"{path}: C_sync {c_sync(parts, k):.0f}")

P = {}
for f in sorted(glob.glob(f"{exp}/parts*.npz")):
    z = np.load(f); P.update({key: z[key] for key in z.files})
for k in (2, 4):
    xyz = [d[f"xyz_{i}"] for i in range(L)]
    w = [d[f"w_{i}"] for i in range(L)]
    write("lon_bands", k, [p.astype(np.int64) for p in partition_layers(xyz, "lon_bands", k, w)])
    seeds = sorted({int(key.split("|")[2]) for key in P if key.startswith(f"mtkahypar_km1|{k}|")})
    candidates = [[P[f"mtkahypar_km1|{k}|{s}|{i}"] for i in range(L)] for s in seeds]
    best = min(candidates, key=lambda parts: c_sync(parts, k))
    write("mtkahypar_km1", k, best)
    seeds = sorted({int(key.split("|")[2]) for key in P if key.startswith(f"metis_rb_u10|{k}|")})
    candidates = [[P[f"metis_rb_u10|{k}|{s}|{i}"] for i in range(L)] for s in seeds]
    write("metis", k, min(candidates, key=lambda parts: c_sync(parts, k)))
