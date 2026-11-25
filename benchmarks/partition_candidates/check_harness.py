"""Load partition files through the message-passing branch's builder on the
real 1-degree model (CPU) and compare halo counts with the viewer's."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.argv_saved = sys.argv; exp = Path(sys.argv[1])
from benchmarks.model_parallel_real import _load_workload
from neural_lam.models.distributed_graph_efm import build_partition_from_graph_efm

args = SimpleNamespace(config_path=Path("sample_data/global_ocean_1.yaml"), graph="global_cluster_graph",
                       hidden_dim=64, processor_layers=2, batch_size=2, ar_steps=1, lr=1e-3, seed=42)
_, model, _ = _load_workload(args, include_global_dataset=False)
d = np.load(exp / "graph.npz"); meta = json.load(open(exp / "graph_meta.json"))
rel = {"g2m": ("g2m", None), "m2g": ("m2g", None), "m2m 0": ("m2m", 0), "m2m 1": ("m2m", 1),
       "up 0->1": ("mesh_up", 0), "down 1->0": ("mesh_down", 0)}

def expected(parts, k):
    out = {}
    for j, m in enumerate(meta["edge_sets"]):
        ei = d[f"ei_{j}"]; sp, dp = parts[m["src"]][ei[0]], parts[m["dst"]][ei[1]]; cut = sp != dp
        pairs = np.unique(np.stack([ei[0, cut], dp[cut]]), axis=1)
        out[m["name"]] = np.bincount(pairs[1], minlength=k).tolist()
    return out

def measured(k, **kw):
    out = {name: [] for name in rel}
    for rank in range(k):
        p = build_partition_from_graph_efm(model, rank=rank, world_size=k, **kw)
        for name, (attr, i) in rel.items():
            part = getattr(p, attr) if i is None else getattr(p, attr)[i]
            out[name].append(part.num_halo_nodes)
    return out

ok = True
for k in (2, 4):
    for name in ("lon_bands", "mtkahypar_km1", "metis"):
        f = np.load(exp / "partitions" / f"{name}_k{k}.npz")
        exp_h = expected([f["grid"], f["mesh_0"], f["mesh_1"]], k)
        got = measured(k, strategy="file", partition_file=exp / "partitions" / f"{name}_k{k}.npz")
        match = exp_h == got; ok &= match
        print(f"{name} k={k}: halo match {match}  g2m {got['g2m']}  m2m0 {got['m2m 0']}")
    got = measured(k, strategy="geographic")
    print(f"geographic k={k} (built-in): g2m {got['g2m']} m2m0 {got['m2m 0']}")
print("ALL MATCH" if ok else "MISMATCH")
