"""Load the hierarchical-graph partition files through the benchmark's builder
(CPU) and compare owners and halo traffic with the viewer's numbers."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks.model_parallel_real import _load_workload
from neural_lam.models.distributed_graph_efm import build_partition_from_graph_efm

parts_dir = Path(sys.argv[1])
args = SimpleNamespace(config_path=Path("sample_data/global_ocean_1.yaml"),
                       graph="global_hierarchical_graph", hidden_dim=64,
                       processor_layers=2, batch_size=2, ar_steps=1, lr=1e-3,
                       seed=42)
_, model, _ = _load_workload(args, include_global_dataset=False)
X = {"g2m": 2, "m2g": 1, "m2m": (6, 4), "mesh_up": (3,), "mesh_down": (1,)}


def build(k, **kw):
    per_rank = [build_partition_from_graph_efm(model, rank=r, world_size=k, **kw)
                for r in range(k)]
    owners = []
    for layer in range(3):
        n = (per_rank[0].full_grid_nodes if layer == 0
             else per_rank[0].mesh_nodes[layer - 1])
        size = sum(int((p.full_grid_nodes if layer == 0
                        else p.mesh_nodes[layer - 1]).numel()) for p in per_rank)
        o = np.full(size, -1)
        for r, p in enumerate(per_rank):
            idx = p.full_grid_nodes if layer == 0 else p.mesh_nodes[layer - 1]
            o[idx.numpy()] = r
        owners.append(o)
    comm = []
    for p in per_rank:
        c = 2 * p.g2m.num_halo_nodes + 1 * p.m2g.num_halo_nodes
        c += sum(x * b.num_halo_nodes for x, b in zip(X["m2m"], p.m2m))
        c += sum(x * b.num_halo_nodes for x, b in zip(X["mesh_up"], p.mesh_up))
        c += sum(x * b.num_halo_nodes for x, b in zip(X["mesh_down"], p.mesh_down))
        comm.append(c)
    return owners, comm


for k in (2, 4):
    pass
    """
    ref = np.load(parts_dir / f"max_split_viewer_k{k}.npz")
    same = all(np.array_equal(geo[i], ref[n])
               for i, n in enumerate(("grid", "mesh_0", "mesh_1")))
    """
    for name in ("nils_weighted_rcb", "nils_weighted_mtkahypar_km1"):
        f = parts_dir / f"{name}_k{k}.npz"
        owners, comm = build(k, strategy="file", partition_file=f)
        z = np.load(f)
        same = all(np.array_equal(owners[i], z[n])
                   for i, n in enumerate(("grid", "mesh_0", "mesh_1")))
        print(f"k={k} {name}: max C {max(comm):,}  owners as in file: {same}")
