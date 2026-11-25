"""Partition files for global_hierarchical_graph (run from the viewer worktree).

Converts Nils's weighted owner files (K format) to the benchmark loader's
format and writes lon_bands with the hidden-256 cost model.
Usage: python make_hier_files.py <nils_dir> <cost_model> <out_dir>
"""
import os
import sys

import numpy as np

from neural_lam.graph_viewer.costs import CostModel, node_weights, partition_costs
from neural_lam.graph_viewer.graph import load_viewer_graph
from neural_lam.graph_viewer.partition import partition_layers

nils_dir, cost, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
g = load_viewer_graph("sample_data/global_ocean_1.yaml", "global_hierarchical_graph")
w = CostModel.load(cost)
nw = node_weights(g, w)
sizes = [len(l.xyz) for l in g.layers]


def write(name, k, parts):
    parts = [np.asarray(p, dtype=np.uint8) for p in parts]
    for p, n in zip(parts, sizes):
        assert p.shape == (n,) and np.bincount(p, minlength=k).min() > 0
    np.savez_compressed(f"{out}/{name}_k{k}.npz", world_size=np.array(k),
                        name=np.array(name), grid=parts[0],
                        mesh_0=parts[1], mesh_1=parts[2])
    c = partition_costs(g, parts, k, w)
    halo = {l.name: c.halo_total(l.name) if hasattr(c, "halo_total") else None
            for l in g.layers}
    print(f"{name}_k{k}: max C {c.communication.max():,.0f}  S {c.slowdown:.3f}  "
          f"mem {100 * c.max_share:.2f}%")


for k in (2, 4):
    for src, name in (("metis", "nils_weighted_metis"),
                      ("chunks", "nils_weighted_chunks")):
        z = np.load(f"{nils_dir}/weighted/{src}-k{k}.npz")
        assert int(z["K"]) == k
        write(name, k, [z["grid"], z["mesh_0"], z["mesh_1"]])
    write("lon_bands", k,
          partition_layers([l.xyz for l in g.layers], "lon_bands", k, nw))
    write("max_split_viewer", k,
          partition_layers([l.xyz for l in g.layers], "per_layer_geo_rcb", k,
                           layer_lon_lat=[l.lon_lat for l in g.layers]))
