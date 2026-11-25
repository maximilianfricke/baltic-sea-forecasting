"""Export a graph, its node weights, uses and exchanges to one npz.

Usage: python export_graph.py <out_dir> [config_path graph_name cost_model]
Defaults: the 1-degree sample data, global_cluster_graph and the realistic
baseline cost model. Needs the graph viewer (branch lassi-balance-metrics).
"""
import json
import os
import sys

import numpy as np

from neural_lam.graph_viewer.costs import CostModel, node_weights
from neural_lam.graph_viewer.graph import load_viewer_graph

out = sys.argv[1]
config, graph_name, cost = (sys.argv[2:5] if len(sys.argv) > 2 else (
    "sample_data/global_ocean_1.yaml",
    "global_cluster_graph",
    "neural_lam/graph_viewer/weights/realistic_baseline.json",
))
os.makedirs(out, exist_ok=True)
g = load_viewer_graph(config, graph_name)
w = CostModel.load(cost)
w.check(g)
nw = node_weights(g, w)
arrays = {}
meta = {"graph": graph_name, "cost_model": w.name,
        "layers": [layer.name for layer in g.layers], "edge_sets": []}
for i, layer in enumerate(g.layers):
    arrays[f"lonlat_{i}"] = layer.lon_lat
    arrays[f"xyz_{i}"] = layer.xyz
    arrays[f"w_{i}"] = nw[i]
for j, es in enumerate(g.edge_sets):
    arrays[f"ei_{j}"] = es.edge_index
    meta["edge_sets"].append({
        "name": es.name, "src": es.src, "dst": es.dst,
        "uses": w.uses[es.name], "exchanges": w.exchanges[es.name],
    })
np.savez(os.path.join(out, "graph.npz"), **arrays)
with open(os.path.join(out, "graph_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
print(meta["layers"], [m["name"] for m in meta["edge_sets"]])
