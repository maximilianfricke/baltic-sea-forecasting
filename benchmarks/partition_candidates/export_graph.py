"""Export the viewer graph and cost-model weights to one npz (project venv)."""
import json, sys, numpy as np
from neural_lam.graph_viewer.graph import load_viewer_graph
from neural_lam.graph_viewer.costs import CostModel, node_weights
g = load_viewer_graph("sample_data/global_ocean_1.yaml", "global_cluster_graph")
w = CostModel.load("neural_lam/graph_viewer/weights/realistic_baseline.json")
nw = node_weights(g, w)
out = {}
meta = {"layers": [l.name for l in g.layers], "edge_sets": []}
for i, l in enumerate(g.layers):
    out[f"lonlat_{i}"] = l.lon_lat
    out[f"xyz_{i}"] = l.xyz
    out[f"w_{i}"] = nw[i]
for j, es in enumerate(g.edge_sets):
    out[f"ei_{j}"] = es.edge_index
    meta["edge_sets"].append({"name": es.name, "src": es.src, "dst": es.dst,
                              "uses": w.uses[es.name], "exchanges": w.exchanges[es.name]})
np.savez(sys.argv[1] + "/graph.npz", **out)
json.dump(meta, open(sys.argv[1] + "/graph_meta.json", "w"), indent=1)
print(meta)
