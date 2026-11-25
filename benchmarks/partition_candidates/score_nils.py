import glob, os, sys, numpy as np
from neural_lam.graph_viewer.graph import load_viewer_graph
from neural_lam.graph_viewer.costs import CostModel, node_weights, partition_costs
from neural_lam.graph_viewer.partition import partition_layers
nils_dir, cost = sys.argv[1:3]
g = load_viewer_graph("sample_data/global_ocean_1.yaml", "global_hierarchical_graph")
w = CostModel.load(cost); nw = node_weights(g, w)
rows = []
for k in (2, 4):
    cands = {}
    for f in sorted(glob.glob(f"{nils_dir}/*/*-k{k}.npz")):
        z = np.load(f); cands[f"Nils {os.path.basename(os.path.dirname(f))} {str(z['partitioner'])}"] = [z["grid"].astype(np.int64), z["mesh_0"].astype(np.int64), z["mesh_1"].astype(np.int64)]
    cands["ours lon_bands"] = partition_layers([l.xyz for l in g.layers], "lon_bands", k, nw)
    cands["ours Max's split"] = partition_layers([l.xyz for l in g.layers], "per_layer_geo_rcb", k, layer_lon_lat=[l.lon_lat for l in g.layers])
    print(f"\n| k={k} | max C | C_sync | S | memory share | grid / m0 / m1 node imbalance |\n|---|---|---|---|---|---|")
    for name, parts in cands.items():
        c = partition_costs(g, parts, k, w)
        ni = " / ".join(f"{c.node_imbalance(l.name):.3f}" for l in g.layers)
        print(f"| {name} | {c.communication.max():,.0f} | {c.communication_sync:,.0f} | {c.slowdown:.3f} | {100*c.max_share:.2f}% | {ni} |")
