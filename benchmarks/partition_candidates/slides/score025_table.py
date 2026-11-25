"""Slide-table numbers on the 0.25-degree graph: weighted METIS on mesh 0
(other layers attached), lon_bands, per-layer geographic split.
Usage: python score025_table.py <config> <cost_model> <pymetis_dir>"""
import sys

import numpy as np

config, cost, pm = sys.argv[1:4]
sys.path.insert(0, pm)

from neural_lam.graph_viewer.costs import CostModel, node_weights, partition_costs
from neural_lam.graph_viewer.graph import load_viewer_graph
from neural_lam.graph_viewer.partition import partition_layers

g = load_viewer_graph(config, "global_cluster_analysis")
w = CostModel.load(cost)
nw = node_weights(g, w)
xyz = [l.xyz for l in g.layers]
print("layers:", [(l.name, len(l)) for l in g.layers])
for k in (2, 4):
    cands = {
        "Weighted METIS on G0": partition_layers(xyz, "mesh0_metis", k, nw, graph=g),
        "lon_bands": partition_layers(xyz, "lon_bands", k, nw),
        "Per-layer split": partition_layers(
            xyz, "per_layer_geo_rcb", k,
            layer_lon_lat=[l.lon_lat for l in g.layers]),
    }
    for name, parts in cands.items():
        c = partition_costs(g, parts, k, w)
        imb = " / ".join(f"{c.node_imbalance(l.name):.3f}" for l in g.layers)
        print(f"k={k} | {name} | S {c.slowdown:.3f} | mem {100 * c.max_share:.2f}% "
              f"| imb {imb} | max C {c.communication.max():,.0f} "
              f"| C_sync {c.communication_sync:,.0f}")
