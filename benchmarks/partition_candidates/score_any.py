"""Score partitions of any graph with the viewer cost model and estimate the
largest hidden size that fits.

Usage: python score_any.py <config> <graph> <cost_model.json> <exp_dir>
       <ks e.g. 2,4> [<also_1deg_graph_exp_dir_for_comparison>]
"""
import sys
import numpy as np
from neural_lam.graph_viewer.graph import load_viewer_graph
from neural_lam.graph_viewer.costs import CostModel, node_weights, partition_costs
from neural_lam.graph_viewer.partition import partition_layers

config, graph, cost, exp, ks = sys.argv[1:6]
ks = [int(k) for k in ks.split(",")]
g = load_viewer_graph(config, graph)
w = CostModel.load(cost)
nw = node_weights(g, w)
P = dict(np.load(f"{exp}/parts_any.npz"))
L = len(g.layers)
GIB = 2 ** 30
# Stored values per sample per hidden unit (one forecast step)
V = sum(w.node[l.name] * len(l) for l in g.layers) + sum(w.edge[es.name] * es.n_edges for es in g.edge_sets)
print(f"graph {graph}: stored values per sample and hidden unit V = {V/1e6:.2f} M "
      f"({V * 4 / GIB * 64:.2f} GiB per sample at hidden 64, fp32)")

def lib(name, k):
    seeds = sorted({int(key.split('|')[2]) for key in P if key.startswith(f"{name}|{k}|")})
    cands = [[P[f"{name}|{k}|{s}|{i}"] for i in range(L)] for s in seeds]
    return min(cands, key=lambda p: partition_costs(g, p, k, w).communication_sync) if cands else None

M_USABLE, M_FIXED = 10.5, 0.16
print(f"\nd_max, 1 GPU (batch 1): {(M_USABLE - M_FIXED) / (V * 4 / GIB):.0f}")
print("| k | strategy | max C | C_sync | S | memory share | max node imb (layer) | d_max b=1 | d_max b=2 |")
print("|---|---|---|---|---|---|---|---|---|")
for k in ks:
    cands = {"Max's split": partition_layers([l.xyz for l in g.layers], "per_layer_geo_rcb", k,
                                             layer_lon_lat=[l.lon_lat for l in g.layers]) if k in (2, 4) else None,
             "lon_bands": lib("lon_bands", k), "rcb": lib("rcb", k),
             "Mt-KaHyPar": lib("mtkahypar_km1", k), "METIS": lib("metis", k)}
    for name, parts in cands.items():
        if parts is None:
            continue
        c = partition_costs(g, parts, k, w)
        imb = max((c.node_imbalance(l.name), l.name) for l in g.layers)
        per_h = c.max_share * V * 4 / GIB
        print(f"| {k} | {name} | {c.communication.max():,.0f} | {c.communication_sync:,.0f} | {c.slowdown:.3f} | "
              f"{100 * c.max_share:.2f}% | {imb[0]:.3f} ({imb[1]}) | {(M_USABLE - M_FIXED) / per_h:.0f} | {(M_USABLE - M_FIXED) / (2 * per_h):.0f} |")
