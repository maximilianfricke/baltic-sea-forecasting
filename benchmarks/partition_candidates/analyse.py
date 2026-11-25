"""Summarise partition-comparison results against cost-model predictions.

Usage (viewer checkout on PYTHONPATH, run from it):
  python analyse.py <results_dir> <exp_dir>
"""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict
import numpy as np
from neural_lam.graph_viewer.graph import load_viewer_graph
from neural_lam.graph_viewer.costs import CostModel, partition_costs
from neural_lam.graph_viewer.partition import partition_layers

res_dir, exp = sys.argv[1], sys.argv[2]
GIB = 2 ** 30
STORED = {64: 427.2e6, 256: 1670.0e6}  # measured values per sample and step
MODELS = {64: CostModel.load("neural_lam/graph_viewer/weights/realistic_baseline.json"),
          256: CostModel.load(f"{exp}/cost_h256.json")}
g = load_viewer_graph("sample_data/global_ocean_1.yaml", "global_cluster_graph")
ORDER = ["geographic", "lon_bands", "mtkahypar_km1", "metis"]

def parts_for(strategy, k):
    if strategy == "geographic":
        return partition_layers([l.xyz for l in g.layers], "per_layer_geo_rcb", k,
                                layer_lon_lat=[l.lon_lat for l in g.layers])
    f = np.load(f"{exp}/partitions/{strategy}_k{k}.npz")
    return [f["grid"], f["mesh_0"], f["mesh_1"]]

pred = {}
for h, w in MODELS.items():
    for k in (2, 4):
        for s in ORDER:
            c = partition_costs(g, parts_for(s, k), k, w)
            pred[(h, k, s)] = dict(share=c.max_share, S=c.slowdown, csync=c.communication_sync)

runs = defaultdict(list)
for path in sorted(glob.glob(f"{res_dir}/*.json")):
    m = re.match(r"h(\d+)-(\d)gpu-(\w+?)-rep(\d)", os.path.basename(path))
    if not m:
        continue
    d = json.load(open(path))
    h, k, s = int(m[1]), int(m[2]), m[3]
    recs = d["records"]
    peaks = [max(r["peak_memory_allocated_bytes"] for r in rm["records"]) for rm in d["rank_metadata"]]
    runs[(h, k, s)].append(dict(
        step=st.median(r["step_ms"] for r in recs),
        fwd=st.median(r["forward_ms"] for r in recs),
        bwd=st.median(r["backward_ms"] for r in recs),
        peak=max(peaks) / GIB, n=len(recs)))

print("| hidden | GPUs | strategy | step ms (reps) | fwd | bwd | step vs geo | pred. S vs geo | peak GiB | Δpeak vs geo MiB | pred. Δ MiB |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for h in (64, 256):
    for k in (2, 4):
        if not runs.get((h, k, "geographic")):
            continue
        base = runs[(h, k, "geographic")]
        b_step = st.mean(r["step"] for r in base); b_peak = st.mean(r["peak"] for r in base)
        for s in ORDER:
            rs = runs.get((h, k, s))
            if not rs:
                continue
            step = st.mean(r["step"] for r in rs); peak = st.mean(r["peak"] for r in rs)
            p, pg = pred[(h, k, s)], pred[(h, k, "geographic")]
            d_pred = (p["share"] - pg["share"]) * STORED[h] * 2 * 4 / 2 ** 20
            print(f"| {h} | {k} | {s} | {step:.1f} ({', '.join(f'{r['step']:.0f}' for r in rs)}) "
                  f"| {st.mean(r['fwd'] for r in rs):.1f} | {st.mean(r['bwd'] for r in rs):.1f} "
                  f"| {step / b_step:.3f} | {p['S'] / pg['S']:.3f} | {peak:.3f} "
                  f"| {(peak - b_peak) * 1024:+.0f} | {d_pred:+.0f} |")
