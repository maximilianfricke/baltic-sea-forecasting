"""New partitions of global_hierarchical_graph (k = 4) to test the step-time
model out of sample: longitude bands with skewed borders (more imbalance),
with random node reassignment (more halo traffic), both, and fully random.

Run from the viewer worktree (branch lassi-balance-metrics):
    python make_test_partitions.py <cost_model.json> <out_dir> <a> <b> <c_us>
a, b, c_us: fitted model time = a + b * imbalance + c_us * 1e-3 * halo (ms).
"""
import os
import sys

import numpy as np

from neural_lam.graph_viewer.costs import CostModel, node_weights, partition_costs
from neural_lam.graph_viewer.graph import load_viewer_graph
from neural_lam.graph_viewer.partition import LongitudeBands, _weighted_quantile

cost, out = sys.argv[1:3]
a, b, c_us = (float(x) for x in sys.argv[3:6])
os.makedirs(out, exist_ok=True)
K = 4
g = load_viewer_graph("sample_data/global_ocean_1.yaml",
                      "global_hierarchical_graph")
w = CostModel.load(cost)
nw = node_weights(g, w)
xyz = [l.xyz for l in g.layers]


def bands(fractions):
    """Longitude bands over all layers pooled, band i holding fractions[i]
    of the total memory weight (equal fractions = lon_bands)."""
    lb = LongitudeBands(K).fit(np.concatenate(xyz), np.concatenate(nw))
    shifted = lb._shift(np.arctan2(np.concatenate(xyz)[:, 1],
                                   np.concatenate(xyz)[:, 0]))
    cum = np.cumsum(fractions)[:-1]
    lb.borders = np.array([_weighted_quantile(shifted, np.concatenate(nw), q)
                           for q in cum])
    return [lb.assign(x) for x in xyz]


def noise(parts, frac, seed):
    """Move a random fraction of every layer's nodes to a random GPU."""
    rng = np.random.default_rng(seed)
    new = []
    for p in parts:
        p = p.copy()
        idx = rng.choice(len(p), size=int(round(frac * len(p))), replace=False)
        p[idx] = rng.integers(0, K, size=len(idx))
        new.append(p)
    return new


def skew(s):
    return [0.25 * (1 + s), 0.25, 0.25, 0.25 * (1 - s)]


even = [0.25] * 4
rng = np.random.default_rng(7)
cands = {
    "test_bands": bands(even),
    "test_noise01": noise(bands(even), 0.01, 1),
    "test_noise03": noise(bands(even), 0.03, 2),
    "test_noise10": noise(bands(even), 0.10, 3),
    "test_noise30": noise(bands(even), 0.30, 5),
    "test_skew10": bands(skew(0.10)),
    "test_skew20": bands(skew(0.20)),
    "test_skew10_noise03": noise(bands(skew(0.10)), 0.03, 4),
    "test_random": [rng.integers(0, K, size=len(x)) for x in xyz],
}
print(f"{'partition':22s} {'imbalance':>9s} {'halo copies':>11s} "
      f"{'MB/sample':>9s} {'mem share':>9s} {'predicted ms':>12s}")
for name, parts in cands.items():
    parts = [np.asarray(p, dtype=np.uint8) for p in parts]
    for p, x in zip(parts, xyz):
        assert len(p) == len(x) and np.bincount(p, minlength=K).min() > 0
    c = partition_costs(g, parts, K, w)
    s, h = c.slowdown, float(c.communication.max())
    pred = a + b * s + c_us * 1e-3 * h
    print(f"{name:22s} {s:9.3f} {h:11,.0f} {h * 1024 / 1e6:9.2f} "
          f"{100 * c.max_share:8.2f}% {pred:12.1f}")
    if name != "test_bands":
        np.savez_compressed(f"{out}/{name}_k{K}.npz", world_size=np.array(K),
                            name=np.array(name), grid=parts[0],
                            mesh_0=parts[1], mesh_1=parts[2])
