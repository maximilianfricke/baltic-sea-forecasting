"""Slide figure: cost-model imbalance S vs measured step time (4 GPUs,
hidden 256, 1-degree hierarchical graph).
Usage: python balance_fig.py <results_dir> <out.png> <jobs,comma,separated>"""
import glob
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

root, out, jobs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
HIGHLIGHT, NEUTRAL = "#4f5b1a", "#7d7d72"
INK, MUTED, GRID = "#1f1f1c", "#6b6b63", "#e4e4dc"

# Cost model, k = 4, hidden-256 cost model (score_nils.py)
S = {"lon_bands": 1.025, "geographic": 1.028, "nils_weighted_chunks": 1.033,
     "nils_weighted_metis": 1.067, "nils_weighted_rcb": 1.077,
     "nils_weighted_mtkahypar_km1": 1.090}
HALO = {"lon_bands": 2090, "geographic": 3561, "nils_weighted_chunks": 69234,
        "nils_weighted_metis": 1070, "nils_weighted_rcb": 1845,
        "nils_weighted_mtkahypar_km1": 945}
names = {"lon_bands": "Longitude bands", "geographic": "Per-layer split",
         "nils_weighted_metis": "METIS on G0",
         "nils_weighted_mtkahypar_km1": "Mt-KaHyPar on G0",
         "nils_weighted_rcb": "RCB on G0",
         "nils_weighted_chunks": "Node-ID chunks"}

steps, run_means = {}, {}
for job in jobs:
    for f in glob.glob(f"{root}/{job}/h256-4gpu-*-rep*.json"):
        key = f.split("h256-4gpu-")[1].rsplit("-rep", 1)[0]
        d = json.load(open(f))
        t = [r["compute_ms"] for r in d["records"]]
        steps.setdefault(key, []).extend(t)
        run_means.setdefault(key, []).append(np.mean(t))
med = {k: np.mean(v) for k, v in steps.items()}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
fig = plt.figure(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("white")
fig.text(0.045, 0.93, "Imbalance alone doesn't predict training time",
         fontsize=30, weight="bold", color=INK)
fig.text(0.045, 0.875,
         "Imbalance from the cost model vs measured training step, 4 GPUs, hidden 256. "
         "Lower left is better.", fontsize=17, color=MUTED)
ax = fig.add_axes([0.09, 0.17, 0.86, 0.64])

# Least-squares fits of mean step time on imbalance alone
def fit(keys):
    x = np.array([S[k] for k in keys])
    t = np.array([med[k] for k in keys])
    slope, icpt = np.polyfit(x, t, 1)
    res = t - (icpt + slope * x)
    return slope, icpt, 1 - res @ res / ((t - t.mean()) @ (t - t.mean()))


all6 = [k for k in S if k in med]
low5 = [k for k in all6 if k != "nils_weighted_chunks"]
xs = np.linspace(1.016, 1.100, 50)
for keys, style, label, ypos in (
        (all6, "-", "all six", 1.0955),
        (low5, "--", "without node-ID chunks", 1.0955)):
    slope, icpt, r2 = fit(keys)
    ax.plot(xs, icpt + slope * xs, style, color=MUTED, linewidth=1.5,
            zorder=1, label=f"linear fit, {label}:  $R^2$ = {r2:.2f}")
    print(f"imbalance-only fit, {label}: R2 {r2:.3f}")

offsets = {"lon_bands": (0.0, -3.4, "center"),
           "geographic": (0.0015, -1.0, "left"),
           "nils_weighted_metis": (-0.0017, 0.0, "right"),
           "nils_weighted_rcb": (-0.0017, 2.4, "right"),
           "nils_weighted_mtkahypar_km1": (-0.0017, -2.0, "right"),
           "nils_weighted_chunks": (0.0017, 0.0, "left")}
for k in S:
    if k not in med:
        continue
    c = HIGHLIGHT if k == "lon_bands" else NEUTRAL
    rm = run_means[k]
    ax.scatter([S[k]] * len(rm), rm, s=22, color=c, alpha=0.45,
               edgecolors="none", zorder=2)
    ax.scatter([S[k]], [med[k]], s=170, color=c, edgecolors="white",
               linewidths=2, zorder=3)
    dx, dy, ha = offsets[k]
    label = f"{names[k]}  {med[k]:.1f} ms"
    ax.text(S[k] + dx, med[k] + dy, label, ha=ha, va="center", fontsize=15,
            color=INK, weight="normal")

k = "nils_weighted_chunks"
if k in med:
    ax.text(S[k] + 0.0017, med[k] - 2.6,
            f"slow from halo traffic: {HALO[k]:,} node copies per step,\n"
            f"20–65× the others (≈ {HALO[k] * 256 * 4 / 1e6:.0f} MB per sample)",
            ha="left", va="top", fontsize=13, color=MUTED, linespacing=1.2)

ax.set_xlim(1.015, 1.101)
ax.set_ylim(238, 284)
ax.legend(loc="upper right", frameon=False, fontsize=15, handlelength=2.5)
ax.set_xlabel(r"$\mathit{imbalance}$ from the cost model "
              "(1.00 = perfectly balanced)",
              color=MUTED, fontsize=15)
ax.set_ylabel("Mean time per training step (ms)", color=MUTED, fontsize=15)
ax.grid(color=GRID, linewidth=1)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=13)

fig.text(0.045, 0.075,
         "Large dot: mean of all measured steps; small dots: mean of each run "
         "(25 steps). Halo traffic of the other partitions: 945–3,561 node copies "
         "per step (1–4 MB per sample).", fontsize=12.5, color=MUTED)
fig.text(0.045, 0.045,
         "1° icosahedral hierarchical graph, batch 2, one AR step, 4 × GTX 1080 Ti "
         f"on one node. Slurm jobs {', '.join(jobs)}.", fontsize=12.5,
         color=MUTED)
fig.savefig(out, dpi=150, facecolor="white")
for k in sorted(med, key=med.get):
    pred = med["lon_bands"] * S[k] / S["lon_bands"]
    print(f"{names[k]:18s} S {S[k]:.3f} median {med[k]:.1f} runs {len(run_means[k])} "
          f"proportional {pred:.1f} -> {100 * (med[k] - med['lon_bands']) / (pred - med['lon_bands'] + 1e-9):.0f}% of it")
