"""Slide figure: halo traffic vs measured step time (4 GPUs,
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


MB = {k: HALO[k] * 256 * 4 / 1e6 for k in HALO}  # fp32, hidden 256, batch 1
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
fig = plt.figure(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("white")
fig.text(0.045, 0.93, "Halo traffic alone doesn't predict training time",
         fontsize=30, weight="bold", color=INK)
fig.text(0.045, 0.875,
         "Halo traffic of the busiest GPU vs measured training step, 4 GPUs, "
         "hidden 256. Lower left is better.", fontsize=17, color=MUTED)
ax = fig.add_axes([0.09, 0.17, 0.86, 0.64])

a = "lon_bands"

# Least-squares fits of mean step time on halo traffic alone (linear in MB)
all6 = [k for k in HALO if k in med]
low5 = [k for k in all6 if k != "nils_weighted_chunks"]
xs = np.geomspace(0.12, 130, 200)
for keys, style, label in ((all6, "-", "all six"),
                           (low5, "--", "without node-ID chunks")):
    x = np.array([MB[k] for k in keys])
    t = np.array([med[k] for k in keys])
    slope, icpt = np.polyfit(x, t, 1)
    res = t - (icpt + slope * x)
    r2 = 1 - res @ res / ((t - t.mean()) @ (t - t.mean()))
    print(f"halo-only fit, {label}: R2 {r2:.3f}")
    xr = xs if keys is all6 else np.geomspace(0.85, 4.2, 50)
    ax.plot(xr, icpt + slope * xr, style, color=MUTED, linewidth=1.5,
            zorder=1, label=f"linear fit, {label}:  $R^2$ = {r2:.2f}")

offsets = {"lon_bands": (0.93, -2.2, "right"),
           "geographic": (1.08, -1.6, "left"),
           "nils_weighted_metis": (0.92, -1.0, "right"),
           "nils_weighted_rcb": (1.08, 0.0, "left"),
           "nils_weighted_mtkahypar_km1": (0.94, 0.6, "right"),
           "nils_weighted_chunks": (0.92, 0.0, "right")}
for k in HALO:
    if k not in med:
        continue
    c = HIGHLIGHT if k == "lon_bands" else NEUTRAL
    rm = run_means[k]
    ax.scatter([MB[k]] * len(rm), rm, s=22, color=c, alpha=0.45,
               edgecolors="none", zorder=2)
    ax.scatter([MB[k]], [med[k]], s=170, color=c, edgecolors="white",
               linewidths=2, zorder=3)
    fx, dy, ha = offsets[k]
    x = MB[k] if ha == "center" else MB[k] * fx
    ax.text(x, med[k] + dy, f"{names[k]}  {med[k]:.1f} ms",
            ha=ha, va="center", fontsize=14.5, color=INK)

ax.set_xscale("log")
ax.set_xlim(0.12, 130)
ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"])
ax.minorticks_off()
ax.set_ylim(238, 284)
ax.legend(loc="upper left", frameon=False, fontsize=15, handlelength=2.5)
ax.set_xlabel("Halo traffic of the busiest GPU "
              "(MB per step per sample, fp32, log scale)", color=MUTED, fontsize=15)
ax.set_ylabel("Mean time per training step (ms)", color=MUTED, fontsize=15)
ax.grid(color=GRID, linewidth=1)
ax.set_axisbelow(True)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"):
    ax.spines[s_].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=13)

fig.text(0.045, 0.075,
         "Large dot: mean of all measured steps; small dots: mean of each run "
         "(25 steps). Halo traffic: node copies from the cost model × 256 "
         "values × 4 bytes, per sample.", fontsize=12.5, color=MUTED)
fig.text(0.045, 0.045,
         "1° icosahedral hierarchical graph, batch 2, one AR step, 4 × GTX 1080 Ti "
         f"on one node. Slurm jobs {', '.join(jobs)}.", fontsize=12.5,
         color=MUTED)
fig.canvas.draw()
box = ax.get_window_extent()
for t in ax.texts:
    e = t.get_window_extent()
    assert box.x0 <= e.x0 and e.x1 <= box.x1 and box.y0 <= e.y0 and e.y1 <= box.y1, t.get_text()
fig.savefig(out, dpi=150, facecolor="white")
for k in sorted(med, key=lambda k: MB[k]):
    print(f"{names[k]:18s} {MB[k]:6.2f} MB  median {med[k]:.1f}")
