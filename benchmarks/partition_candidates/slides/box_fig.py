"""Slide figure: per-step training time (box plot) and peak memory per GPU for
the partitions of jobs 6088 and 6104 (1-degree hierarchical graph, 4 GPUs,
hidden 256). Usage: python box_fig.py <results_dir> <out.png>"""
import glob
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

root, out, jobs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
HIGHLIGHT, NEUTRAL = "#4f5b1a", "#7d7d72"
INK, MUTED, GRID = "#1f1f1c", "#6b6b63", "#e4e4dc"

names = {
    "lon_bands": "Longitude bands",
    "geographic": "Per-layer split",
    "nils_weighted_metis": "METIS on G0",
    "nils_weighted_mtkahypar_km1": "Mt-KaHyPar on G0",
    "nils_weighted_rcb": "RCB on G0",
    "nils_weighted_chunks": "Node-ID chunks",
}
steps, peaks, runs = {}, {}, {}
for job in jobs:
    for f in glob.glob(f"{root}/{job}/h256-4gpu-*-rep*.json"):
        key = f.split("h256-4gpu-")[1].rsplit("-rep", 1)[0]
        d = json.load(open(f))
        steps.setdefault(key, []).extend(r["compute_ms"] for r in d["records"])
        peaks.setdefault(key, []).append(
            d["summary"]["peak_memory_allocated_bytes_per_gpu"] / 2**30)
        runs.setdefault(key, []).append(job)
n_runs = sorted({len(v) for v in runs.values()})
runs_text = (f"{n_runs[0]}" if len(n_runs) == 1
             else f"{n_runs[0]}–{n_runs[-1]}")

order = sorted(steps, key=lambda k: np.mean(steps[k]))  # fastest mean on top
order = order[::-1]  # matplotlib draws the first row at the bottom
y = np.arange(len(order))
best_time = min(steps, key=lambda k: np.mean(steps[k]))
best_mem = min(peaks, key=lambda k: np.mean(peaks[k]))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
fig = plt.figure(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("white")
fig.text(0.045, 0.93, "Training step time and peak memory by partitioning strategy",
         fontsize=30, weight="bold", color=INK)
fig.text(0.045, 0.875,
         "Measured training step on 4 GPUs, hidden 256, ranked by mean step "
         "time. Lower is better; bold: best.", fontsize=17, color=MUTED)

ax1 = fig.add_axes([0.205, 0.17, 0.48, 0.62])
ax2 = fig.add_axes([0.73, 0.17, 0.225, 0.62], sharey=ax1)

for i, k in enumerate(order):
    c = HIGHLIGHT if k == "lon_bands" else NEUTRAL
    ax1.boxplot(
        steps[k], positions=[i], orientation="horizontal", widths=0.5, whis=1.5,
        patch_artist=True, showfliers=True,
        boxprops=dict(facecolor=c, edgecolor=c, alpha=0.28, linewidth=0),
        medianprops=dict(color=c, linewidth=3),
        whiskerprops=dict(color=c, linewidth=2),
        capprops=dict(color=c, linewidth=2),
        flierprops=dict(marker="o", markersize=4, markerfacecolor=c,
                        markeredgecolor="none", alpha=0.6),
    )
    mu = np.mean(steps[k])
    ax1.plot([mu], [i], marker="D", markersize=9, color="white",
             markeredgecolor=INK, markeredgewidth=1.5, zorder=4)
    ax1.text(np.percentile(steps[k], 75) + 1.2 if k != "nils_weighted_chunks"
             else np.percentile(steps[k], 25) - 1.2,
             i + 0.36, f"mean {mu:.1f} ms",
             ha="left" if k != "nils_weighted_chunks" else "right",
             va="center", fontsize=14, color=INK,
             weight="bold" if k == best_time else "normal")
    p = np.mean(peaks[k])
    ax2.plot([p], [i], "o", markersize=13, color=c,
             markeredgecolor="white", markeredgewidth=2, zorder=3)
    ax2.text(p, i + 0.36, f"{p:.2f}", ha="center", va="center",
             fontsize=14, color=INK,
             weight="bold" if k == best_mem else "normal")

ax1.set_yticks(y)
ax1.set_yticklabels([names[k] for k in order], fontsize=17)
for lab, k in zip(ax1.get_yticklabels(), order):
    if k == "lon_bands":
        lab.set_color(HIGHLIGHT)
        pass
ax1.set_xlabel("Time per training step (ms)", color=MUTED, fontsize=15)
ax1.set_xlim(236, 290)
ax2.set_xlabel("Peak memory per GPU (GiB)", color=MUTED, fontsize=15)
ax2.set_xlim(4.5, 4.85)
ax2.tick_params(axis="y", left=False, labelleft=False)
for ax in (ax1, ax2):
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=13)
ax1.tick_params(axis="y", length=0, labelsize=17, colors=INK)

fig.text(0.045, 0.075,
         f"Box: middle half of all measured steps (25 per run, {runs_text} runs "
         "per strategy), line: median, diamond: mean, whiskers: 1.5 × IQR. Peak memory is "
         "identical across runs.",
         fontsize=12.5, color=MUTED)
fig.text(0.045, 0.045,
         "1° icosahedral hierarchical graph, batch 2, one AR step, 4 × GTX 1080 Ti "
         f"on one node. Slurm jobs {', '.join(jobs)}.",
         fontsize=12.5, color=MUTED)
fig.savefig(out, dpi=150, facecolor="white")
for k in order[::-1]:
    print(f"{names[k]:18s} n={len(steps[k])} median {np.median(steps[k]):.1f} "
          f"IQR {np.percentile(steps[k], 25):.1f}-{np.percentile(steps[k], 75):.1f} "
          f"peak {np.mean(peaks[k]):.3f} jobs {sorted(runs[k])}")
