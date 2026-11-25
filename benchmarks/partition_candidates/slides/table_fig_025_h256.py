"""Slide-sized table: Nils's weighted METIS vs lon_bands vs Max's split (4 GPUs)."""
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OLIVE = "#4f5b1a"
OLIVE_LIGHT = "#eef0e4"
GREY = "#6b6b63"
RULE = "#c9c9bf"
RED = "#b3261e"

rows = [
    # name, subtitle, S, mem, grid, g0, g1, g2, halo
    ("Weighted METIS on G0", "split G0, other layers follow",
     1.024, 25.43, 1.006, 1.069, 1.073, 1.086, 5144),
    ("lon_bands", "same longitude cuts, every layer balanced",
     1.010, 25.04, 1.006, 1.018, 1.018, 1.017, 8890),
    ("Per-layer split", "each layer cut on its own",
     1.033, 25.44, 1.000, 1.000, 1.000, 1.000, 32496),
]
cols = [
    ("Slowdown from\nimbalance", lambda r: f"{r[2]:.3f}", 2),
    ("Memory on\nbusiest GPU", lambda r: f"{r[3]:.2f}%", 3),
    ("Grid", lambda r: f"{r[4]:.3f}", 4),
    ("G0 mesh", lambda r: f"{r[5]:.3f}", 5),
    ("G1 mesh", lambda r: f"{r[6]:.3f}", 6),
    ("G2 mesh", lambda r: f"{r[7]:.3f}", 7),
    ("Halo traffic,\nbusiest GPU", lambda r: f"{r[8]:,}\n({r[8]*256*4/1e6:.1f} MB)", 8),
]

fig = plt.figure(figsize=(16, 9), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(0.7, 8.25, "Per-layer balance vs. weighting one layer",
        fontsize=30, weight="bold", va="center")
ax.text(0.7, 7.6, "0.25° graph, 4 GPUs, hidden 256, cost-model scores. Lower is better in every column.",
        fontsize=17, color=GREY, va="center")

x_name = 0.7
x_cols = [5.7, 7.35, 8.95, 10.2, 11.45, 12.7, 14.45]
y_head, y_group = 6.45, 6.95
row_h = 1.25
y0 = 5.35

# group label over the three layer columns
ax.text((x_cols[2] + x_cols[5]) / 2, y_group, "Node imbalance (max / mean)",
        fontsize=15, ha="center", va="center", color=GREY)
ax.plot([x_cols[2] - 0.55, x_cols[5] + 0.55], [y_group - 0.22] * 2,
        color=RULE, lw=1.2)

for (label, _, _), x in zip(cols, x_cols):
    ax.text(x, y_head, label, fontsize=15, ha="center", va="center",
            weight="bold", linespacing=1.15)
ax.text(x_name, y_head, "Partition", fontsize=15, va="center", weight="bold")
ax.plot([0.6, 15.4], [y_head - 0.5] * 2, color="black", lw=1.5)

best = {i: min(r[i] for r in rows) for _, _, i in cols}
for k, r in enumerate(rows):
    y = y0 - k * row_h
    if r[0] == "lon_bands":
        ax.add_patch(Rectangle((0.6, y - row_h / 2), 14.8, row_h,
                               color=OLIVE_LIGHT, zorder=0))
    ax.text(x_name, y + 0.17, r[0], fontsize=19, va="center", weight="bold",
            color=OLIVE if r[0] == "lon_bands" else "black")
    ax.text(x_name, y - 0.25, r[1], fontsize=13.5, va="center", color=GREY)
    for (_, fmt, i), x in zip(cols, x_cols):
        is_best = r[i] == best[i]
        color = RED if (i in (5, 6, 7) and r[i] > 1.05) else "black"
        ax.text(x, y, fmt(r), fontsize=19, ha="center", va="center",
                weight="bold" if is_best else "normal", color=color)
    ax.plot([0.6, 15.4], [y - row_h / 2] * 2, color=RULE, lw=1)

ax.text(0.7, 1.0,
        "Bold: best in column.  Red: mesh layer more than 5% above the mean, "
        "the cost of weighting G0 by its grid nodes.",
        fontsize=14, color=GREY, va="center")
ax.text(0.7, 0.62,
        "Halo traffic: node copies sent per forward step; MB at fp32, batch 1. "
        "Slowdown = Σ uses × slowest GPU's work / Σ uses × mean work.",
        fontsize=12.5, color=GREY, va="center")
ax.text(0.7, 0.3,
        "0.25° graph: 675,219 grid nodes, mesh 33,777 / 8,409 / 2,088. Cost model only: at hidden 256 "
        "a training step needs ≈ 17 GiB per GPU on 4 GPUs, more than a 1080 Ti has.",
        fontsize=12.5, color=GREY, va="center")

fig.savefig(sys.argv[1], dpi=150, facecolor="white")
