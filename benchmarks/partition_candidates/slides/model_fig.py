"""Slide figures: fitted step-time model T = a + b*S + c*H against the
measured partitions (2D partial-effect panels and a 3D plane).
Usage: python model_fig.py <results_dir> <out_2d.png> <out_3d.png> <jobs>"""
import glob
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

root, out2, out3, jobs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
HIGHLIGHT, NEUTRAL = "#4f5b1a", "#7d7d72"
INK, MUTED, GRID, PLANE = "#1f1f1c", "#6b6b63", "#e4e4dc", "#b9c08f"

S = {"lon_bands": 1.025, "geographic": 1.028, "nils_weighted_chunks": 1.033,
     "nils_weighted_metis": 1.067, "nils_weighted_rcb": 1.077,
     "nils_weighted_mtkahypar_km1": 1.090}
HALO = {"lon_bands": 2090, "geographic": 3561, "nils_weighted_chunks": 69234,
        "nils_weighted_metis": 1070, "nils_weighted_rcb": 1845,
        "nils_weighted_mtkahypar_km1": 945}
names = {"lon_bands": "Longitude bands", "geographic": "Per-layer split",
         "nils_weighted_metis": "METIS on G0",
         "nils_weighted_mtkahypar_km1": "Mt-KaHyPar on G0",
         "nils_weighted_rcb": "RCB on G0", "nils_weighted_chunks": "Node-ID chunks"}
MB_PER_COPY = 256 * 4 / 1e6

steps = {}
for job in jobs:
    for f in glob.glob(f"{root}/{job}/h256-4gpu-*-rep*.json"):
        k = f.split("h256-4gpu-")[1].rsplit("-rep", 1)[0]
        steps.setdefault(k, []).extend(
            r["compute_ms"] for r in json.load(open(f))["records"])
keys = [k for k in S if k in steps]
T = np.array([np.mean(steps[k]) for k in keys])
X = np.array([[1, S[k], HALO[k]] for k in keys], dtype=float)
(a, b, c), *_ = np.linalg.lstsq(X, T, rcond=None)
pred = X @ np.array([a, b, c])
rms = np.sqrt(np.mean((T - pred) ** 2))
r2 = 1 - ((T - pred) ** 2).sum() / ((T - T.mean()) ** 2).sum()
eq = (r"$\mathit{time}$ = " f"{a:.1f} ms + {b:.1f} ms × "
      r"$\mathit{imbalance}$" f" + {c * 1e3:.2f} µs × " r"$\mathit{traffic}$"
      f"     $R^2$ = {r2:.3f}, RMS error {rms:.1f} ms")
print(eq)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})


def style(ax):
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=13)


def header(fig, title, sub):
    fig.patch.set_facecolor("white")
    fig.text(0.045, 0.93, title, fontsize=30, weight="bold", color=INK)
    fig.text(0.045, 0.875, sub, fontsize=17, color=MUTED)


def footer(fig, first):
    fig.text(0.045, 0.085, first, fontsize=12.5, color=MUTED)
    fig.text(0.045, 0.055,
             "imbalance and traffic (halo node copies of the busiest GPU per "
             "step) from the cost model; times: means of all measured steps.",
             fontsize=12.5, color=MUTED)
    fig.text(0.045, 0.025,
             "1° icosahedral hierarchical graph, hidden 256, batch 2, "
             f"4 × GTX 1080 Ti on one node. Slurm jobs {', '.join(jobs)}.",
             fontsize=12.5, color=MUTED)


def check_fit(fig, axes):
    fig.canvas.draw()
    for ax in axes:
        box = ax.get_window_extent()
        for t in ax.texts:
            e = t.get_window_extent()
            assert (box.x0 <= e.x0 and e.x1 <= box.x1 and box.y0 <= e.y0
                    and e.y1 <= box.y1), t.get_text()


# ---- 2D: partial effects ------------------------------------------------
fig = plt.figure(figsize=(16, 9), dpi=150)
header(fig, "Imbalance plus halo traffic fits all six partitions",
       eq)
axS = fig.add_axes([0.075, 0.19, 0.4, 0.58])
axH = fig.add_axes([0.56, 0.19, 0.4, 0.58], sharey=axS)

# Left: time minus the halo term, against S
yS = T - c * X[:, 2]
xs = np.linspace(1.015, 1.1, 2)
axS.plot(xs, a + b * xs, color=HIGHLIGHT, linewidth=2, zorder=1)
textS = {"lon_bands": (1.018, 248.0, "left"),
         "geographic": (1.036, 236.6, "left"),
         "nils_weighted_chunks": (1.045, 240.2, "left"),
         "nils_weighted_metis": (1.066, 243.8, "right"),
         "nils_weighted_rcb": (1.075, 256.0, "right"),
         "nils_weighted_mtkahypar_km1": (1.093, 245.0, "right")}
for k, x, y in zip(keys, X[:, 1], yS):
    col = HIGHLIGHT if k == "lon_bands" else NEUTRAL
    axS.scatter([x], [y], s=150, color=col, edgecolors="white", linewidths=2,
                zorder=3)
    tx, ty, ha = textS[k]
    axS.annotate(names[k], xy=(x, y), xytext=(tx, ty), ha=ha, va="center",
                 fontsize=13.5, color=INK,
                 weight="bold" if k == "lon_bands" else "normal",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                 shrinkA=2, shrinkB=6))
axS.set_xlim(1.015, 1.1)
axS.set_xlabel("Load imbalance S", color=MUTED, fontsize=15)
axS.set_ylabel("Step time without the halo term (ms)", color=MUTED,
               fontsize=15)
axS.set_title(f"Imbalance: {b * 0.01:.1f} ms per 0.01 of S", fontsize=17,
              color=INK, loc="left", pad=12)

# Right: time minus the imbalance term (relative to longitude bands' S)
s0 = S["lon_bands"]
yH = T - b * (X[:, 1] - s0)
hs = np.linspace(0, 75, 2)
axH.plot(hs, a + b * s0 + c * hs / MB_PER_COPY, color=HIGHLIGHT, linewidth=2,
         zorder=1)
for k, h, y in zip(keys, X[:, 2] * MB_PER_COPY, yH):
    col = HIGHLIGHT if k == "lon_bands" else NEUTRAL
    axH.scatter([h], [y], s=150, color=col, edgecolors="white", linewidths=2,
                zorder=3)
k = "nils_weighted_chunks"
i = keys.index(k)
axH.text(X[i, 2] * MB_PER_COPY - 2.5, yH[i], names[k], ha="right",
         va="center", fontsize=13.5, color=INK)
low = [j for j, k in enumerate(keys) if k != "nils_weighted_chunks"]
axH.annotate("the other five partitions:\n1–4 MB, all within ±3 ms of the line",
             xy=(4.5, float(np.mean(yH[low]))), xytext=(12, 252),
             ha="left", va="center", fontsize=13.5, color=INK,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=4))
axH.set_xlim(-3, 80)
axH.set_xlabel("Halo traffic of the busiest GPU (MB per forward step)",
               color=MUTED, fontsize=15)
axH.tick_params(axis="y", labelleft=False)
axH.set_title(f"Halo: {c * 1e3 / MB_PER_COPY / 1e3:.2f} ms per MB "
              f"({c * 1e3:.2f} µs per node copy)", fontsize=17, color=INK,
              loc="left", pad=12)
axS.set_ylim(234, 284)
for ax in (axS, axH):
    style(ax)
footer(fig, "Each panel removes the other term's contribution: left, measured "
       "time minus c × H; right, measured time minus b × (S − S of longitude "
       "bands). Line: the fitted model.")
check_fit(fig, (axS, axH))
fig.savefig(out2, dpi=150, facecolor="white")

# ---- 3D: plane and points ----------------------------------------------
fig = plt.figure(figsize=(16, 9), dpi=150)
header(fig, "Imbalance and halo traffic explain 99% of the differences", eq)
ax = fig.add_axes([0.02, 0.1, 0.84, 0.76], projection="3d",
                  computed_zorder=False)
sg, hg = np.meshgrid(np.linspace(1.02, 1.095, 20), np.linspace(0, 75, 20))
from matplotlib.colors import LinearSegmentedColormap, Normalize
zg = a + b * sg + c * hg / MB_PER_COPY
cmap = LinearSegmentedColormap.from_list("olive", ["#eef0e4", "#b9c08f",
                                                   "#7d8a3c", "#3d4714"])
norm = Normalize(238, 282)
ax.plot_surface(sg, hg, zg, facecolors=cmap(norm(zg)), alpha=0.75,
                linewidth=0, shade=False, rstride=1, cstride=1, zorder=1)
ax.plot_wireframe(sg, hg, zg, color="white", linewidth=0.5, alpha=0.6,
                  rstride=4, cstride=4, zorder=2)
cax = fig.add_axes([0.83, 0.3, 0.012, 0.42])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label("Model step time (ms)", color=MUTED, fontsize=14)
cb.outline.set_visible(False)
cb.ax.tick_params(colors=MUTED, labelsize=12)
lab = {"lon_bands": (-0.002, 0, 3.0), "geographic": (0.006, 0, -4.5),
       "nils_weighted_chunks": (0, 0, 3.0), "nils_weighted_metis": (0.001, 0, -4.5),
       "nils_weighted_rcb": (0, 0, 3.5),
       "nils_weighted_mtkahypar_km1": (0.004, 0, -6.0)}
for k, s_, h, t, p in zip(keys, X[:, 1], X[:, 2] * MB_PER_COPY, T, pred):
    col = HIGHLIGHT if k == "lon_bands" else "#4a4a43"
    ax.plot([s_, s_], [h, h], [p, t], color=col, linewidth=2, zorder=4)
    ax.scatter([s_], [h], [t], s=110, color=col, edgecolors="white",
               linewidths=1.5, depthshade=False, zorder=5)
    dx, dy, dz = lab[k]
    ax.text(s_ + dx, h + dy, t + dz, f"{names[k]} ({t - p:+.1f})",
            fontsize=12.5, color=INK, ha="center", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
            weight="normal")
ax.set_xlabel(r"$\mathit{imbalance}$", color=MUTED, fontsize=14, labelpad=14)
ax.set_ylabel(r"$\mathit{traffic}$ (MB per step)", color=MUTED, fontsize=14,
              labelpad=14)
ax.set_zlabel("Step time (ms)", color=MUTED, fontsize=14, labelpad=10)
ax.set_xlim(1.02, 1.095)
ax.set_ylim(0, 75)
ax.set_zlim(238, 282)
ax.tick_params(colors=MUTED, labelsize=11)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor("white")
    axis.pane.set_edgecolor(GRID)
    axis._axinfo["grid"]["color"] = GRID
ax.view_init(elev=18, azim=-58)
ax.set_box_aspect((1.4, 1.2, 0.8))
footer(fig, "Plane: fitted model, coloured by step time. Numbers in brackets: "
       "measured minus model (ms). 6 strategies, 3 fitted parameters.")
fig.savefig(out3, dpi=150, facecolor="white")
