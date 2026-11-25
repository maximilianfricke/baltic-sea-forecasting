"""3D slide with out-of-sample points: plane fitted on the first six
strategies, new partitions (not used in the fit) drawn on it.
Usage: python model3d_oos.py <results_dir> <out.png> <fit_jobs> <test_job> <title>"""
import glob
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0])
from oos_fig_data import HALO, IMB, NAMES  # noqa: E402

root, out = sys.argv[1:3]
fit_jobs, test_job, title = sys.argv[3].split(","), sys.argv[4], sys.argv[5]
OLIVE, INK, MUTED, GRID = "#4f5b1a", "#1f1f1c", "#6b6b63", "#e4e4dc"
NEW = "#8a3b12"  # new partitions: a distinct, dark warm hue
MB = 256 * 4 / 1e6


def load(jobs):
    steps = {}
    for job in jobs:
        for f in glob.glob(f"{root}/{job}/h256-4gpu-*-rep*.json"):
            k = f.split("h256-4gpu-")[1].rsplit("-rep", 1)[0]
            d = json.load(open(f))
            if d.get("status") != "completed":
                continue
            steps.setdefault(k, []).extend(r["compute_ms"] for r in d["records"])
    return {k: np.mean(v) for k, v in steps.items()}


fit_t, test_t = load(fit_jobs), load([test_job])
old = [k for k in IMB if not k.startswith("test_") and k in fit_t]
new = [k for k in IMB if k.startswith("test_") and k in test_t]
X = np.array([[1, IMB[k], HALO[k]] for k in old], float)
coef, *_ = np.linalg.lstsq(X, np.array([fit_t[k] for k in old]), rcond=None)
pred = {k: coef @ [1, IMB[k], HALO[k]] for k in IMB}


def r2(keys, meas):
    m = np.array([meas[k] for k in keys])
    p = np.array([pred[k] for k in keys])
    return 1 - ((m - p) ** 2).sum() / ((m - m.mean()) ** 2).sum(), \
        np.sqrt(np.mean((m - p) ** 2))


r2_old, rms_old = r2(old, fit_t)
r2_new, rms_new = r2(new, test_t)
print(f"fit R2 {r2_old:.3f} RMS {rms_old:.2f}; new R2 {r2_new:.3f} RMS {rms_new:.2f}")
for k in new:
    print(f"  {NAMES[k]:22s} pred {pred[k]:6.1f} meas {test_t[k]:6.1f} "
          f"err {test_t[k] - pred[k]:+5.1f}")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
fig = plt.figure(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("white")
fig.text(0.045, 0.93, title, fontsize=30, weight="bold", color=INK)
fig.text(0.045, 0.875,
         r"$\mathit{time}$ = " f"{coef[0]:.1f} ms + {coef[1]:.1f} ms × "
         r"$\mathit{imbalance}$" f" + {coef[2] * 1e3:.2f} µs × "
         r"$\mathit{traffic}$" f",  fitted on 6 strategies ($R^2$ = {r2_old:.3f})",
         fontsize=17, color=MUTED)
fig.text(0.045, 0.83,
         f"{len(new)} new partitions, not used in the fit:  $R^2$ = {r2_new:.2f},"
         f"  RMS error {rms_new:.1f} ms,  every one slower than predicted",
         fontsize=17, color=NEW)
ax = fig.add_axes([0.0, 0.1, 0.84, 0.72], projection="3d",
                  computed_zorder=False)
sg, hg = np.meshgrid(np.linspace(1.0, 1.21, 22), np.linspace(0, 75, 20))
zg = coef[0] + coef[1] * sg + coef[2] * hg / MB
zmin, zmax = 236, 292
cmap = LinearSegmentedColormap.from_list("olive", ["#eef0e4", "#b9c08f",
                                                   "#7d8a3c", "#3d4714"])
norm = Normalize(zmin, zmax)
ax.plot_surface(sg, hg, zg, facecolors=cmap(norm(zg)), alpha=0.7,
                linewidth=0, shade=False, rstride=1, cstride=1, zorder=1)
ax.plot_wireframe(sg, hg, zg, color="white", linewidth=0.5, alpha=0.6,
                  rstride=4, cstride=4, zorder=2)
cax = fig.add_axes([0.84, 0.3, 0.012, 0.42])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label("Model step time (ms)", color=MUTED, fontsize=14)
cb.outline.set_visible(False)
cb.ax.tick_params(colors=MUTED, labelsize=12)

for k in old:
    s_, h, t = IMB[k], HALO[k] * MB, fit_t[k]
    ax.plot([s_, s_], [h, h], [pred[k], t], color="#4a4a43", lw=1.5, zorder=4)
    ax.scatter([s_], [h], [t], s=70, color="#4a4a43", edgecolors="white",
               linewidths=1.2, depthshade=False, zorder=5)
LAB = {"test_noise01": (-0.016, -4.5), "test_noise03": (-0.02, 0.0),
       "test_noise10": (-0.016, 4.0), "test_noise30": (0, 3.5),
       "test_skew10": (0.012, -5.5), "test_skew10_noise03": (0.0, 4.5),
       "test_skew20": (0, 3.5), "test_random": (0, 3.5)}
for k in new:
    s_, h, t = IMB[k], HALO[k] * MB, test_t[k]
    ax.plot([s_, s_], [h, h], [pred[k], t], color=NEW, lw=2, zorder=4)
    ax.scatter([s_], [h], [t], s=120, marker="D", color=NEW,
               edgecolors="white", linewidths=1.5, depthshade=False, zorder=5)
    dx, dz = LAB.get(k, (0, 3.5))
    ax.text(s_ + dx, h, t + dz, f"{NAMES[k]} ({t - pred[k]:+.1f})", fontsize=11.5,
            color=INK, ha="center", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.2))
ax.scatter([], [], [], s=70, color="#4a4a43", label="fitted (6 strategies)")
ax.scatter([], [], [], s=120, marker="D", color=NEW,
           label="new partitions, not in the fit")
ax.legend(loc="upper left", bbox_to_anchor=(0.06, 0.95), frameon=False,
          fontsize=14)
ax.set_xlabel(r"$\mathit{imbalance}$", color=MUTED, fontsize=14, labelpad=14)
ax.set_ylabel(r"$\mathit{traffic}$ (MB per step per sample)", color=MUTED,
              fontsize=14, labelpad=14)
ax.set_zlabel("Step time (ms)", color=MUTED, fontsize=14, labelpad=10)
ax.set_xlim(1.0, 1.21)
ax.set_ylim(0, 75)
ax.set_zlim(zmin, zmax)
ax.tick_params(colors=MUTED, labelsize=11)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor("white")
    axis.pane.set_edgecolor(GRID)
    axis._axinfo["grid"]["color"] = GRID
ax.view_init(elev=18, azim=-58)
ax.set_box_aspect((1.4, 1.2, 0.8))
fig.text(0.045, 0.085,
         "Plane: model fitted on the 6 strategies only, coloured by step time. "
         "Numbers in brackets: measured minus predicted (ms).",
         fontsize=12.5, color=MUTED)
fig.text(0.045, 0.055,
         "New partitions: longitude bands with random node reassignment "
         "(noise), shifted borders (skew), both, and fully random.",
         fontsize=12.5, color=MUTED)
fig.text(0.045, 0.025,
         "1° icosahedral hierarchical graph, hidden 256, batch 2, "
         f"4 × GTX 1080 Ti. Fit: jobs {', '.join(fit_jobs)}; new: job {test_job}.",
         fontsize=12.5, color=MUTED)
fig.savefig(out, dpi=150, facecolor="white")
