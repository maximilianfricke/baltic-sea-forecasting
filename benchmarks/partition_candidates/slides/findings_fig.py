"""Slide: controlled tests of the two model terms (job 6133). Left: skew
family (imbalance only). Right: noise family (halo only), with chunks.
Usage: python findings_fig.py <results_dir> <out.png> <fit_jobs> <test_job>"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oos_fig_data import HALO, IMB, NAMES  # noqa: E402

root, out = sys.argv[1:3]
fit_jobs, test_job = sys.argv[3].split(","), sys.argv[4]
OLIVE, NEW, INK, MUTED, GREY, GRID = ("#4f5b1a", "#8a3b12", "#1f1f1c",
                                      "#6b6b63", "#9a9a8e", "#e4e4dc")
MB = 256 * 4 / 1e6


def load(jobs):
    steps = {}
    for job in jobs:
        for f in glob.glob(f"{root}/{job}/h256-4gpu-*-rep*.json"):
            k = f.split("h256-4gpu-")[1].rsplit("-rep", 1)[0]
            d = json.load(open(f))
            if d.get("status") == "completed":
                steps.setdefault(k, []).extend(
                    r["compute_ms"] for r in d["records"])
    return {k: np.mean(v) for k, v in steps.items()}


fit_t, t = load(fit_jobs), load([test_job])
old = [k for k in IMB if not k.startswith("test_") and k in fit_t]
X = np.array([[1, IMB[k], HALO[k]] for k in old], float)
coef, *_ = np.linalg.lstsq(X, np.array([fit_t[k] for k in old]), rcond=None)
ref = "lon_bands"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
fig = plt.figure(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("white")
fig.text(0.045, 0.93, "Controlled tests: both costs are ~25% higher than fitted",
         fontsize=30, weight="bold", color=INK)
fig.text(0.045, 0.875, "Longitude bands changed in one way at a time, all "
         "measured in one job. 4 GPUs, hidden 256.", fontsize=17, color=MUTED)
axL = fig.add_axes([0.075, 0.19, 0.4, 0.58])
axR = fig.add_axes([0.56, 0.19, 0.4, 0.58], sharey=axL)

# ---- left: imbalance only --------------------------------------------------
sk = [ref, "test_skew10", "test_skew20"]
xs = np.array([IMB[k] for k in sk])
ys = np.array([t[k] for k in sk])
b, a = np.polyfit(xs, ys, 1)
xx = np.linspace(1.0, 1.22, 2)
axL.plot(xx, a + b * xx, color=NEW, lw=2, zorder=2,
         label=f"measured slope: {b:.0f} ms per unit")
axL.plot(xx, t[ref] + coef[1] * (xx - IMB[ref]), "--", color=MUTED, lw=1.5,
         zorder=1, label=f"model slope: {coef[1]:.0f} ms per unit")
for k, x, y in zip(sk, xs, ys):
    axL.scatter(x, y, s=150, color=OLIVE if k == ref else NEW,
                edgecolors="white", linewidths=2, zorder=3,
                marker="o" if k == ref else "D")
    axL.annotate(NAMES[k], (x, y), xytext=(-10, 10), textcoords="offset points",
                 ha="right", fontsize=13.5, color=INK)
axL.set_xlim(1.0, 1.22)
axL.set_xlabel(r"$\mathit{imbalance}$  (skewed band borders)", color=MUTED,
               fontsize=15)
axL.set_ylabel("Mean time per training step (ms)", color=MUTED, fontsize=15)
axL.set_title("Imbalance only", loc="left", fontsize=17, color=INK, pad=12)
axL.legend(loc="upper left", frameon=False, fontsize=14)

# ---- right: halo only ------------------------------------------------------
nz = [ref, "test_noise01", "test_noise03", "test_noise10", "test_noise30",
      "test_random"]
hx = np.array([HALO[k] for k in nz], float)
# remove the small imbalance differences with the measured imbalance slope
hy = np.array([t[k] - b * (IMB[k] - IMB[ref]) for k in nz])
c, c0 = np.polyfit(hx, hy, 1)
res = hy - (c0 + c * hx)
r2 = 1 - res @ res / ((hy - hy.mean()) @ (hy - hy.mean()))
hh = np.linspace(0, 75, 2)
axR.plot(hh, c0 + c * hh / MB, color=NEW, lw=2, zorder=2,
         label=f"measured: {c * 1e3:.2f} µs per copy ($R^2$ = {r2:.3f})")
axR.plot(hh, t[ref] + coef[2] * (hh / MB - HALO[ref]), "--", color=MUTED,
         lw=1.5, zorder=1, label=f"model: {coef[2] * 1e3:.2f} µs per copy")
lab = {ref: (8, -14, "left"), "test_noise01": (8, 4, "left"),
       "test_noise03": (8, -12, "left"), "test_noise10": (8, -2, "left"),
       "test_noise30": (8, -2, "left"), "test_random": (-10, 6, "right")}
for k, x, y in zip(nz, hx * MB, hy):
    axR.scatter(x, y, s=150, color=OLIVE if k == ref else NEW,
                edgecolors="white", linewidths=2, zorder=3,
                marker="o" if k == ref else "D")
    if k == "test_noise01":
        continue
    if k == "test_noise03":
        axR.annotate("noise 1%, 3%", (x, y), xytext=(60, -24),
                     textcoords="offset points", ha="left", va="center",
                     fontsize=13.5, color=INK,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                     shrinkB=6))
        continue
    dx, dy, ha = lab[k]
    axR.annotate(NAMES[k], (x, y), xytext=(dx, dy), textcoords="offset points",
                 ha=ha, va="center", fontsize=13.5, color=INK)
# chunks: other jobs, shifted to this job's longitude-bands level
ck = "nils_weighted_chunks"
cy = (fit_t[ck] - fit_t[ref]) + t[ref] - b * (IMB[ck] - IMB[ref])
axR.scatter(HALO[ck] * MB, cy, s=150, facecolors="white", edgecolors=NEW,
            linewidths=2, zorder=3, marker="D")
axR.annotate("Node-ID chunks\n(contiguous halo)", (HALO[ck] * MB, cy),
             xytext=(0, -38), textcoords="offset points", ha="center",
             fontsize=13.5, color=INK)
axR.set_xlim(0, 75)
axR.set_xlabel(r"$\mathit{traffic}$ (MB per step per sample; random node "
               "reassignment)", color=MUTED, fontsize=15)
axR.tick_params(axis="y", labelleft=False)
axR.set_title("Halo traffic only", loc="left", fontsize=17, color=INK, pad=12)
axR.legend(loc="upper left", frameon=False, fontsize=14)

axL.set_ylim(238, 292)
for ax in (axL, axR):
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=13)

print(f"imbalance slope {b:.1f} (model {coef[1]:.1f}); halo {c * 1e3:.3f} us "
      f"(model {coef[2] * 1e3:.3f}), R2 {r2:.3f}; chunks at {cy:.1f}, line "
      f"{c0 + c * HALO[ck]:.1f}")
fig.text(0.045, 0.085,
         "Dashed: the 6-strategy model's slope, drawn through longitude bands. "
         "Right: times corrected for small imbalance differences with the "
         "left slope.", fontsize=12.5, color=MUTED)
fig.text(0.045, 0.055,
         "Hollow: node-ID chunks from earlier jobs, shifted to this job's "
         "longitude-bands level; its contiguous halo costs less per copy.",
         fontsize=12.5, color=MUTED)
fig.text(0.045, 0.025,
         "1° icosahedral hierarchical graph, batch 2, 4 × GTX 1080 Ti. "
         f"New partitions: job {test_job}; model and chunks: jobs "
         f"{', '.join(fit_jobs)}.", fontsize=12.5, color=MUTED)
fig.savefig(out, dpi=150, facecolor="white")
