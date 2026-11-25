"""Slide: how the cost model computes imbalance and halo traffic.
Usage: python formula_fig.py <out.png>"""
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = sys.argv[1]
OLIVE, OLIVE_LIGHT = "#4f5b1a", "#f3f4ec"
INK, MUTED, RULE = "#1f1f1c", "#6b6b63", "#d9d9cf"

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans"})
fig = plt.figure(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

ax.text(0.7, 8.3, "Two numbers from the cost model", fontsize=30,
        weight="bold", color=INK, va="center")
ax.text(0.7, 7.7, "Computed from the graph and the partition alone, "
        "before any GPU run.", fontsize=17, color=MUTED, va="center")

Y0, Y1 = 2.4, 7.3
boxes = []
for x0 in (0.6, 8.2):
    boxes.append((x0, x0 + 7.2))
    ax.add_patch(FancyBboxPatch((x0, Y0), 7.2, Y1 - Y0,
                                boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor=OLIVE_LIGHT, edgecolor="none"))


def word(w):
    return r"\mathrm{" + w.replace("_", r"\_") + "}"


def count(w):
    return r"\#" + word(w)


GL = r"(\mathrm{gpu},\mathrm{layer})"
SUM = r"\sum_{\mathrm{layer}}"
LY = r"(\mathrm{layer})"
FS = float(sys.argv[2]) if len(sys.argv) > 2 else 19
MAX = r"\underset{\mathrm{gpu}}{\mathrm{max}}"
MEAN = r"\underset{\mathrm{gpu}}{\mathrm{mean}}"

# ---- Imbalance ----------------------------------------------------------
L = 0.95
ax.text(L, 6.85, "Imbalance", fontsize=24, weight="bold", color=OLIVE,
        va="center")
ax.text(L, 6.2,
        rf"${word('work')}{GL} = {count('edges')} + 0.75 \times "
        rf"{count('nodes')}$",
        fontsize=20, color=INK, va="center")
ax.text(L, 5.55,
        "edges this gpu receives in this layer, and the\n"
        "nodes it updates at their receiving end",
        fontsize=14, color=MUTED, va="center", linespacing=1.3)
ax.text(L, 4.75, rf"${word('imbalance')} =$", fontsize=20, color=INK,
        va="center")
ax.text(L + 0.25, 3.75,
        rf"$\dfrac{{{SUM}\ {word('passes')}{LY} \times "
        rf"{MAX}\{{{word('work')}{GL}\}}}}"
        rf"{{{SUM}\ {word('passes')}{LY} \times "
        rf"{MEAN}\{{{word('work')}{GL}\}}}}$",
        fontsize=FS, color=INK, va="center")
ax.text(L, 2.72, "1.00 = perfectly balanced", fontsize=16, color=MUTED,
        va="center")

# ---- Halo traffic -------------------------------------------------------
R = 8.55
ax.text(R, 6.85, "Halo traffic", fontsize=24, weight="bold", color=OLIVE,
        va="center")
ax.text(R, 6.2,
        rf"${word('halo')}{GL} = {count('remote_senders')}$",
        fontsize=20, color=INK, va="center")
ax.text(R, 5.55,
        "nodes on other GPUs with an edge of this layer\n"
        "into this gpu, each counted once",
        fontsize=14, color=MUTED, va="center", linespacing=1.3)
ax.text(R, 4.75, rf"${word('traffic')} =$", fontsize=20, color=INK,
        va="center")
ax.text(R + 0.25, 3.75,
        rf"${MAX}\left\{{{SUM}\ {word('exchanges')}{LY} "
        rf"\times {word('halo')}{GL}\right\}}$",
        fontsize=FS, color=INK, va="center")
ax.text(R, 2.72, "node copies per step", fontsize=16, color=MUTED,
        va="center")

# ---- passes and exchanges table ----------------------------------------
layers = ["g2m", "m2m 0", "m2m 1", "up 0→1", "down 1→0", "m2g"]
passes = [3, 6, 4, 3, 1, 1]
exchanges = [2, 6, 4, 3, 1, 1]
ax.plot([0.6, 15.4], [2.15] * 2, color=RULE, lw=1)
rows = [(1.85, "Layer", layers, True),
        (1.45, "passes(layer): passes per step", passes, False),
        (1.05, "exchanges(layer): exchanges per step", exchanges, False)]
x0, dx = 6.4, 1.6
for y_r, label, vals, bold in rows:
    ax.text(0.7, y_r, label, fontsize=15, color=INK if bold else MUTED,
            va="center", weight="bold" if bold else "normal")
    for i, v in enumerate(vals):
        ax.text(x0 + i * dx, y_r, str(v), fontsize=15, color=INK,
                ha="center", va="center", weight="bold" if bold else "normal")

ax.text(0.7, 0.4,
        "gpu: one of the GPUs.  layer: one message-passing layer of the model, "
        "i.e. one set of edges of the same kind (grid → mesh, within a mesh "
        "level, between levels, mesh → grid).",
        fontsize=12.5, color=MUTED, va="center")

# Every text inside its box and inside the page
fig.canvas.draw()
to_data = ax.transData.inverted()
for t in ax.texts:
    (ex0, ey0), (ex1, ey1) = to_data.transform(t.get_window_extent())
    assert 0 <= ex0 and ex1 <= 16, t.get_text()[:40]
    if Y0 < (ey0 + ey1) / 2 < Y1:
        assert any(b0 <= ex0 and ex1 <= b1 - 0.15 for b0, b1 in boxes), \
            t.get_text()[:40]
        assert Y0 + 0.1 <= ey0 and ey1 <= Y1 - 0.1, t.get_text()[:40]
fig.savefig(out, dpi=150, facecolor="white")
