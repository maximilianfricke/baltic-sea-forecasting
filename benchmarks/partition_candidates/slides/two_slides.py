"""Two slides: imbalance and halo traffic, one formula block and one small
diagram each. Usage: python two_slides.py <imbalance.png> <halo.png>"""
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

out_imb, out_halo = sys.argv[1:3]
OLIVE, OLIVE_LIGHT, OLIVE_PALE = "#4f5b1a", "#c9cfa6", "#f3f4ec"
INK, MUTED, GREY, RULE = "#1f1f1c", "#6b6b63", "#9a9a8e", "#d9d9cf"
plt.rcParams.update({"font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans"})


def w(s):
    return r"\mathrm{" + s.replace("_", r"\_") + "}"


def f(s):
    """Function names in italics."""
    return r"\mathit{" + s + "}"


GL = r"(\mathrm{gpu},\mathrm{layer})"
LY = r"(\mathrm{layer})"
SUM = r"\sum_{\mathrm{layer}}"
MAX = r"\underset{\mathrm{gpu}}{\mathrm{max}}"
MEAN = r"\underset{\mathrm{gpu}}{\mathrm{mean}}"


def page(title, sub):
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.7, 8.3, title, fontsize=30, weight="bold", color=INK,
            va="center")
    ax.text(0.7, 7.7, sub, fontsize=17, color=MUTED, va="center")
    return fig, ax


def table(ax, label, values):
    layers = ["g2m", "m2m 0", "m2m 1", "up 0→1", "down 1→0", "m2g"]
    ax.plot([0.6, 15.4], [1.45] * 2, color=RULE, lw=1)
    x0, dx = 6.4, 1.6
    ax.text(0.7, 1.05, "layer", fontsize=15, color=MUTED, va="center")
    ax.text(0.7, 0.55, label, fontsize=15, color=MUTED, va="center")
    for i, (name, v) in enumerate(zip(layers, values)):
        ax.text(x0 + i * dx, 1.05, name, fontsize=15, color=INK,
                ha="center", va="center", weight="bold")
        ax.text(x0 + i * dx, 0.55, str(v), fontsize=15, color=INK,
                ha="center", va="center")


def check(fig, ax):
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for t in ax.texts:
        (x0, y0), (x1, y1) = inv.transform(t.get_window_extent())
        assert 0.3 <= x0 and x1 <= 15.7 and 0.1 <= y0 and y1 <= 8.9, \
            t.get_text()[:40]


# ---- Slide 1: imbalance -------------------------------------------------
fig, ax = page("Imbalance", "Every layer waits for its slowest GPU.")
ax.text(0.7, 6.3, rf"${f('work')}{GL} = \#{w('edges')} + 0.75 \times "
        rf"\#{w('nodes')}$", fontsize=24, color=INK, va="center")
ax.text(0.7, 5.5,
        "#edges: edges of the layer that end at nodes on this gpu\n"
        "#nodes: nodes on this gpu that the layer updates",
        fontsize=14, color=MUTED, va="center", linespacing=1.4)
ax.text(0.7, 4.0,
        rf"${f('imbalance')} = \dfrac{{{SUM}\ {f('passes')}{LY} \times "
        rf"{MAX}\{{{f('work')}{GL}\}}}}"
        rf"{{{SUM}\ {f('passes')}{LY} \times {MEAN}\{{{f('work')}{GL}\}}}}$",
        fontsize=22, color=INK, va="center")
ax.text(0.7, 2.55, "1.00 = perfectly balanced", fontsize=18, color=MUTED,
        va="center")

# Diagram: work of four GPUs in one layer
work = [3.0, 2.6, 3.6, 2.8]
base, scale = 2.4, 1.05
xs = [11.0, 12.15, 13.3, 14.45]
top = base + max(work) * scale
mean = base + sum(work) / len(work) * scale
for i, (x, v) in enumerate(zip(xs, work)):
    h = v * scale
    slow = v == max(work)
    ax.add_patch(FancyBboxPatch((x - 0.35, base), 0.7, h,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=OLIVE if slow else GREY,
                                edgecolor="none"))
    if not slow:
        ax.add_patch(FancyBboxPatch((x - 0.35, base + h + 0.04), 0.7,
                                    top - base - h - 0.04,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor="none", edgecolor=GREY,
                                    linestyle=(0, (3, 2)), linewidth=1.2))
    ax.text(x, base - 0.3, f"GPU {i}", fontsize=14, color=MUTED,
            ha="center", va="center")
ax.plot([10.5, 14.95], [top, top], color=OLIVE, lw=1.5)
ax.plot([10.5, 14.95], [mean, mean], color=MUTED, lw=1.5, ls="--")
ax.text(10.4, top, "max", fontsize=15, color=OLIVE, ha="right",
        va="center", weight="bold")
ax.text(10.4, mean, "mean", fontsize=15, color=MUTED, ha="right",
        va="center")
ax.add_patch(FancyBboxPatch((11.35, 1.67), 0.36, 0.26,
                            boxstyle="round,pad=0,rounding_size=0.04",
                            facecolor="none", edgecolor=GREY,
                            linestyle=(0, (3, 2)), linewidth=1.2))
ax.text(11.85, 1.8, "waiting for the slowest GPU", fontsize=14,
        color=MUTED, va="center")
ax.text(12.7, 6.85, "work in one layer", fontsize=16, color=INK,
        ha="center", va="center", weight="bold")
table(ax, rf"${f('passes')}(\mathrm{{layer}})$ per step", [3, 6, 4, 3, 1, 1])
check(fig, ax)
fig.savefig(out_imb, dpi=150, facecolor="white")

# ---- Slide 2: halo traffic ---------------------------------------------
fig, ax = page("Halo traffic",
               "Node states a GPU must copy from other GPUs before a layer.")
ax.text(0.7, 6.3, rf"${f('halo')}{GL} = \#{w('remote_senders')}$",
        fontsize=24, color=INK, va="center")
ax.text(0.7, 5.5,
        "#remote_senders: nodes on other GPUs with an edge of the layer\n"
        "into a node on this gpu, each counted once",
        fontsize=14, color=MUTED, va="center", linespacing=1.4)
ax.text(0.7, 4.0,
        rf"${f('traffic')} = {MAX}\left\{{{SUM}\ {f('exchanges')}{LY} "
        rf"\times {f('halo')}{GL}\right\}}$",
        fontsize=22, color=INK, va="center")
ax.text(0.7, 2.55, "node copies per step", fontsize=18, color=MUTED,
        va="center")

# Diagram: two GPUs, edges of one layer crossing the cut
for x0, name in ((9.9, "GPU A"), (12.85, "GPU B")):
    ax.add_patch(FancyBboxPatch((x0, 2.3), 2.55, 4.0,
                                boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor=OLIVE_PALE, edgecolor="none"))
    ax.text(x0 + 1.275, 6.65, name, fontsize=16, color=INK, ha="center",
            va="center", weight="bold")
A = {"a1": (11.7, 5.4), "a2": (11.7, 3.3), "a3": (10.6, 4.4)}
B = {"b1": (13.65, 5.75), "b2": (13.65, 4.6), "b3": (13.65, 3.0),
     "b4": (14.75, 4.2)}
edges_cut = [("a1", "b1"), ("a1", "b2"), ("a2", "b3")]
edges_in = [("a3", "a1"), ("a3", "a2"), ("b2", "b4"), ("b1", "b4")]
P = {**A, **B}
for s, d in edges_in:
    ax.add_patch(FancyArrowPatch(P[s], P[d], arrowstyle="-|>",
                                 mutation_scale=14, color=GREY, lw=1.3,
                                 shrinkA=13, shrinkB=13))
for s, d in edges_cut:
    ax.add_patch(FancyArrowPatch(P[s], P[d], arrowstyle="-|>",
                                 mutation_scale=16, color=OLIVE, lw=2.2,
                                 shrinkA=13, shrinkB=13))
for k, (x, y) in P.items():
    halo = k in ("a1", "a2")
    ax.add_patch(Circle((x, y), 0.2, facecolor=OLIVE if halo else "white",
                        edgecolor=OLIVE if halo else GREY, lw=2, zorder=3))
ax.text(14.12, 1.85, rf"${f('halo')}(\mathrm{{B}},\mathrm{{layer}}) = 2$",
        fontsize=17, color=OLIVE,
        ha="center", va="center", weight="bold")
ax.text(10.6, 1.85, "a sender counts once,\neven with two edges", fontsize=13,
        color=MUTED, ha="center", va="center", linespacing=1.2)
table(ax, rf"${f('exchanges')}(\mathrm{{layer}})$ per step",
      [2, 6, 4, 3, 1, 1])
check(fig, ax)
fig.savefig(out_halo, dpi=150, facecolor="white")
