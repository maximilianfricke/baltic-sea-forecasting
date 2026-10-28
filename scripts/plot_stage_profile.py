"""Plot CUDA-event timings: python scripts/plot_stage_profile.py results.csv."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def samples(rows, key, labels, steps):
    totals = defaultdict(float)
    for row in rows:
        totals[(row[key], int(row["step"]))] += float(row["cuda_ms"])
    return [[totals[(label, step)] for step in steps] for label in labels]


def bars(ax, labels, values, colors):
    means = [mean(v) for v in values]
    errors = [stdev(v) if len(v) > 1 else 0 for v in values]
    ax.barh(labels, means, xerr=errors, capsize=3, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("CUDA-event elapsed time per training step (ms)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(m + e for m, e in zip(means, errors)) * 1.25 or 1)
    for i, value in enumerate(means):
        ax.text(value + errors[i], i, f"  {value:.2f}", va="center", fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Figure destination (default: figures/ next to the results CSV).",
    )
    args = parser.parse_args()
    paths = {kind: args.results.with_suffix(f".{kind}.csv")
             for kind in ("phases", "stages")}
    data = {}
    for kind, path in paths.items():
        with path.open() as handle:
            data[kind] = list(csv.DictReader(handle))
    configs = sorted({(r["hidden_dim"], r["processor_layers"])
                      for r in data["phases"]})
    out = args.output_dir or args.results.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    for width, depth in configs:
        selected = {
            kind: [r for r in rows if (r["hidden_dim"], r["processor_layers"])
                   == (width, depth)] for kind, rows in data.items()
        }
        steps = sorted({int(r["step"]) for r in selected["phases"]})
        title = f"Njord: width {width}, depth {depth}, n={len(steps)} steps"
        fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
        labels = ["forward", "backward", "optimizer", "step"]
        bars(ax, ["Forward + loss", "Backward (includes recomputation)",
                  "Optimizer update", "Whole training step"],
             samples(selected["phases"], "phase", labels, steps),
             ["#2166ac", "#b2182b", "#4d9221", "#777777"])
        ax.set_title(title + "\nTraining phases; error bars: sample SD")
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"phases-w{width}-d{depth}.{suffix}", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(
            1, 2, figsize=(11, 3.9), layout="constrained", sharex=True
        )
        stages = ["embeddings", "g2m", "mesh", "m2g"]
        upper = 0
        for ax, phase, heading in zip(
            axes, ("forward", "recompute"),
            ("Original forward", "Checkpoint recomputation within backward"),
        ):
            rows = [r for r in selected["stages"] if r["phase"] == phase]
            bars(ax, ["Feature embeddings", "Grid → mesh", "Mesh processing",
                      "Mesh → grid"], samples(rows, "stage", stages, steps),
                 ["#8073ac", "#2166ac", "#4d9221", "#d6604d"])
            upper = max(upper, ax.get_xlim()[1])
            ax.set_title(heading)
        axes[0].set_xlim(0, upper)
        fig.suptitle(title + "\nStage sums; error bars: sample SD")
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"stages-w{width}-d{depth}.{suffix}", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
        for phase, color in zip(labels, (
            "#2166ac", "#b2182b", "#4d9221", "#777777"
        )):
            values = samples(selected["phases"], "phase", [phase], steps)[0]
            ax.plot(steps, values, "o-", color=color, label=phase)
        ax.set(xlabel="Measured training step", ylabel="CUDA-event time (ms)",
               title=title + "\nPer-step variability", ylim=(0, None))
        ax.grid(alpha=0.2)
        ax.legend()
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"steps-w{width}-d{depth}.{suffix}", dpi=200)
        plt.close(fig)
    print(f"Figures: {out}")


if __name__ == "__main__":
    main()
