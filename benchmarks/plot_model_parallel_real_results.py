"""Create final tables and plots for the real-data 1-vs-2-GPU benchmark."""

from __future__ import annotations

# Standard library
import argparse
import csv
import json
from pathlib import Path

# Third-party
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#1F4E79"
ORANGE = "#D97706"
LIGHT_BLUE = "#8CBED6"
GRAY = "#667085"
LIGHT_GRAY = "#D0D5DD"
GREEN = "#16803C"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-gpu", type=Path, required=True)
    parser.add_argument("--two-gpu", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(one, two, comparison):
    if one.get("world_size") != 1 or two.get("world_size") != 2:
        raise ValueError("expected one- and two-GPU result JSONs")
    if not comparison.get("valid_comparison"):
        raise ValueError("comparison JSON is not marked valid")
    if not all(comparison["checks"].values()):
        raise ValueError("at least one controlled-comparison check failed")


def _configure_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def _save(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _mean(result, key):
    return result["summary"][key]["mean"]


def _plot_time_and_memory(one, two, output_dir):
    labels = ["1 GPU", "2 GPUs"]
    data_ms = np.array(
        [_mean(one, "data_distribution_ms"), _mean(two, "data_distribution_ms")]
    )
    model_ms = np.array([_mean(one, "step_ms"), _mean(two, "step_ms")])
    total_ms = data_ms + model_ms
    memory_gib = np.array(
        [
            one["summary"]["peak_memory_allocated_bytes_per_gpu"] / 2**30,
            two["summary"]["peak_memory_allocated_bytes_per_gpu"] / 2**30,
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    positions = np.arange(2)
    axes[0].bar(positions, data_ms, color=LIGHT_BLUE, label="Data distribution")
    axes[0].bar(
        positions,
        model_ms,
        bottom=data_ms,
        color=BLUE,
        label="Model step",
    )
    for index, total in enumerate(total_ms):
        data_share = data_ms[index] / total * 100
        axes[0].text(
            index,
            data_ms[index] / 2,
            f"Data {data_share:.1f}%",
            ha="center",
            va="center",
            color="#17324D",
            fontweight="bold",
        )
        axes[0].text(
            index,
            total + 45,
            f"{total:.0f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Mean time per global batch [ms]")
    axes[0].set_title("End-to-end time decomposition")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(0, total_ms.max() * 1.18)

    colors = [BLUE, ORANGE]
    axes[1].bar(positions, memory_gib, color=colors, width=0.65)
    reduction = (1 - memory_gib[1] / memory_gib[0]) * 100
    for index, value in enumerate(memory_gib):
        axes[1].text(
            index,
            value + 0.08,
            f"{value:.2f} GiB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[1].annotate(
        f"{reduction:.1f}% less per GPU",
        xy=(1, memory_gib[1]),
        xytext=(0.5, memory_gib[0] * 0.82),
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.5},
        ha="center",
        color=GREEN,
        fontweight="bold",
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Peak allocated memory per GPU [GiB]")
    axes[1].set_title("Per-GPU memory footprint")
    axes[1].set_ylim(0, memory_gib.max() * 1.2)

    fig.suptitle(
        "Njord real-data model-parallel benchmark",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir, "final-time-and-memory")


def _plot_speedup(comparison, output_dir):
    phase_specs = [
        ("Data*", "data_distribution_ms", GRAY),
        ("Forward", "forward_ms", BLUE),
        ("Backward", "backward_ms", BLUE),
        ("Compute", "compute_ms", ORANGE),
        ("End-to-end", "end_to_end_ms", GREEN),
    ]
    labels = [item[0] for item in phase_specs]
    speedups = [
        comparison["phases"][item[1]]["speedup"] for item in phase_specs
    ]
    colors = [item[2] for item in phase_specs]
    efficiencies = [speedup / 2 * 100 for speedup in speedups]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    positions = np.arange(len(labels))
    bars = ax.bar(positions, speedups, color=colors, width=0.68)
    ax.axhline(2.0, color="#B42318", linestyle="--", linewidth=1.4)
    ax.text(
        len(labels) - 0.5,
        2.03,
        "Ideal 2-GPU speedup",
        color="#B42318",
        ha="right",
        va="bottom",
    )
    ax.axhline(1.0, color=LIGHT_GRAY, linewidth=1.0)
    for bar, speedup, efficiency in zip(bars, speedups, efficiencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.045,
            f"{speedup:.2f}×\n({efficiency:.0f}% eff.)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Speedup: 1-GPU time / 2-GPU time")
    ax.set_ylim(0, 2.35)
    ax.set_title("Strong scaling from one to two GTX 1080 Ti GPUs")
    ax.text(
        0.01,
        -0.20,
        "* Data speedup is cache-sensitive; compute speedup is the robust "
        "model-parallel result.",
        transform=ax.transAxes,
        color=GRAY,
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, output_dir, "final-speedup-and-efficiency")


def _plot_step_stability(one, two, output_dir):
    records_one = one["records"]
    records_two = two["records"]
    steps = np.arange(1, min(len(records_one), len(records_two)) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.0), sharex=True)
    specs = [
        ("compute_ms", "Compute time [ms]", "GPU compute is highly stable"),
        (
            "end_to_end_ms",
            "End-to-end time [ms]",
            "I/O variability dominates end-to-end time",
        ),
    ]
    for axis, (key, ylabel, title) in zip(axes, specs):
        values_one = np.array(
            [record[key] for record in records_one[: len(steps)]]
        )
        values_two = np.array(
            [record[key] for record in records_two[: len(steps)]]
        )
        axis.plot(
            steps,
            values_one,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=BLUE,
            label="1 GPU",
        )
        axis.plot(
            steps,
            values_two,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=ORANGE,
            label="2 GPUs",
        )
        axis.axhline(values_one.mean(), color=BLUE, alpha=0.45, linestyle="--")
        axis.axhline(
            values_two.mean(), color=ORANGE, alpha=0.45, linestyle="--"
        )
        cv_one = values_one.std() / values_one.mean() * 100
        cv_two = values_two.std() / values_two.mean() * 100
        axis.text(
            0.99,
            0.94,
            f"Coefficient of variation: 1 GPU {cv_one:.2f}%, "
            f"2 GPUs {cv_two:.2f}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=GRAY,
            fontsize=9,
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(loc="center right")
    axes[1].set_xlabel("Measured training step")
    axes[1].set_xticks(steps)
    fig.suptitle(
        "Per-step stability across 20 measured batches",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir, "final-per-step-stability")


def _table_rows(one, two):
    rows = []
    timing_specs = [
        ("Data distribution", "data_distribution_ms"),
        ("Forward", "forward_ms"),
        ("Backward", "backward_ms"),
        ("Compute", "compute_ms"),
        ("End-to-end", "end_to_end_ms"),
    ]
    for label, key in timing_specs:
        first = _mean(one, key)
        second = _mean(two, key)
        rows.append((label, "ms", first, second, first / second, "speedup"))

    first = _mean(one, "gradient_sync_ms")
    second = _mean(two, "gradient_sync_ms")
    rows.insert(
        3,
        (
            "Gradient synchronization",
            "ms",
            first,
            second,
            second - first,
            "overhead",
        ),
    )

    first = one["summary"]["observed_samples_per_second"]
    second = two["summary"]["observed_samples_per_second"]
    rows.append(
        (
            "Observed throughput",
            "samples/s",
            first,
            second,
            second / first,
            "gain",
        )
    )
    first = one["summary"]["compute_samples_per_second"]
    second = two["summary"]["compute_samples_per_second"]
    rows.append(
        (
            "Compute throughput",
            "samples/s",
            first,
            second,
            second / first,
            "gain",
        )
    )
    first = one["summary"]["peak_memory_allocated_bytes_per_gpu"] / 2**30
    second = two["summary"]["peak_memory_allocated_bytes_per_gpu"] / 2**30
    rows.append(
        (
            "Peak memory per GPU",
            "GiB",
            first,
            second,
            1 - second / first,
            "reduction",
        )
    )
    return rows


def _write_tables(one, two, comparison, output_dir):
    rows = _table_rows(one, two)
    csv_path = output_dir / "final-comparison-table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["metric", "unit", "one_gpu", "two_gpu", "comparison", "kind"]
        )
        writer.writerows(rows)

    lines = [
        "# Final real-data model-parallel benchmark",
        "",
        "| Metric | 1 GPU | 2 GPUs | Comparison |",
        "|---|---:|---:|---:|",
    ]
    for label, unit, first, second, value, kind in rows:
        if kind == "reduction":
            comparison_text = f"{value * 100:.1f}% less"
        elif kind == "overhead":
            comparison_text = f"+{value:.3f} ms overhead"
        else:
            comparison_text = f"{value:.3f}x"
        lines.append(
            f"| {label} | {first:.3f} {unit} | {second:.3f} {unit} | "
            f"{comparison_text} |"
        )
    numerical = comparison["numerical_equivalence"]
    lines.extend(
        [
            "",
            f"- Compute parallel efficiency: "
            f"{comparison['phases']['compute_ms']['speedup'] / 2:.1%}",
            f"- End-to-end parallel efficiency: "
            f"{comparison['headline']['parallel_efficiency']:.1%}",
            f"- Mean-loss relative difference: "
            f"{numerical['measured_mean_loss_relative_difference']:.3e}",
            "- Data timing is cache-sensitive; numerical equivalence is not "
            "a forecast-accuracy evaluation.",
            "",
        ]
    )
    (output_dir / "final-comparison-table.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    args = _parse_args()
    one = _load(args.one_gpu)
    two = _load(args.two_gpu)
    comparison = _load(args.comparison)
    _validate(one, two, comparison)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()
    _plot_time_and_memory(one, two, args.output_dir)
    _plot_speedup(comparison, args.output_dir)
    _plot_step_stability(one, two, args.output_dir)
    _write_tables(one, two, comparison, args.output_dir)
    print(f"Plots and tables written to {args.output_dir}")


if __name__ == "__main__":
    main()
