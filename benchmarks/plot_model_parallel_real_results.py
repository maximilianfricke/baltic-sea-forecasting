"""Create tables and plots for the real-data 1/2/4-GPU benchmark."""

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
GREEN = "#16803C"
RED = "#B42318"
PURPLE = "#7A5AF8"
COLORS = [BLUE, ORANGE, GREEN]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-gpu", type=Path, required=True)
    parser.add_argument("--two-gpu", type=Path, required=True)
    parser.add_argument("--four-gpu", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(results, gpu_counts, comparison):
    actual_world_sizes = [result.get("world_size") for result in results]
    if actual_world_sizes != gpu_counts:
        raise ValueError(
            f"expected world sizes {gpu_counts}, got {actual_world_sizes}"
        )
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


def _gpu_labels(gpu_counts):
    return [
        f"{count} GPU" if count == 1 else f"{count} GPUs"
        for count in gpu_counts
    ]


def _plot_time_and_memory(results, gpu_counts, output_dir):
    labels = _gpu_labels(gpu_counts)
    total_ms = np.array([_mean(result, "end_to_end_ms") for result in results])
    model_ms = np.array([_mean(result, "step_ms") for result in results])
    effective_input_ms = np.maximum(total_ms - model_ms, 0)
    memory_gib = np.array(
        [
            result["summary"]["peak_memory_allocated_bytes_per_gpu"] / 2**30
            for result in results
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    positions = np.arange(len(results))
    axes[0].bar(
        positions,
        effective_input_ms,
        color=LIGHT_BLUE,
        label="Effective input + waiting",
    )
    axes[0].bar(
        positions,
        model_ms,
        bottom=effective_input_ms,
        color=BLUE,
        label="Model step",
    )
    for index, total in enumerate(total_ms):
        input_share = effective_input_ms[index] / total * 100
        axes[0].text(
            index,
            effective_input_ms[index] / 2,
            f"{input_share:.0f}% input/wait",
            ha="center",
            va="center",
            color="#17324D",
            fontweight="bold",
        )
        axes[0].text(
            index,
            total + 35,
            f"{total:.0f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Mean time per global batch [ms]")
    axes[0].set_title("End-to-end critical-path time")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(0, total_ms.max() * 1.18)

    axes[1].bar(positions, memory_gib, color=COLORS[: len(results)], width=0.65)
    for index, value in enumerate(memory_gib):
        reduction = (1 - value / memory_gib[0]) * 100
        label = f"{value:.2f} GiB"
        if index:
            label += f"\n−{reduction:.1f}%"
        axes[1].text(
            index,
            value + memory_gib.max() * 0.035,
            label,
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Peak allocated memory per GPU [GiB]")
    axes[1].set_title("Per-GPU memory footprint")
    axes[1].set_ylim(0, memory_gib.max() * 1.23)

    fig.suptitle(
        "Njord real-data model-parallel benchmark",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir, "final-time-and-memory")


def _plot_speedup(comparison, gpu_counts, output_dir):
    phase_specs = [
        ("Data read", "data_read_ms", GRAY),
        ("Compute", "compute_ms", ORANGE),
        ("End-to-end", "end_to_end_ms", GREEN),
    ]
    labels = _gpu_labels(gpu_counts)
    keys = [f"{count}_gpu" for count in gpu_counts]
    annotation_offsets = {
        "Data read": (-9, -17),
        "Compute": (0, 9),
        "End-to-end": (12, -2),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for label, phase, color in phase_specs:
        speedups = [comparison["phases"][phase]["speedup"][key] for key in keys]
        axes[0].plot(
            gpu_counts,
            speedups,
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
        for count, value in zip(gpu_counts[1:], speedups[1:]):
            axes[0].annotate(
                f"{value:.2f}×",
                (count, value),
                xytext=annotation_offsets[label],
                textcoords="offset points",
                ha="center",
                color=color,
            )
    axes[0].plot(
        gpu_counts,
        gpu_counts,
        linestyle="--",
        linewidth=1.4,
        color=RED,
        label="Ideal",
    )
    axes[0].set_xticks(gpu_counts, labels)
    axes[0].set_ylabel("Speedup relative to 1 GPU")
    axes[0].set_ylim(0.8, max(gpu_counts) * 1.08)
    axes[0].set_title("Strong scaling")
    axes[0].legend(loc="upper left")

    end_to_end_efficiency = [
        comparison["headline"]["parallel_efficiency"][key] * 100 for key in keys
    ]
    compute_efficiency = [
        comparison["phases"]["compute_ms"]["speedup"][key] / count * 100
        for key, count in zip(keys, gpu_counts)
    ]
    axes[1].plot(
        gpu_counts,
        compute_efficiency,
        marker="o",
        linewidth=2.0,
        color=ORANGE,
        label="Compute",
    )
    axes[1].plot(
        gpu_counts,
        end_to_end_efficiency,
        marker="o",
        linewidth=2.0,
        color=GREEN,
        label="End-to-end",
    )
    axes[1].axhline(100, linestyle="--", linewidth=1.4, color=RED)
    for count, value in zip(gpu_counts[1:], end_to_end_efficiency[1:]):
        axes[1].annotate(
            f"{value:.0f}%",
            (count, value),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            color=GREEN,
        )
    axes[1].set_xticks(gpu_counts, labels)
    axes[1].set_ylabel("Parallel efficiency [%]")
    axes[1].set_ylim(0, 108)
    axes[1].set_title("Efficiency falls as I/O and communication dominate")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    _save(fig, output_dir, "final-speedup-and-efficiency")


def _plot_step_stability(results, gpu_counts, output_dir):
    record_count = min(len(result["records"]) for result in results)
    steps = np.arange(1, record_count + 1)
    labels = _gpu_labels(gpu_counts)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True)
    specs = [
        ("step_ms", "Model step [ms]", "Model-step variability"),
        (
            "end_to_end_ms",
            "End-to-end time [ms]",
            "End-to-end variability is dominated by input latency",
        ),
    ]
    for axis, (key, ylabel, title) in zip(axes, specs):
        cv_labels = []
        for result, label, color in zip(results, labels, COLORS):
            values = np.array(
                [record[key] for record in result["records"][:record_count]]
            )
            axis.plot(
                steps,
                values,
                linewidth=1.15,
                alpha=0.85,
                color=color,
                label=label,
            )
            axis.axhline(
                np.median(values),
                color=color,
                alpha=0.55,
                linestyle="--",
                linewidth=1.0,
            )
            coefficient = values.std() / values.mean() * 100
            cv_labels.append(f"{label}: {coefficient:.1f}%")
        axis.text(
            0.99,
            0.95,
            "CV — " + ", ".join(cv_labels),
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
    fig.suptitle(
        f"Per-step stability across {record_count} measured batches",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir, "final-per-step-stability")


def _plot_phase_variability(results, gpu_counts, output_dir):
    labels = _gpu_labels(gpu_counts)
    positions = np.arange(1, len(results) + 1)
    forward_values = [
        [record["forward_ms"] for record in result["records"]]
        for result in results
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    box = axes[0].boxplot(
        forward_values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": RED,
            "markeredgecolor": RED,
            "markersize": 4,
        },
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.55},
    )
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for index, result in enumerate(results, start=1):
        p95 = result["summary"]["forward_ms"]["p95"]
        axes[0].text(index, p95 + 10, f"p95 {p95:.0f}", ha="center", color=GRAY)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Forward time [ms]")
    axes[0].set_title("Forward-pass jitter grows at four GPUs")

    overhead_specs = [
        ("gradient_sync_ms", "Gradient sync", BLUE),
        ("metric_reduce_ms", "Metric reduction", ORANGE),
        ("optimizer_ms", "Optimizer", GREEN),
    ]
    width = 0.23
    for offset, (key, label, color) in enumerate(overhead_specs):
        values = [_mean(result, key) for result in results]
        bars = axes[1].bar(
            positions + (offset - 1) * width,
            values,
            width=width,
            color=color,
            label=label,
        )
        for bar, value in zip(bars, values):
            if value >= 0.1:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.18,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Mean overhead [ms]")
    axes[1].set_title("Synchronization overhead becomes visible at four GPUs")
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    _save(fig, output_dir, "final-phase-variability")


def _partition_halo_totals(result):
    relation_names = ["g2m", "m2g", "m2m", "mesh_up", "mesh_down"]
    totals = []
    for payload in result["rank_metadata"]:
        relations = payload["partition"]["relations"]
        totals.append(
            {
                name: sum(item["halo_nodes"] for item in relations[name])
                for name in relation_names
            }
        )
    return relation_names, totals


def _plot_partition_balance(four_gpu, output_dir):
    ranks = np.arange(4)
    partitions = [payload["partition"] for payload in four_gpu["rank_metadata"]]
    owned_grid = np.array(
        [partition["owned_interior_grid_nodes"] for partition in partitions]
    )
    owned_mesh = np.array(
        [
            sum(partition["owned_mesh_nodes_by_level"])
            for partition in partitions
        ]
    )
    owned_edges = np.array(
        [
            sum(
                item["owned_edges"]
                for relation in partition["relations"].values()
                for item in relation
            )
            for partition in partitions
        ]
    )
    owned_series = [
        ("Grid nodes", owned_grid, BLUE),
        ("Mesh nodes", owned_mesh, ORANGE),
        ("Graph edges", owned_edges, GREEN),
    ]

    relation_names, halo_totals = _partition_halo_totals(four_gpu)
    relation_labels = {
        "g2m": "Grid→mesh",
        "m2g": "Mesh→grid",
        "m2m": "Within mesh",
        "mesh_up": "Mesh up",
        "mesh_down": "Mesh down",
    }
    relation_colors = [BLUE, LIGHT_BLUE, ORANGE, GREEN, PURPLE]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    width = 0.23
    for offset, (label, values, color) in enumerate(owned_series):
        deviation = (values / values.mean() - 1) * 100
        axes[0].bar(
            ranks + (offset - 1) * width,
            deviation,
            width=width,
            color=color,
            label=label,
        )
    axes[0].axhline(0, color=GRAY, linewidth=1.0)
    axes[0].set_xticks(ranks, [f"Rank {rank}" for rank in ranks])
    axes[0].set_ylabel("Deviation from mean load [%]")
    axes[0].set_title("Owned compute load is balanced")
    axes[0].legend(loc="lower left")

    bottom = np.zeros(len(ranks))
    for relation, color in zip(relation_names, relation_colors):
        values = np.array([totals[relation] for totals in halo_totals])
        axes[1].bar(
            ranks,
            values,
            bottom=bottom,
            color=color,
            label=relation_labels[relation],
        )
        bottom += values
    for rank, total in zip(ranks, bottom):
        axes[1].text(
            rank,
            total + bottom.max() * 0.025,
            f"{int(total)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[1].set_xticks(ranks, [f"Rank {rank}" for rank in ranks])
    axes[1].set_ylabel("Remote halo nodes")
    axes[1].set_title("Halo communication is geographically imbalanced")
    axes[1].set_ylim(0, bottom.max() * 1.16)
    axes[1].legend(loc="upper left")

    fig.suptitle(
        "Four-GPU partition balance and communication footprint",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir, "final-partition-balance")


def _table_rows(results):
    timing_specs = [
        ("Data read (mean)", "ms", "data_read_ms", "mean", "lower"),
        ("Forward (mean)", "ms", "forward_ms", "mean", "lower"),
        ("Forward (median)", "ms", "forward_ms", "median", "lower"),
        ("Backward (mean)", "ms", "backward_ms", "mean", "lower"),
        (
            "Gradient synchronization (mean)",
            "ms",
            "gradient_sync_ms",
            "mean",
            "overhead",
        ),
        (
            "Metric reduction (mean)",
            "ms",
            "metric_reduce_ms",
            "mean",
            "overhead",
        ),
        ("Optimizer (mean)", "ms", "optimizer_ms", "mean", "overhead"),
        ("Model step (mean)", "ms", "step_ms", "mean", "lower"),
        ("Model step (median)", "ms", "step_ms", "median", "lower"),
        ("End-to-end (mean)", "ms", "end_to_end_ms", "mean", "lower"),
        ("End-to-end (median)", "ms", "end_to_end_ms", "median", "lower"),
    ]
    rows = []
    for label, unit, key, statistic, direction in timing_specs:
        values = [result["summary"][key][statistic] for result in results]
        rows.append((label, unit, values, direction))

    rows.append(
        (
            "Observed throughput",
            "samples/s",
            [
                result["summary"]["observed_samples_per_second"]
                for result in results
            ],
            "higher",
        )
    )
    rows.append(
        (
            "Peak memory per GPU",
            "GiB",
            [
                result["summary"]["peak_memory_allocated_bytes_per_gpu"] / 2**30
                for result in results
            ],
            "reduction",
        )
    )
    rows.append(
        (
            "Data staging",
            "ms",
            [
                result["rank_metadata"][0]["initialization_timings"][
                    "data_staging_ms"
                ]
                for result in results
            ],
            "overhead",
        )
    )
    return rows


def _comparison(values, index, direction):
    if direction == "higher":
        return values[index] / values[0]
    if direction == "lower":
        return values[0] / values[index]
    if direction == "reduction":
        return 1 - values[index] / values[0]
    if direction == "overhead":
        return values[index] - values[0]
    raise ValueError(f"unknown comparison direction: {direction}")


def _comparison_text(value, unit, direction):
    if direction == "reduction":
        return f"{value:.1%} less"
    if direction == "overhead":
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.3f} {unit}"
    return f"{value:.3f}×"


def _write_tables(results, gpu_counts, comparison, output_dir):
    rows = _table_rows(results)
    csv_path = output_dir / "final-comparison-table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "metric",
                "unit",
                "one_gpu",
                "two_gpu",
                "four_gpu",
                "comparison_1_to_2",
                "comparison_1_to_4",
                "comparison_kind",
                "four_gpu_efficiency",
            ]
        )
        for label, unit, values, direction in rows:
            comparison_two = _comparison(values, 1, direction)
            comparison_four = _comparison(values, 2, direction)
            efficiency = (
                comparison_four / gpu_counts[2]
                if direction in {"higher", "lower"}
                else ""
            )
            writer.writerow(
                [
                    label,
                    unit,
                    *values,
                    comparison_two,
                    comparison_four,
                    direction,
                    efficiency,
                ]
            )

    lines = [
        "# Final real-data model-parallel benchmark",
        "",
        "Controlled strong-scaling comparison on GTX 1080 Ti GPUs with "
        "238 measured batches, five warm-up batches, and a constant global "
        "batch size of two.",
        "",
        "| Metric | 1 GPU | 2 GPUs | 4 GPUs | 1→2 | 1→4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, unit, values, direction in rows:
        comparison_two = _comparison(values, 1, direction)
        comparison_four = _comparison(values, 2, direction)
        lines.append(
            f"| {label} | {values[0]:.3f} {unit} | {values[1]:.3f} {unit} | "
            f"{values[2]:.3f} {unit} | "
            f"{_comparison_text(comparison_two, unit, direction)} | "
            f"{_comparison_text(comparison_four, unit, direction)} |"
        )

    numerical = comparison["numerical_equivalence"]
    preflight_differences = numerical["relative_difference_from_1_gpu"]
    memory_values = rows[-2][2]
    _, halo_totals = _partition_halo_totals(results[2])
    halo_counts = [sum(totals.values()) for totals in halo_totals]
    four_gpu_sync = comparison["phases"]["gradient_sync_ms"]["mean_ms"][
        "4_gpu"
    ]
    lines.extend(
        [
            "",
            "## Key findings",
            "",
            f"- End-to-end speedup is "
            f"{comparison['phases']['end_to_end_ms']['speedup']['2_gpu']:.3f}× "
            f"on two GPUs and "
            f"{comparison['phases']['end_to_end_ms']['speedup']['4_gpu']:.3f}× "
            "on four GPUs.",
            f"- End-to-end parallel efficiency falls to "
            f"{comparison['headline']['parallel_efficiency']['2_gpu']:.1%} "
            "and "
            f"{comparison['headline']['parallel_efficiency']['4_gpu']:.1%}.",
            f"- Per-GPU peak memory falls by "
            f"{(1 - memory_values[1] / memory_values[0]):.1%} on two GPUs and "
            f"{(1 - memory_values[2] / memory_values[0]):.1%} on four GPUs.",
            f"- Data-read speedup reaches only "
            f"{comparison['phases']['data_read_ms']['speedup']['4_gpu']:.3f}× "
            "at four GPUs, so input latency dominates end-to-end scaling.",
            f"- Four-GPU gradient synchronization averages "
            f"{four_gpu_sync:.3f} ms; "
            "forward-pass jitter further limits compute scaling.",
            "- Owned grid nodes are exactly balanced, but total halo nodes "
            "range "
            f"from {min(halo_counts)} to {max(halo_counts)} across four ranks "
            f"({max(halo_counts) / min(halo_counts):.2f}× imbalance).",
            f"- Preflight-loss relative differences are "
            f"{preflight_differences['2_gpu']:.3e} for two GPUs and "
            f"{preflight_differences['4_gpu']:.3e} for four GPUs.",
            "- Numerical equivalence checks one-step training consistency, not "
            "forecast accuracy; forecast accuracy requires equal validation "
            "rollouts.",
            "- Multi-rank phase maxima are reported independently and should "
            "not be summed; end-to-end time is the measured critical-path "
            "total.",
            "",
        ]
    )
    (output_dir / "final-comparison-table.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    args = _parse_args()
    results = [
        _load(args.one_gpu),
        _load(args.two_gpu),
        _load(args.four_gpu),
    ]
    gpu_counts = [1, 2, 4]
    comparison = _load(args.comparison)
    _validate(results, gpu_counts, comparison)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()
    _plot_time_and_memory(results, gpu_counts, args.output_dir)
    _plot_speedup(comparison, gpu_counts, args.output_dir)
    _plot_step_stability(results, gpu_counts, args.output_dir)
    _plot_phase_variability(results, gpu_counts, args.output_dir)
    _plot_partition_balance(results[2], args.output_dir)
    _write_tables(results, gpu_counts, comparison, args.output_dir)
    print(f"Plots and tables written to {args.output_dir}")


if __name__ == "__main__":
    main()
