"""Plot the rank-local DataLoader tuning experiments."""

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
GREEN = "#16803C"
LIGHT_BLUE = "#8CBED6"
GRAY = "#667085"
RED = "#B42318"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-dir", type=Path, required=True)
    parser.add_argument("--tuning-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _find(directory, suffix):
    matches = list(directory.glob(f"*{suffix}.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one result ending in {suffix}.json in {directory}, "
            f"found {len(matches)}"
        )
    return _load(matches[0])


def _configure_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def _save(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _mean(result, key):
    return result["summary"][key]["mean"]


def _four_gpu_loader_plot(configs, output_dir):
    labels = [label for label, _ in configs]
    results = [result for _, result in configs]
    positions = np.arange(len(results))
    colors = [GREEN, ORANGE, RED]
    input_ms = np.array(
        [_mean(result, "data_distribution_ms") for result in results]
    )
    step_ms = np.array([_mean(result, "step_ms") for result in results])
    total_ms = np.array([_mean(result, "end_to_end_ms") for result in results])
    p95_ms = np.array(
        [result["summary"]["end_to_end_ms"]["p95"] for result in results]
    )
    throughput = np.array(
        [result["summary"]["observed_samples_per_second"] for result in results]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    bars = axes[0].bar(positions, throughput, color=colors)
    for bar, value in zip(bars, throughput):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + throughput.max() * 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Observed samples/s")
    axes[0].set_ylim(0, throughput.max() * 1.16)
    axes[0].set_title("Four-GPU throughput is best without workers")

    axes[1].bar(
        positions,
        input_ms,
        color=LIGHT_BLUE,
        label="Input + rank waiting",
    )
    axes[1].bar(
        positions,
        step_ms,
        bottom=input_ms,
        color=BLUE,
        label="Model step",
    )
    axes[1].scatter(
        positions,
        p95_ms,
        color=RED,
        marker="D",
        s=38,
        label="End-to-end p95",
        zorder=3,
    )
    for index, (mean_value, p95_value) in enumerate(zip(total_ms, p95_ms)):
        axes[1].text(
            index,
            mean_value + total_ms.max() * 0.025,
            f"mean {mean_value:.0f} ms\np95 {p95_value:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Critical-path time [ms]")
    axes[1].set_ylim(0, max(p95_ms) * 1.18)
    axes[1].set_title("Workers slow both input and model execution")
    axes[1].legend(loc="upper left")

    fig.suptitle(
        "Four-GPU rank-local DataLoader tuning",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir, "data-loader-four-gpu-overview")


def _rank_means(result, key):
    values = []
    for payload in result["rank_metadata"]:
        records = payload["records"]
        values.append(np.mean([record[key] for record in records]))
    return np.array(values)


def _rank_wait_plot(configs, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), sharey=True)
    ranks = np.arange(4)
    for axis, (label, result) in zip(axes, configs):
        read_ms = _rank_means(result, "data_read_ms")
        wait_ms = _rank_means(result, "data_rank_wait_ms")
        axis.bar(ranks, read_ms, color=BLUE, label="Read/decode/collate")
        axis.bar(
            ranks,
            wait_ms,
            bottom=read_ms,
            color=ORANGE,
            label="Wait for slowest rank",
        )
        for rank, (read_value, wait_value) in enumerate(zip(read_ms, wait_ms)):
            axis.text(
                rank,
                read_value + wait_value + 45,
                f"{read_value + wait_value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_xticks(ranks, [f"R{rank}" for rank in ranks])
        axis.set_title(label)
        axis.set_xlabel("Rank")
    axes[0].set_ylabel("Mean per-rank data phase [ms]")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Multiprocessing creates rank stragglers; synchronous loading is balanced",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir, "data-loader-four-gpu-rank-wait")


def _best_tuned_plot(best_results, output_dir):
    gpu_counts = np.array([1, 2, 4])
    labels = ["1 GPU\n2 workers", "2 GPUs\n2/rank", "4 GPUs\n0 workers"]
    throughput = np.array(
        [result["summary"]["observed_samples_per_second"] for result in best_results]
    )
    speedup = throughput / throughput[0]
    efficiency = speedup / gpu_counts * 100
    memory = np.array(
        [
            result["summary"]["peak_memory_allocated_bytes_per_gpu"] / 2**30
            for result in best_results
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].plot(
        gpu_counts,
        speedup,
        marker="o",
        linewidth=2,
        color=GREEN,
        label="Best observed end-to-end",
    )
    axes[0].plot(
        gpu_counts,
        gpu_counts,
        linestyle="--",
        color=RED,
        label="Ideal",
    )
    for count, value, eff in zip(gpu_counts, speedup, efficiency):
        axes[0].annotate(
            f"{value:.2f}×\n{eff:.0f}% eff.",
            (count, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )
    axes[0].set_xticks(gpu_counts, labels)
    axes[0].set_ylim(0, 4.3)
    axes[0].set_ylabel("Speedup relative to tuned 1 GPU")
    axes[0].set_title("Even per-scale tuning does not yield strong scaling")
    axes[0].legend(loc="upper left")

    bars = axes[1].bar(np.arange(3), memory, color=[BLUE, ORANGE, GREEN])
    for bar, value in zip(bars, memory):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.2f} GiB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[1].set_xticks(np.arange(3), labels)
    axes[1].set_ylim(0, memory.max() * 1.2)
    axes[1].set_ylabel("Peak allocated memory per GPU [GiB]")
    axes[1].set_title("Memory scaling remains close to ideal")

    fig.suptitle(
        "Best observed loader setting at each GPU count",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Diagnostic system tuning, not a controlled strong-scaling comparison",
        ha="center",
        color=GRAY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, output_dir, "data-loader-best-observed-scaling")


def _write_table(configs, output_dir):
    rows = []
    for label, result in configs:
        summary = result["summary"]
        rows.append(
            {
                "configuration": label,
                "workers_per_rank": result["workload"]["num_workers"],
                "prefetch_factor": result["workload"].get("prefetch_factor"),
                "data_read_mean_ms": summary["data_read_ms"]["mean"],
                "data_rank_wait_mean_ms": summary["data_rank_wait_ms"]["mean"],
                "model_step_mean_ms": summary["step_ms"]["mean"],
                "end_to_end_mean_ms": summary["end_to_end_ms"]["mean"],
                "end_to_end_p95_ms": summary["end_to_end_ms"]["p95"],
                "observed_samples_per_second": summary[
                    "observed_samples_per_second"
                ],
                "peak_memory_gib_per_gpu": summary[
                    "peak_memory_allocated_bytes_per_gpu"
                ]
                / 2**30,
            }
        )
    path = output_dir / "data-loader-four-gpu-summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    one_gpu = _find(args.controlled_dir, "1gpu-4347")
    two_gpu = _find(args.controlled_dir, "2gpu-4344")
    four_gpu_two_workers = _find(args.controlled_dir, "4gpu-4348")
    four_gpu_one_worker = _find(args.tuning_dir, "4gpu-4352")
    four_gpu_zero_workers = _find(args.tuning_dir, "4gpu-4354")

    four_gpu_configs = [
        ("0 workers", four_gpu_zero_workers),
        ("1/rank, P4", four_gpu_one_worker),
        ("2/rank, P2", four_gpu_two_workers),
    ]
    _four_gpu_loader_plot(four_gpu_configs, args.output_dir)
    _rank_wait_plot(four_gpu_configs, args.output_dir)
    _best_tuned_plot(
        [one_gpu, two_gpu, four_gpu_zero_workers],
        args.output_dir,
    )
    _write_table(four_gpu_configs, args.output_dir)
    print(f"Loader tuning plots and table written to {args.output_dir}")


if __name__ == "__main__":
    main()
