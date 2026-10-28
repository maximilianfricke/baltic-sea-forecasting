"""Create publication-ready plots from NJORD benchmark JSON files."""

# Standard library
import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

# Third-party
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.common import (  # noqa: E402
    label as _label,
    load_results,
    mean as _mean,
    nested as _nested,
    timing_mode as _timing_mode,
    workload as _workload,
)


COLORS = {
    "forward": "#4477AA",
    "backward": "#EE6677",
    "optimizer": "#228833",
    "data": "#CCBB44",
    "off": "#4477AA",
    "on": "#EE6677",
}


def _representative_by_label(results):
    """Use the median-time repeat for dense overview plots."""
    grouped = defaultdict(list)
    for result in results:
        grouped[_label(result)].append(result)
    representatives = []
    for label in sorted(grouped):
        subset = sorted(
            grouped[label],
            key=lambda item: _mean(item, "step_ms") or float("inf"),
        )
        representatives.append(subset[len(subset) // 2])
    representatives.sort(
        key=lambda item: (
            _nested(item, "arguments", "hidden_dim", default=float("inf")),
            _label(item),
        )
    )
    return representatives


def _failure_memory_gib(result):
    match = re.search(
        r"process has ([0-9.]+) GiB memory in use",
        result.get("error") or "",
    )
    return float(match.group(1)) if match else None


def plot_capacity_outcomes(results, failures, output_dir):
    """Show successful configurations and OOM boundaries together."""
    completed = [
        result
        for result in results
        if _workload(result) == "training"
        and _nested(result, "arguments", "hidden_dim") is not None
        and _nested(result, "summary", "gpu_peak_memory_allocated_bytes")
        is not None
    ]
    failed = [
        result
        for result in failures
        if _workload(result) == "training"
        and _nested(result, "arguments", "hidden_dim") is not None
    ]
    if not completed or not failed:
        return

    fig, (memory_ax, outcome_ax) = plt.subplots(1, 2, figsize=(11, 5))
    for state in (False, True):
        color = COLORS["on" if state else "off"]
        state_label = f"Checkpointing {'on' if state else 'off'}"
        successful = [
            item
            for item in completed
            if bool(_nested(item, "model", "gradient_checkpointing"))
            == state
        ]
        if successful:
            successful.sort(
                key=lambda item: _nested(item, "arguments", "hidden_dim")
            )
            x = [_nested(item, "arguments", "hidden_dim") for item in successful]
            y = [
                _nested(item, "summary", "gpu_peak_memory_allocated_bytes")
                / 1024**3
                for item in successful
            ]
            memory_ax.plot(
                x,
                y,
                marker="s" if state else "o",
                color=color,
                label=f"{state_label}: completed",
            )
        failed_state = [
            item
            for item in failed
            if bool(_nested(item, "model", "gradient_checkpointing"))
            == state
            and _failure_memory_gib(item) is not None
        ]
        if failed_state:
            memory_ax.scatter(
                [_nested(item, "arguments", "hidden_dim") for item in failed_state],
                [_failure_memory_gib(item) for item in failed_state],
                marker="X",
                s=90,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                label=f"{state_label}: OOM",
                zorder=4,
            )

    total_memory = _nested(
        completed[0], "hardware", "gpu", "total_memory_bytes"
    )
    if total_memory:
        memory_ax.axhline(
            total_memory / 1024**3,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            label="Physical GPU memory",
        )
    memory_ax.set_xlabel("Hidden dimension")
    memory_ax.set_ylabel("GPU memory [GiB]")
    memory_ax.set_title("Measured capacity boundary")
    memory_ax.grid(alpha=0.25)
    memory_ax.legend(frameon=False, fontsize=8)

    dimensions = sorted(
        {
            _nested(item, "arguments", "hidden_dim")
            for item in completed + failed
        }
    )
    for state, y in ((False, 0), (True, 1)):
        outcome_ax.scatter(
            dimensions,
            [y] * len(dimensions),
            s=55,
            facecolor="none",
            edgecolor="#BBBBBB",
            linewidth=1,
        )
    for item in completed:
        outcome_ax.scatter(
            _nested(item, "arguments", "hidden_dim"),
            int(bool(_nested(item, "model", "gradient_checkpointing"))),
            marker="o",
            s=90,
            color="#228833",
            label="Completed",
            zorder=3,
        )
    for item in failed:
        outcome_ax.scatter(
            _nested(item, "arguments", "hidden_dim"),
            int(bool(_nested(item, "model", "gradient_checkpointing"))),
            marker="X",
            s=100,
            color="#CC3311",
            label="OOM" if "OutOfMemoryError" in (item.get("error") or "") else "Failed",
            zorder=3,
        )
    handles, labels = outcome_ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    outcome_ax.legend(unique.values(), unique.keys(), frameon=False)
    outcome_ax.set_xticks(dimensions)
    outcome_ax.set_yticks((0, 1), ("off", "on"))
    outcome_ax.set_xlabel("Hidden dimension")
    outcome_ax.set_ylabel("Gradient checkpointing")
    outcome_ax.set_title("Configuration outcome")
    outcome_ax.grid(alpha=0.2)
    fig.suptitle("Single-GPU capacity on GTX 1080 Ti")
    _save(fig, output_dir, "capacity-outcomes")


def _save(fig, output_dir, name):
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(
        output_dir / f"{name}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_time_breakdown(results, output_dir):
    results = [
        result
        for result in results
        if _workload(result) == "training"
        and _timing_mode(result) == "phase"
    ]
    results = _representative_by_label(results)
    if not results:
        return
    labels = [_label(result) for result in results]
    metrics = [
        ("forward_ms", "Forward", COLORS["forward"]),
        ("backward_ms", "Backward", COLORS["backward"]),
        ("optimizer_and_hooks_ms", "Optimizer", COLORS["optimizer"]),
        ("data_wait_ms", "Data wait", COLORS["data"]),
    ]
    x = np.arange(len(results))
    bottoms = np.zeros(len(results))
    fig, ax = plt.subplots(
        figsize=(max(8, min(20, len(results) * 0.55)), 6)
    )
    for metric, name, color in metrics:
        values = np.array([_mean(result, metric) or 0 for result in results])
        ax.bar(x, values, bottom=bottoms, label=name, color=color)
        bottoms += values
    ax.set_ylabel("Mean time per batch [ms]")
    ax.set_title("End-to-end training-time breakdown")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir, "time-breakdown")


def plot_step_distributions(results, output_dir):
    results = [
        result
        for result in results
        if _workload(result) == "training"
        and _timing_mode(result) == "phase"
    ]
    results = _representative_by_label(results)
    values = [
        [step["step_ms"] for step in result.get("steps", [])]
        for result in results
    ]
    keep = [(result, data) for result, data in zip(results, values) if data]
    if not keep:
        return
    fig, ax = plt.subplots(
        figsize=(max(8, min(20, len(keep) * 0.55)), 6)
    )
    ax.boxplot(
        [data for _, data in keep],
        tick_labels=[_label(result) for result, _ in keep],
        showmeans=True,
    )
    ax.set_ylabel("Compute step [ms]")
    ax.set_title("Measured-step distribution after warm-up")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir, "step-distributions")


def plot_baseline_repeatability(results, output_dir):
    """Visualize run-to-run variation for comparable aggregate baselines."""
    grouped = defaultdict(list)
    for result in results:
        if (
            _workload(result) != "training"
            or _timing_mode(result) != "aggregate"
        ):
            continue
        if _mean(result, "step_ms") is None:
            continue
        controls = (
            _label(result),
            _nested(result, "hardware", "gpu", "name"),
            _nested(result, "arguments", "hidden_dim"),
            _nested(result, "arguments", "processor_layers"),
            _nested(result, "arguments", "batch_size"),
            _nested(result, "arguments", "ar_steps_train"),
            _nested(result, "arguments", "precision"),
            bool(_nested(result, "model", "gradient_checkpointing")),
        )
        grouped[controls].append(result)

    repeated = [subset for subset in grouped.values() if len(subset) >= 2]
    if not repeated:
        return
    subset = max(repeated, key=len)
    subset.sort(
        key=lambda item: int(
            _nested(item, "hardware", "slurm_job_id", default=0)
        )
    )
    jobs = [
        str(_nested(item, "hardware", "slurm_job_id", default=index + 1))
        for index, item in enumerate(subset)
    ]
    step_times = [_mean(item, "step_ms") for item in subset]
    throughputs = [
        _nested(item, "summary", "samples_per_second") for item in subset
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for axis, values, ylabel in (
        (axes[0], step_times, "End-to-end time per batch [ms]"),
        (axes[1], throughputs, "Throughput [samples/s]"),
    ):
        x = np.arange(len(values))
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        axis.scatter(x, values, s=70, color="#4477AA", zorder=3)
        axis.axhline(mean, color="#228833", linewidth=1.5, label="Mean")
        axis.axhspan(
            mean - stdev,
            mean + stdev,
            color="#228833",
            alpha=0.13,
            label="Mean ± 1 SD",
        )
        axis.set_xticks(x, [f"Job {job}" for job in jobs])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

    cv = statistics.stdev(throughputs) / statistics.mean(throughputs) * 100
    fig.suptitle(
        f"Clean single-GPU baseline repeatability (throughput CV {cv:.2f}%)"
    )
    _save(fig, output_dir, "baseline-repeatability")


def plot_checkpoint_tradeoff(results, output_dir):
    checkpoint_results = [
        result
        for result in results
        if _workload(result) == "training"
        and _nested(result, "model", "gradient_checkpointing") is not None
        and _nested(
            result, "summary", "gpu_peak_memory_allocated_bytes"
        )
        is not None
    ]
    groups = defaultdict(list)
    for result in checkpoint_results:
        key = tuple(
            _nested(result, "arguments", argument)
            for argument in (
                "hidden_dim",
                "processor_layers",
                "batch_size",
                "num_workers",
                "ar_steps_train",
                "precision",
            )
        )
        groups[key].append(result)
    paired = []
    for subset in groups.values():
        states = {
            bool(_nested(item, "model", "gradient_checkpointing"))
            for item in subset
        }
        if states == {False, True}:
            paired.extend(subset)
    if not paired:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for state in (False, True):
        subset = [
            result
            for result in paired
            if bool(
                _nested(result, "model", "gradient_checkpointing")
            )
            == state
        ]
        ax.scatter(
            [
                _nested(
                    result,
                    "summary",
                    "gpu_peak_memory_allocated_bytes",
                )
                / 1024**3
                for result in subset
            ],
            [_mean(result, "step_ms") for result in subset],
            s=65,
            color=COLORS["on" if state else "off"],
            marker="s" if state else "o",
            label=f"Checkpointing {'on' if state else 'off'}",
        )
        for result in subset:
            ax.annotate(
                f"h={_nested(result, 'arguments', 'hidden_dim')}",
                (
                    _nested(
                        result,
                        "summary",
                        "gpu_peak_memory_allocated_bytes",
                    )
                    / 1024**3,
                    _mean(result, "step_ms"),
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("Peak allocated accelerator memory [GiB]")
    ax.set_ylabel("Mean compute step [ms]")
    ax.set_title("Checkpointing memory-time trade-off")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, output_dir, "checkpoint-tradeoff")


def plot_memory_compute_frontier(results, output_dir):
    """Show every training run in the memory-versus-runtime plane."""
    usable = [
        result
        for result in results
        if _workload(result) == "training"
        and _mean(result, "step_ms") is not None
        and _nested(
            result, "summary", "gpu_peak_memory_allocated_bytes"
        )
        is not None
    ]
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for state in (False, True):
        subset = [
            result
            for result in usable
            if bool(
                _nested(result, "model", "gradient_checkpointing")
            )
            == state
        ]
        if not subset:
            continue
        ax.scatter(
            [
                _nested(
                    result,
                    "summary",
                    "gpu_peak_memory_allocated_bytes",
                )
                / 1024**3
                for result in subset
            ],
            [_mean(result, "step_ms") for result in subset],
            s=50,
            alpha=0.75,
            marker="s" if state else "o",
            color=COLORS["on" if state else "off"],
            label=f"Checkpointing {'on' if state else 'off'}",
        )
    ax.set_xlabel("Peak allocated accelerator memory [GiB]")
    ax.set_ylabel("Mean compute step [ms]")
    ax.set_title("Memory-compute design space")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, output_dir, "memory-compute-frontier")


def _plot_scaling_axis(
    results,
    output_dir,
    argument,
    filename,
    title,
    xlabel,
):
    usable = [
        result
        for result in results
        if _workload(result) == "training"
        and _nested(result, "arguments", argument) is not None
    ]
    distinct = {
        _nested(result, "arguments", argument) for result in usable
    }
    if len(distinct) < 2:
        return
    controlled_arguments = (
        "hidden_dim",
        "processor_layers",
        "batch_size",
        "num_workers",
        "ar_steps_train",
        "precision",
    )
    grouped = defaultdict(list)
    for result in usable:
        state = bool(_nested(result, "model", "gradient_checkpointing"))
        controls = tuple(
            _nested(result, "arguments", name)
            for name in controlled_arguments
            if name != argument
        )
        grouped[(state, controls)].append(result)
    selected = {}
    for (state, _), subset in grouped.items():
        distinct_x = {
            _nested(item, "arguments", argument) for item in subset
        }
        if len(distinct_x) < 2:
            continue
        current = selected.get(state)
        if current is None or len(distinct_x) > len(
            {
                _nested(item, "arguments", argument) for item in current
            }
        ):
            selected[state] = subset
    if not selected:
        return
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.5), layout="constrained"
    )
    for state, subset in selected.items():
        by_x = defaultdict(list)
        for item in subset:
            by_x[_nested(item, "arguments", argument)].append(item)
        x = sorted(by_x)
        label = f"Checkpointing {'on' if state else 'off'}"
        color = COLORS["on" if state else "off"]
        axes[0].plot(
            x,
            [
                statistics.median(
                    _mean(item, "step_ms") for item in by_x[value]
                )
                for value in x
            ],
            marker="o",
            label=label,
            color=color,
        )
        memory = []
        for value in x:
            samples = [
                _nested(
                    item,
                    "summary",
                    "gpu_peak_memory_allocated_bytes",
                )
                / 1024**3
                for item in by_x[value]
                if _nested(
                    item,
                    "summary",
                    "gpu_peak_memory_allocated_bytes",
                )
                is not None
            ]
            if samples:
                memory.append(statistics.median(samples))
        if len(memory) == len(x):
            axes[1].plot(
                x, memory, marker="o", label=label, color=color
            )
    axes[0].set_ylabel("Mean compute step [ms]")
    axes[1].set_ylabel("Peak allocated memory [GiB]")
    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle(title)
    _save(fig, output_dir, filename)


def plot_dataloader_scaling(results, output_dir):
    usable = [
        result
        for result in results
        if _workload(result) == "training"
        and _nested(result, "arguments", "num_workers") is not None
    ]
    workers = {
        _nested(result, "arguments", "num_workers") for result in usable
    }
    if len(workers) < 2:
        return
    grouped = defaultdict(list)
    for result in usable:
        controls = (
            _nested(result, "arguments", "hidden_dim"),
            _nested(result, "arguments", "processor_layers"),
            _nested(result, "arguments", "batch_size"),
            _nested(result, "arguments", "ar_steps_train"),
            _nested(result, "arguments", "precision"),
            bool(_nested(result, "model", "gradient_checkpointing")),
        )
        grouped[controls].append(result)
    subset = max(
        grouped.values(),
        key=lambda items: len(
            {
                _nested(item, "arguments", "num_workers") for item in items
            }
        ),
    )
    by_workers = defaultdict(list)
    for result in subset:
        by_workers[_nested(result, "arguments", "num_workers")].append(
            result
        )
    ordered = sorted(by_workers.items())
    if len(ordered) < 2:
        return
    x = [item[0] for item in ordered]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        x,
        [
            statistics.median(
                _mean(result, "data_wait_ms") for result in repeats
            )
            for _, repeats in ordered
        ],
        marker="o",
        label="Data wait",
    )
    ax.plot(
        x,
        [
            statistics.median(
                _mean(result, "end_to_end_wall_ms")
                for result in repeats
            )
            for _, repeats in ordered
        ],
        marker="s",
        label="End-to-end",
    )
    ax.set_xlabel("DataLoader workers")
    ax.set_ylabel("Mean time per batch [ms]")
    ax.set_title("Input-pipeline scaling")
    ax.set_xticks(x)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, output_dir, "dataloader-scaling")


def plot_multi_gpu_scaling(results, output_dir):
    usable = [
        result
        for result in results
        if _nested(result, "distributed", "world_size", default=1) >= 1
    ]
    world_sizes = {
        _nested(result, "distributed", "world_size", default=1)
        for result in usable
    }
    if len(world_sizes) < 2:
        return
    modes = defaultdict(list)
    for result in usable:
        workload = _workload(result)
        mode = _nested(result, "distributed", "scaling_mode")
        if mode in (None, "single"):
            label = _label(result).lower()
            mode = (
                "strong"
                if "strong" in label
                else "weak"
                if "weak" in label
                else None
            )
        if mode in ("strong", "weak"):
            modes[(workload, mode)].append(result)
    for (workload, mode), mode_results in modes.items():
        by_size = defaultdict(list)
        for result in mode_results:
            size = _nested(result, "distributed", "world_size", default=1)
            by_size[size].append(result)
        if len(by_size) < 2 or 1 not in by_size:
            continue
        sizes = sorted(by_size)
        step_times = [
            statistics.median(
                _mean(item, "step_ms") for item in by_size[size]
            )
            for size in sizes
        ]
        throughput = [
            statistics.median(
                _nested(item, "summary", "samples_per_second")
                for item in by_size[size]
            )
            for size in sizes
        ]
        base_time = step_times[sizes.index(1)]
        base_throughput = throughput[sizes.index(1)]
        if mode == "strong":
            speedups = [base_time / value for value in step_times]
            efficiencies = [
                value / size for value, size in zip(speedups, sizes)
            ]
        else:
            speedups = [value / base_throughput for value in throughput]
            efficiencies = [
                value / (base_throughput * size)
                for value, size in zip(throughput, sizes)
            ]
        fig, axes = plt.subplots(
            1, 2, figsize=(10, 4.5), layout="constrained"
        )
        axes[0].plot(sizes, speedups, marker="o", label="Measured")
        axes[0].plot(sizes, sizes, linestyle="--", label="Ideal")
        axes[0].set_ylabel("Speedup")
        axes[1].plot(
            sizes,
            efficiencies,
            marker="o",
            label="Parallel efficiency",
        )
        axes[1].set_ylabel("Parallel efficiency")
        for axis in axes:
            axis.set_xlabel("GPU count")
            axis.set_xticks(sizes)
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
        fig.suptitle(
            f"Multi-GPU {workload} {mode} scaling "
            "(median across repeats)"
        )
        _save(
            fig,
            output_dir,
            f"multi-gpu-{workload}-{mode}-scaling",
        )


def plot_inference_scaling(results, output_dir):
    inference = [
        result for result in results if _workload(result) == "inference"
    ]
    if not inference:
        return

    def draw(argument, fixed_argument, fixed_value, filename, title, xlabel):
        subset = [
            result
            for result in inference
            if _nested(result, "arguments", fixed_argument) == fixed_value
            and _nested(result, "arguments", argument) is not None
        ]
        best = {}
        for result in subset:
            x_value = _nested(result, "arguments", argument)
            current = best.get(x_value)
            if current is None or _mean(result, "step_ms") < _mean(
                current, "step_ms"
            ):
                best[x_value] = result
        if len(best) < 2:
            return
        ordered = sorted(best.items())
        x = [item[0] for item in ordered]
        selected = [item[1] for item in ordered]
        fig, axes = plt.subplots(
            1, 3, figsize=(15.5, 4.8), layout="constrained"
        )
        axes[0].plot(
            x,
            [_mean(item, "step_ms") for item in selected],
            marker="o",
        )
        axes[0].set_ylabel("Latency [ms]")
        axes[1].plot(
            x,
            [
                _nested(item, "summary", "samples_per_second")
                for item in selected
            ],
            marker="o",
        )
        axes[1].set_ylabel("Members/s")
        memory = [
            _nested(
                item, "summary", "gpu_peak_memory_allocated_bytes"
            )
            / 1024**3
            for item in selected
        ]
        axes[2].plot(x, memory, marker="o")
        axes[2].set_ylabel("Memory [GiB]")
        for axis in axes:
            axis.set_xlabel(xlabel)
            axis.set_xticks(x)
            axis.grid(alpha=0.25)
        fig.suptitle(title)
        _save(fig, output_dir, filename)

    draw(
        "benchmark_inference_ensemble_size",
        "ar_steps_eval",
        1,
        "inference-ensemble-scaling",
        "Inference ensemble scaling",
        "Ensemble size",
    )
    draw(
        "ar_steps_eval",
        "benchmark_inference_ensemble_size",
        1,
        "inference-horizon-scaling",
        "Inference forecast-horizon scaling",
        "Horizon steps",
    )


def plot_precision_comparison(results, output_dir):
    grouped = defaultdict(list)
    for result in results:
        if _workload(result) != "training":
            continue
        controls = (
            _nested(result, "arguments", "hidden_dim"),
            _nested(result, "arguments", "processor_layers"),
            _nested(result, "arguments", "batch_size"),
            _nested(result, "arguments", "num_workers"),
            _nested(result, "arguments", "ar_steps_train"),
            bool(_nested(result, "model", "gradient_checkpointing")),
        )
        grouped[controls].append(result)
    candidates = [
        subset
        for subset in grouped.values()
        if len(
            {
                _nested(item, "arguments", "precision") for item in subset
            }
        )
        > 1
    ]
    if not candidates:
        return
    subset = max(candidates, key=len)
    subset.sort(key=lambda item: str(_nested(item, "arguments", "precision")))
    labels = [str(_nested(item, "arguments", "precision")) for item in subset]
    x = np.arange(len(subset))
    fig, axes = plt.subplots(
        1, 2, figsize=(9, 4.5), layout="constrained"
    )
    axes[0].bar(x, [_mean(item, "step_ms") for item in subset])
    axes[0].set_ylabel("Mean compute step [ms]")
    memory = [
        _nested(item, "summary", "gpu_peak_memory_allocated_bytes")
        / 1024**3
        for item in subset
    ]
    axes[1].bar(x, memory)
    axes[1].set_ylabel("Peak allocated memory [GiB]")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Precision")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Numerical-precision comparison")
    _save(fig, output_dir, "precision-comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(
        args.results, statuses=("completed",), require=True
    )
    failures = load_results(args.results, statuses=("failed",))

    plot_capacity_outcomes(results, failures, args.output_dir)
    plot_time_breakdown(results, args.output_dir)
    plot_step_distributions(results, args.output_dir)
    plot_baseline_repeatability(results, args.output_dir)
    plot_checkpoint_tradeoff(results, args.output_dir)
    plot_memory_compute_frontier(results, args.output_dir)
    plot_dataloader_scaling(results, args.output_dir)
    _plot_scaling_axis(
        results,
        args.output_dir,
        "hidden_dim",
        "hidden-dimension-scaling",
        "Hidden-dimension scaling",
        "Hidden dimension",
    )
    _plot_scaling_axis(
        results,
        args.output_dir,
        "processor_layers",
        "processor-layer-scaling",
        "Processor-layer scaling",
        "Processor layers",
    )
    _plot_scaling_axis(
        results,
        args.output_dir,
        "batch_size",
        "batch-size-scaling",
        "Batch-size scaling",
        "Batch size",
    )
    _plot_scaling_axis(
        results,
        args.output_dir,
        "ar_steps_train",
        "autoregressive-step-scaling",
        "Autoregressive-unroll scaling",
        "Autoregressive training steps",
    )
    plot_multi_gpu_scaling(results, args.output_dir)
    plot_inference_scaling(results, args.output_dir)
    plot_precision_comparison(results, args.output_dir)


if __name__ == "__main__":
    main()
