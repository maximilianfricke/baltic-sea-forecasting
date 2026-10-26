"""Create publication-ready plots from NJORD benchmark JSON files."""

# Standard library
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Third-party
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COLORS = {
    "forward": "#4477AA",
    "backward": "#EE6677",
    "optimizer": "#228833",
    "data": "#CCBB44",
    "off": "#4477AA",
    "on": "#EE6677",
}


def _nested(mapping, *keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _mean(result, metric):
    return _nested(result, "summary", metric, "mean")


def _load(paths):
    results = []
    expanded = []
    for path in paths:
        expanded.extend(path.rglob("*.json") if path.is_dir() else [path])
    for path in expanded:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            continue
        result["_path"] = str(path)
        results.append(result)
    if not results:
        raise ValueError("No completed benchmark results found")
    return results


def _label(result):
    return result.get("label") or Path(result["_path"]).stem


def _workload(result):
    return _nested(result, "protocol", "workload", default="training")


def _save(fig, output_dir, name):
    if not fig.get_constrained_layout():
        fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{name}.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_time_breakdown(results, output_dir):
    results = [result for result in results if _workload(result) == "training"]
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
    fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.15), 5))
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
    results = [result for result in results if _workload(result) == "training"]
    values = [
        [step["step_ms"] for step in result.get("steps", [])]
        for result in results
    ]
    keep = [(result, data) for result, data in zip(results, values) if data]
    if not keep:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(keep) * 1.15), 5))
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
        subset.sort(key=lambda item: _nested(item, "arguments", argument))
        x = [_nested(item, "arguments", argument) for item in subset]
        label = f"Checkpointing {'on' if state else 'off'}"
        color = COLORS["on" if state else "off"]
        axes[0].plot(
            x,
            [_mean(item, "step_ms") for item in subset],
            marker="o",
            label=label,
            color=color,
        )
        memory = [
            _nested(
                item, "summary", "gpu_peak_memory_allocated_bytes"
            )
            / 1024**3
            for item in subset
            if _nested(
                item, "summary", "gpu_peak_memory_allocated_bytes"
            )
            is not None
        ]
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
    best_by_workers = {
        _nested(result, "arguments", "num_workers"): result
        for result in subset
    }
    ordered = sorted(best_by_workers.items())
    if len(ordered) < 2:
        return
    x = [item[0] for item in ordered]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        x,
        [_mean(item[1], "data_wait_ms") for item in ordered],
        marker="o",
        label="Data wait",
    )
    ax.plot(
        x,
        [_mean(item[1], "end_to_end_wall_ms") for item in ordered],
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


def plot_communication_estimate(results, output_dir):
    training = [result for result in results if _workload(result) == "training"]
    best_by_hidden = {}
    for result in training:
        hidden = _nested(result, "arguments", "hidden_dim")
        payload = _nested(
            result,
            "model",
            "trainable_parameter_bytes",
        )
        if payload is None:
            parameters = _nested(
                result, "model", "parameters_trainable"
            )
            payload = parameters * 4 if parameters is not None else None
        if hidden is not None and payload is not None:
            best_by_hidden.setdefault(hidden, payload)
    if len(best_by_hidden) < 2:
        return
    ordered = sorted(best_by_hidden.items())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [hidden for hidden, _ in ordered],
        [payload / 1024**2 for _, payload in ordered],
        marker="o",
    )
    ax.set_xlabel("Hidden dimension")
    ax.set_ylabel("Gradient payload per training step [MiB]")
    ax.set_title("Estimated DDP gradient synchronization payload")
    ax.grid(alpha=0.25)
    _save(fig, output_dir, "communication-payload-estimate")


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


def write_table(results, output_dir):
    fields = (
        "label",
        "gpu",
        "gpus",
        "hidden_dim",
        "processor_layers",
        "batch_size",
        "workers",
        "checkpointing",
        "step_mean_ms",
        "step_p95_ms",
        "data_wait_ms",
        "end_to_end_ms",
        "samples_per_second",
        "peak_memory_gib",
    )
    with (output_dir / "benchmark-table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            peak = _nested(
                result, "summary", "gpu_peak_memory_allocated_bytes"
            )
            writer.writerow(
                {
                    "label": _label(result),
                    "gpu": _nested(result, "hardware", "gpu", "name"),
                    "gpus": _nested(
                        result, "distributed", "world_size", default=1
                    ),
                    "hidden_dim": _nested(
                        result, "arguments", "hidden_dim"
                    ),
                    "processor_layers": _nested(
                        result, "arguments", "processor_layers"
                    ),
                    "batch_size": _nested(
                        result, "arguments", "batch_size"
                    ),
                    "workers": _nested(
                        result, "arguments", "num_workers"
                    ),
                    "checkpointing": _nested(
                        result, "model", "gradient_checkpointing"
                    ),
                    "step_mean_ms": _mean(result, "step_ms"),
                    "step_p95_ms": _nested(
                        result, "summary", "step_ms", "p95"
                    ),
                    "data_wait_ms": _mean(result, "data_wait_ms"),
                    "end_to_end_ms": _mean(
                        result, "end_to_end_wall_ms"
                    ),
                    "samples_per_second": _nested(
                        result, "summary", "samples_per_second"
                    ),
                    "peak_memory_gib": peak / 1024**3 if peak else None,
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = _load(args.results)

    plot_time_breakdown(results, args.output_dir)
    plot_step_distributions(results, args.output_dir)
    plot_checkpoint_tradeoff(results, args.output_dir)
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
    plot_communication_estimate(results, args.output_dir)
    plot_precision_comparison(results, args.output_dir)
    write_table(results, args.output_dir)


if __name__ == "__main__":
    main()
