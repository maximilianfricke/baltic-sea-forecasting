"""Generate a concise Markdown analysis from benchmark JSON files."""

# Standard library
import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.common import label, load_results, mean, nested, workload


def controlled_training(results, **values):
    selected = []
    for result in results:
        if workload(result) != "training":
            continue
        if all(
            nested(result, "arguments", key) == value
            for key, value in values.items()
        ):
            selected.append(result)
    return selected


def percent_change(new, old):
    return (new / old - 1) * 100


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _failure_evidence(result):
    error = result.get("error") or ""
    match = re.search(
        r"process has ([0-9.]+) GiB memory in use", error
    )
    if "OutOfMemoryError" in error:
        used = f" at {match.group(1)} GiB in use" if match else ""
        return f"CUDA OOM{used}"
    return "failed; see raw JSON"


def make_report(results, failures):
    scope_reference = next(
        (
            result
            for result in results
            if label(result) == "hidden-h64-ckpt-off"
        ),
        next(
            (
                result
                for result in results
                if nested(result, "arguments", "hidden_dim") == 64
                and workload(result) == "training"
            ),
            results[0],
        ),
    )
    lines = [
        "# Benchmark analysis",
        "",
        "This report is generated from benchmark JSON files. Completed runs "
        "provide timing and memory metrics; failed runs are retained as "
        "capacity evidence. Local CPU results are exploratory and must not "
        "be compared numerically with the official CUDA cluster baseline.",
        "",
    ]
    graph = nested(scope_reference, "model", "graph", default={})
    lines.extend(
        [
            "## Experiment scope",
            "",
            f"- Accelerator: "
            f"{nested(scope_reference, 'hardware', 'gpu', 'name')}",
            f"- Model: {nested(scope_reference, 'model', 'class')}",
            f"- Grid nodes: {graph.get('grid_node_count'):,}",
            f"- Mesh nodes: {graph.get('mesh_node_count'):,}",
            f"- Trainable parameters: "
            f"{nested(scope_reference, 'model', 'parameters_trainable'):,}",
            "",
        ]
    )

    # This is the budgeted, realistic workload adapted from Graph-EFM's
    # original unrolled stage-3/4 training scripts at the fork point.
    canonical = controlled_training(
        results,
        hidden_dim=64,
        processor_layers=2,
        batch_size=2,
        ar_steps_train=4,
        precision="32",
    )
    canonical = [
        item
        for item in canonical
        if nested(item, "model", "gradient_checkpointing")
    ]
    if canonical:
        base = min(canonical, key=lambda item: mean(item, "step_ms"))
        peak_gib = nested(
            base, "summary", "gpu_peak_memory_allocated_bytes"
        ) / 1024**3
        lines.extend(
            [
                "## Canonical realistic baseline",
                "",
                f"- Mean compute step: {fmt(mean(base, 'step_ms'))} ms",
                f"- Mean data wait: {fmt(mean(base, 'data_wait_ms'))} ms",
                f"- End-to-end throughput: "
                f"{fmt(nested(base, 'summary', 'samples_per_second'))} "
                "samples/s",
                f"- Peak allocated accelerator memory: "
                f"{fmt(peak_gib)} GiB",
                "",
            ]
        )

    capacity_failures = [
        result
        for result in failures
        if workload(result) == "training"
        and nested(result, "arguments", "hidden_dim") is not None
    ]
    if capacity_failures:
        lines.extend(
            [
                "## Capacity and fit failures",
                "",
                "| Hidden dimension | Checkpointing | Outcome | Evidence |",
                "|---:|:---:|:---:|---|",
            ]
        )
        for result in sorted(
            capacity_failures,
            key=lambda item: (
                nested(item, "arguments", "hidden_dim"),
                bool(nested(item, "model", "gradient_checkpointing")),
            ),
        ):
            checkpoint = (
                "on"
                if nested(result, "model", "gradient_checkpointing")
                else "off"
            )
            lines.append(
                f"| {nested(result, 'arguments', 'hidden_dim')} | "
                f"{checkpoint} | failed | {_failure_evidence(result)} |"
            )
        lines.append("")

    hidden_on = sorted(
        (
            result
            for result in results
            if label(result).startswith("hidden-h")
            and nested(result, "model", "gradient_checkpointing")
        ),
        key=lambda item: nested(item, "arguments", "hidden_dim"),
    )
    if hidden_on:
        base = hidden_on[0]
        largest = hidden_on[-1]
        base_e2e = mean(base, "end_to_end_wall_ms")
        backward_share = 100 * mean(base, "backward_ms") / base_e2e
        wait_share = 100 * mean(base, "data_wait_ms") / base_e2e
        base_peak = nested(
            base, "summary", "gpu_peak_memory_allocated_bytes"
        )
        base_parameters = nested(base, "model", "trainable_parameter_bytes")
        lines.extend(
            [
                "## Decision-relevant observations",
                "",
                f"- Backward accounts for {backward_share:.1f}% of the h=64 "
                "end-to-end batch time; input waiting accounts for only "
                f"{wait_share:.1f}%.",
                f"- From h=64 to h={nested(largest, 'arguments', 'hidden_dim')}, "
                f"step time grows {mean(largest, 'step_ms') / mean(base, 'step_ms'):.2f}x "
                f"and peak allocated memory grows {nested(largest, 'summary', 'gpu_peak_memory_allocated_bytes') / base_peak:.2f}x.",
                f"- At h=64, trainable parameters occupy only "
                f"{base_parameters / 1024**2:.2f} MiB versus "
                f"{base_peak / 1024**3:.2f} GiB peak GPU allocation. "
                "Activations and graph intermediates therefore dominate memory.",
                "- The phase-timed runs synchronize every batch. Use the "
                "separate aggregate/no-telemetry repetitions for headline "
                "throughput.",
                "",
            ]
        )

    pairs = defaultdict(dict)
    for result in results:
        if workload(result) != "training":
            continue
        key = (
            nested(result, "arguments", "hidden_dim"),
            nested(result, "arguments", "processor_layers"),
            nested(result, "arguments", "batch_size"),
            nested(result, "arguments", "num_workers"),
            nested(result, "arguments", "ar_steps_train"),
            nested(result, "arguments", "precision"),
        )
        state = bool(nested(result, "model", "gradient_checkpointing"))
        pairs[key][state] = result
    overheads = []
    savings = []
    for pair in pairs.values():
        if set(pair) != {False, True}:
            continue
        off, on = pair[False], pair[True]
        overheads.append(
            percent_change(mean(on, "step_ms"), mean(off, "step_ms"))
        )
        off_mem = nested(off, "summary", "gpu_peak_memory_allocated_bytes")
        on_mem = nested(on, "summary", "gpu_peak_memory_allocated_bytes")
        if off_mem and on_mem:
            savings.append(-percent_change(on_mem, off_mem))
    if overheads:
        memory_sentence = (
            "Median observed allocated-memory reduction is "
            f"{fmt(statistics.median(savings), 1)}%."
            if savings
            else "Allocated-memory reduction is unavailable for this pair."
        )
        lines.extend(
            [
                "## Gradient checkpointing",
                "",
                f"Across {len(overheads)} controlled pairs, median compute "
                f"overhead is {fmt(statistics.median(overheads), 1)}%. "
                f"{memory_sentence}",
                "",
            ]
        )

    sweep_groups = (
        ("hidden-h", "Hidden dimension and checkpointing", "hidden_dim"),
        ("workers-w", "Input workers", "num_workers"),
        ("layers-p", "Processor depth", "processor_layers"),
        ("batch-b", "Batch size", "batch_size"),
        ("ar-steps", "Autoregressive training unroll", "ar_steps_train"),
        ("precision-", "Numerical precision", "precision"),
    )
    sweep_results = {}
    lines.extend(["## Single-GPU controlled sweeps", ""])
    for prefix, title, argument in sweep_groups:
        subset = [
            result
            for result in results
            if label(result).startswith(prefix)
            and workload(result) == "training"
        ]
        if not subset:
            continue
        controls = (
            "hidden_dim",
            "processor_layers",
            "batch_size",
            "num_workers",
            "ar_steps_train",
            "precision",
        )
        references = list(subset)
        for candidate in results:
            if workload(candidate) != "training":
                continue
            for reference in references:
                same_controls = all(
                    nested(candidate, "arguments", name)
                    == nested(reference, "arguments", name)
                    for name in controls
                    if name != argument
                )
                same_checkpointing = nested(
                    candidate, "model", "gradient_checkpointing"
                ) == nested(reference, "model", "gradient_checkpointing")
                if same_controls and same_checkpointing:
                    if candidate not in subset:
                        subset.append(candidate)
                    break
        subset.sort(
            key=lambda result: (
                isinstance(nested(result, "arguments", argument), str),
                nested(result, "arguments", argument),
                bool(nested(result, "model", "gradient_checkpointing")),
            )
        )
        sweep_results[argument] = subset
        lines.extend(
            [
                f"### {title}",
                "",
                "| Setting | Checkpointing | Step [ms] | End-to-end [ms] "
                "| Samples/s | Peak GiB |",
                "|---:|:---:|---:|---:|---:|---:|",
            ]
        )
        for result in subset:
            peak = nested(
                result, "summary", "gpu_peak_memory_allocated_bytes"
            )
            checkpoint = (
                "on"
                if nested(result, "model", "gradient_checkpointing")
                else "off"
            )
            lines.append(
                f"| {nested(result, 'arguments', argument)} | "
                f"{checkpoint} | "
                f"{fmt(mean(result, 'step_ms'))} | "
                f"{fmt(mean(result, 'end_to_end_wall_ms'))} | "
                f"{fmt(nested(result, 'summary', 'samples_per_second'))} | "
                f"{fmt(peak / 1024**3 if peak is not None else None)} |"
            )
        lines.append("")

    workers = sweep_results.get("num_workers", [])
    precision = sweep_results.get("precision", [])
    hidden_off = [
        result
        for result in results
        if label(result).startswith("hidden-h")
        and not nested(result, "model", "gradient_checkpointing")
    ]
    lines.extend(["## Local observations", ""])
    if workers:
        best_worker = min(
            workers, key=lambda result: mean(result, "end_to_end_wall_ms")
        )
        lines.append(
            f"- The fastest input configuration used "
            f"{nested(best_worker, 'arguments', 'num_workers')} worker(s), "
            f"at {fmt(mean(best_worker, 'end_to_end_wall_ms'))} ms end-to-end."
        )
    if len(precision) >= 2:
        by_precision = {
            str(nested(item, "arguments", "precision")): item
            for item in precision
        }
        if "32" in by_precision and "16-mixed" in by_precision:
            fp32 = by_precision["32"]
            mixed = by_precision["16-mixed"]
            precision_change = percent_change(
                mean(mixed, "step_ms"), mean(fp32, "step_ms")
            )
            lines.append(
                f"- Mixed FP16 changed compute time by "
                f"{fmt(precision_change, 1)}% "
                "relative to FP32."
            )
    if len(hidden_off) >= 2:
        hidden_off.sort(
            key=lambda result: nested(result, "arguments", "hidden_dim")
        )
        smallest, largest = hidden_off[0], hidden_off[-1]
        lines.append(
            f"- Increasing hidden dimension from "
            f"{nested(smallest, 'arguments', 'hidden_dim')} to "
            f"{nested(largest, 'arguments', 'hidden_dim')} increased compute "
            f"time by "
            f"{mean(largest, 'step_ms') / mean(smallest, 'step_ms'):.1f}x."
        )
    lines.append("")

    inference = [
        result for result in results if workload(result) == "inference"
    ]
    if inference:
        lines.extend(
            [
                "## Inference",
                "",
                "| Configuration | Horizon | Ensemble | Step [ms] "
                "| Forecast members/s |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for result in sorted(inference, key=label):
            ensemble_size = nested(
                result,
                "arguments",
                "benchmark_inference_ensemble_size",
            )
            lines.append(
                f"| {label(result)} | "
                f"{nested(result, 'arguments', 'ar_steps_eval')} | "
                f"{ensemble_size} | "
                f"{fmt(mean(result, 'step_ms'))} | "
                f"{fmt(nested(result, 'summary', 'samples_per_second'))} |"
            )
        lines.append("")

    distributed = [
        result
        for result in results
        if nested(result, "distributed", "world_size", default=1) > 1
    ]
    lines.extend(["## Multi-GPU status", ""])
    if distributed:
        lines.append(
            f"The input contains {len(distributed)} completed multi-GPU runs."
        )
        families = defaultdict(list)
        for result in results:
            mode = nested(result, "distributed", "scaling_mode")
            if mode in ("strong", "weak"):
                families[(workload(result), mode)].append(result)
        for (workload_name, mode), subset in sorted(families.items()):
            by_size = defaultdict(list)
            for result in subset:
                by_size[nested(result, "distributed", "world_size")].append(
                    result
                )
            if 1 not in by_size:
                continue
            base_step = statistics.median(
                mean(item, "step_ms") for item in by_size[1]
            )
            base_throughput = statistics.median(
                nested(item, "summary", "samples_per_second")
                for item in by_size[1]
            )
            lines.extend(
                [
                    "",
                    f"### {workload_name.title()} {mode} scaling",
                    "",
                    "| GPUs | Step [ms] | Samples/s | Speedup | Efficiency |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
            for size in sorted(by_size):
                step = statistics.median(
                    mean(item, "step_ms") for item in by_size[size]
                )
                throughput = statistics.median(
                    nested(item, "summary", "samples_per_second")
                    for item in by_size[size]
                )
                speedup = (
                    base_step / step
                    if mode == "strong"
                    else throughput / base_throughput
                )
                lines.append(
                    f"| {size} | {fmt(step)} | {fmt(throughput)} | "
                    f"{fmt(speedup)} | {fmt(speedup / size)} |"
                )
    else:
        lines.append(
            "No multi-GPU result is present yet. Speedup, efficiency, "
            "communication time, and per-rank imbalance must be measured "
            "after the distributed implementation is available."
        )
    lines.extend(
        [
            "",
            "## Measurement notes",
            "",
            "- CUDA timings use accelerator events and resettable allocator "
            "peaks.",
            "- CPU smoke tests use wall clock and are validation only.",
            "- Distributed communication is not yet measured and no "
            "multi-GPU performance claim is made.",
            "- Current official runs use the physical-repeat I/O proxy. It "
            "has independent Zarr chunks and a long epoch but repeats the "
            "same scientific values, so it is an I/O proxy rather than new "
            "training data.",
            "- Report medians across at least three independent cluster runs.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completed = load_results(
        args.results, statuses=("completed",), require=True
    )
    failed = load_results(args.results, statuses=("failed",))
    report = make_report(completed, failed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
