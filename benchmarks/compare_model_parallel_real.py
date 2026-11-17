"""Compare controlled one-, two- and four-GPU Graph-EFM benchmarks."""

from __future__ import annotations

# Standard library
import argparse
import json
import math
from pathlib import Path


PHASES = (
    "data_distribution_ms",
    "data_read_ms",
    "host_to_device_ms",
    "batch_broadcast_ms",
    "forward_ms",
    "backward_ms",
    "gradient_sync_ms",
    "optimizer_ms",
    "compute_ms",
    "metric_reduce_ms",
    "step_ms",
    "end_to_end_ms",
)

WORKLOAD_KEYS = (
    "data_kind",
    "data_mode",
    "dataset_samples",
    "global_surface_grid_nodes",
    "graph",
    "hidden_dim",
    "learning_rate",
    "num_workers",
    "processor_layers",
    "batch_size_global",
    "ar_steps",
    "precision",
    "progress_interval",
    "seed",
    "warmup_steps",
    "measurement_steps",
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-gpu", type=Path, required=True)
    parser.add_argument("--two-gpu", type=Path, required=True)
    parser.add_argument("--four-gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError(f"benchmark did not complete: {path}")
    if result.get("benchmark") != "real_data_model_parallel_graph_efm":
        raise ValueError(f"not a real-data model-parallel result: {path}")
    return result


def _relative_difference(first, second):
    scale = max(abs(first), abs(second), 1e-12)
    return abs(first - second) / scale


def _fingerprints_match(first, second):
    if first["shapes"] != second["shapes"]:
        return False
    if first["dtypes"] != second["dtypes"]:
        return False
    return all(
        math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-7)
        for left, right in zip(first["sums"], second["sums"])
    )


def _gpu_names(result):
    return sorted(
        payload["hardware"]["gpu"]["name"]
        for payload in result["rank_metadata"]
    )


def compare(results, paths):
    expected_world_sizes = (1, 2, 4)
    actual_world_sizes = tuple(result["world_size"] for result in results)
    if actual_world_sizes != expected_world_sizes:
        raise ValueError(
            f"expected world sizes {expected_world_sizes}, got "
            f"{actual_world_sizes}"
        )

    workload_differences = {}
    for key in WORKLOAD_KEYS:
        values = [result["workload"].get(key) for result in results]
        serialized = {json.dumps(value, sort_keys=True) for value in values}
        if len(serialized) != 1:
            workload_differences[key] = values

    all_gpu_names = [_gpu_names(result) for result in results]
    same_gpu_model = len(
        {name for names in all_gpu_names for name in names}
    ) == 1
    fingerprints = [
        result["preflight"]["batch_fingerprint"] for result in results
    ]
    same_batch = all(
        _fingerprints_match(fingerprints[0], fingerprint)
        for fingerprint in fingerprints[1:]
    )
    losses = [result["preflight"]["loss"] for result in results]
    losses_close = all(
        math.isclose(losses[0], loss, rel_tol=1e-4, abs_tol=1e-2)
        for loss in losses[1:]
    )

    phases = {}
    labels = ("1_gpu", "2_gpu", "4_gpu")
    for phase in PHASES:
        means = [result["summary"][phase]["mean"] for result in results]
        phases[phase] = {
            "mean_ms": dict(zip(labels, means)),
            "speedup": {
                "1_gpu": 1.0,
                "2_gpu": means[0] / means[1] if means[1] else None,
                "4_gpu": means[0] / means[2] if means[2] else None,
            },
        }

    throughputs = [
        result["summary"]["observed_samples_per_second"]
        for result in results
    ]
    speedups = [throughput / throughputs[0] for throughput in throughputs]
    checks = {
        "non_instrumented_results": all(
            result["interpretation"]["comparable_across_world_sizes"]
            for result in results
        ),
        "same_workload": not workload_differences,
        "same_gpu_model": same_gpu_model,
        "same_global_preflight_batch": same_batch,
        "preflight_loss_numerically_close": losses_close,
    }

    return {
        "schema_version": 2,
        "comparison": "real_data_model_parallel_1_vs_2_vs_4_gpu",
        "inputs": dict(zip(labels, map(str, paths))),
        "valid_comparison": all(checks.values()),
        "checks": checks,
        "workload_differences": workload_differences,
        "hardware": dict(zip(labels, all_gpu_names)),
        "headline": {
            "observed_samples_per_second": dict(zip(labels, throughputs)),
            "observed_speedup": dict(zip(labels, speedups)),
            "parallel_efficiency": {
                "1_gpu": 1.0,
                "2_gpu": speedups[1] / 2,
                "4_gpu": speedups[2] / 4,
            },
            "peak_memory_bytes_per_gpu": {
                label: result["summary"][
                    "peak_memory_allocated_bytes_per_gpu"
                ]
                for label, result in zip(labels, results)
            },
            "data_staging_ms": {
                label: result["rank_metadata"][0][
                    "initialization_timings"
                ].get("data_staging_ms", 0.0)
                for label, result in zip(labels, results)
            },
        },
        "numerical_equivalence": {
            "preflight_loss": dict(zip(labels, losses)),
            "relative_difference_from_1_gpu": {
                "2_gpu": _relative_difference(losses[0], losses[1]),
                "4_gpu": _relative_difference(losses[0], losses[2]),
            },
            "accuracy_note": (
                "This checks one-step numerical training equivalence, not "
                "forecast accuracy. Accuracy requires equal validation "
                "rollouts."
            ),
        },
        "phases": phases,
    }


def _print_summary(result):
    print(f"Valid controlled comparison: {result['valid_comparison']}")
    headline = result["headline"]
    print("\nThroughput, speedup and parallel efficiency:")
    print(f"{'GPUs':>4} {'samples/s':>12} {'speedup':>10} {'efficiency':>12}")
    for label, gpu_count in (("1_gpu", 1), ("2_gpu", 2), ("4_gpu", 4)):
        print(
            f"{gpu_count:>4} "
            f"{headline['observed_samples_per_second'][label]:>12.3f} "
            f"{headline['observed_speedup'][label]:>9.3f}x "
            f"{headline['parallel_efficiency'][label]:>11.1%}"
        )

    numerical = result["numerical_equivalence"]
    print("\nPreflight loss:")
    for label, value in numerical["preflight_loss"].items():
        print(f"  {label}: {value:.8g}")

    print("\nPhase means:")
    print(f"{'phase':<24} {'1 GPU':>11} {'2 GPU':>11} {'4 GPU':>11}")
    for phase, values in result["phases"].items():
        means = values["mean_ms"]
        print(
            f"{phase:<24} {means['1_gpu']:>10.3f} "
            f"{means['2_gpu']:>10.3f} {means['4_gpu']:>10.3f}"
        )


def main():
    args = _parse_args()
    paths = (args.one_gpu, args.two_gpu, args.four_gpu)
    results = tuple(_load(path) for path in paths)
    comparison = compare(results, paths)
    _print_summary(comparison)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nComparison written to {args.output}")


if __name__ == "__main__":
    main()
