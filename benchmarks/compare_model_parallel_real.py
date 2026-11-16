"""Compare the controlled one- and two-GPU real-data benchmark results."""

from __future__ import annotations

# Standard library
import argparse
import json
import math
from pathlib import Path


PHASES = (
    "data_distribution_ms",
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
    "dataset",
    "dataset_samples",
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


def compare(one, two, one_path, two_path):
    if one["world_size"] != 1 or two["world_size"] != 2:
        raise ValueError("expected a world-size-1 and a world-size-2 result")

    workload_differences = {
        key: [one["workload"].get(key), two["workload"].get(key)]
        for key in WORKLOAD_KEYS
        if one["workload"].get(key) != two["workload"].get(key)
    }
    one_gpu_names = _gpu_names(one)
    two_gpu_names = _gpu_names(two)
    same_gpu_model = len(set(one_gpu_names + two_gpu_names)) == 1
    same_batch = _fingerprints_match(
        one["preflight"]["batch_fingerprint"],
        two["preflight"]["batch_fingerprint"],
    )

    phase_comparison = {}
    for phase in PHASES:
        one_mean = one["summary"][phase]["mean"]
        two_mean = two["summary"][phase]["mean"]
        phase_comparison[phase] = {
            "one_gpu_mean_ms": one_mean,
            "two_gpu_mean_ms": two_mean,
            "speedup": one_mean / two_mean if two_mean else None,
        }

    one_throughput = one["summary"]["observed_samples_per_second"]
    two_throughput = two["summary"]["observed_samples_per_second"]
    observed_speedup = two_throughput / one_throughput
    preflight_one = one["preflight"]["loss"]
    preflight_two = two["preflight"]["loss"]
    mean_loss_one = one["summary"]["loss"]["mean"]
    mean_loss_two = two["summary"]["loss"]["mean"]

    checks = {
        "non_instrumented_results": (
            one["interpretation"]["comparable_across_world_sizes"]
            and two["interpretation"]["comparable_across_world_sizes"]
        ),
        "same_workload": not workload_differences,
        "same_gpu_model": same_gpu_model,
        "same_preflight_batch": same_batch,
        "preflight_loss_numerically_close": math.isclose(
            preflight_one,
            preflight_two,
            rel_tol=1e-4,
            abs_tol=1e-2,
        ),
    }
    return {
        "schema_version": 1,
        "comparison": "real_data_model_parallel_1_vs_2_gpu",
        "inputs": {
            "one_gpu": str(one_path),
            "two_gpu": str(two_path),
        },
        "valid_comparison": all(checks.values()),
        "checks": checks,
        "workload_differences": workload_differences,
        "hardware": {
            "one_gpu_models": one_gpu_names,
            "two_gpu_models": two_gpu_names,
        },
        "headline": {
            "one_gpu_observed_samples_per_second": one_throughput,
            "two_gpu_observed_samples_per_second": two_throughput,
            "observed_speedup": observed_speedup,
            "parallel_efficiency": observed_speedup / 2,
            "one_gpu_peak_memory_bytes_per_gpu": one["summary"][
                "peak_memory_allocated_bytes_per_gpu"
            ],
            "two_gpu_peak_memory_bytes_per_gpu": two["summary"][
                "peak_memory_allocated_bytes_per_gpu"
            ],
        },
        "numerical_equivalence": {
            "preflight_loss_one_gpu": preflight_one,
            "preflight_loss_two_gpu": preflight_two,
            "preflight_relative_difference": _relative_difference(
                preflight_one,
                preflight_two,
            ),
            "measured_mean_loss_one_gpu": mean_loss_one,
            "measured_mean_loss_two_gpu": mean_loss_two,
            "measured_mean_loss_relative_difference": _relative_difference(
                mean_loss_one,
                mean_loss_two,
            ),
            "accuracy_note": (
                "This checks numerical training equivalence, not forecast "
                "accuracy. Accuracy requires equal-length validation rollouts."
            ),
        },
        "phases": phase_comparison,
    }


def _print_summary(result):
    headline = result["headline"]
    numerical = result["numerical_equivalence"]
    print(f"Valid controlled comparison: {result['valid_comparison']}")
    print(
        "Observed throughput: "
        f"{headline['one_gpu_observed_samples_per_second']:.3f} -> "
        f"{headline['two_gpu_observed_samples_per_second']:.3f} samples/s"
    )
    print(
        f"Speedup: {headline['observed_speedup']:.3f}x; "
        f"parallel efficiency: {headline['parallel_efficiency']:.1%}"
    )
    print(
        "Preflight loss relative difference: "
        f"{numerical['preflight_relative_difference']:.3e}"
    )
    print("\nPhase means and speedups:")
    print(f"{'phase':<24} {'1 GPU ms':>12} {'2 GPU ms':>12} {'speedup':>10}")
    for phase, values in result["phases"].items():
        speedup = values["speedup"]
        speedup_text = "n/a" if speedup is None else f"{speedup:.3f}x"
        print(
            f"{phase:<24} {values['one_gpu_mean_ms']:>12.3f} "
            f"{values['two_gpu_mean_ms']:>12.3f} {speedup_text:>10}"
        )


def main():
    args = _parse_args()
    one = _load(args.one_gpu)
    two = _load(args.two_gpu)
    result = compare(one, two, args.one_gpu, args.two_gpu)
    _print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nComparison written to {args.output}")


if __name__ == "__main__":
    main()
