"""Convert NJORD benchmark JSON files into one comparison table."""

# Standard library
import argparse
import csv
import json
import sys
from pathlib import Path

from benchmarks.common import nested as _nested, result_paths

FIELDS = (
    "file",
    "status",
    "label",
    "started_at_utc",
    "finished_at_utc",
    "error",
    "git_commit",
    "git_dirty",
    "slurm_job_id",
    "gpu",
    "torch",
    "cuda_runtime",
    "world_size",
    "scaling_mode",
    "workload",
    "global_batch_size",
    "model",
    "hidden_dim",
    "processor_layers",
    "gradient_checkpointing",
    "precision",
    "batch_size",
    "num_workers",
    "ar_steps_train",
    "inference_ensemble_size",
    "warmup_steps",
    "measurement_steps",
    "parameters",
    "grid_nodes",
    "mesh_nodes",
    "total_edges",
    "forward_mean_ms",
    "backward_mean_ms",
    "optimizer_mean_ms",
    "data_wait_mean_ms",
    "step_mean_ms",
    "step_std_ms",
    "step_p95_ms",
    "step_p99_ms",
    "end_to_end_mean_ms",
    "end_to_end_p95_ms",
    "end_to_end_p99_ms",
    "samples_per_second",
    "compute_samples_per_second",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "process_peak_rss_gib",
    "gradient_payload_bytes",
    "ring_allreduce_bytes_per_rank",
    "job_to_fit_start_seconds",
    "data_staging_seconds",
    "data_staging_gib",
    "measurement_started_at_utc",
    "measurement_finished_at_utc",
    "timing_mode",
    "gpu_telemetry_enabled",
    "gpu_telemetry_interval_ms",
    "pytorch_profiler_enabled",
    "data_location",
    "original_dataset_samples",
    "effective_samples_per_epoch",
    "dataset_world_size_hint",
    "samples_per_rank_per_epoch",
    "batches_per_epoch",
    "samples_repeated_within_epoch",
)


def _compact_error(error):
    """Keep the comparison CSV single-line; raw JSON retains the traceback."""
    if not error:
        return None
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    for line in reversed(lines):
        if "Error:" in line:
            return line[-1000:]
    return lines[-1][-1000:] if lines else None


def _row(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or "schema_version" not in result:
        return None
    graph = _nested(result, "model", "graph") or {}
    total_edges = sum(
        value
        for key, value in graph.items()
        if key.endswith("_edge_count") and isinstance(value, int)
    )
    peak_bytes = _nested(
        result, "summary", "gpu_peak_memory_allocated_bytes"
    )
    peak_reserved_bytes = _nested(
        result, "summary", "gpu_peak_memory_reserved_bytes"
    )
    process_rss_bytes = _nested(
        result, "summary", "process_peak_rss_bytes"
    )
    return {
        "file": str(path),
        "status": result.get("status"),
        "label": result.get("label"),
        "started_at_utc": result.get("started_at_utc"),
        "finished_at_utc": result.get("finished_at_utc"),
        "error": _compact_error(result.get("error")),
        "git_commit": _nested(result, "git", "commit"),
        "git_dirty": _nested(result, "git", "dirty"),
        "slurm_job_id": _nested(result, "hardware", "slurm_job_id"),
        "gpu": _nested(result, "hardware", "gpu", "name"),
        "torch": _nested(result, "hardware", "torch"),
        "cuda_runtime": _nested(result, "hardware", "cuda_runtime"),
        "world_size": _nested(result, "distributed", "world_size"),
        "scaling_mode": _nested(
            result, "distributed", "scaling_mode"
        ),
        "workload": _nested(result, "protocol", "workload"),
        "global_batch_size": _nested(
            result, "distributed", "global_batch_size"
        ),
        "model": _nested(result, "model", "class"),
        "hidden_dim": _nested(result, "arguments", "hidden_dim"),
        "processor_layers": _nested(
            result, "arguments", "processor_layers"
        ),
        "gradient_checkpointing": _nested(
            result, "model", "gradient_checkpointing"
        ),
        "precision": _nested(result, "arguments", "precision"),
        "batch_size": _nested(result, "arguments", "batch_size"),
        "num_workers": _nested(result, "arguments", "num_workers"),
        "ar_steps_train": _nested(
            result, "arguments", "ar_steps_train"
        ),
        "inference_ensemble_size": _nested(
            result, "arguments", "benchmark_inference_ensemble_size"
        ),
        "warmup_steps": _nested(result, "protocol", "warmup_steps"),
        "measurement_steps": _nested(
            result, "protocol", "completed_measurement_steps"
        ),
        "parameters": _nested(result, "model", "parameters_total"),
        "grid_nodes": graph.get("grid_node_count"),
        "mesh_nodes": graph.get("mesh_node_count"),
        "total_edges": total_edges,
        "forward_mean_ms": _nested(
            result, "summary", "forward_ms", "mean"
        ),
        "backward_mean_ms": _nested(
            result, "summary", "backward_ms", "mean"
        ),
        "optimizer_mean_ms": _nested(
            result, "summary", "optimizer_and_hooks_ms", "mean"
        ),
        "data_wait_mean_ms": _nested(
            result, "summary", "data_wait_ms", "mean"
        ),
        "step_mean_ms": _nested(result, "summary", "step_ms", "mean"),
        "step_std_ms": _nested(result, "summary", "step_ms", "std"),
        "step_p95_ms": _nested(result, "summary", "step_ms", "p95"),
        "step_p99_ms": _nested(result, "summary", "step_ms", "p99"),
        "end_to_end_mean_ms": _nested(
            result, "summary", "end_to_end_wall_ms", "mean"
        ),
        "end_to_end_p95_ms": _nested(
            result, "summary", "end_to_end_wall_ms", "p95"
        ),
        "end_to_end_p99_ms": _nested(
            result, "summary", "end_to_end_wall_ms", "p99"
        ),
        "samples_per_second": _nested(
            result, "summary", "samples_per_second"
        ),
        "compute_samples_per_second": _nested(
            result, "summary", "compute_samples_per_second"
        ),
        "peak_allocated_gib": (
            peak_bytes / 1024**3 if peak_bytes is not None else None
        ),
        "peak_reserved_gib": (
            peak_reserved_bytes / 1024**3
            if peak_reserved_bytes is not None
            else None
        ),
        "process_peak_rss_gib": (
            process_rss_bytes / 1024**3
            if process_rss_bytes is not None
            else None
        ),
        "gradient_payload_bytes": _nested(
            result,
            "distributed",
            "estimated_gradient_payload_bytes_per_step",
        ),
        "ring_allreduce_bytes_per_rank": _nested(
            result,
            "distributed",
            "estimated_ring_allreduce_bytes_per_rank_per_step",
        ),
        "job_to_fit_start_seconds": _nested(
            result, "startup", "job_to_fit_start_seconds"
        ),
        "data_staging_seconds": _nested(
            result, "startup", "data_staging_seconds"
        ),
        "data_staging_gib": (
            _nested(result, "startup", "data_staging_bytes") / 1024**3
            if _nested(result, "startup", "data_staging_bytes")
            is not None
            else None
        ),
        "measurement_started_at_utc": _nested(
            result, "protocol", "measurement_started_at_utc"
        ),
        "measurement_finished_at_utc": _nested(
            result, "protocol", "measurement_finished_at_utc"
        ),
        "timing_mode": _nested(
            result, "instrumentation", "timing_mode"
        ),
        "gpu_telemetry_enabled": _nested(
            result, "instrumentation", "gpu_telemetry_enabled"
        ),
        "gpu_telemetry_interval_ms": _nested(
            result, "instrumentation", "gpu_telemetry_interval_ms"
        ),
        "pytorch_profiler_enabled": _nested(
            result, "instrumentation", "pytorch_profiler_enabled"
        ),
        "data_location": _nested(
            result, "startup", "data_location"
        ),
        "original_dataset_samples": _nested(
            result, "dataset", "original_samples"
        ),
        "effective_samples_per_epoch": _nested(
            result, "dataset", "effective_samples_per_epoch"
        ),
        "dataset_world_size_hint": _nested(
            result, "dataset", "world_size_hint"
        ),
        "samples_per_rank_per_epoch": _nested(
            result, "dataset", "samples_per_rank_per_epoch"
        ),
        "batches_per_epoch": _nested(
            result, "dataset", "batches_per_epoch"
        ),
        "samples_repeated_within_epoch": _nested(
            result, "dataset", "samples_repeated_within_epoch"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    writer = csv.DictWriter(
        sys.stdout, fieldnames=FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    for path in result_paths(args.results):
        try:
            row = _row(path)
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping {path}: {error}", file=sys.stderr)
            continue
        if row is not None:
            writer.writerow(row)


if __name__ == "__main__":
    main()
