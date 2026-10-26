"""Convert NJORD benchmark JSON files into one comparison table."""

# Standard library
import argparse
import csv
import json
import sys
from pathlib import Path


FIELDS = (
    "file",
    "status",
    "label",
    "started_at_utc",
    "git_commit",
    "git_dirty",
    "gpu",
    "world_size",
    "model",
    "hidden_dim",
    "processor_layers",
    "gradient_checkpointing",
    "parameters",
    "grid_nodes",
    "mesh_nodes",
    "total_edges",
    "step_mean_ms",
    "step_p95_ms",
    "end_to_end_mean_ms",
    "samples_per_second",
    "compute_samples_per_second",
    "peak_allocated_gib",
)


def _nested(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _row(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    graph = _nested(result, "model", "graph") or {}
    total_edges = sum(
        value
        for key, value in graph.items()
        if key.endswith("_edge_count") and isinstance(value, int)
    )
    peak_bytes = _nested(
        result, "summary", "gpu_peak_memory_allocated_bytes"
    )
    return {
        "file": str(path),
        "status": result.get("status"),
        "label": result.get("label"),
        "started_at_utc": result.get("started_at_utc"),
        "git_commit": _nested(result, "git", "commit"),
        "git_dirty": _nested(result, "git", "dirty"),
        "gpu": _nested(result, "hardware", "gpu", "name"),
        "world_size": _nested(result, "distributed", "world_size"),
        "model": _nested(result, "model", "class"),
        "hidden_dim": _nested(result, "arguments", "hidden_dim"),
        "processor_layers": _nested(
            result, "arguments", "processor_layers"
        ),
        "gradient_checkpointing": _nested(
            result, "model", "gradient_checkpointing"
        ),
        "parameters": _nested(result, "model", "parameters_total"),
        "grid_nodes": graph.get("grid_node_count"),
        "mesh_nodes": graph.get("mesh_node_count"),
        "total_edges": total_edges,
        "step_mean_ms": _nested(result, "summary", "step_ms", "mean"),
        "step_p95_ms": _nested(result, "summary", "step_ms", "p95"),
        "end_to_end_mean_ms": _nested(
            result, "summary", "end_to_end_wall_ms", "mean"
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
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for path in args.results:
        writer.writerow(_row(path))


if __name__ == "__main__":
    main()
