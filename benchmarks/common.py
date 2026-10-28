"""Shared benchmark-result loading and access helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def nested(mapping: Mapping[str, Any], *keys: str, default=None):
    """Read a nested mapping value without repeated type checks."""
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key)
    return default if value is None else value


def mean(result: Mapping[str, Any], metric: str):
    """Return the mean of a benchmark summary metric."""
    return nested(result, "summary", metric, "mean")


def label(result: Mapping[str, Any]) -> str:
    """Return the explicit run label or fall back to the source filename."""
    source = Path(str(result.get("_path", "benchmark")))
    return str(result.get("label") or source.stem)


def workload(result: Mapping[str, Any]) -> str:
    return str(nested(result, "protocol", "workload", default="training"))


def timing_mode(result: Mapping[str, Any]) -> str:
    return str(
        nested(result, "instrumentation", "timing_mode", default="phase")
    )


def result_paths(inputs: Iterable[Path]) -> list[Path]:
    """Expand JSON files and directories, excluding profiler traces."""
    paths: set[Path] = set()
    for input_path in inputs:
        if input_path.is_file() and input_path.suffix == ".json":
            paths.add(input_path)
        elif input_path.is_dir():
            for root, _, names in os.walk(input_path, followlinks=True):
                for name in names:
                    if name.endswith(".json") and not name.endswith(
                        ".pt.trace.json"
                    ):
                        paths.add(Path(root) / name)
    return sorted(paths)


def load_results(
    inputs: Iterable[Path],
    *,
    statuses: Iterable[str] | None = None,
    require: bool = False,
) -> list[dict[str, Any]]:
    """Load valid benchmark JSON documents from files or directories."""
    accepted = set(statuses) if statuses is not None else None
    results = []
    for path in result_paths(inputs):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict) or "schema_version" not in result:
            continue
        if accepted is not None and result.get("status") not in accepted:
            continue
        result["_path"] = str(path)
        results.append(result)
    if require and not results:
        expected = "benchmark results"
        if accepted:
            expected = f"benchmark results with status {sorted(accepted)}"
        raise ValueError(f"No {expected} found")
    return results
