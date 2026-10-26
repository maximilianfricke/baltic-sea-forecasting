"""Summarize and plot nvidia-smi telemetry captured by Slurm jobs."""

# Standard library
import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Third-party
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
JOB_ID = re.compile(r"gpu-(\d+)\.csv$")


def number(value):
    match = NUMBER.search(value or "")
    return float(match.group()) if match else None


def timestamp(value):
    value = value.strip()
    for pattern in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            pass
    raise ValueError(f"Unsupported nvidia-smi timestamp: {value}")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def load(path):
    by_gpu = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {key.strip(): value.strip() for key, value in raw.items()}
            gpu = int(number(row["index"]))
            parsed_time = timestamp(row["timestamp"])
            by_gpu[gpu].append(
                {
                    "time": parsed_time,
                    "epoch": parsed_time.timestamp(),
                    "name": row["name"],
                    "utilization": number(row["utilization.gpu [%]"]),
                    "memory_utilization": number(
                        row["utilization.memory [%]"]
                    ),
                    "memory_mib": number(row["memory.used [MiB]"]),
                    "power_w": number(row["power.draw [W]"]),
                    "temperature_c": number(row["temperature.gpu"]),
                    "sm_clock_mhz": number(row["clocks.current.sm [MHz]"]),
                    "memory_clock_mhz": number(
                        row["clocks.current.memory [MHz]"]
                    ),
                }
            )
    return by_gpu


def _benchmark_windows(inputs):
    """Index benchmark measurement windows by Slurm job ID."""
    windows = defaultdict(list)
    if not inputs:
        return windows
    paths = set()
    for input_path in inputs:
        if input_path.is_file():
            paths.add(input_path)
        elif input_path.is_dir():
            for root, _, names in os.walk(input_path, followlinks=True):
                for name in names:
                    if name.endswith(".json") and not name.endswith(
                        ".pt.trace.json"
                    ):
                        paths.add(Path(root) / name)
    for path in sorted(paths):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict) or "schema_version" not in result:
            continue
        job_id = result.get("hardware", {}).get("slurm_job_id")
        protocol = result.get("protocol", {})
        start = protocol.get("measurement_started_at_utc")
        finish = protocol.get("measurement_finished_at_utc")
        if not job_id or not start or not finish:
            continue
        windows[str(job_id)].append(
            {
                "label": result.get("label", path.stem),
                "start_epoch": datetime.fromisoformat(start).timestamp(),
                "finish_epoch": datetime.fromisoformat(finish).timestamp(),
                "steps": protocol.get("completed_measurement_steps"),
                "global_batch_size": result.get("distributed", {}).get(
                    "global_batch_size"
                ),
            }
        )
    return windows


def _select_window(by_gpu, window):
    return {
        gpu: [
            row
            for row in rows
            if window["start_epoch"] <= row["epoch"]
            <= window["finish_epoch"]
        ]
        for gpu, rows in by_gpu.items()
    }


def energy_wh(rows):
    total_joules = 0.0
    for left, right in zip(rows, rows[1:]):
        elapsed = (right["time"] - left["time"]).total_seconds()
        total_joules += (left["power_w"] + right["power_w"]) / 2 * elapsed
    return total_joules / 3600


def summarize(path, by_gpu, scope="full-job", window=None):
    summaries = []
    for gpu, rows in sorted(by_gpu.items()):
        if not rows:
            continue
        rows.sort(key=lambda row: row["time"])
        utilization = [row["utilization"] for row in rows]
        power = [row["power_w"] for row in rows]
        item = {
                "file": str(path),
                "scope": scope,
                "gpu": gpu,
                "name": rows[0]["name"],
                "samples": len(rows),
                "duration_s": (
                    rows[-1]["time"] - rows[0]["time"]
                ).total_seconds(),
                "gpu_utilization_mean_pct": statistics.fmean(utilization),
                "gpu_utilization_p95_pct": percentile(utilization, 0.95),
                "power_mean_w": statistics.fmean(power),
                "power_peak_w": max(power),
                "energy_wh": energy_wh(rows),
                "memory_peak_mib": max(row["memory_mib"] for row in rows),
                "temperature_peak_c": max(
                    row["temperature_c"] for row in rows
                ),
            }
        if window is not None:
            steps = window.get("steps")
            global_batch_size = window.get("global_batch_size")
            joules = item["energy_wh"] * 3600
            item["measurement_steps"] = steps
            item["energy_per_step_j"] = (
                joules / steps if steps else None
            )
            item["energy_per_sample_j"] = (
                joules / (steps * global_batch_size)
                if steps and global_batch_size
                else None
            )
        else:
            item["measurement_steps"] = None
            item["energy_per_step_j"] = None
            item["energy_per_sample_j"] = None
        summaries.append(item)
    return summaries


def plot(path, by_gpu, output_dir, suffix="", title=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    metrics = (
        ("utilization", "GPU utilization [%]"),
        ("power_w", "Power [W]"),
        ("memory_mib", "Device memory [MiB]"),
        ("sm_clock_mhz", "SM clock [MHz]"),
    )
    for gpu, rows in sorted(by_gpu.items()):
        if not rows:
            continue
        rows.sort(key=lambda row: row["time"])
        origin = rows[0]["time"]
        seconds = [(row["time"] - origin).total_seconds() for row in rows]
        for axis, (metric, ylabel) in zip(axes.flat, metrics):
            axis.plot(
                seconds,
                [row[metric] for row in rows],
                label=f"GPU {gpu}",
            )
            axis.set_xlabel("Time [s]")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    for axis in axes.flat:
        axis.legend(frameon=False)
    fig.suptitle(title or f"GPU telemetry: {path.stem}")
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{path.stem}{suffix}-telemetry.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--benchmark-results",
        nargs="+",
        type=Path,
        help="Benchmark JSONs used to isolate post-warm-up measurement windows.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for path in args.telemetry:
        paths.extend(path.rglob("*gpu-*.csv") if path.is_dir() else [path])
    windows_by_job = _benchmark_windows(args.benchmark_results)
    all_summaries = []
    for path in paths:
        by_gpu = load(path)
        if not by_gpu:
            continue
        all_summaries.extend(summarize(path, by_gpu))
        plot(path, by_gpu, args.output_dir)
        match = JOB_ID.search(path.name)
        job_id = match.group(1) if match else None
        for index, window in enumerate(windows_by_job.get(job_id, []), 1):
            selected = _select_window(by_gpu, window)
            if not any(selected.values()):
                continue
            scope = f"measurement:{window['label']}"
            all_summaries.extend(
                summarize(path, selected, scope=scope, window=window)
            )
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", window["label"])
            plot(
                path,
                selected,
                args.output_dir,
                suffix=f"-measurement-{index}-{safe_label}",
                title=f"GPU telemetry measurement window: {window['label']}",
            )

    if all_summaries:
        with (args.output_dir / "telemetry-summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=all_summaries[0].keys()
            )
            writer.writeheader()
            writer.writerows(all_summaries)


if __name__ == "__main__":
    main()
