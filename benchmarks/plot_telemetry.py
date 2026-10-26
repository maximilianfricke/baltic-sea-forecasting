"""Summarize and plot nvidia-smi telemetry captured by Slurm jobs."""

# Standard library
import argparse
import csv
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
            by_gpu[gpu].append(
                {
                    "time": timestamp(row["timestamp"]),
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


def energy_wh(rows):
    total_joules = 0.0
    for left, right in zip(rows, rows[1:]):
        elapsed = (right["time"] - left["time"]).total_seconds()
        total_joules += (left["power_w"] + right["power_w"]) / 2 * elapsed
    return total_joules / 3600


def summarize(path, by_gpu):
    summaries = []
    for gpu, rows in sorted(by_gpu.items()):
        rows.sort(key=lambda row: row["time"])
        utilization = [row["utilization"] for row in rows]
        power = [row["power_w"] for row in rows]
        summaries.append(
            {
                "file": str(path),
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
        )
    return summaries


def plot(path, by_gpu, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    metrics = (
        ("utilization", "GPU utilization [%]"),
        ("power_w", "Power [W]"),
        ("memory_mib", "Device memory [MiB]"),
        ("sm_clock_mhz", "SM clock [MHz]"),
    )
    for gpu, rows in sorted(by_gpu.items()):
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
    fig.suptitle(f"GPU telemetry: {path.stem}")
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{path.stem}-telemetry.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for path in args.telemetry:
        paths.extend(path.rglob("*gpu-*.csv") if path.is_dir() else [path])
    all_summaries = []
    for path in paths:
        by_gpu = load(path)
        if not by_gpu:
            continue
        all_summaries.extend(summarize(path, by_gpu))
        plot(path, by_gpu, args.output_dir)

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
