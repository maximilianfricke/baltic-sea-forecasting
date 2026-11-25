"""Reproduce the original-versus-distributed AR4 analysis from archived JSON.

This reads measurements only. It never changes model code or launches jobs.
Stacked timings select one critical rank for each measured step, rather than
adding the independent phase maxima in the published aggregate summaries.
"""

import csv
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.compare_model_parallel_real import (
    WORKLOAD_KEYS,
    _fingerprints_match,
)

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "benchmark_results/model-parallel-real"
OUT = BASE / "analysis/ar4-original-vs-mp-4981-4944-4949-4952"
MP = BASE / "full/ar4-workers8-prefetch2-4944-4949-4952"
SPECS = [
    ("original", "Original\n1 GPU", 1, BASE / "baseline/"
     "original-ar4-workers8-prefetch2-4981/"
     "original-graph-efm-ar4-1gpu-4981.json"),
    *[(f"mp{n}", f"Message passing\n{n} GPU" + ("s" if n > 1 else ""),
       n, MP / f"real-mp-max-ar4-ckpt-full-rank-local-{n}gpu-{job}.json")
      for n, job in ((1, 4944), (2, 4949), (4, 4952))],
]
PHASES = (
    "data_distribution_ms", "data_read_ms", "host_to_device_ms",
    "batch_broadcast_ms", "data_rank_wait_ms", "forward_ms", "backward_ms",
    "gradient_sync_ms", "optimizer_ms", "compute_ms", "metric_reduce_ms",
    "step_ms", "end_to_end_ms",
)
STACK = (
    "data_distribution_ms", "forward_ms", "backward_ms",
    "gradient_sync_ms", "optimizer_ms", "metric_reduce_ms",
)
INTENTIONAL = {"data_mode", "checkpoint_early_stop", "data_phase_rank_barrier"}
COLORS = ["#737d86", "#3166a0", "#168979", "#d88b42"]


def avg(records, key):
    return statistics.fmean(record[key] for record in records)


def relative(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def source_blob(commit, filename):
    return subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{filename}"], cwd=REPO, text=True
    ).strip()


def validate(runs):
    original = runs[0]
    differences = {}
    for run in runs:
        assert run["status"] == "completed"
        assert not run["git"]["tracked_dirty"]
        assert len(run["records"]) == 50
        assert run["summary"]["measured_samples"] == 100
        assert sum(r["batch_samples"] for r in run["records"]) == 100
        assert run["workload"]["warmup_steps"] == 10
        assert run["workload"]["measurement_steps"] == 50
        parameters = run["model"].get(
            "trainable_parameters", run["model"].get("trainable_parameters_per_rank")
        )
        assert parameters == 645020
        different = {
            k: [original["workload"].get(k), run["workload"].get(k)]
            for k in WORKLOAD_KEYS
            if original["workload"].get(k) != run["workload"].get(k)
        }
        assert set(different) <= INTENTIONAL, different
        differences[str(run["slurm_job_id"])] = different
        assert _fingerprints_match(
            original["preflight"]["batch_fingerprint"],
            run["preflight"]["batch_fingerprint"],
        )
        assert math.isclose(original["preflight"]["loss"],
                            run["preflight"]["loss"], rel_tol=1e-5)
        for key in ("train_samples", "train_time_steps", "train_time_start",
                    "train_time_end", "state_variables", "forcing_variables",
                    "source_benchmark_attributes"):
            assert original["workload"]["dataset"][key] == run["workload"]["dataset"][key]
        for phase in PHASES:
            assert math.isclose(avg(run["records"], phase),
                                run["summary"][phase]["mean"], abs_tol=1e-5)
        for i, record in enumerate(run["records"]):
            assert record["measurement_step"] == i
            assert math.isfinite(record["loss"])
            assert math.isclose(record["loss"], original["records"][i]["loss"],
                                rel_tol=1e-4, abs_tol=1e-2)
        for rank in run["rank_metadata"]:
            hardware = rank["hardware"]
            first = original["rank_metadata"][0]["hardware"]
            for key in ("python", "torch", "cuda_runtime", "hostname", "platform"):
                assert first[key] == hardware[key], key
            assert first["gpu"]["name"] == hardware["gpu"]["name"]
            for record in rank["records"]:
                assert math.isclose(sum(record[k] for k in STACK),
                                    record["end_to_end_ms"], abs_tol=.02)
        for filename, blob in original["original_source"]["matching_git_blobs"].items():
            assert source_blob(run["git"]["commit"], filename) == blob, filename
    return {
        "passed": True,
        "all_original_model_loader_source_blobs_match": True,
        "same_gpu_model_node_python_torch_cuda": True,
        "independent_repetitions_per_configuration": 1,
        "original_vs_distributed_intentional_differences": differences,
    }


def describe(spec, run, runs):
    key, label, n, path = spec
    critical = [max((rank["records"][i] for rank in run["rank_metadata"]),
                    key=lambda record: record["end_to_end_ms"])
                for i in range(len(run["records"]))]
    phases = {phase: avg(critical, phase) for phase in PHASES}
    total = run["summary"]["end_to_end_ms"]["mean"]
    assert math.isclose(sum(phases[k] for k in STACK), total, abs_tol=.02)
    residual = phases["data_distribution_ms"] - sum(phases[k] for k in
        ("data_read_ms", "host_to_device_ms", "data_rank_wait_ms", "batch_broadcast_ms"))
    throughput = run["summary"]["observed_samples_per_second"]
    rank_details = []
    for rank in run["rank_metadata"]:
        entry = {
            "rank": rank["rank"],
            "phase_mean_ms": {p: avg(rank["records"], p) for p in PHASES},
            "initialization_timings": rank["initialization_timings"],
        }
        if "partition" in rank:
            partition = rank["partition"]
            entry.update({
                "owned_grid_nodes": partition["owned_interior_grid_nodes"],
                "owned_mesh_nodes": partition["owned_mesh_nodes_by_level"],
                "owned_edges": sum(item["owned_edges"] for levels in
                    partition["relations"].values() for item in levels),
                "halo_entries_by_relation": {name: sum(item["halo_nodes"] for item in levels)
                    for name, levels in partition["relations"].items()},
            })
            entry["total_halo_entries"] = sum(entry["halo_entries_by_relation"].values())
        rank_details.append(entry)
    original = runs[0]
    data_share = phases["data_distribution_ms"] / total
    payload = sum(math.prod(shape) * (8 if dtype == "torch.int64" else 4)
                  for shape, dtype in zip(run["preflight"]["batch_fingerprint"]["shapes"],
                                          run["preflight"]["batch_fingerprint"]["dtypes"]))
    return {
        "key": key, "label": label.replace("\n", " "), "gpus": n,
        "job_id": str(run["slurm_job_id"]), "source": str(path.relative_to(REPO)),
        "observed_samples_per_second": throughput,
        "speedup_vs_original": throughput / original["summary"]["observed_samples_per_second"],
        "speedup_vs_mp1": throughput / runs[1]["summary"]["observed_samples_per_second"],
        "parallel_efficiency_vs_mp1": throughput / runs[1]["summary"]["observed_samples_per_second"] / n,
        "gpu_seconds_per_sample": n / throughput,
        "published_phase_mean_ms": {p: run["summary"][p]["mean"] for p in PHASES},
        "critical_rank_phase_mean_ms": phases,
        "input_unattributed_ms": residual,
        "visible_input_fraction": data_share,
        "maximum_speedup_if_only_visible_input_removed": 1 / (1 - data_share),
        "end_to_end_p95_ms": run["summary"]["end_to_end_ms"]["p95"],
        "within_run_cv": run["summary"]["end_to_end_ms"]["std"] / total,
        "peak_allocated_gib_per_gpu": run["summary"]["peak_memory_allocated_bytes_per_gpu"] / 2**30,
        "peak_reserved_gib_per_gpu": run["summary"]["peak_memory_reserved_bytes_per_gpu"] / 2**30,
        "mean_loss": avg(run["records"], "loss"),
        "mean_negative_likelihood": avg(run["records"], "negative_likelihood"),
        "mean_kl": avg(run["records"], "kl"),
        "max_relative_loss_error_vs_original": max(relative(a["loss"], b["loss"])
            for a, b in zip(original["records"], run["records"])),
        "all_measured_losses_exactly_equal_to_original": all(a["loss"] == b["loss"]
            for a, b in zip(original["records"], run["records"])),
        "block_mean_end_to_end_ms": [avg(run["records"][i:i+10], "end_to_end_ms")
                                     for i in range(0, 50, 10)],
        "measurement_window_seconds": run["summary"]["observed_window_ms"] / 1000,
        "outer_loop_ms_per_step": (run["summary"]["observed_window_ms"]
            - sum(r["end_to_end_ms"] for r in run["records"])) / 50,
        "initialization_seconds_max_rank": max(r["initialization_timings"]
            ["total_before_training_ms"] for r in run["rank_metadata"]) / 1000,
        "staging_seconds": run["rank_metadata"][0]["initialization_timings"]["data_staging_ms"] / 1000,
        "benchmark_timestamp_elapsed_seconds": (datetime.fromisoformat(run["finished_at_utc"])
            - datetime.fromisoformat(run["started_at_utc"])).total_seconds(),
        "global_batch_payload_bytes": payload,
        "ranks": rank_details,
    }


def save(fig, name, note):
    fig.text(.5, .012, note, ha="center", fontsize=8, color="#53645f")
    fig.tight_layout(rect=(0, .045, 1, .965))
    fig.savefig(OUT / "plots" / f"{name}.png", dpi=190, facecolor="white")
    fig.savefig(OUT / "plots" / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def plots(rows, runs):
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.titlepad": 14,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "legend.frameon": False, "grid.alpha": .2,
        "pdf.fonttype": 42,
    })
    labels = [spec[1] for spec in SPECS]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5))
    for ax, values, title, ylabel, fmt in (
        (axes[0], [r["observed_samples_per_second"] for r in rows],
         "Two GPUs deliver the highest throughput", "Global samples / second", ".3f"),
        (axes[1], [r["peak_allocated_gib_per_gpu"] for r in rows],
         "Four GPUs reduce peak memory per device", "Peak allocated GiB / GPU", ".2f"),
    ):
        ax.bar(x, values, color=COLORS, width=.62)
        ax.set(xticks=x, xticklabels=labels, ylabel=ylabel, title=title,
               ylim=(0, max(values)*1.22))
        ax.grid(axis="y")
        for i, value in enumerate(values):
            ax.text(i, value + max(values)*.025, format(value, fmt),
                    ha="center", weight="bold", fontsize=11)
    save(fig, "01-overview", "H64 / L2 / B2 / AR4, checkpointing, 8 workers per rank, prefetch 2. One independent run per case.")

    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    stack = (
        ("data_distribution_ms", "Visible input phase", "#b8cedb"),
        ("forward_ms", "Forward (including halos)", "#3166a0"),
        ("backward_ms", "Backward + checkpoint replay", "#77a0c3"),
        ("gradient_sync_ms", "Parameter-gradient reduction", "#d88b42"),
        ("optimizer_ms", "Optimizer", "#168979"),
        ("metric_reduce_ms", "Metric reduction", "#8b739c"),
    )
    bottom = np.zeros(4)
    for key, label, color in stack:
        values = np.array([r["critical_rank_phase_mean_ms"][key] for r in rows])/1000
        ax.barh(x, values, left=bottom, color=color, height=.58, label=label)
        bottom += values
    for i, value in enumerate(bottom):
        ax.text(value + .025, i, f"{value:.3f} s", va="center", weight="bold")
    ax.set(yticks=x, yticklabels=[s[1].replace("\n", " ") for s in SPECS],
           xlabel="Seconds per global batch of two samples", xlim=(0, 2.3),
           title="Critical-rank timing: model intervals dominate the measured step")
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, fontsize=8)
    save(fig, "02-critical-rank-phases", "One rank selected by maximum end-to-end elapsed time for EACH step. These components add to the total.")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    for base_key, title, ax in (("original", "Original → message passing on one GPU", axes[0]),
                                 ("mp2", "Message passing: two → four GPUs", axes[1])):
        left, right = (rows[0], rows[1]) if base_key == "original" else (rows[2], rows[3])
        keys = STACK
        names = ["Input", "Forward", "Backward", "Weight\nsync", "Optimizer", "Metrics"]
        deltas = [right["critical_rank_phase_mean_ms"][k] - left["critical_rank_phase_mean_ms"][k] for k in keys]
        axes_colors = ["#168979" if d < 0 else "#c96744" for d in deltas]
        ax.bar(np.arange(6), deltas, color=axes_colors, width=.63)
        for i, value in enumerate(deltas):
            ax.annotate(f"{value:+.1f}", (i, value), xytext=(0, 4 if value >= 0 else -14),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.axhline(0, color="#8a949e", lw=.8)
        ax.set(xticks=np.arange(6), xticklabels=names, ylabel="Change in ms / global batch",
               title=title, ylim=(-145, 95))
        ax.grid(axis="y")
    save(fig, "03-phase-deltas", "Negative is faster; positive is slower. Input, checkpoint policy and execution structure differ between original and distributed cases.")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5))
    axes[0].plot([1, 2, 4], [1, 2, 4], "--", color="#a1abb4", label="Ideal relative to MP 1 GPU")
    axes[0].plot([1, 2, 4], [r["speedup_vs_mp1"] for r in rows[1:]], "o-", color=COLORS[1], label="Versus distributed 1 GPU")
    axes[0].plot([1, 2, 4], [r["speedup_vs_original"] for r in rows[1:]], "s-", color=COLORS[2], label="Versus original 1 GPU")
    for i, row in enumerate(rows[1:]):
        axes[0].annotate(f"{row['speedup_vs_original']:.2f}× / {row['speedup_vs_mp1']:.2f}×",
                         (row["gpus"], row["speedup_vs_original"]), xytext=(10 if i == 0 else 0, 12),
                         textcoords="offset points", ha="left" if i == 0 else "center", fontsize=9,
                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
    axes[0].set(xticks=[1, 2, 4], xlabel="GPUs", ylabel="Throughput speedup", ylim=(0, 4.4), xlim=(.65, 4.5),
                title="Keep the two speedup baselines separate")
    axes[0].legend(loc="upper left", fontsize=8)
    costs = [r["gpu_seconds_per_sample"] for r in rows]
    axes[1].bar(x, costs, color=COLORS, width=.62)
    for i, value in enumerate(costs):
        axes[1].text(i, value+.04, f"{value:.3f}", ha="center", weight="bold")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Allocated GPU-seconds / global sample",
                ylim=(0, 3.0), title="One GPU uses the least GPU time per sample")
    for ax in axes:
        ax.grid(axis="y")
    save(fig, "04-scaling-and-resource-cost", "GPU-seconds/sample = GPU count / measured throughput; excludes startup and is not a statement of course billing rules.")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for run, row, color in zip(runs, rows, COLORS):
        values = np.array([r["end_to_end_ms"] for r in run["records"]])/1000
        axes[0].plot(np.arange(1, 51), values, color=color, alpha=.3, lw=.8)
        axes[0].plot(np.arange(3, 49), [np.median(values[i-2:i+3]) for i in range(2, 48)],
                     color=color, label=row["label"], lw=1.8)
        axes[1].plot(np.arange(1, 6), np.array(row["block_mean_end_to_end_ms"])/1000,
                     "o-", color=color, label=row["label"])
    axes[0].set(xlabel="Measured optimizer step", ylabel="End-to-end seconds", title="Faint: raw steps; solid: five-step median")
    axes[1].set(xticks=[1, 2, 3, 4, 5], xlabel="Consecutive ten-step block", ylabel="Mean end-to-end seconds",
                title="Same speed ordering in all five blocks")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y")
    save(fig, "05-step-stability", "Blocks are within-run descriptions, not independent repeated experiments or confidence intervals.")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    for run, row, color in zip(runs, rows, COLORS):
        axes[0].plot(np.arange(1, 51), [r["loss"] for r in run["records"]], color=color,
                     label=row["label"], lw=1.4, alpha=.8)
    for run, row, color in zip(runs[1:], rows[1:], COLORS[1:]):
        errors = [relative(a["loss"], b["loss"]) for a, b in zip(runs[0]["records"], run["records"])]
        axes[1].plot(np.arange(1, 51), np.maximum(errors, 1e-12), color=color, lw=1, label=row["label"])
    axes[0].set(yscale="log", xlabel="Measured optimizer step", ylabel="Training loss", title="All four loss trajectories overlap")
    axes[1].set(yscale="log", xlabel="Measured optimizer step", ylabel="Relative loss difference from original",
                ylim=(3e-13, 1e-5), title="Original and MP 1 GPU: exactly equal at all 50 steps")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y")
    save(fig, "06-loss-agreement", "Zero errors are drawn at 10⁻¹² for log visibility. Loss agreement is not a validation of forecast skill or every gradient.")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    bottom = np.zeros(4)
    for key, label, color in (("data_read_ms", "Visible loader wait", "#b8cedb"),
                               ("host_to_device_ms", "Host-to-device timer", "#3166a0"),
                               ("data_rank_wait_ms", "Selected-rank data wait", "#d88b42"),
                               ("input_unattributed_ms", "Uninstrumented input remainder", "#8b739c")):
        values = [r[key] if key == "input_unattributed_ms" else r["critical_rank_phase_mean_ms"][key] for r in rows]
        axes[0].bar(x, values, bottom=bottom, color=color, label=label, width=.62)
        bottom += values
    axes[0].set(xticks=x, xticklabels=labels, ylabel="ms / global batch", title="Input timer contains more than loader wait and copy")
    axes[0].legend(fontsize=8, loc="upper right")
    shares = [100*r["visible_input_fraction"] for r in rows]
    axes[1].bar(x, shares, color=COLORS, width=.62)
    for i, value in enumerate(shares):
        axes[1].text(i, value+.12, f"{value:.1f}%", ha="center", weight="bold")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Visible input / measured end-to-end time (%)",
                title="The AR4 experiment is not 90% visible input", ylim=(0, 8))
    for ax in axes:
        ax.grid(axis="y")
    save(fig, "07-input-diagnostics", "Worker decoding can overlap compute or contend for CPU. Visible queue wait is not total decoding service time or raw disk I/O.")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    ranks = rows[3]["ranks"]
    edges = [r["owned_edges"] for r in ranks]
    axes[0].bar(x, edges, color=COLORS[1], width=.62)
    for i, value in enumerate(edges):
        axes[0].text(i, value+500, f"{value:,}", ha="center", fontsize=10)
    axes[0].set(xticks=x, xticklabels=[f"Rank {i}" for i in x], ylabel="Receiver-owned edges",
                title="All ranks own 10,587 grid nodes; edge totals differ little", ylim=(0, 81000))
    bottom = np.zeros(4)
    for name, label, color in (("g2m", "Grid → mesh", "#3166a0"), ("m2g", "Mesh → grid", "#b8cedb"),
                               ("m2m", "Within meshes", "#d88b42"), ("mesh_up", "Up", "#168979"),
                               ("mesh_down", "Down", "#8b739c")):
        values = [r["halo_entries_by_relation"][name] for r in ranks]
        axes[1].bar(x, values, bottom=bottom, color=color, width=.62, label=label)
        bottom += values
    for i, value in enumerate(bottom):
        axes[1].text(i, value+20, str(int(value)), ha="center", weight="bold")
    axes[1].set(xticks=x, xticklabels=[f"Rank {i}" for i in x], ylabel="Halo entries summed across relations",
                title="Halo footprint is uneven despite equal grid counts", ylim=(0, 1600))
    axes[1].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(axis="y")
    save(fig, "08-partition-footprint", "Static halo entries are not measured traffic, latency or proof of the four-GPU slowdown; nodes can recur across relations.")


def write_outputs(analysis):
    rows = analysis["rows"]
    (OUT / "comparison.json").write_text(json.dumps(analysis, indent=2) + "\n")
    csv_fields = ("key", "job_id", "gpus", "observed_samples_per_second", "speedup_vs_original",
                  "speedup_vs_mp1", "parallel_efficiency_vs_mp1", "gpu_seconds_per_sample",
                  "peak_allocated_gib_per_gpu", "peak_reserved_gib_per_gpu", "end_to_end_p95_ms",
                  "within_run_cv", "visible_input_fraction", "max_relative_loss_error_vs_original")
    with (OUT / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields + PHASES)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in csv_fields}, **row["published_phase_mean_ms"]})
    with (OUT / "critical-rank-phases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("key", *PHASES, "input_unattributed_ms"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"key": row["key"], **row["critical_rank_phase_mean_ms"],
                             "input_unattributed_ms": row["input_unattributed_ms"]})
    with (OUT / "rank-metrics.csv").open("w", newline="") as stream:
        fields = ("key", "rank", "owned_grid_nodes", "owned_edges", "total_halo_entries", *PHASES)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for rank in row["ranks"]:
                writer.writerow({"key": row["key"], **{k: rank.get(k) for k in fields[1:5]},
                                 **rank["phase_mean_ms"]})


def report(analysis):
    rows = analysis["rows"]
    a, b, c, d = rows
    names = ["Original 1 GPU", "MP 1 GPU", "MP 2 GPUs", "MP 4 GPUs"]

    def table(headers, body):
        return "\n".join([
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(x) for x in row) + " |" for row in body],
        ])

    overview = table(
        ["Case / job", "Samples/s", "Speedup vs original", "Speedup vs MP 1", "Mean E2E ms", "p95 E2E ms", "Peak GiB/GPU"],
        [[f"{name} / {r['job_id']}", f"{r['observed_samples_per_second']:.3f}",
          f"{r['speedup_vs_original']:.3f}×", "—" if i == 0 else f"{r['speedup_vs_mp1']:.3f}×",
          f"{r['published_phase_mean_ms']['end_to_end_ms']:.3f}", f"{r['end_to_end_p95_ms']:.3f}",
          f"{r['peak_allocated_gib_per_gpu']:.3f}"] for i, (name, r) in enumerate(zip(names, rows))],
    )
    phases = table(["Published phase mean, ms", *names], [
        [p, *[f"{r['published_phase_mean_ms'][p]:.3f}" for r in rows]] for p in PHASES])
    critical = table(["Additive critical-rank component, ms", *names], [
        [p, *[f"{r['critical_rank_phase_mean_ms'][p]:.3f}" for r in rows]] for p in STACK])
    input_table = table(["Input component, selected critical rank (ms)", *names], [
        [p, *[f"{r['critical_rank_phase_mean_ms'][p]:.3f}" for r in rows]]
        for p in ("data_read_ms", "host_to_device_ms", "data_rank_wait_ms")
    ] + [["Uninstrumented remainder", *[f"{r['input_unattributed_ms']:.3f}" for r in rows]],
         ["Total visible input", *[f"{r['critical_rank_phase_mean_ms']['data_distribution_ms']:.3f}" for r in rows]]])
    delta = {p: d["critical_rank_phase_mean_ms"][p] - c["critical_rank_phase_mean_ms"][p] for p in STACK}
    delta_table = table(["Two → four GPUs", "Change, ms/global batch"],
                        [[p, f"{delta[p]:+.3f}"] for p in STACK])
    stability = table(["Case", "Within-run CV", "Block means, ms (five consecutive groups of ten steps)"],
                      [[name, f"{r['within_run_cv']:.1%}", ", ".join(f"{x:.1f}" for x in r["block_mean_end_to_end_ms"])]
                       for name, r in zip(names, rows)])
    costs = table(["Case", "GPU-seconds/sample", "Relative to MP 1 GPU", "Peak reserved GiB/GPU"],
                  [[name, f"{r['gpu_seconds_per_sample']:.3f}", f"{r['gpu_seconds_per_sample']/b['gpu_seconds_per_sample']:.2f}×",
                    f"{r['peak_reserved_gib_per_gpu']:.3f}"] for name, r in zip(names, rows)])
    startup = table(["Case", "Staging (s)", "Init through preflight, excl. staging (s)", "Measured window (s)", "Benchmark timestamps (s)"],
                    [[name, f"{r['staging_seconds']:.1f}", f"{r['initialization_seconds_max_rank']:.1f}",
                      f"{r['measurement_window_seconds']:.1f}", f"{r['benchmark_timestamp_elapsed_seconds']:.1f}"]
                     for name, r in zip(names, rows)])
    rank_table = table(["Rank", "Owned grid", "Owned mesh levels", "Incoming edges", "Halo entries across relations", "G2M halo entries"],
                       [[r["rank"], r["owned_grid_nodes"], str(r["owned_mesh_nodes"]), r["owned_edges"],
                         r["total_halo_entries"], r["halo_entries_by_relation"]["g2m"]] for r in d["ranks"]])
    report_text = f"""# Original GraphEFM versus distributed message passing: complete AR4 comparison

Source archive: `{analysis['source_archive_commit'][:7]}`. Jobs **4981 / 4944 / 4949 / 4952**.
Read-only analysis: no model, loader, partitioning or training-job code was changed.

## 1. What the measurements support

**Two GPUs are fastest in this series.** They deliver {c['observed_samples_per_second']:.3f} samples/s:
{c['speedup_vs_original']:.3f}× the original one-GPU reference and {c['speedup_vs_mp1']:.3f}× our distributed
one-GPU implementation. Four GPUs still outperform the original, but deliver
{100*(1-d['observed_samples_per_second']/c['observed_samples_per_second']):.1f}% less throughput than two GPUs.

**Memory distribution works well.** Peak allocated memory falls from {a['peak_allocated_gib_per_gpu']:.3f}
GiB in the original to {d['peak_allocated_gib_per_gpu']:.3f} GiB per GPU on four GPUs, a
{100*(1-d['peak_allocated_gib_per_gpu']/a['peak_allocated_gib_per_gpu']):.1f}% reduction. A capacity benefit and a throughput benefit are separate results.

**The current AR4 hot loop is not 90% visible data loading.** Visible input accounts for
{a['visible_input_fraction']:.1%} in the original and {b['visible_input_fraction']:.1%} / {c['visible_input_fraction']:.1%} /
{d['visible_input_fraction']:.1%} in MP 1/2/4. Forward and backward intervals now dominate.
They include communication and scheduling gaps; these are not isolated GPU arithmetic times.

**Numerical consistency is strong.** Original and MP 1 GPU have exactly the same scalar loss at
every one of the 50 measured optimizer steps. Maximum per-step relative loss differences on
2/4 GPUs are {c['max_relative_loss_error_vs_original']:.3g} / {d['max_relative_loss_error_vs_original']:.3g}.

![Throughput and peak GPU memory](plots/01-overview.png)

{overview}

## 2. What was actually held constant

All four cases use `global_cluster_graph`, 42,348 ocean grid nodes, 8,456 fine and 2,097 coarse
mesh nodes, H64 on grid/mesh/edge/latent representations, two decoder processor layers,
one layer in each encoder, one hidden layer per MLP, global batch size two, two input states
and **four autoregressive forecast steps**. There are 645,020 trainable parameters per rank.
They use FP32, the learned isotropic prior and learned diagonal posterior, WMSE likelihood
plus KL with beta one, CRPS weight zero, no observation-noise sampling, original clamping
and density feedback. AdamW uses LR 0.001, betas (0.9, 0.95), weight decay 0.1.

Every case uses eight persistent spawn workers **per rank**, prefetch factor two **per worker**,
shuffle seed 42, no pinned host memory and no dropped final batch. This is 8 / 8 / 16 / 32
workers in total across original / MP1 / MP2 / MP4. Prefetch two can queue up to two batches
per worker; it does not mean two workers or two batches total. Worker count does not allocate CPU cores.

All cases run on `studgpu-node02.inf.ethz.ch`, NVIDIA GeForce GTX 1080 Ti, Python 3.12.3,
PyTorch 2.14.1+cu126, CUDA runtime 12.6. Each measures 50 optimizer steps after ten warmups,
or 100 measured samples. Each optimizer step contains four forecast steps. The measurement is
not a full training epoch or a convergence/forecast-quality study.

The original benchmark code is commit `675c2bc`; MP runs use `aa5829a`. These are intentionally
different harness commits. This analysis verifies that all 13 recorded original model,
message-passing, loss and loader source blobs match the frozen global-model source `76b30d7`
in **both** commits. Tracked files were clean in every run.

The original and distributed paths use the same seed/scenario sequence by different mechanisms:
the original samples with the default CUDA generator and checkpoint RNG preservation; MP draws
explicit global latent noise outside checkpoints and selects local top-mesh nodes.
Shapes/dtypes/sums of the first effective batch match after canonicalizing unused empty tensors.
This is a fingerprint, not a cryptographic hash of every input value. Its agreement plus the
full measured loss trajectory provides stronger evidence than either check alone.

## 3. Important boundaries of the original-versus-MP comparison

The original reference really calls the unchanged `GraphEFM.training_step` and original
`WeatherDataset`; only metric logging is captured by an adapter. Both paths use the same
standalone CUDA timing/AdamW harness, rather than the complete Lightning trainer lifecycle.
Validation loops, schedulers, checkpoint-file saving and external experiment logging are not timed.

Three recorded settings differ intentionally:

| Setting | Original | Distributed |
| --- | --- | --- |
| Input layout | Original full rectangular Zarr, ocean mask applied during sampling | Prebuilt rank-local ocean shards |
| Non-reentrant checkpoint early stopping | Original default: enabled | Disabled to replay the full collective schedule |
| Data-phase rank barrier | None | Present for multiple ranks; effectively no-op for one |

The original stages its full directory into job-local temporary storage. MP stages the selected
shard layout; its graph/config/statistics still come from the scratch source. Startup paths
therefore also differ. The original full-grid data has 61,200 rectangular cells before masking;
the effective state, forcing and atmosphere tensors contain the same 42,348 ocean nodes.

Consequently the original-to-MP1 result is a **bundle of input-layout and execution changes**.
It is not a measurement of an isolated halo overhead: MP1 has no inter-GPU halo traffic at all.
Even the compute interval can be influenced by background loader CPU work, tensor layout,
Python dispatch and checkpoint scheduling. An observed improvement is real for these runs;
its cause needs controlled experiments and independent repetitions.

## 4. Full phase comparison and the aggregation trap

The following means reproduce the result JSON summaries. Multi-rank summaries take the
maximum rank for each metric independently at every step. They are useful worst-rank
diagnostics, but **their phase maxima must not be added to build a time stack**.
Also, `compute_ms`, `step_ms` and `end_to_end_ms` are nested totals, not additional phases.

{phases}

For additive attribution, the analysis selects the rank with the largest end-to-end interval
at each step, then takes every phase from that same rank. The resulting components sum to
the measured maximum end-to-end time:

{critical}

![Critical-rank additive phase breakdown](plots/02-critical-rank-phases.png)

Observed throughput counts actual measured samples over the complete measured wall-clock window.
It is the primary speedup metric. It differs slightly from `2 / mean(end_to_end_seconds)`
because loop bookkeeping lies outside the per-step interval. That outer-loop remainder is
{a['outer_loop_ms_per_step']:.2f} / {b['outer_loop_ms_per_step']:.2f} / {c['outer_loop_ms_per_step']:.2f} /
{d['outer_loop_ms_per_step']:.2f} ms/step in the four cases; it is not silently assigned to data or halos.

## 5. Original one GPU → distributed one GPU

MP1 improves measured throughput by **{100*(b['speedup_vs_original']-1):.1f}%** and reduces mean
end-to-end time by {a['published_phase_mean_ms']['end_to_end_ms']-b['published_phase_mean_ms']['end_to_end_ms']:.1f} ms
({100*(1-b['published_phase_mean_ms']['end_to_end_ms']/a['published_phase_mean_ms']['end_to_end_ms']):.1f}%).
The input phase falls by 58.5 ms, forward by 84.8 ms, backward by 115.9 ms, optimizer by 17.3 ms
and metric reduction by 1.3 ms. The model-step interval falls from 1,759.7 to 1,540.3 ms.

This does **not** show a 17.5% improvement caused by halo communication. No halo transfer is
needed with one rank. It shows that the distributed training structure plus sharded input
does not incur an observed net single-device slowdown in this experiment.

The optimizer is especially useful as a diagnostic: it is the same AdamW and parameter count,
yet its recorded interval is 17.95 versus 0.63 ms. A change of this size cannot simply be
attributed to fewer graph messages. CUDA events span elapsed stream time between recorded
events, which may include idle time while the host schedules work. The original log explicitly
warns about eight loader workers with a suggested CPU maximum of two. Full-grid background
sample preparation is a plausible source of extra host pressure, but CPU utilization, actual
run-time affinity per rank and kernel traces are not recorded. This remains a hypothesis.

![Phase changes](plots/03-phase-deltas.png)

## 6. Distributed scaling: why two GPUs currently win

The correct internal strong-scaling denominator is distributed MP1, not the slower original:
MP2 is **{c['speedup_vs_mp1']:.3f}×** ({c['parallel_efficiency_vs_mp1']:.1%} efficiency);
MP4 is **{d['speedup_vs_mp1']:.3f}×** ({d['parallel_efficiency_vs_mp1']:.1%} efficiency).
The original-relative factors {c['speedup_vs_original']:.3f}× / {d['speedup_vs_original']:.3f}× answer a
different question: the improvement of the complete new execution over the original reference.

From MP1 to MP2, published forward drops from 535.2 to 435.8 ms and backward from 1,004.3 to
697.0 ms. These savings dominate the added gradient synchronization and other coordination.
From MP2 to MP4, mean E2E instead rises by 78.1 ms ({100*(d['published_phase_mean_ms']['end_to_end_ms']/c['published_phase_mean_ms']['end_to_end_ms']-1):.1f}%).
The exact critical-rank decomposition is:

{delta_table}

Backward contributes about 63 ms of the extra 78 ms. Forward adds about 12 ms.
The visible input phase actually improves by about 8 ms, so it does not explain this reversal.
The separately recorded parameter-gradient reduction is only one communication stage;
halo transfers and checkpoint replay are inside forward/backward. Its small size does not
prove that all communication is small. Conversely, larger forward/backward intervals do not
prove that the network alone caused the slowdown.

Plausible mechanisms to distinguish are smaller local kernels with more launch overhead,
foreground CPU contention as rank/worker counts rise, halo latency and packing, unequal
boundary work, repeated checkpoint communication, and global synchronization. None has an
isolated measured contribution yet.

![Strong scaling and GPU time per sample](plots/04-scaling-and-resource-cost.png)

## 7. Data loading, workers, prefetch and the remaining timer gap

{input_table}

`data_read_ms` measures waiting for a prepared CPU batch from the loader, including immediate
queue/collation work encountered by the caller. With background workers it does **not** measure
all Zarr reading/decompression service time. A 6 ms wait can coexist with substantial decoding
work while the previous graph step runs. `host_to_device_ms` includes the host-side transfer
call and completion wait; it is not an isolated DMA-kernel or raw PCIe-bandwidth measurement.

There is an uninstrumented remainder inside the enclosing data interval: 39.9 ms in the
original, 22.9 ms in MP1 and smaller values on the selected multi-GPU ranks. Code places the
inner transfer timestamp before returning from `stream.next`, while the outer timestamp is
after returning and replacing the previous batch. Cleanup, IPC tensor lifetime and scheduling
are possible contributors. Existing timers cannot identify which. It must not be mislabeled
as extra disk reads, decompression or halo communication.

If **only** the visible input interval vanished and every model interval stayed unchanged,
the maximum throughput factors would be {a['maximum_speedup_if_only_visible_input_removed']:.3f}× /
{b['maximum_speedup_if_only_visible_input_removed']:.3f}× / {c['maximum_speedup_if_only_visible_input_removed']:.3f}× /
{d['maximum_speedup_if_only_visible_input_removed']:.3f}×. For the fastest two-GPU case this is only
about 3.2%. This conditional bound is not a bound on all loader improvements: reducing
background CPU contention could also shorten the model intervals.

More workers or prefetch are therefore not the first blind sweep to run now. Freeze 8/P2
for attribution. A preload/cached-batch mode is useful as a **bounded diagnostic**, with
explicit host/GPU memory budgets and setup time, rather than a promise to preload an
arbitrarily large production dataset. Larger models may hide more input latency but do not
repair the measured scaling mechanism; increasing width now would change the question.

The historical AR1 measurements had a different forecast length and loader policy. Their
large visible input fraction and this AR4 result can both be correct. The improvement cannot
be credited to AR4 alone because the worker/prefetch setup changed too.

![Input phase diagnostics](plots/07-input-diagnostics.png)

## 8. Numerical agreement and run-to-run uncertainty

All first-batch fingerprints and initial losses pass the comparison tolerance. The original
initial loss is 1,102,826,112, equal to MP1 and MP4; MP2 differs by 128 (roundoff scale).
Mean measured loss is {a['mean_loss']:.3f}, {b['mean_loss']:.3f}, {c['mean_loss']:.3f},
{d['mean_loss']:.3f}. Original and MP1 are exactly equal at every recorded scalar-loss step;
the multi-GPU maximum relative differences are below 2.2e-7.

This supports preservation of this training trajectory. It does not establish every full-H64
GPU gradient or parameter element is identical, validate a scientific forecast ensemble,
or demonstrate long-run convergence. Separate earlier equivalence tests checked forecasts,
gradients and AdamW updates on a small CUDA/NCCL graph, and the physical H64 case on CPU.
The four-run performance result is additional evidence, not a replacement for those tests.

![Measured loss agreement](plots/06-loss-agreement.png)

{stability}

Original > MP1 > MP4 > MP2 in elapsed time in all five ten-step blocks. The fastest case is
not selected by a single unusually fast batch. However there is **one independent run per
configuration**. These 50 steps share one job, loader state, evolving model and machine
conditions. They must not be treated as 50 independent replications or used to manufacture
confidence intervals for a configuration's true speed. The original's 15.4% within-run CV
also argues for repeated paired measurements before attributing its improvement causally.

![Timing stability](plots/05-step-stability.png)

## 9. Memory, preflight lifetime and resource efficiency

Allocated peaks are 3.408 / 3.495 / 1.801 / 0.965 GiB per GPU. MP4 reduces per-device memory
by 72.4% relative to MP1 and 71.7% relative to the original. This allows larger problems to
fit even when four GPUs do not maximize throughput for H64. Reserved memory is allocator
capacity and should not be substituted for live allocated memory.

MP1's allocated peak is 89.4 MiB above the original. A code-visible measurement difference
matters here: `model_parallel_real.py` retains `preflight_batch` throughout measurement;
the original runner reassigns its `batch` variable and releases the first input batch.
The effective global input batch is {a['global_batch_payload_bytes']:,} bytes (93.1 MiB).
This can influence the apparent one-GPU memory difference, alongside graph buffers/layout.
It is not an exact allocation attribution; do not subtract the payload from the peaks and
claim a corrected model-only peak. The next measurement should align buffer lifetimes first.

{costs}

GPU-seconds/sample equals allocated GPU count divided by observed samples/s in the steady
window. It excludes initialization and is not the course's billing formula. MP2 uses about
1.48× MP1 GPU time per sample; MP4 about 3.15×. For this configuration MP1 minimizes this
resource metric, MP2 minimizes wall time, and MP4 minimizes peak memory per GPU.

## 10. Four-way ownership: balance counts versus boundary cost

{rank_table}

Grid counts are identical and receiver-owned edge totals differ by only about 1.2% across
ranks. Static halo entries vary from 404 to 1,262, driven particularly by G2M. These entries
are distinct remote sender states **within a relation**, summed over relations. The same
global node may occur in several relations. They are not the number of cut edges, actual
traffic bytes, message calls or communication time.

Near-equal per-rank elapsed time does not prove equal useful work: ranks can wait for each
other. Conversely unequal halo counts are not proof of the measured slowdown. To turn the
partition footprint into actionable weights, record relation use counts, actual batch and
feature dimensions, directed send/receive bytes and packing/wait/compute timings for forward,
checkpoint replay and reverse-gradient traffic.

![Four-GPU partition footprint](plots/08-partition-footprint.png)

The current communication code posts grouped `isend`/`irecv` operations and waits before
using received features. Transfers between peers can be concurrent within that exchange;
there is no deliberately scheduled local-edge MLP computation before that wait. This is
different from an implementation that explicitly overlaps interior work with halo arrival.

## 11. Startup and what a short job costs

{startup}

Staging is outside initialization and the measured window. Initialization includes model,
loader startup and preflight; those substages cannot all be labeled worker spawning.
Benchmark timestamp elapsed includes warmup and other benchmark work but excludes staging,
earlier interpreter/process-group setup and final process cleanup. The original accounting
file records `COMPLETED|0:0`, not a full Slurm elapsed/resource record, so no exact original
job duration or bill is inferred. Short-run end-to-end job costs differ from steady throughput.

## 12. Recommended next experiments, in order

### A. Establish a reproducible timing and memory reference

**Question:** Does the observed ordering persist, and how much of original → MP1 is input
layout versus checkpoint policy versus training execution structure?

Keep H64/L2/B2/AR4, FP32, global batch two, AdamW, seeds and 8/P2 fixed. Prepare a one-GPU
control ladder: (A) original/full-grid/early-stop-on, (B) original/ws1 ocean shard/early-stop-on,
(C) original/ws1 shard/early-stop-off, (D) MP1/ws1 shard/early-stop-off. This reuses the
original model and changes only the harness; no changes to its scientific equations are needed.
Align preflight batch lifetimes, queue state, warmup and measurement counts. Save the
historical measurements unchanged; new runs get new identifiers.

Record the actual CPU affinity/cpuset per rank and inherited worker affinity, allocated CPUs,
GPU UUID/topology, GPU/CPU utilization, source/shard hashes and runtime versions. The log's
two-core warning is evidence of a constraint, not a complete measurement of CPU utilization.
Use at least three independent repetitions of the selected cases, rotating execution order.
Do not silently give the one-GPU case extra CPU resources when moving to a different allocation.
Only claim an isolated effect if its control differs in the intended single factor.

**Deliverable / exit:** unchanged numerical checks, aligned buffer lifetime, per-case medians
and ranges from independent runs, and an explicit account of each remaining configuration difference.

### B. Profile current two- and four-GPU execution before optimizing communication

**Question:** Is the four-GPU penalty primarily foreground CPU launch starvation, smaller
local kernels, halo/wait cost, replay or some combination?

Use separate short traced jobs alongside an untraced timing reference. Instrument embeddings,
G2M, mesh-0 and mesh-1 layers, up/down, M2G, loss, checkpoint replay, parameter-gradient
reduction and optimizer. Capture CPU and CUDA activity on every rank. Count actual posted
P2P messages/bytes and annotate pack, posting, waiting and dependent compute without adding
global synchronization at every layer. CPU `Work.wait` duration alone is not network latency.

Add a bounded compute-only diagnostic using the same prepared batches across cases, after
stopping background loader work; retain the original end-to-end mode as a separate experiment.
This reveals whether background decoding affects model dispatch. Add timestamps around the
return/cleanup boundary of `stream.next` to explain the 5–40 ms input remainder. Profiled
timings are for attribution; report speedups from untraced repetitions.

**Deliverable / exit:** a per-rank critical timeline and measured attribution of the dominant
two → four-GPU increase. A later implementation is selected by that evidence, not halo counts alone.

### C. Integrate the team's external partition files on the current AR4 runtime

**Question:** Can the team's owner maps reduce the measured boundary/imbalance cost without
changing model/data semantics?

Port the NPZ owner-array loader/validation from `lassi-partition-benchmark`, not its old AR1
benchmark wholesale. Bind owner arrays to exact graph/node/edge ordering, world size and a
partition hash. Rebuild matching data shards and manifests; existing geographic shards cannot
be used with arbitrary new owner arrays. Keep scientific tensors, temporal sample order and
8/P2 unchanged. Check output, loss, gradients and AdamW equivalence before timing.
Compare current geographic ownership with the team's proposed owners at H64 first, using
per-relation measured compute/communication weights. The partition design remains the team's task.

**Deliverable / exit:** validated owner/shard identities, unchanged numerical results, measured
per-rank balance and halo traffic, then repeatable untraced throughput and memory improvements.

### D. Implement one targeted runtime change after attribution

If CPU gaps dominate, test explicit CPU placement and a bounded CPU-thread policy before
adding workers; any reduction from 8/P2 is a separately labeled tuning experiment. If packing
or allocation dominates, reuse stable buffers and index plans and validate stream lifetimes.
If waiting dominates and independent local edges exist, overlap transfers with local edge
MLPs, then combine local/remote contributions using the full receiver degree and apply the
node update once. Preserve edge ordering, reverse halo gradients and the complete checkpoint
communication schedule. If small kernels dominate, investigate batching/fusion of equivalent
operations before increasing the model merely to make scaling look better.

Alex's branch offers related halo/Lightning and overlap work, but its grid/G2M/M2G remain
replicated and its archived overlap runs hit CUDA OOM. Align interfaces and memory budgets
before reuse. Nils's stage hooks and Lassi's trace attribution can help, but must wrap our
distributed path, not only the original model. Their numerical timing results use different
hardware/workloads and are not direct baselines for this series.

**Deliverable / exit:** one factor changed, all numerical/communication-order checks passed,
bounded memory, and improvement reproduced against the unchanged case.

### E. Characterize capacity and larger models after the mechanism is understood

Only then vary one of hidden width, processor depth or batch size at a time. Width changes
MLP work, activation memory and halo feature bytes; depth changes relation-use counts and
checkpoint work; batch size changes memory and arithmetic intensity. They are different
experiments and may have different scaling crossover points. Keep AR4 and the original H64
reference as anchors. Larger production datasets need a chunk/shard/cache strategy, not an
unbounded preload promise. CRPS/ensemble training and scientific forecast validation remain
separate workload extensions; this four-run comparison measures the ELBO configuration only.

## 13. Reproduction and evidence

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/njord-analysis-mpl \\
  .venv/bin/python -m benchmarks.analyze_ar4_reference_comparison
```

`comparison.json` includes validated settings, source-file SHA256 hashes, both speedup
baselines, critical-rank components, per-rank data, startup, loss agreement and limitations.
`metrics.csv` reproduces published phase means; `critical-rank-phases.csv` contains additive
components; `rank-metrics.csv` retains per-rank means. PNG and vector PDF figures are under
`plots/`. The HTML presentation accompanies this report. All archived input JSONs are read-only.

Source-code checkpoints for interpretation: `benchmarks/original_graph_efm.py`,
`benchmarks/model_parallel_real.py`, `neural_lam/models/distributed_graph_efm_training.py`,
`neural_lam/distributed_message_passing.py`, and `benchmarks/TEAM_BRANCH_REVIEW_2026-11-22.md`.
The code-visible timer and preflight-lifetime findings should be verified in the next trace,
not silently treated as measured causal explanations.
"""
    (OUT / "analysis.md").write_text(report_text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    runs = [json.loads(path.read_text()) for _, _, _, path in SPECS]
    validation = validate(runs)
    rows = [describe(spec, run, runs) for spec, run in zip(SPECS, runs)]
    analysis = {
        "analysis_version": 1,
        "source_archive_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "validation": validation,
        "source_manifest": [{"path": str(path.relative_to(REPO)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for *_, path in SPECS],
        "rows": rows,
        "limitations": [
            "One independent run per configuration; 50 within-run measurements are not 50 independent experiments.",
            "Original and MP use different input layouts, checkpoint early-stop policy and CUDA RNG handling.",
            "Original model/loader source matches frozen global-model; both paths use a standalone timing harness, not the full Lightning training lifecycle.",
            "Visible loader wait excludes overlapped background decoding; CPU contention can affect model intervals.",
            "Halo time, replay time, CPU utilization, per-rank CPU affinity, GPU utilization and interconnect topology were not separately measured.",
            "Peak allocated memory includes all live buffers; MP retains the first preflight batch throughout the measurement unlike original.",
            "physical_repeat_16 repeats 31 original timesteps; this is a performance proxy rather than a forecast-skill study.",
            "Raw shape/dtype/sum fingerprints and loss trajectories are strong numerical evidence, not exhaustive input/gradient hashes.",
            "Aggregate phase means are maxima selected independently per step; use critical-rank CSV when adding phases.",
        ],
    }
    write_outputs(analysis)
    plots(rows, runs)
    report(analysis)
    print(json.dumps({"validated": True, "output": str(OUT), "cases": [
        {k: row[k] for k in ("key", "observed_samples_per_second", "speedup_vs_original", "peak_allocated_gib_per_gpu")}
        for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
