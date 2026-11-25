"""Rebuild the AR4 analysis from immutable 1/2/4-GPU result JSONs.

No training code is modified. Timing stacks always use the SAME critical
rank per step; independently aggregated phase maxima must not be added.
"""

import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SPECS = ((1, 4944), (2, 4949), (4, 4952))
RUNS = [json.loads((ROOT / (
    f"real-mp-max-ar4-ckpt-full-rank-local-{gpu}gpu-{job}.json"
)).read_text()) for gpu, job in SPECS]
COLORS = ("#275d9a", "#d58124", "#24917c")
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True,
    "grid.alpha": .2, "legend.frameon": False,
})


def mean(records, key):
    return statistics.fmean(record[key] for record in records)


def describe(result):
    ranks = result["rank_metadata"]
    critical = [max((rank["records"][i] for rank in ranks),
                    key=lambda record: record["end_to_end_ms"])
                for i in range(len(result["records"]))]
    phase_keys = ("data_distribution_ms", "forward_ms", "backward_ms",
                  "gradient_sync_ms", "optimizer_ms", "metric_reduce_ms")
    phases = {key: mean(critical, key) for key in phase_keys}
    total = result["summary"]["end_to_end_ms"]["mean"]
    assert math.isclose(sum(phases.values()), total, abs_tol=.001)
    rank_rows = []
    for rank in ranks:
        partition = rank["partition"]
        relations = partition["relations"]
        rank_rows.append({
            "rank": rank["rank"],
            "grid_nodes": partition["owned_interior_grid_nodes"],
            "mesh_nodes_by_level": partition["owned_mesh_nodes_by_level"],
            "owned_edges": sum(item["owned_edges"]
                               for levels in relations.values() for item in levels),
            "halo_entries_across_relations": sum(item["halo_nodes"]
                               for levels in relations.values() for item in levels),
            "halo_entries_by_relation": {
                name: sum(item["halo_nodes"] for item in levels)
                for name, levels in relations.items()
            },
            "phase_mean_ms": {key: mean(rank["records"], key) for key in
                (*phase_keys, "data_read_ms", "data_rank_wait_ms", "step_ms")},
            "initialization_ms": rank["initialization_timings"],
        })
    throughput = result["summary"]["observed_samples_per_second"]
    reference_throughput = RUNS[0]["summary"]["observed_samples_per_second"]
    reference_records = RUNS[0]["records"]
    loss_error = max(abs(left["loss"] - right["loss"]) /
                     max(abs(left["loss"]), abs(right["loss"]), 1.)
                     for left, right in zip(reference_records, result["records"]))
    speedup = throughput / reference_throughput
    return {
        "gpus": result["world_size"], "job_id": result["slurm_job_id"],
        "samples_per_second": throughput, "speedup": speedup,
        "parallel_efficiency": speedup / result["world_size"],
        "gpu_seconds_per_sample_steady_state": result["world_size"] / throughput,
        "peak_memory_per_gpu_gib": result["summary"]["peak_memory_allocated_bytes_per_gpu"] / 2**30,
        "end_to_end_mean_ms": total,
        "end_to_end_p95_ms": result["summary"]["end_to_end_ms"]["p95"],
        "within_run_cv": result["summary"]["end_to_end_ms"]["std"] / total,
        "published_data_phase_share": result["summary"]["data_distribution_ms"]["mean"] / total,
        "critical_rank_phase_mean_ms": phases,
        "loss_mean": result["summary"]["loss"]["mean"],
        "max_per_step_loss_relative_error_from_one_gpu": loss_error,
        "mean_nll": mean(result["records"], "negative_likelihood"),
        "mean_kl": mean(result["records"], "kl"),
        "ten_step_block_means_ms": [mean(result["records"][i:i+10], "end_to_end_ms")
                                   for i in range(0, len(result["records"]), 10)],
        "ranks": rank_rows,
    }


def validate():
    assert [result["world_size"] for result in RUNS] == [1, 2, 4]
    assert len({result["git"]["commit"] for result in RUNS}) == 1
    assert not any(result["git"]["tracked_dirty"] for result in RUNS)
    assert len({rank["hardware"]["hostname"] for result in RUNS
                for rank in result["rank_metadata"]}) == 1
    expected = {"ar_steps": 4, "hidden_dim": 64, "processor_layers": 2,
                "batch_size_global": 2, "num_workers": 8, "prefetch_factor": 2,
                "gradient_checkpointing": True, "warmup_steps": 10,
                "measurement_steps": 50}
    for result in RUNS:
        assert result["status"] == "completed"
        assert len(result["records"]) == 50
        assert result["summary"]["measured_samples"] == 100
        for key, value in expected.items():
            assert result["workload"][key] == value, key
        first = RUNS[0]["preflight"]["batch_fingerprint"]
        current = result["preflight"]["batch_fingerprint"]
        assert first["shapes"] == current["shapes"]
        assert first["dtypes"] == current["dtypes"]
        assert all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)
                   for a, b in zip(first["sums"], current["sums"]))
        assert all(math.isfinite(record["loss"]) for record in result["records"])


def overview(rows):
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.7))
    x = np.arange(3)
    labels = ["1 GPU", "2 GPUs", "4 GPUs"]
    phases = [row["critical_rank_phase_mean_ms"] for row in rows]
    groups = (
        ("Visible input / rank wait", [p["data_distribution_ms"] for p in phases], "#abc7e1"),
        ("Forward (includes halo)", [p["forward_ms"] for p in phases], "#275d9a"),
        ("Backward + checkpoint replay", [p["backward_ms"] for p in phases], "#5988b8"),
        ("Sync / optimizer / metrics", [sum(p[k] for k in
          ("gradient_sync_ms", "optimizer_ms", "metric_reduce_ms")) for p in phases], "#d58124"),
    )
    bottom = np.zeros(3)
    for label, values, color in groups:
        values = np.asarray(values) / 1000
        axes[0].bar(x, values, bottom=bottom, color=color, width=.62, label=label)
        bottom += values
    for i, value in enumerate(bottom):
        axes[0].text(i, value + .025, f"{value:.3f} s", ha="center", weight="bold")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Seconds per global batch",
                title="Critical-rank timing", ylim=(0, 2.05))
    axes[0].legend(loc="upper right", fontsize=8)
    observed = [row["speedup"] for row in rows]
    axes[1].plot([1, 2, 4], [1, 2, 4], "--", color="#9ba4b2", label="Ideal scaling")
    axes[1].plot([1, 2, 4], observed, "o-", color=COLORS[0], linewidth=2, label="Measured throughput")
    for g, value, row in zip((1, 2, 4), observed, rows):
        axes[1].annotate(f"{value:.2f}x\n{row['parallel_efficiency']:.1%} efficiency",
                         (g, value), xytext=(0, 12), textcoords="offset points",
                         ha="center", fontsize=9,
                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1})
    axes[1].set(xticks=[1, 2, 4], xlabel="GPUs", ylabel="Speedup over distributed 1 GPU",
                title="2 GPUs fastest in these runs", ylim=(0, 4.4), xlim=(.6, 4.5))
    axes[1].legend(loc="upper left", fontsize=8)
    memory = [row["peak_memory_per_gpu_gib"] for row in rows]
    axes[2].bar(x, memory, color=COLORS, width=.62)
    axes[2].plot(x, [memory[0] / g for g in (1, 2, 4)], "o--", color="#525c69", label="Ideal 1/N memory")
    for i, value in enumerate(memory):
        axes[2].text(i, value + .10, f"{value:.2f} GiB", ha="center", weight="bold",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1})
    axes[2].set(xticks=x, xticklabels=labels, ylabel="Peak allocated GiB per GPU",
                title="Memory decreases by 72%", ylim=(0, 4.2))
    axes[2].legend(loc="upper right", fontsize=8)
    for axis in axes:
        axis.grid(axis="y")
    fig.suptitle("GraphEFM AR4 | H64 / L2 / B2 | checkpointing | 8 workers per rank, prefetch 2", fontsize=13)
    fig.text(.5, .015, "Same code and node; 10 warmup + 50 measured steps; one run per GPU count. Timing stacks select one critical rank per step.",
             ha="center", fontsize=8, color="#526070")
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(OUT / "ar4-scaling-overview.png", dpi=190)
    plt.close(fig)


def stability_and_halos(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    for result, color in zip(RUNS, COLORS):
        raw = np.array([r["end_to_end_ms"] for r in result["records"]]) / 1000
        axes[0].plot(np.arange(1, len(raw)+1), raw, color=color, alpha=.28, linewidth=1)
        axes[0].plot(np.arange(3, len(raw)-1),
                     [np.median(raw[i-2:i+3]) for i in range(2, len(raw)-2)],
                     color=color, linewidth=2,
                     label=f"{result['world_size']} {'GPU' if result['world_size'] == 1 else 'GPUs'}, 5-step median")
    axes[0].set(xlabel="Measured optimizer step", ylabel="End-to-end seconds",
                title="Four GPUs slower in all five 10-step blocks")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y")
    ranks = rows[2]["ranks"]
    bottom = np.zeros(4)
    names = ("g2m", "m2g", "m2m", "mesh_up", "mesh_down")
    labels = ("Grid to mesh", "Mesh to grid", "Within meshes", "Up", "Down")
    colors = ("#275d9a", "#abc7e1", "#d58124", "#24917c", "#8864a8")
    for name, label, color in zip(names, labels, colors):
        values = [rank["halo_entries_by_relation"][name] for rank in ranks]
        axes[1].bar(np.arange(4), values, bottom=bottom, color=color, label=label, width=.65)
        bottom += values
    for i, value in enumerate(bottom):
        axes[1].text(i, value+20, f"{int(value)}", ha="center", weight="bold")
    axes[1].set(xticks=np.arange(4), xticklabels=[f"Rank {i}" for i in range(4)],
                ylabel="Halo node entries summed across relations",
                title="Equal grid counts, unequal halo footprint", ylim=(0, 1550))
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Run variability and four-GPU partition footprint", fontsize=13)
    fig.text(.5, .015, "Raw timing lines are faint; no independent repeat runs yet. Halo entries are static counts, not measured bytes or communication time.",
             ha="center", fontsize=8, color="#526070")
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(OUT / "ar4-stability-and-halos.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    validate()
    rows = [describe(result) for result in RUNS]
    result = {
        "analysis_commit": "a766208",
        "benchmark_commit": RUNS[0]["git"]["commit"],
        "same_code_commit_and_node": True,
        "independent_runs_per_gpu_count": 1,
        "rows": rows,
        "limitations": [
            "No original unpartitioned global-model GPU timing in this series.",
            "Forward/backward include halo communication, CPU launch gaps and checkpoint work; these are not isolated kernel-only timings.",
            "Visible DataLoader wait is not worker read/decompression service time.",
            "Halo entry counts are not per-step bytes and may repeat nodes across relations.",
            "physical_repeat_16 is an I/O proxy with repeated scientific values, not a full accuracy dataset.",
        ],
    }
    (OUT / "analysis-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    overview(rows)
    stability_and_halos(rows)
    print(json.dumps({"validated": True, "output": str(OUT),
                      "speedups": [row["speedup"] for row in rows]}, indent=2))
