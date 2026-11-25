"""CPU-only analysis of the three completed PhysicsNeMo full-stage jobs."""

import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .bench import cases, median_summary
from .partitions import METHODS

LABELS = dict(
    original="original",
    chunks="chunks",
    rcb="RCB",
    metis="METIS",
    mtkahypar_km1="Mt-KaHyPar",
)
COLORS = dict(zip(METHODS, ("#2674b8", "#2f8b57", "#d68114", "#b74356")))


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def key(row):
    return (
        int(row["K"]),
        row["partitioner"],
        int(row["hidden_dim"]),
        row["checkpointing"],
        int(row["live_loader"]),
    )


def rank_max(rows, field):
    """Combine halo phases on each rank before taking the slowest rank."""
    steps = defaultdict(list)
    for row in rows:
        value = (
            (float(row["halo_forward_ms"]) + float(row["halo_recompute_ms"]))
            if field == "halo_total_ms"
            else float(row[field])
        )
        steps[int(row["step"])].append(value)
    return [max(steps[step]) for step in range(20)]


def validate(roots, output, accounting=None, stage="full"):
    text = (
        accounting.read_text()
        if accounting
        else subprocess.check_output(
            [
                "sacct",
                "-j",
                ",".join(root.name.rsplit("job", 1)[-1] for root in roots),
                "--format=JobID,State,ExitCode,Elapsed,NodeList",
                "--parsable2",
            ],
            text=True,
        )
    )
    import io

    jobs = {
        row["JobID"]: row
        for row in csv.DictReader(io.StringIO(text), delimiter="|")
    }
    problems, commits, contents = [], set(), []
    sizes, counts = (
        ((1, 2, 4), (5, 16, 16)) if stage == "full" else ((2, 4), (6, 6))
    )
    expected_commit = (
        "1f09517b2659aec07ae3fdd7fad8c964922d7708"
        if stage == "full"
        else "4a20e0d3cfd31249369ca3edb4c4d5b4675b3086"
    )
    for K, root, expected in zip(sizes, roots, counts):
        job = root.name.rsplit("job", 1)[-1]
        state = jobs.get(job, {})
        if state.get("State") != "COMPLETED" or state.get("ExitCode") != "0:0":
            problems.append(f"Job {job}: {state}")
        rows, raw = read_csv(root / "bench.csv"), read_csv(
            root / "raw_steps.csv"
        )
        if len(rows) != expected or len(raw) != expected * 20 * K:
            problems.append(
                f"K={K}: rows={len(rows)}, raw={len(raw)}; expected {expected}/{expected*20*K}"
            )
        if state.get("NodeList") != "studgpu-node02":
            problems.append(
                f"Job {job}: unexpected node {state.get('NodeList')}"
            )
        expected_cases = {
            (
                c["partitioner"],
                c["hidden_dim"],
                c["checkpointing"],
                int(c["live_loader"]),
                c["repeat"],
            )
            for c in cases(K, stage)
        }
        actual_cases = {
            (
                r["partitioner"],
                int(r["hidden_dim"]),
                r["checkpointing"],
                int(r["live_loader"]),
                int(r["repeat"]),
            )
            for r in rows
        }
        if expected_cases != actual_cases:
            problems.append(f"K={K}: case matrix differs from full-stage plan")
        indexed = defaultdict(list)
        for row in raw:
            indexed[row["case_id"]].append(row)
        for row in rows:
            if row["status"] != "ok" or row["check"] != "pass":
                plausible = (
                    row["checkpointing"] == "off" or int(row["hidden_dim"]) > 64
                )
                problems.append(
                    f"K={K} {row['case_id']}: status={row['status']}, check={row['check']}, OOM plausible={plausible}"
                )
            if not np.isclose(
                float(row["check_loss"]),
                float(row["reference_loss"]),
                rtol=2e-5,
                atol=0,
            ):
                problems.append(f"K={K} {row['case_id']}: loss mismatch")
            points = indexed[row["case_id"]]
            if {(int(p["rank"]), int(p["step"])) for p in points} != {
                (r, s) for r in range(K) for s in range(20)
            } or len(points) != K * 20:
                problems.append(
                    f"K={K} {row['case_id']}: missing/duplicate raw steps"
                )
        for row in rows + raw:
            if stage == "balance" and (
                row.get("reference_commit")
                != "1f09517b2659aec07ae3fdd7fad8c964922d7708"
                or not row.get("partitions_dir", "").endswith(
                    "/1deg-unweighted"
                )
            ):
                problems.append(f"Job {job}: balance reference/root differs")
            commits.add(row["commit"])
            if not row["hostname"].startswith("studgpu-node02"):
                problems.append(f"Unexpected hostname {row['hostname']}")
        for filename in ("summary.csv", "partition_metrics.csv"):
            commits.update(r["commit"] for r in read_csv(root / filename))
        metadata = json.loads((root / "metadata.json").read_text())
        commits.add(metadata["git_commit"])
        contents.append((root, rows, indexed, metadata))
    if commits != {expected_commit}:
        problems.append(f"Unexpected commits: {sorted(commits)}")
    print(
        "Validation deviations: "
        + ("none" if not problems else "\n" + "\n".join(problems)),
        flush=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    suffix = "-balance" if stage == "balance" else ""
    (output / f"job-accounting{suffix}.txt").write_text(text)
    (output / f"validation{suffix}.json").write_text(
        json.dumps(
            dict(deviations=problems, commits=sorted(commits), jobs=jobs),
            indent=2,
        )
        + "\n"
    )
    if problems:
        raise SystemExit("Resolve listed deviations before analysis")
    return contents


def summarize(contents, reference_groups=None):
    groups, outliers = {}, []
    for root, rows, indexed, metadata in contents:
        static = read_csv(root / "partition_metrics.csv")
        for row in rows:
            k = key(row)
            group = groups.setdefault(
                k, dict(key=k, launches=[], raw=[], static=static)
            )
            raw = indexed[row["case_id"]]
            group["launches"].append(row)
            group["raw"].append(raw)
            times = rank_max(raw, "e2e_ms")
            med = statistics.median(times)
            maximum = max(times)
            outliers.append(
                dict(
                    K=k[0],
                    case_id=row["case_id"],
                    median_ms=med,
                    max_ms=maximum,
                    max_over_median=maximum / med,
                    slowest_step=times.index(maximum),
                )
            )
    metrics = (
        "e2e_ms",
        "gpu_ms",
        "halo_total_ms",
        "halo_forward_ms",
        "halo_recompute_ms",
        "halo_forward_bytes_per_step",
        "load_wait_ms",
    )
    for k, g in groups.items():
        g["stats"] = {
            m: median_summary([rank_max(r, m) for r in g["raw"]])
            for m in metrics
        }
        g["memory_min"] = (
            min(float(r["peak_memory_min_mib"]) for r in g["launches"]) / 1024
        )
        g["memory_max"] = (
            max(float(r["peak_memory_max_mib"]) for r in g["launches"]) / 1024
        )
        g["memory_peak"] = (
            statistics.median(
                float(r["peak_memory_max_mib"]) for r in g["launches"]
            )
            / 1024
        )
        g["compute_min"] = min(
            float(r["compute_only_ms_min"]) for r in g["launches"]
        )
        g["compute_max"] = max(
            float(r["compute_only_ms_max"]) for r in g["launches"]
        )
        g["compute_slowest_median"] = statistics.median(
            float(r["compute_only_ms_max"]) for r in g["launches"]
        )
        selected = [
            r
            for r in g["static"]
            if (int(r["K"]), r["partitioner"], int(r["hidden_dim"])) == k[:3]
        ]
        balance = []
        for name in ("grid", "mesh_0", "mesh_1"):
            values = {float(r[name + "_imbalance"]) for r in selected}
            assert len(values) == (0 if k[0] == 1 else 1)
            balance.append(next(iter(values)) if values else 1.0)
        g["balance"] = balance
    baselines = dict(reference_groups or {})
    baselines.update(groups)
    for k, g in groups.items():
        K, method, hidden, checkpointing, live = k
        baseline = (
            (2, "mtkahypar_km1", 256, "on", live)
            if hidden == 256
            else (1, "original", hidden, "on", live)
        )
        g["baseline"] = baseline
        g["speedup"] = {
            m: baselines[baseline]["stats"][m]["median"]
            / g["stats"][m]["median"]
            for m in ("e2e_ms", "gpu_ms")
        }
        g["compute_bound"] = (
            baselines[baseline]["compute_slowest_median"]
            / g["compute_slowest_median"]
        )
    return groups, outliers


def case_label(k):
    K, method, hidden, checkpointing, live = k
    return (
        f"K{K} {LABELS[method]} H{hidden}"
        + (" off" if checkpointing == "off" else "")
        + (" live" if live else "")
    )


def number(value):
    return f"{value:,.1f}"


def table(groups, selected, kind):
    if kind == "scaling":
        headers = [
            "H",
            "K",
            "Model",
            "Repeats / CI scope",
            "Step ms [95% CI]",
            "Speedup",
            "Peak GiB (max rank)",
            "Baseline",
        ]
    elif kind == "partitioners":
        headers = [
            "K",
            "Partitioner",
            "Step ms [95% CI]",
            "Launch medians ms (r0/r1/r2)",
            "Max-rank compute ms",
            "Step − compute ms",
            "Forward halo MiB/step",
            "Grid/G0/G1 max:mean",
        ]
    else:
        headers = [
            "K",
            "Case",
            "Repeats / CI scope",
            "Step ms [95% CI]",
            "Speedup wall / GPU",
            "Peak GiB (max rank)",
            "Load wait ms",
            "Baseline / configuration",
        ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    rows = []
    for k in selected:
        K, method, hidden, checkpointing, live = k
        if k not in groups:
            lines.append(
                f"| {hidden} | {K} | original | not measured | — | — | — | H256 uses K=2 |"
            )
            rows.append(dict(K=K, hidden_dim=hidden, status="not_measured"))
            continue
        g = groups[k]
        stat = g["stats"]["e2e_ms"]
        scope = stat["ci_scope"]
        scope_text = (
            "launch+blocks" if scope == "launch_and_cycle_blocks" else scope
        )
        repeat_text = f"{len(g['launches'])} / {scope_text}"
        time_text = f"{number(stat['median'])} [{number(stat['ci95_low'])}, {number(stat['ci95_high'])}]"
        base = g["baseline"]
        baseline = (
            f"K={base[0]} H{base[2]} ckpt on {'live' if base[4] else 'cached'}"
        )
        if checkpointing == "off":
            baseline += "; different configuration"
        launch_medians = [
            statistics.median(rank_max(raw, "e2e_ms")) for raw in g["raw"]
        ]
        compute = g["compute_slowest_median"]
        residual = stat["median"] - compute
        balance = "/".join(f"{v:.3f}" for v in g["balance"])
        if kind == "scaling":
            cells = [
                hidden,
                K,
                LABELS[method],
                repeat_text,
                time_text,
                f"{g['speedup']['e2e_ms']:.2f}×",
                f"{g['memory_max']:.2f}",
                baseline,
            ]
        elif kind == "partitioners":
            cells = [
                K,
                LABELS[method],
                time_text,
                " / ".join(number(v) for v in launch_medians),
                number(compute),
                number(residual),
                f"{g['stats']['halo_forward_bytes_per_step']['median']/2**20:.2f}",
                balance,
            ]
        else:
            cells = [
                K,
                case_label(k),
                repeat_text,
                time_text,
                f"{g['speedup']['e2e_ms']:.2f}× / {g['speedup']['gpu_ms']:.2f}×",
                f"{g['memory_max']:.2f}",
                f"{g['stats']['load_wait_ms']['median']:.3f}",
                baseline,
            ]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
        row = dict(
            commit=g["launches"][0]["commit"],
            hostname=g["launches"][0]["hostname"],
            K=K,
            partitioner=method,
            hidden_dim=hidden,
            checkpointing=checkpointing,
            live_loader=live,
            status="ok",
            repeats=len(g["launches"]),
            ci_scope=scope,
            baseline=baseline,
            peak_memory_max_gib=g["memory_max"],
            e2e_speedup=g["speedup"]["e2e_ms"],
            gpu_speedup=g["speedup"]["gpu_ms"],
            launch_medians_ms="/".join(str(v) for v in launch_medians),
            max_rank_compute_ms=compute,
            step_minus_compute_ms=residual,
            grid_imbalance=g["balance"][0],
            G0_imbalance=g["balance"][1],
            G1_imbalance=g["balance"][2],
        )
        for metric, stats in g["stats"].items():
            row.update(
                {
                    metric + "_" + name: value
                    for name, value in stats.items()
                    if name != "ci_scope"
                }
            )
        rows.append(row)
    return "\n".join(lines), rows


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def figures(groups, output, unweighted=None):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, hidden in zip(axes, (64, 128)):
        methods = METHODS if hidden == 64 else ("mtkahypar_km1",)
        for method in methods:
            keys = [(1, "original", hidden, "on", 0)] + [
                (K, method, hidden, "on", 0) for K in (2, 4)
            ]
            ax.plot(
                (1, 2, 4),
                [groups[k]["speedup"]["e2e_ms"] for k in keys],
                "o-",
                color=COLORS[method],
                label=LABELS[method],
            )
            ax.plot(
                (1, 2, 4),
                [groups[k]["compute_bound"] for k in keys],
                "--",
                color=COLORS[method],
                alpha=0.75,
            )
        ax.plot(
            (1, 2, 4), (1, 2, 4), ":", color="#555555", label="Ideal linear"
        )
        ax.set(
            xlabel="GPUs K",
            ylabel="Speedup vs original K=1",
            title=f"H{hidden}: checkpointing on, cached",
            xticks=(1, 2, 4),
            ylim=(0.8, 4.3),
        )
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color="black", linestyle="--"))
        labels.append("Compute-only bound")
        ax.legend(handles, labels, fontsize=8)
    save_figure(fig, output, "figure1-speedup")

    selected = [(K, m, 64, "on", 0) for K in (2, 4) for m in METHODS]
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    points = [(k, groups[k], "weighted") for k in selected]
    points += [(k, g, "unweighted") for k, g in (unweighted or {}).items()]
    for k, g, variant in points:
        stat = g["stats"]["e2e_ms"]
        ax.errorbar(
            g["compute_slowest_median"],
            stat["median"],
            yerr=[
                [stat["median"] - stat["ci95_low"]],
                [stat["ci95_high"] - stat["median"]],
            ],
            fmt="o" if k[0] == 2 else "s",
            color=COLORS[k[1]],
            capsize=4,
            markersize=7,
            markerfacecolor="none" if variant == "unweighted" else COLORS[k[1]],
            label=f"K={k[0]} {LABELS[k[1]]} {variant}",
        )
    ax.plot((250, 850), (250, 850), ":", color="gray", label="Equal durations")
    ax.set(
        xlabel="Profiled max-rank compute-only time [ms]",
        ylabel="Median rank-max step time [ms]",
        title="H64 partitioners: separately profiled compute and timed steps",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, output, "figure2-step-vs-compute")

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    values = [
        np.concatenate([rank_max(raw, "e2e_ms") for raw in groups[k]["raw"]])
        for k in selected
    ]
    positions = np.arange(len(selected))
    boxes = ax.boxplot(
        values,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    rng = np.random.default_rng(42)
    for i, (k, data, box) in enumerate(zip(selected, values, boxes["boxes"])):
        box.set_facecolor(COLORS[k[1]])
        box.set_alpha(0.2)
        ax.scatter(
            i + rng.uniform(-0.17, 0.17, len(data)),
            data,
            color=COLORS[k[1]],
            s=12,
            alpha=0.6,
        )
    ax.set_xticks(
        positions,
        [f"K{k[0]}\n{LABELS[k[1]]}\nn=60" for k in selected],
        fontsize=9,
    )
    ax.set(
        ylabel="Rank-max step time [ms]",
        title="H64 partitioners: all twenty steps × three launches",
    )
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output, "figure3-step-distributions")


def save_figure(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=200)
    fig.savefig(output / (name + ".pdf"))
    plt.close(fig)


def findings(groups):
    notes = []
    for hidden in (64, 128):
        base = groups[(1, "original", hidden, "on", 0)]
        two, four = [
            groups[(K, "mtkahypar_km1", hidden, "on", 0)] for K in (2, 4)
        ]
        notes.append(
            f"H{hidden} scaling: original K=1 {base['stats']['e2e_ms']['median']:.1f} ms; Mt-KaHyPar K=2 {two['stats']['e2e_ms']['median']:.1f} ms ({two['speedup']['e2e_ms']:.2f}×), K=4 {four['stats']['e2e_ms']['median']:.1f} ms ({four['speedup']['e2e_ms']:.2f}×), Table 1. H128 has one launch per point."
        )
    h256 = groups[(4, "mtkahypar_km1", 256, "on", 0)]
    notes.append(
        f"H256: K=4 {h256['stats']['e2e_ms']['median']:.1f} ms, {h256['speedup']['e2e_ms']:.2f}× versus K=2 {groups[h256['baseline']]['stats']['e2e_ms']['median']:.1f} ms; no K=1 performance result exists (Table 1)."
    )
    for K in (2, 4):
        a = groups[(K, "chunks", 64, "on", 0)]["stats"]["e2e_ms"]
        b = groups[(K, "mtkahypar_km1", 64, "on", 0)]["stats"]["e2e_ms"]
        separate = (
            a["ci95_low"] > b["ci95_high"] or b["ci95_low"] > a["ci95_high"]
        )
        notes.append(
            f"Chunks vs Mt-KaHyPar K={K}: {a['median']:.1f} vs {b['median']:.1f} ms; CIs [{a['ci95_low']:.1f}, {a['ci95_high']:.1f}] vs [{b['ci95_low']:.1f}, {b['ci95_high']:.1f}] ms. "
            + (
                "The CIs do not overlap, supporting a difference in this experiment."
                if separate
                else "No significant difference under the requested non-overlap criterion."
            )
            + " Table 2."
        )
    rcb = groups[(2, "rcb", 64, "on", 0)]["stats"]["e2e_ms"]
    mt = groups[(2, "mtkahypar_km1", 64, "on", 0)]["stats"]["e2e_ms"]
    notes.append(
        f"K=2 RCB is lower than Mt-KaHyPar with non-overlapping CIs: {rcb['median']:.1f} [{rcb['ci95_low']:.1f}, {rcb['ci95_high']:.1f}] ms, versus {mt['median']:.1f} [{mt['ci95_low']:.1f}, {mt['ci95_high']:.1f}] ms (Table 2). All four K=4 CIs overlap, so their numerical median order is not interpreted as a ranking."
    )
    g = groups[(4, "chunks", 64, "on", 0)]
    med = [statistics.median(rank_max(r, "e2e_ms")) for r in g["raw"]]
    notes.append(
        f"K=4 launch variation is substantial: chunks launch medians are {' / '.join(number(v) for v in med)} ms (Table 2). CIs resample launches so this variation is retained."
    )
    for K in (2, 4):
        a, b = [
            groups[(K, m, 64, "on", 0)] for m in ("chunks", "mtkahypar_km1")
        ]
        av, bv = [
            g["stats"]["halo_forward_bytes_per_step"]["median"] / 2**20
            for g in (a, b)
        ]
        ar, br = [
            g["stats"]["e2e_ms"]["median"] - g["compute_slowest_median"]
            for g in (a, b)
        ]
        notes.append(
            f"K={K} chunks/Mt-KaHyPar forward halo volume is {av:.2f}/{bv:.2f} MiB per step, while step-minus-profiled-compute is {ar:.1f}/{br:.1f} ms (Table 2). This residual includes more than communication and cannot be read as transport time."
        )
    a, b = [groups[(2, m, 64, "on", 0)] for m in ("rcb", "mtkahypar_km1")]
    notes.append(
        f"At K=2, RCB/Mt-KaHyPar grid imbalance is {a['balance'][0]:.3f}/{b['balance'][0]:.3f}, and max-rank profiled compute is {a['compute_slowest_median']:.1f}/{b['compute_slowest_median']:.1f} ms (Table 2). Node-count balance alone cannot establish which partition-dependent work determines the critical path; the weighted/unweighted comparison is in Table 4."
    )
    for K in (2, 4):
        on, off = [
            groups[(K, "mtkahypar_km1", 64, c, 0)] for c in ("on", "off")
        ]
        notes.append(
            f"K={K} checkpointing off: {off['stats']['e2e_ms']['median']:.1f} ms versus on {on['stats']['e2e_ms']['median']:.1f} ms; peak memory {off['memory_max']:.2f} versus {on['memory_max']:.2f} GiB/GPU (Tables 1 and 3). Off speedup {off['speedup']['e2e_ms']:.2f}× uses K=1 checkpointing on and is explicitly a different configuration."
        )
    for K in (1, 2, 4):
        method = "original" if K == 1 else "mtkahypar_km1"
        cached, live = [groups[(K, method, 64, "on", l)] for l in (0, 1)]
        notes.append(
            f"K={K} live/cached: {live['stats']['e2e_ms']['median']:.1f}/{cached['stats']['e2e_ms']['median']:.1f} ms, exposed next(iterator) wait {live['stats']['load_wait_ms']['median']:.3f} ms (Tables 1 and 3). "
            + (
                "The original single-worker full-sample loader remains exposed."
                if K == 1
                else "The small host wait indicates effective prefetching, but does not exclude data-readiness waits inside GPU halo events or launch-to-launch drift."
            )
        )
    return notes


def balance_table(weighted, unweighted):
    headers = [
        "K",
        "Partitioner",
        "Assignment",
        "Grid/G0/G1 max:mean",
        "Forward halo MiB/step",
        "Max-rank compute ms",
        "Median step ms [95% CI]",
        "Launch medians ms",
        "Slowdown U/W",
    ]
    text = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    rows, notes = [], []
    for k in sorted(unweighted):
        w, u = weighted[k], unweighted[k]
        ratio = u["stats"]["e2e_ms"]["median"] / w["stats"]["e2e_ms"]["median"]
        for variant, g in (("weighted", w), ("unweighted", u)):
            st = g["stats"]["e2e_ms"]
            medians = [
                statistics.median(rank_max(r, "e2e_ms")) for r in g["raw"]
            ]
            cells = [
                k[0],
                LABELS[k[1]],
                variant,
                "/".join(f"{v:.3f}" for v in g["balance"]),
                f"{g['stats']['halo_forward_bytes_per_step']['median']/2**20:.2f}",
                number(g["compute_slowest_median"]),
                f"{number(st['median'])} [{number(st['ci95_low'])}, {number(st['ci95_high'])}]",
                " / ".join(number(v) for v in medians),
                f"{ratio:.3f}" if variant == "unweighted" else "—",
            ]
            text.append("| " + " | ".join(map(str, cells)) + " |")
            rows.append(
                dict(
                    K=k[0],
                    partitioner=k[1],
                    assignment=variant,
                    commit=g["launches"][0]["commit"],
                    hostname=g["launches"][0]["hostname"],
                    repeats=3,
                    ci_scope=st["ci_scope"],
                    grid_imbalance=g["balance"][0],
                    G0_imbalance=g["balance"][1],
                    G1_imbalance=g["balance"][2],
                    halo_forward_bytes=g["stats"][
                        "halo_forward_bytes_per_step"
                    ]["median"],
                    max_rank_compute_ms=g["compute_slowest_median"],
                    step_ms=st["median"],
                    ci95_low=st["ci95_low"],
                    ci95_high=st["ci95_high"],
                    launch_medians_ms="/".join(map(str, medians)),
                    slowdown_unweighted_weighted=(
                        ratio if variant == "unweighted" else 1.0
                    ),
                )
            )
        a, b = w["stats"]["e2e_ms"], u["stats"]["e2e_ms"]
        separate = (
            a["ci95_high"] < b["ci95_low"] or b["ci95_high"] < a["ci95_low"]
        )
        comparison = (
            "The CIs do not overlap."
            if separate
            else "No significant difference under the non-overlap criterion."
        )
        notes.append(
            f"{LABELS[k[1]]} K={k[0]} weighted/unweighted: {a['median']:.1f}/{b['median']:.1f} ms, unweighted/weighted={ratio:.3f}; grid imbalance {w['balance'][0]:.3f}/{u['balance'][0]:.3f}, profiled max-rank compute {w['compute_slowest_median']:.1f}/{u['compute_slowest_median']:.1f} ms. {comparison} (Table 4)."
        )
    return "\n".join(text), rows, notes


def write_report(groups, outliers, contents, output, tables, notes):
    meta = contents[0][3]
    gpu = read_csv(contents[0][0] / "gpus.csv")[0]
    driver = next(
        v.strip() for k, v in gpu.items() if k.strip() == "driver_version"
    )
    cached = [
        r for r in outliers if r["K"] > 1 and "-on-cached-" in r["case_id"]
    ]
    count = sum(r["slowest_step"] == 4 for r in cached)
    text = (
        f"""# GraphEFM full benchmark

## Validation and setup

No deviations: jobs 5490/5491/5492 COMPLETED, exit code 0:0; 5/16/16 case rows,
100/640/1280 raw rows, all 37 cases status=ok and check=pass; no OOMs.
Benchmark commit `{meta['git_commit']}` is identical in all measurement rows.
Node studgpu-node02; GTX 1080 Ti, PCIe, about 11 GiB per GPU; driver {driver},
torch {meta['torch']}, CUDA {meta['cuda']}, NCCL {'.'.join(map(str,meta['nccl']))}.
Workload: 1° GraphEFM, FP32, batch 2, AR4, two processor layers, AdamW lr=1e-3,
betas=(0.9,0.95), weight decay=0.1. Initialization seed 42, check seed 1042,
training seed 1042+repeat. Timed kernels are nondeterministic. All data staging
succeeded. One loader worker/rank, prefetch=2, two allocated CPUs/GPU.

## Timing and statistics

Each launch uses five warmup and twenty measured steps, cycling four distinct
batches [0,1], [2,3], [4,5], [6,7], plus one separately profiled step. All raw
steps and outliers are retained. Cached timing excludes loading and H2D:
copy to GPU, synchronize, barrier, start wall/CUDA-event timers, train, synchronize
the final event. Thus cached e2e_ms is a synchronized model-training step, not
application end-to-end time. Live mode starts the wall timer before next(iterator),
includes H2D, and has no pre-timer barrier. Its GPU/halo events may include
waiting for other ranks' data. The CPU cache contains four batches; only one
batch resides on each GPU during a step. Peak allocated memory excludes the
separate profiler run and is reset after warmup.

For each measured step, take the maximum over ranks, then pool all launches
and take the median. Speedup is a ratio of these medians, never a mean of ratios.
95% CIs use a nonparametric percentile bootstrap over launches and four-step
cycle blocks within launches, 4,000 resamples, seed 42. H64 cached checkpoint-on
cases have three launches (scope launch_and_cycle_blocks). Every other case has
one launch (scope within_launch_only); its CI does not measure run-to-run variation.
With only three launches and five timed cycles, drift and serial dependence
remain incompletely sampled. Differences are interpreted only for non-overlapping
95% timing CIs; "no significant difference" below refers to this requested
criterion, not a formal hypothesis test.

Compute-only time is the summed duration of non-NCCL CUDA kernels from one extra
profiled step per rank. Take the maximum rank per launch, then the median of
these launch maxima. Table 2's step-minus-compute subtracts this profile statistic
from the pooled timed median. It mixes communication, host gaps, synchronization
and differences between a profiled step and timed steps; it is not a measurement
of communication alone. The compute-only speedup bound uses matching K=1 compute
divided by this max-rank statistic. It is idealized, not an achievable prediction.

## Table 1 — scaling

K>1 uses Mt-KaHyPar km1; K=1 is original GraphEFM. All cached, checkpointing on.
Peak memory is the maximum across ranks and launches. H256 uses K=2 as baseline;
no K=1 H256 performance case was requested (its forward-only loss check passed).
H128/H256 have one launch; the H64 points have three.

{tables[0]}

## Table 2 — H64 partitioners

Three launches per case, all cached and checkpointing on. Compute is the median
of the profiled max-rank kernel sums. Halo bytes are predicted forward received
bytes per step, taking the maximum rank; recomputation/backward bytes are excluded.
Grid/G0/G1 imbalance is maximum owned-node count divided by mean count, from the
actual GraphPartitions. Per-launch medians show variation hidden by a pooled value.

{tables[1]}

## Table 3 — checkpointing off and live loader

Every row has repeats=1 / within_launch_only. Checkpoint-off speedup compares
against checkpoint-on original K=1: this is a different configuration. Live
speedup uses the live original K=1 baseline at the same width. GPU speedup is
reported alongside end-to-end speedup because live host loading is timed separately.

{tables[2]}

## Figures

![Figure 1: speedup and compute-only bound](figure1-speedup.png)

Figure 1: H64/H128 cached checkpoint-on scaling, ratios of medians. Dashed colored
curves are the separate compute-only bound; dotted gray is ideal linear scaling.

![Figure 2: step versus max-rank compute](figure2-step-vs-compute.png)

Figure 2: eight weighted H64 partitioner cases{' and four unweighted cases (hollow markers)' if len(tables)==4 else ''}. X is median profiled max-rank compute;
Y is pooled timed median with its 95% CI. The diagonal denotes equal durations,
not an equality expected across these two separate measurements.

![Figure 3: all rank-max timed steps](figure3-step-distributions.png)

Figure 3: all sixty rank-max timed steps per partitioner case (twenty × three
launches), box plots with individual jittered points; no category-connecting lines
and no outliers removed. All figures have PNG and vector PDF versions.

## Findings

"""
        + "\n".join("- " + n for n in notes)
        + f"""

## Known effects and scope

- K=4 has substantial launch-to-launch variation (Table 2). Sequential K jobs
  do not control node activity, clocks or filesystem-cache state across time.
- The recurring single outlier step is kept. Among {len(cached)} distributed cached
  checkpoint-on launches, {count} have their slowest timed step at zero-based
  index 4. Some K=4 H64 maxima are about 2× their launch median; K=2 outliers
  are often smaller. See outliers.csv and Figure 3; no cause is established here.
- K=1 uses set_checkpoint_early_stop(True), the original behavior. K>1 uses
  False, fully recomputing AR steps to preserve collective order. This slightly
  disadvantages distributed checkpoint-on runs.
- Original K=1 live reads full samples with one worker on two CPUs. K>1 live
  reads owned-rank samples. Live speedup includes that input-pipeline change.
- NCCL kernel durations include waiting for other ranks. Timed halo counters
  cover forward/recomputation; backward exchange and loss/parameter reductions
  are included only in complete profiler NCCL activity. No application-level
  compute/communication overlap was added.
- The chosen subset conserves quota: 1°, training only, H64 for all partitioners,
  H128/H256 distributed runs only for Mt-KaHyPar, three launches only for primary
  H64 cached cases. K=1 checkpointing off was omitted because the earlier run
  did not fit. Wider and live/checkpoint-off cases have one launch.
- These findings apply to this 1° GraphEFM training workload on 1–4 GTX 1080 Ti
  GPUs on one node. They do not predict 0.25°, other GPUs, multiple nodes or a
  complete training run. The short, repeated-batch experiment does not measure
  variation over a changing training dataset or later optimizer trajectory.

Reporting follows [Hoefler and Belli](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf):
absolute reference times, ratio-of-median speedups, labelled CI scope, all raw
samples, and explicit subsets and limits. Balance jobs use commit `4a20e0d` and reuse the
validated `1f09517` K=1 reference after checking that original model, adapter,
dataset and vendored library implementations are unchanged. The numerical workload
and timing path are the same; partition selection and provenance fields were added.
"""
    )
    if len(tables) == 4:
        section = """## Table 4 — weighted vs unweighted

Jobs 5497 (K=2) and 5498 (K=4) COMPLETED on studgpu-node02; all 12 cases
passed, with 240/480 raw rows. No validation deviations. Each case has three
launches and launch_and_cycle_blocks CI scope. Compute is the median of
per-launch maximum-rank profiled kernel sums. Halo bytes are forward received
bytes per step, maximum rank. Slowdown is the ratio unweighted/weighted medians;
values below one are numerical observations, not a resolved advantage when CIs
overlap. Jobs were sequential; clocks and node activity were not controlled
between the full and balance runs.

""" + tables[3] + "\n\n"
        text = text.replace("## Figures\n", section + "## Figures\n")
    (output / "RESULTS.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        nargs=3,
        help="K=1, K=2, K=4 result directories, in that order",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results/report-full",
    )
    parser.add_argument(
        "--balance-results",
        type=Path,
        nargs=2,
        help="K=2 and K=4 balance result directories",
    )
    parser.add_argument(
        "--balance-accounting", type=Path, help="Saved balance sacct output"
    )
    parser.add_argument(
        "--accounting",
        type=Path,
        help="Saved sacct output for offline reproduction",
    )
    args = parser.parse_args()
    contents = validate(args.results, args.output, args.accounting)
    groups, outliers = summarize(contents)
    for root, rows, _, _ in contents:
        for old in read_csv(root / "summary.csv"):
            new = groups[key(old)]["stats"]["e2e_ms"]
            assert np.isclose(
                new["median"], float(old["e2e_ms_median"]), rtol=1e-12
            )
            assert np.isclose(
                new["ci95_low"], float(old["e2e_ms_ci95_low"]), rtol=1e-12
            )
            assert np.isclose(
                new["ci95_high"], float(old["e2e_ms_ci95_high"]), rtol=1e-12
            )
    primary = [
        (K, "original" if K == 1 else "mtkahypar_km1", h, "on", 0)
        for h in (64, 128, 256)
        for K in (1, 2, 4)
    ]
    partitioners = [(K, m, 64, "on", 0) for K in (2, 4) for m in METHODS]
    other = [k for k in groups if k[3] == "off" or k[4]]
    texts = []
    for name, kind, keys in [
        ("table1", "scaling", primary),
        ("table2", "partitioners", partitioners),
        ("table3", "other", other),
    ]:
        text, rows = table(groups, keys, kind)
        texts.append(text)
        (args.output / (name + ".md")).write_text(text + "\n")
        save_csv(args.output / (name + ".csv"), rows)
    save_csv(args.output / "outliers.csv", outliers)
    unweighted = {}
    notes = findings(groups)
    if args.balance_results:
        extra = validate(
            args.balance_results,
            args.output,
            args.balance_accounting,
            stage="balance",
        )
        unweighted, _ = summarize(extra, groups)
        for root, _, _, _ in extra:
            for row in read_csv(root / "summary.csv"):
                stat = unweighted[key(row)]["stats"]["e2e_ms"]
                for name in ("median", "ci95_low", "ci95_high"):
                    assert np.isclose(
                        stat[name], float(row["e2e_ms_" + name]), rtol=1e-12
                    )
        text, rows, extra_notes = balance_table(groups, unweighted)
        texts.append(text)
        (args.output / "table4.md").write_text(text + "\n")
        save_csv(args.output / "table4.csv", rows)
        notes += extra_notes
    figures(groups, args.output, unweighted)
    write_report(groups, outliers, contents, args.output, texts, notes)
    for i, text in enumerate(texts, 1):
        print(f"\nTable {i}\n" + text, flush=True)
    print("\nFindings\n" + "\n".join("- " + n for n in notes), flush=True)
    print(f"Outputs: {args.output}", flush=True)


if __name__ == "__main__":
    main()
