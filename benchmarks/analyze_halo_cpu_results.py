"""Read-only analysis of archived paired halo benchmarks and CPU placement.

Writes an English offline report, scientific plots and source-hashed JSON.
Does not change model/runtime code, run training, or submit cluster work.
"""

import base64
from collections import Counter
from html import escape
import hashlib
import json
import math
from pathlib import Path
import statistics as stats


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "benchmark_results/halo-transport"
OUT = ROOT / "analysis/cpu-transport-2026-11-25"
CASES = ((1, 6114), (2, 6117), (4, 6121), (4, 6131))
PHASES = ("forward_ms", "backward_ms", "gradient_sync_ms", "optimizer_ms", "metric_reduce_ms")
SOURCE_COMMIT = "8554302"
NILS = "69939c991289a5f5901fbd24ef64526a56199716"


def read(path):
    return json.loads(path.read_text())


def describe(gpus, job):
    directory = ROOT / f"full-{gpus}gpu-{job}"
    runs = [read(directory / f"repeat-{i}.json") for i in range(3)]
    published = read(directory / "comparison.json")
    assert all(r["status"] == "completed" and all(v["passed"] for v in r["validation"]) for r in runs)
    assert len({r["controls_sha256"] for r in runs}) == 1
    result = {"job": job, "gpus": gpus, "host": runs[0]["rank_metadata"][0]["hardware"]["hostname"],
              "published": published, "cases": {},
              "rank_masks": [r["cpu_affinity"] for r in runs[0]["rank_metadata"]],
              "controls": runs[0]["controls"], "source_sha256": {}}
    for path in sorted(directory.glob("*")):
        if path.is_file():
            result["source_sha256"][str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for backend in ("p2p", "physicsnemo"):
        critical, all_rank = [], []
        medians = []
        critical_ids = []
        for run in runs:
            case = run["cases"][backend]
            assert len(case["records"]) == 50
            medians.append(stats.median(row["model_wall_ms"] for row in case["records"]))
            for i, row in enumerate(case["records"]):
                critical.append(case["rank_records"][row["critical_rank"]][i])
                critical_ids.append(row["critical_rank"])
            all_rank.extend(row for rank in case["rank_records"] for row in rank)
        assert medians == published["per_launch_model_wall_medians_ms"][backend]
        means = {k: stats.fmean(row[k] for row in critical) for k in ("model_wall_ms", *PHASES)}
        means["wrapper_ms"] = means["model_wall_ms"] - sum(means[k] for k in PHASES)
        assert means["wrapper_ms"] >= 0
        result["cases"][backend] = {
            "launch_medians_ms": medians, "median_of_launch_medians_ms": stats.median(medians),
            "critical_rank_means_ms": means,
            "all_rank_optimizer_median_ms": stats.median(row["optimizer_ms"] for row in all_rank),
            "critical_rank_counts": dict(Counter(critical_ids)),
            "peak_allocated_gib": max(r["cases"][backend]["peak_allocated_bytes_per_gpu"] for r in runs) / 2**30}
    p, n = result["cases"]["p2p"], result["cases"]["physicsnemo"]
    result["reported_backend_ratio"] = p["median_of_launch_medians_ms"] / n["median_of_launch_medians_ms"]
    assert math.isclose(result["reported_backend_ratio"], published["physicsnemo_speedup_over_p2p"], rel_tol=1e-12)
    result["paired_backend_ratios"] = [a / b for a, b in zip(p["launch_medians_ms"], n["launch_medians_ms"])]
    result["median_paired_backend_ratio"] = stats.median(result["paired_backend_ratios"])
    result["candidate_order"] = [r["order"] for r in runs]
    return result


def build_data():
    rows = [describe(*case) for case in CASES]
    old, new = rows[2:]
    matches = {k: value == new["controls"][k] for k, value in old["controls"].items() if k != "source_hashes"}
    assert all(matches.values()), "Model/input/partition controls changed in the setup comparison"
    repairs = []
    for path in sorted((ROOT / "full-4gpu-6131").glob("cpu-launch-*-rank-*.json")):
        audit = read(path)
        assert all(c["passed"] for c in audit["checkpoints"])
        changes = [{"stage": c["stage"], "main_before": c["main_before"],
                    "repaired_threads": len(c["repaired_thread_ids"])}
                   for c in audit["checkpoints"] if c["repaired_thread_ids"]]
        assert [c["stage"] for c in changes] == ["before_torch_import"]
        repairs.append({"file": str(path.relative_to(REPO)), "rank": audit["rank"],
                        "target": audit["rank_cpu_ids"], "changes": changes})
    assert len(repairs) == 12
    placement = new["controls"]["cpu_placement"]
    assert sorted(cpu for rank in placement["rank_cpu_ids"] for cpu in rank) == placement["job_cpu_ids"]
    topology = (ROOT / "full-4gpu-6131/cpu-topology.txt").read_text().splitlines()[1:]
    cpu_numa = {int(row.split()[0]): int(row.split()[1]) for row in topology if row.strip()}
    rank_numa = [sorted({cpu_numa[cpu] for cpu in ids}) for ids in placement['rank_cpu_ids']]
    before_after = {b: {"old_ms": old["cases"][b]["median_of_launch_medians_ms"],
                       "new_ms": new["cases"][b]["median_of_launch_medians_ms"],
                       "observed_ratio": old["cases"][b]["median_of_launch_medians_ms"] / new["cases"][b]["median_of_launch_medians_ms"]}
                    for b in ("p2p", "physicsnemo")}
    return {"checked_on": "2026-11-25", "archive_commit": SOURCE_COMMIT, "nils_commit": NILS,
            "runs": rows, "old_new_common_controls_equal": matches,
            "new_batch_entry": read(ROOT / "full-4gpu-6131/batch-cpu-mask.json"),
            "new_rank_repairs": repairs, "new_rank_numa_nodes": rank_numa, "before_after": before_after,
            "interpretation": {
                "cpu_fact": "Old main-thread masks shared two logical CPUs; new ranks have disjoint pairs covering eight CPUs.",
                "necessity": "Fixed pairs are an explicit experimental policy, not a requirement of GraphEFM or PhysicsNeMo.",
                "no_runtime_repairs": "Only initial rank placement changed masks in all 12 full-run rank reports.",
                "cpu_causality_limit": "Launch method, placement, environment policy and node/topology changed together; fixed-pair necessity is not isolated.",
                "backend_fact": "Both backends use identical new placement. A small observed PN advantage survives.",
                "nils_limit": "No raw Nils initial/post-init affinity or package metadata is committed; no special affinity fix appears in his driver.",
                "nccl_lead": "Installed package is 2.29.3. Version-tagged upstream init source saves affinitySave but sets comm->cpuAffinity at exit; verify the actually loaded library and pre/post-init masks before declaring root cause."}}


def plots(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    old, new = data["runs"][2:]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    colors = {"p2p": "#365f8a", "physicsnemo": "#28725a"}
    for i, backend in enumerate(colors):
        pos = np.array([0, 1]) + (i - .5) * .28
        vals = [r["cases"][backend]["median_of_launch_medians_ms"] for r in (old, new)]
        axes[0].bar(pos, vals, .26, label=backend, color=colors[backend])
        for x, value, run in zip(pos, vals, (old, new)):
            axes[0].text(x, value + 35, f"{value:,.0f}", ha="center", fontsize=9)
            axes[0].scatter([x] * 3, run["cases"][backend]["launch_medians_ms"], color="black", s=10, zorder=3)
    axes[0].set_xticks([0, 1], ["6121: old setup", "6131: verified placement"])
    axes[0].set_ylabel("Training step (ms): median of launch medians")
    axes[0].set_ylim(0, 2250)
    axes[0].legend(frameon=False)
    axes[0].set_title("Large setup-level improvement", loc="left", weight="bold")
    ratios = new["paired_backend_ratios"]
    axes[1].bar(range(3), [(r - 1) * 100 for r in ratios], color="#28725a", width=.55)
    for i, ratio in enumerate(ratios):
        axes[1].text(i, (ratio - 1) * 100 + .3, f"{100 * (ratio - 1):.2f}%", ha="center")
    axes[1].set_xticks(range(3), ["P2P first", "PN first", "P2P first"])
    axes[1].set_ylabel("Observed PN throughput gain over paired P2P (%)")
    axes[1].set_ylim(0, 12)
    axes[1].set_title("Smaller backend effect; order may matter", loc="left", weight="bold")
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.text(.06, -.02, "Left: before/after includes launch, placement and node/topology changes. Right: identical CPU policy within each pair.\nThree paired launches per full job; loading and H2D are excluded from this named timer.", fontsize=9)
    fig.tight_layout(w_pad=3)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"setup-versus-backend.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    rows = [(r, b) for r in (old, new) for b in colors]
    bottom = np.zeros(4)
    phase_names = (*PHASES, "wrapper_ms")
    phase_colors = ("#365f8a", "#b27539", "#28725a", "#7969a3", "#777777", "#b9bdc3")
    for phase, color in zip(phase_names, phase_colors):
        values = np.array([r["cases"][b]["critical_rank_means_ms"][phase] for r, b in rows])
        ax.bar(range(4), values, bottom=bottom, label=phase.removesuffix("_ms").replace("_", " "), color=color, width=.6)
        bottom += values
    ax.set_xticks(range(4), ["6121 P2P", "6121 PN", "6131 P2P", "6131 PN"])
    ax.set_ylabel("Mean cached model wall step (ms)")
    ax.set_title("Forward and backward account for most of the observed reduction", loc="left", weight="bold")
    ax.legend(ncol=3, frameon=False, fontsize=9)
    ax.set_ylim(0, 2500)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    fig.text(.09, -.015, "Means over 150 measured steps/backend. Components follow one critical rank per step; they are not independently maximized phases.\nIntervals include CPU scheduling and communication waits, not just GPU arithmetic. Setup-level comparison is not a pure CPU ablation.", fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"critical-rank-phases.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure(name, caption):
    encoded = base64.b64encode((OUT / f"{name}.png").read_bytes()).decode()
    return f'<figure><img style="max-width:100%;height:auto" src="data:image/png;base64,{encoded}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'


def render_section(table):
    path = OUT / "analysis.json"
    if not path.exists():
        return ""
    data = read(path)
    old, new = data["runs"][2:]
    prefix = f"https://github.com/maximilianfricke/baltic-sea-forecasting/blob/{SOURCE_COMMIT}/"
    comparisons = table(("Full job / GPUs", "CPU policy", "P2P ms", "PN ms", "PN throughput gain"), [
        (str(r["job"]) + f" / {r['gpus']}", "disjoint pairs" if r["job"] == 6131 else "inherited mask",
         f"{r['cases']['p2p']['median_of_launch_medians_ms']:,.2f}",
         f"{r['cases']['physicsnemo']['median_of_launch_medians_ms']:,.2f}",
         f"{100 * (r['reported_backend_ratio'] - 1):.2f}%") for r in data["runs"]],
        "halo-cpu-summary", highlights={(i, j): ("key", "Cached model timer; each job has its own policy and allocation.") for i in range(4) for j in (2, 3, 4)})
    pairs = table(("Repeat / order", "P2P ms", "PN ms", "PN gain"), [
        (f"{i + 1}: {'P2P first' if i % 2 == 0 else 'PN first'}", f"{p:.2f}", f"{n:.2f}", f"{100 * (p / n - 1):.2f}%")
        for i, (p, n) in enumerate(zip(new["cases"]["p2p"]["launch_medians_ms"], new["cases"]["physicsnemo"]["launch_medians_ms"]))],
        "halo-paired-launches", highlights={(i, j): ("best", "PN was faster within this matched pair; three launches do not settle order effects.") for i in range(3) for j in (2, 3)})
    phase_rows = []
    for r in (old, new):
        for b in ("p2p", "physicsnemo"):
            m = r["cases"][b]["critical_rank_means_ms"]
            phase_rows.append((f"{r['job']} / {b}", *[f"{m[k]:.2f}" for k in (*PHASES, "model_wall_ms")]))
    phase_table = table(("Job / backend", "Forward", "Backward", "Grad sync", "Optimizer", "Metrics", "Wall ms"), phase_rows,
                        "halo-critical-phases", highlights={(i, j): ("key", "Means use the wall-critical rank at each measured step.") for i in range(4) for j in (1, 2, 6)})
    placement = table(("Rank", "CPU IDs", "NUMA node", "Later mask repairs"), [
        (i, ", ".join(map(str, ids)), ", ".join(map(str, data['new_rank_numa_nodes'][i])), "0 across all three launches")
        for i, ids in enumerate(new["controls"]["cpu_placement"]["rank_cpu_ids"])], "halo-placement",
        highlights={(i, 1): ("key", "Distinct CPU pairs cover the verified eight-CPU allocation.") for i in range(4)})
    p = new["published"]
    return f'''<div id="cpu-transport-review">
<h3>3.5 CPU placement and the controlled transport results</h3>
<p><strong>Latest evidence: jobs 6114, 6117, 6121 and 6131, pulled at {SOURCE_COMMIT}.</strong> The large improvement is a setup-level change affecting both backends. A smaller PhysicsNeMo advantage remains within matched pairs. The older live-input AR4 tables above are historical experiments, not the latest scaling curve.</p>
{comparisons}
<p>The official statistic is the ratio of medians of three per-launch step medians. Each launch has 10 warmup and 50 measured updates per backend. Inputs are four pre-built CPU batches; Zarr reading, H2D and pre-step barriers are outside this timer. Forecasts, loss, gradients and two unmasked AdamW updates pass the paired equivalence gate.</p>
{figure('setup-versus-backend', 'Setup change and backend change are different comparisons. Dots on the left are the three launch medians; the right plot exposes per-pair variation and candidate order.')}
<p><strong>Observed setup change:</strong> P2P 1,974.17 → 759.45 ms (2.60×); PN 1,805.69 → 734.87 ms (2.46×). Both improve substantially. Model configuration, graph, partition, weights, exact cached input hashes, deterministic policy and checkpoint replay match. The setup also changed launch method, CPU placement, OpenMP affinity policy and node/topology. Under the user's assumption of negligible node performance differences, CPU/startup placement is the leading explanation; this is not an isolated fixed-pair experiment.</p>
<details><summary>Forward/backward and optimizer evidence</summary>{phase_table}
{figure('critical-rank-phases', 'Phase means follow the rank with the maximum model wall time at each step, avoiding sums of unrelated rank maxima.')}
<p>Most of the reduction lies in forward and backward, where host launch delays and halo waits are included. The median optimizer interval across all ranks falls from 19.05 to 6.81 ms for P2P and from 18.87 to 0.67 ms for PN. The parameter set is unchanged. These intervals are not kernel-only measurements, so this does not prove an optimizer arithmetic optimization.</p></details>
<h4>Why does CPU placement matter, and is fixed pinning necessary?</h4>
<p>There are three separate decisions: Slurm reserves CPUs, a process/thread has an allowed affinity mask, and the OS schedules that thread within its mask. Our job contains one Slurm task; torchrun creates four child ranks inside it. Slurm does not automatically assign distinct CPU pairs to those children. Linux children inherit affinity across fork/exec. With a healthy shared eight-CPU mask, fixed pairs need not be required. With all four main threads restricted to the same two CPUs, unused reserved capacity cannot help them.</p>
<p><strong>The requirement is sufficient effective CPU capacity. Disjoint pairs are our explicit, reproducible policy, not a mathematical requirement of GraphEFM or PhysicsNeMo.</strong> GPU kernel dispatch, packing and communication progress still need host execution even with zero DataLoader workers. Delayed ranks can make other ranks wait repeatedly during AR4 and checkpoint replay.</p>
{placement}
<p><strong>New trace fact:</strong> batch entry and the srun launcher already see CPUs 10–17. The only mask change in all 12 rank reports is our initial placement before importing torch. No later recorded checkpoint repairs a mask after imports, thread setup, CUDA/NCCL, cache setup or warmup. This supports simplifying the eventual implementation to placement plus verification. It does not establish which component narrowed job 6121, whose pre-init masks were never captured.</p>
<details><summary>A version-specific NCCL lead and the missing causal test</summary>
<p>The package archive lists <code>nvidia-nccl-cu12==2.29.3</code>. In the corresponding upstream <code>initTransportsRank</code> source, the original affinity is saved, GPU-local affinity is applied, and the exit path applies <code>comm-&gt;cpuAffinity</code> again instead of the saved value. That is a concrete mechanism capable of retaining a narrow mask in the executing thread. Group initialization can execute work in the caller. This matches the kind of symptom seen in 6121, but the actually loaded NCCL version and pre/post-init masks must be measured before calling it the proven cause.</p>
<p>Our initialization passes <code>device_id</code> and eagerly connects NCCL. Nils initializes without that argument and then calls a barrier. The invocation/thread path and the intersection of allocated CPUs with GPU-local CPUs may therefore matter. The 6131 allocation is entirely on CPU NUMA node 1 while the GPUs report node 0; no later automatic narrowing was observed. This is a topology-specific hypothesis, not proof that NCCL or Slurm is universally broken.</p>
<p>A short probe should record batch/worker masks before imports, after thread setup, after CUDA setup, after eager or lazy NCCL creation and after the first collective; also record <code>torch.cuda.nccl.version()</code> and per-thread masks. Run eager/lazy cases in fresh processes within the same allocation. Separately compare a shared full CPU pool with disjoint pairs under the same allocation. These tests distinguish inherited/startup narrowing, NCCL behavior and the necessity of fixed pairs.</p></details>
<h4>Why does Nils not need our workaround?</h4>
<p><strong>His code does not contain a special CPU-affinity fix either.</strong> Both use torchrun, NCCL and one PyTorch CPU compute thread per rank. Nils records <code>initial_affinity</code> and <code>cpu_affinity</code>, but his raw metadata is not committed, so his actual allocation and masks remain unknown. A healthy inherited mask could make explicit pinning unnecessary. PhysicsNeMo does not automatically fix our host allocation: both candidates in job 6121 shared the same restricted masks.</p>
<p>The original global-model training enables deterministic algorithms and has no explicit per-rank CPU pinning. Our archived original-model job 5031 had a normal two-CPU mask for one GPU. Nils also differs in graph size, ownership inheritance, compact CSC layout, deterministic policy, distribution-argument checks and one-GPU reference. His requirements pin the same Torch/CUDA wheel as ours; a different loaded CUDA module alone is not evidence of a different Torch runtime. His 2.42× H64 scaling and our backend ratios answer different questions.</p>
<h4>Does PhysicsNeMo still help when CPU placement is the same?</h4>
<p><strong>Yes, a small advantage is observed in all three new matched pairs.</strong> Both candidates use exactly the same CPU masks. The reported throughput gain is <strong>{100 * (new['reported_backend_ratio'] - 1):.2f}%</strong>, equivalent to <strong>{100 * (1 - 1 / new['reported_backend_ratio']):.2f}% less step time</strong>. Its archived paired-launch bootstrap ratio interval is [{p['paired_launch_bootstrap_ci95'][0]:.4f}, {p['paired_launch_bootstrap_ci95'][1]:.4f}], based on only three independent pairs.</p>
{pairs}
<p>The median of the three individual paired gains is {100 * (new['median_paired_backend_ratio'] - 1):.2f}%; this differs from the published ratio-of-medians estimator. The PN-first launch has the largest gain, so order/carryover remains a candidate explanation for part of the variation. The 14.4% smoke result is not a steady-state result. Evidence supports a modest, workload-specific transport benefit, not a universal 2–3× PhysicsNeMo gain.</p>
<h4>What should be cleaned up after the reference is fixed?</h4>
<ol><li>Centralize the workload and CPU policy; preserve the same reference configuration and recorded hashes.</li>
<li>Keep CPU/Slurm setup in the launcher. After a same-allocation placement test, prefer one initial placement plus assertions over repeated blanket helper-thread repairs. No later repair was needed in 6131.</li>
<li>Keep one training/timing pipeline with a selectable halo transport. The current global monkeypatch is useful for this experiment, but an explicit interface would make future maintenance clearer.</li>
<li>Audit the duplicated distributed training flow against original GraphEFM before considering Nils's original-layer wrapper as a replacement. Model-source identity alone does not prove every distributed training detail.</li>
<li>Separate original-reference, cached-transport and live-input experiments. Retain raw archives and compact provenance; reduce duplicate reporting, not evidence.</li></ol>
<p><strong>Next evidence, in order:</strong> retrieve Nils's initial/post-init affinity and runtime metadata; run the short eager/lazy placement probe; decide whether fixed pairs are needed; then collect matching one-/two-GPU references under the chosen policy. If the small backend difference matters, use at least four counterbalanced paired launches before making it a design decision. Change deterministic/distribution-validation policy only in separately labelled tests. Code cleanup follows this stable experiment definition; it is not itself a performance claim.</p>
<details><summary>Primary sources and traceable local evidence</summary><ul>
<li><a href="{prefix}benchmark_results/halo-transport/full-4gpu-6131/comparison.json">Job 6131 summary</a>; <a href="{prefix}benchmark_results/halo-transport/full-4gpu-6131/batch-cpu-mask.json">batch-entry mask</a>; <a href="{prefix}benchmark_results/halo-transport/full-4gpu-6131/cpu-launch-0-rank-0.json">rank-0 stage trace</a>.</li>
<li><a href="https://github.com/maximilianfricke/baltic-sea-forecasting/blob/{NILS}/dphpc/pnemo/bench.py#L308">Nils's worker and metadata</a>.</li>
<li><a href="https://slurm.schedmd.com/cpu_management.html">Slurm: allocation and binding are separate</a>; <a href="https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html">Linux affinity and inheritance</a>.</li>
<li><a href="https://github.com/NVIDIA/nccl/blob/v2.29.3-1/src/init.cc#L954">NCCL version-tagged initialization</a>; <a href="https://github.com/NVIDIA/nccl/blob/v2.29.3-1/src/init.cc#L1334">exit affinity assignment</a>; <a href="https://github.com/NVIDIA/nccl/blob/v2.29.3-1/src/group.cc#L26">caller execution path</a>.</li></ul>
<p>Every archived input file used here has a SHA256 in the accompanying analysis JSON. Common model/input/partition controls match between 6121 and 6131. Source hashes show the benchmark/runtime setup changed; original model computation files did not. Smoke jobs are functional evidence only.</p></details>
</div>'''


def standalone_table(headers, rows, table_id=None, highlights=None, **kwargs):
    highlights = highlights or {}
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = ""
    for i, row in enumerate(rows):
        cells = []
        for j, value in enumerate(row):
            content = escape(str(value))
            if (i, j) in highlights:
                tone, reason = highlights[i, j]
                content = f'<strong class="value-{tone}" title="{escape(reason)}">{content}</strong>'
            elif j == 0:
                content = f"<strong>{content}</strong>"
            cells.append(f"<td>{content}</td>")
        body += "<tr>" + "".join(cells) + "</tr>"
    return f'<div class="table-wrap"><table id="{table_id or ""}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_data()
    (OUT / "analysis.json").write_text(json.dumps(data, indent=2) + "\n")
    plots(data)
    body = render_section(standalone_table)
    css = "body{font:16px/1.6 system-ui,sans-serif;color:#202428;margin:0 auto;max-width:1080px;padding:24px}h3,h4{line-height:1.3}h3{font-size:27px}h4{font-size:21px;margin-top:32px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #d4dbe1}th{background:#f0f3f5}figure{margin:22px 0}figcaption{font-size:13px;color:#555}details{border:1px solid #d4dbe1;padding:12px;margin:18px 0}summary{cursor:pointer;font-weight:600}.value-best{color:#28725a}.value-worst{color:#a32622}code{background:#f1f3f5;padding:1px 4px}a{color:#245e85}@media print{details{display:block}figure,table{break-inside:avoid}}"
    (OUT / "analysis.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CPU placement and PhysicsNeMo analysis</title><style>' + css + '</style><main>' + body + '</main></html>')
    print(OUT / "analysis.html")


if __name__ == "__main__":
    main()
