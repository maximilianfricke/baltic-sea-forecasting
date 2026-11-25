"""Combine presentation evidence with archived MP runs, preserving timing scopes.

Run with --pptx to extract the source and rebuild the evidence and plots. The
report renderer only reads the resulting JSON and PNGs, using the standard
library. Nils's rounded slide labels are observations, not recovered raw runs.
"""

import argparse
import base64
import csv
import hashlib
from html import escape
import io
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "benchmark_results/model-parallel-real/analysis/pnemo-comparison-2026-11-25"
NILS_COMMIT = "69939c9"
RUN_DIR = REPO / "benchmark_results/model-parallel-real/full/ar4-workers8-prefetch2-4944-4949-4952"
BLUE, GREEN, PURPLE, RED = "#365f8a", "#28725a", "#7969a3", "#a32622"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract_evidence(pptx):
    """Read visible text and preserve the exact source figures, without edits."""
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    figures = {4: ("image13.png", "slide4-balance.png"),
               6: ("image15.png", "slide6-speedup.png"),
               7: ("image17.png", "slide7-checkpoint.png")}
    (OUT / "source-figures").mkdir(parents=True, exist_ok=True)
    slides, images = [], []
    with zipfile.ZipFile(pptx) as archive:
        names = sorted((n for n in archive.namelist()
                        if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                       key=lambda n: int(re.search(r"slide(\d+)", n)[1]))
        for name in names:
            number = int(re.search(r"slide(\d+)", name)[1])
            root = ET.fromstring(archive.read(name))
            slides.append({"slide": number,
                           "text": "\n".join(x.text or "" for x in root.findall(".//a:t", ns))})
            if number in figures:
                source, target = figures[number]
                payload = archive.read("ppt/media/" + source)
                path = OUT / "source-figures" / target
                path.write_bytes(payload)
                images.append({"slide": number, "path": str(path.relative_to(OUT)), "sha256": sha(payload)})
    text = "\n".join(row["text"] for row in slides)
    for required in ("H64 2.42", "H128 3.42", "732", "484", "541", "320", "23,428"):
        assert required in text, f"Source slide text changed: {required}"
    return {"filename": pptx.name, "original_path": str(pptx), "sha256": sha(pptx.read_bytes()),
            "slides": slides, "figures": images, "raw_measurements_available": False,
            "extraction": "Slide text and exact PNGs; chart labels manually verified. No pixel-estimated timings."}


def build_data(pptx):
    import numpy as np

    source = extract_evidence(pptx)
    ours = []
    for gpu, job in ((1, 4944), (2, 4949), (4, 4952)):
        path = RUN_DIR / f"real-mp-max-ar4-ckpt-full-rank-local-{gpu}gpu-{job}.json"
        raw = path.read_bytes()
        run = json.loads(raw)
        w, s = run["workload"], run["summary"]
        assert w["ar_steps"] == 4 and w["num_workers"] == 8 and w["measurement_steps"] == 50
        assert w["deterministic_algorithms"] is True and run["world_size"] == gpu
        ours.append(dict(gpus=gpu, job=job, source=str(path.relative_to(REPO)), sha256=sha(raw),
                         wall_samples_per_second=s["observed_samples_per_second"],
                         model_ms=s["step_ms"]["mean"], total_ms=s["end_to_end_ms"]["mean"],
                         input_ms=s["data_distribution_ms"]["mean"],
                         peak_gib=s["peak_memory_allocated_bytes_per_gpu"] / 2**30,
                         mesh_sizes=[sum(r["partition"]["owned_mesh_nodes_by_level"][i]
                                         for r in run["rank_metadata"]) for i in range(2)]))
    for row in ours:
        row["wall_speedup"] = row["wall_samples_per_second"] / ours[0]["wall_samples_per_second"]
        row["model_speedup"] = ours[0]["model_ms"] / row["model_ms"]
        row["input_share_percent"] = row["input_ms"] / row["total_ms"] * 100
        row["memory_fraction_of_1gpu"] = row["peak_gib"] / ours[0]["peak_gib"]
    # These are rounded labels printed on slides 6/7, not raw measurements.
    nils = {
        "commit": NILS_COMMIT, "graph": "global_hierarchical_graph", "mesh_sizes": [7193, 1812],
        "widths": [
            {"hidden": 64, "speedup_1_to_2": 1.79, "speedup_1_to_4": 2.42,
             "speedup_2_to_4": 1.35, "four_gpu_ci95": [2.38, 2.50], "launches": 3,
             "ci_scope": "launches and four-step cycle blocks", "slide": 6},
            {"hidden": 128, "speedup_1_to_2": 1.82, "speedup_1_to_4": 3.42,
             "speedup_2_to_4": 1.88, "four_gpu_ci95": [3.42, 3.43], "launches": 1,
             "ci_scope": "within one launch only", "slide": 6},
            {"hidden": 256, "speedup_1_to_2": None, "speedup_1_to_4": None,
             "speedup_2_to_4": 1.93, "four_gpu_ci95": None, "launches": 1,
             "ci_scope": "within one launch only; no 1-GPU training baseline", "slide": 6}],
        "checkpoint": [
            {"gpus": 2, "on_ms": 732, "off_ms": 484, "on_gib": 1.62, "off_gib": 5.85, "slide": 7},
            {"gpus": 4, "on_ms": 541, "off_ms": 320, "on_gib": .87, "off_gib": 3.11, "slide": 7}],
        "on_memory_gib": [3.08, 1.62, .87],
        "one_gpu_on_ms": None,
        "one_gpu_off": "Slide reports earlier failure to fit; no matched timing case",
        "four_gpu_off_vs_one_gpu_on": 4.09,
        "halo_counts": {"chunks": 23428, "RCB": 623, "METIS": 386, "Mt-KaHyPar km1": 331},
        "weighted_balance": {"grid_max_mean": [1.336, 1.016], "mesh0_max_mean": [1.007, 1.082],
                             "mesh1_max_mean": [1.009, 1.066], "two_gpu_time_gain_percent": [2, 4],
                             "four_gpu_difference": "No significant difference reported under the report's CI criterion"},
    }
    for row in nils["checkpoint"]:
        row["speedup"] = row["on_ms"] / row["off_ms"]
        row["time_saved_percent"] = 100 * (1 - row["off_ms"] / row["on_ms"])
        row["memory_multiplier"] = row["off_gib"] / row["on_gib"]
    owner_bytes = subprocess.check_output(["git", "show", f"{NILS_COMMIT}:dphpc/pnemo/owners/weighted/metis-k4.npz"], cwd=REPO)
    with np.load(io.BytesIO(owner_bytes)) as maps:
        sizes = [len(maps[k]) for k in ("grid", "mesh_0", "mesh_1")]
        assert sizes == [42348, *nils["mesh_sizes"]]
        nils["owner_evidence"] = dict(path="dphpc/pnemo/owners/weighted/metis-k4.npz", sha256=sha(owner_bytes),
                                     graph_fingerprint=str(maps["graph_fingerprint"]),
                                     counts={k: np.bincount(maps[k], minlength=4).tolist() for k in ("grid", "mesh_0", "mesh_1")})
    data = dict(checked_on="2026-11-25", source=source, ours=ours, nils=nils,
                caveat="Independent experiments, different graphs and input/timing policies. No causal backend speedup is established.")
    (OUT / "comparison-data.json").write_text(json.dumps(data, indent=2) + "\n")
    with (OUT / "observations.csv").open("w", newline="") as handle:
        columns = ("experiment", "hidden", "gpus", "metric", "value", "scope", "source")
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for row in ours:
            for metric in ("wall_speedup", "model_speedup", "model_ms", "total_ms", "peak_gib"):
                writer.writerow(dict(experiment="Max MP", hidden=64, gpus=row["gpus"], metric=metric,
                                     value=row[metric], scope="Live input; one launch; means except whole-window wall speedup", source=row["source"]))
        for row in nils["widths"]:
            for gpu, metric in ((2, "speedup_1_to_2"), (4, "speedup_1_to_4"), (4, "speedup_2_to_4")):
                if row[metric] is not None:
                    writer.writerow(dict(experiment="Nils PhysicsNeMo", hidden=row["hidden"], gpus=gpu, metric=metric,
                                         value=row[metric], scope="Cached model step; rounded slide label; " + row["ci_scope"], source="pnemo-results.pptx slide 6"))
    return data


def build_plots(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    ours, nils = data["ours"], data["nils"]
    for mobile in (False, True):
        suffix = "-mobile" if mobile else ""
        size = (4.0, 4.5) if mobile else (8.5, 4.3)

        fig, ax = plt.subplots(figsize=size, layout="constrained")
        ax.plot([1, 2, 4], [1, 2, 4], ":", color="#929aa0", label="Ideal")
        ax.plot([1, 2, 4], [r["model_speedup"] for r in ours], "o--", color=BLUE, label="Max H64: live model interval")
        for row, color in zip(nils["widths"][:2], (GREEN, PURPLE)):
            y = [1, row["speedup_1_to_2"], row["speedup_1_to_4"]]
            ax.plot([1, 2, 4], y, "o-", color=color, label=f"Nils H{row['hidden']}: cached")
            low, high = row["four_gpu_ci95"]
            ax.errorbar([4], [y[-1]], yerr=[[y[-1]-low], [high-y[-1]]], color=color, capsize=4)
            ax.annotate(f"{y[-1]:.2f}×", (4, y[-1]), xytext=(-5, 10), textcoords="offset points", ha="right", color=color)
        ax.annotate(f"{ours[-1]['model_speedup']:.2f}×", (4, ours[-1]["model_speedup"]), xytext=(-5, -18), textcoords="offset points", ha="right", color=BLUE)
        ax.set(xlabel="GPUs", ylabel="Speedup vs own 1-GPU reference" if mobile else "Speedup over each experiment's 1 GPU", xticks=[1, 2, 4], ylim=(.8, 4.25))
        ax.grid(axis="y", alpha=.2); ax.legend(loc="upper left", fontsize=8 if mobile else 10)
        fig.savefig(OUT / f"scaling{suffix}.png", dpi=190)
        if not mobile: fig.savefig(OUT / "scaling.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=size, layout="constrained")
        labels = ["Max H64\nmodel interval", "Nils H64\ncached", "Nils H128\ncached", "Nils H256\ncached"]
        values = [ours[1]["model_ms"] / ours[2]["model_ms"], *[r["speedup_2_to_4"] for r in nils["widths"]]]
        bars = ax.barh(range(4), values, color=[BLUE, GREEN, PURPLE, "#b27539"], height=.56)
        ax.axvline(1, color="#82909a", linewidth=1); ax.axvline(2, color="#82909a", linestyle=":")
        for bar, value in zip(bars, values):
            ax.text(value + .035, bar.get_y() + bar.get_height()/2, f"{value:.2f}×", va="center", fontsize=10)
        ax.set(yticks=range(4), yticklabels=labels, xlim=(0, 2.25), xlabel="Speedup from 2 to 4 GPUs" if mobile else "Speedup when doubling from 2 to 4 GPUs")
        ax.tick_params(axis="y", labelsize=9 if mobile else 11); ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
        fig.savefig(OUT / f"doubling{suffix}.png", dpi=190)
        if not mobile: fig.savefig(OUT / "doubling.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=size, layout="constrained")
        for row, color in zip(nils["checkpoint"], (BLUE, GREEN)):
            ax.plot([row["on_gib"], row["off_gib"]], [row["on_ms"], row["off_ms"]], "o-", color=color, label=f"{row['gpus']} GPUs")
            for mode, dx, dy in (("on", 7, 9), ("off", -7, -19)):
                ax.annotate(f"{row['gpus']} GPUs, {mode}", (row[mode+"_gib"], row[mode+"_ms"]),
                            xytext=(dx, dy), textcoords="offset points", ha="left" if mode=="on" else "right", fontsize=9 if mobile else 11, color=color)
        ax.set(xlabel="Peak memory per GPU (GiB)" if mobile else "Peak allocated memory per GPU (GiB)", ylabel="Cached training step (ms)", xlim=(.2, 6.7), ylim=(250, 810))
        ax.grid(alpha=.2); ax.legend(fontsize=10)
        fig.savefig(OUT / f"checkpoint-tradeoff{suffix}.png", dpi=190)
        if not mobile: fig.savefig(OUT / "checkpoint-tradeoff.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=size, layout="constrained")
        ax.plot([1, 2, 4], [1, .5, .25], ":", color="#929aa0", label="Ideal 1 / GPUs")
        ax.plot([1, 2, 4], [r["memory_fraction_of_1gpu"] for r in ours], "o-", color=BLUE, label="Max H64")
        mem = nils["on_memory_gib"]
        ax.plot([1, 2, 4], [v/mem[0] for v in mem], "s--", color=GREEN, label="Nils H64")
        ax.set(xlabel="GPUs", ylabel="Memory / own 1-GPU peak" if mobile else "Memory per GPU / own 1-GPU peak", xticks=[1, 2, 4], ylim=(.15, 1.08))
        ax.grid(axis="y", alpha=.2); ax.legend(fontsize=10)
        fig.savefig(OUT / f"memory-scaling{suffix}.png", dpi=190)
        if not mobile: fig.savefig(OUT / "memory-scaling.pdf")
        plt.close(fig)


def embedded_plot(stem, alt):
    def img(suffix, css):
        path = OUT / (stem + suffix + ".png")
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f'<img class="{css}" src="data:image/png;base64,{encoded}" alt="{escape(alt, quote=True)}" loading="lazy">'
    return '<div class="comparison-plot">' + img("", "comparison-desktop") + img("-mobile", "comparison-mobile") + '</div>'


def render_section(table):
    """Return the new results section using the report's table styling helper."""
    path = OUT / "comparison-data.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text()); ours, n = data["ours"], data["nils"]
    for row in ours:
        assert sha((REPO / row["source"]).read_bytes()) == row["sha256"]
    key = lambda reason: ("key", reason)
    setup = [
        ("Model", "AR4, batch 2, decoder L2, prior/posterior L1", "Same architecture settings"),
        ("Ocean / mesh 0 / mesh 1", "42,348 / 8,456 / 2,097", "42,348 / 7,193 / 1,812"),
        ("Numerical policy in timing", "Deterministic algorithms on", "Deterministic algorithms off"),
        ("CPU input", "Rank-local Zarr; 8 workers/rank; prefetch 2; not pinned", "Rank-local memory-mapped raw .pt; 1 worker/rank; prefetch 2; pinned"),
        ("Main timer", "Live wall throughput; separate model CUDA-event interval", "Cached model step, after copy/sync/barrier; excludes loading and H2D"),
        ("Batch selection", "Shuffled windows from the physically repeated store", "First eight samples, grouped into four batches and cycled"),
        ("Worker activity while timing", "Workers remain active and prefetch", "Cached loader is destroyed before timing; separate live cases exist"),
        ("Warmup / measured steps", "10 / 50; one launch per GPU count", "5 / 20; H64 cached: 3 launches, wider/off: 1"),
        ("One-GPU reference", "MP path with rank-local ocean input", "Original GraphEFM; cached, or original full input in live mode"),
        ("Checkpoint replay", "Full replay on 1/2/4 GPUs", "Original 1 GPU uses early stop; distributed uses full replay"),
    ]
    setup_html = table(("Comparison control", "Max message-passing", "Nils PhysicsNeMo"), setup, "pnemo-setup-table",
                       highlights={(i,j):key("Scope difference must be aligned before causal comparison.") for i in (1,2,4,6,7,8,9) for j in (1,2)})
    rows = [
        ("Max H64: observed wall throughput", "1.353×", "1.269×", "0.938×", "1 per case; no independent-run CI"),
        ("Max H64: model interval only", f"{ours[1]['model_speedup']:.3f}×", f"{ours[2]['model_speedup']:.3f}×", f"{ours[1]['model_ms']/ours[2]['model_ms']:.3f}×", "Same archived runs; input excluded from interval"),
        ("Nils H64: cached model step", "1.79×", "2.42×", "1.35×", "3 launches; 4-GPU CI [2.38, 2.50]"),
        ("Nils H128: cached model step", "1.82×", "3.42×", "1.88×", "1 launch; CI [3.42, 3.43] is within-launch only"),
        ("Nils H256: cached model step", "—", "—", "1.93×", "1 launch; no 1-GPU training reference"),
    ]
    marks = {(i,j):key("Speedup against this experiment's own reference, not the other implementation.") for i in range(5) for j in (1,2,3)}
    for i in (0,1): marks[i,3] = ("worst", "Four GPUs are slower than two within this Max experiment.")
    for i in (2,3,4): marks[i,3] = ("best", "Four GPUs improve over two within this Nils experiment.")
    scaling_table = table(("Experiment / metric", "1 to 2 GPUs", "1 to 4 GPUs", "2 to 4 GPUs", "Evidence scope"), rows, "pnemo-scaling-table", highlights=marks)
    cp_rows = [(r["gpus"], r["on_ms"], r["off_ms"], f"{r['speedup']:.2f}×", f"{r['on_gib']:.2f}", f"{r['off_gib']:.2f}", f"{r['memory_multiplier']:.2f}×") for r in n["checkpoint"]]
    cp_marks = {(i,j):key("Rounded labels from Nils slide 7.") for i in range(2) for j in range(1,7)}
    for i in range(2):
        cp_marks[i,2] = ("best", "Less step time when checkpointing is disabled at this fixed GPU count.")
        cp_marks[i,5] = ("worst", "Higher peak memory when checkpointing is disabled at this fixed GPU count.")
    cp_table = table(("GPUs", "On ms", "Off ms", "Speedup", "On GiB/GPU", "Off GiB/GPU", "Memory factor"), cp_rows, "pnemo-checkpoint-table", numeric=tuple(range(7)), highlights=cp_marks)
    memory_rows = [("Max H64", *[f"{r['peak_gib']:.3f}" for r in ours], f"{100*(1-ours[-1]['memory_fraction_of_1gpu']):.1f}%"),
                   ("Nils H64", *[f"{v:.2f}" for v in n["on_memory_gib"]], f"{100*(1-n['on_memory_gib'][-1]/n['on_memory_gib'][0]):.1f}%")]
    memory_marks = {(i,j):key("Compare memory scaling within a row; graphs and retained buffers differ between rows.") for i in range(2) for j in (1,2,3,4)}
    memory_table = table(("Checkpointing on", "1 GPU GiB", "2 GPUs GiB/GPU", "4 GPUs GiB/GPU", "Per-GPU reduction at 4"), memory_rows, "pnemo-memory-table", numeric=(1,2,3,4), highlights=memory_marks)
    partition_rows = [(k, f"{v:,}") for k,v in n["halo_counts"].items()]
    partition_table = table(("Nils 4-GPU partition", "Reported max halo nodes/GPU"), partition_rows, "pnemo-halo-table", numeric=(1,),
                            highlights={(0,1):("worst","Largest reported halo count in this Nils comparison."), (3,1):("best","Smallest reported halo count in this Nils comparison.")})
    base = f"https://github.com/maximilianfricke/baltic-sea-forecasting/blob/{NILS_COMMIT}/"
    return f'''<h3 id="pnemo-comparison">3.4 Combined findings with Nils’s PhysicsNeMo results</h3>
<p><strong>Spatial distribution reduces memory in both implementations. Nils also demonstrates substantially better four-GPU scaling in his experiment, especially at H128/H256.</strong> The new evidence comes from slides 2–7 of <cite>pnemo-results.pptx</cite>, supplied on 25 November. Values below preserve the rounded slide labels. The raw Package E archive is still unavailable here, so no independent reanalysis of its confidence intervals or profiler traces is claimed.</p>
<div class="note warning"><strong>These are independent experiments.</strong> Same AR4 architecture settings and GPU model do not make them a controlled backend comparison. The mesh, deterministic policy, input activity, samples, checkpoint baseline and statistics differ. Each speedup uses its own reference.</div>
{scaling_table}
<figure>{embedded_plot('scaling','Independent scaling curves: Max H64 model interval, Nils H64 and H128 cached model steps')}<figcaption>Input is excluded from our plotted model interval, but our background loaders remain active. Nils’s cached workers have stopped. Only Nils’s published four-GPU confidence intervals appear. H128 has one launch, so its narrow interval does not establish run-to-run stability.</figcaption></figure>
<figure>{embedded_plot('doubling','Incremental speedup from two to four GPUs across the independent experiments')}<figcaption>One means no gain from adding GPUs; two is ideal doubling. Nils’s H256 result is <strong>2 to 4 GPUs only</strong>, not a four-GPU speedup over one. Our plotted ratio uses model-step means; Nils’s labels use ratios of medians.</figcaption></figure>
<p><strong>Input wait alone does not explain the gap.</strong> Our H64 model interval is 1,540 / 1,146 / 1,232 ms on 1 / 2 / 4 GPUs, giving 1.345× and 1.251× model-interval speedups. Removing the visible 2–4% input phase therefore leaves the four-GPU slowdown in place. Background CPU contention can still inflate model intervals even when the next batch is already ready. Cached timing removes that worker activity as well as exposed input time.</p>
<details><summary>Exact setup differences and what the plots can establish</summary>{setup_html}<p>The committed Nils ownership map confirms 7,193 / 1,812 mesh nodes, versus 8,456 / 2,097 in our result metadata. Recreate ownership on the chosen common graph before reusing a partition. Both paths retain the original edge/node MLP formulas and exchange complete required sender features. PhysicsNeMo’s exact graph exchange does not establish that PyG itself requires neighbour sampling: our PyG operators also use all assigned edges.</p><p>Our model interval is a CUDA-event mean from a live run; Nils’s main cached step is a synchronized wall-time median. Both include training work and communication, but they are not identical metrics. His live cases also change original full-store input at K=1 to rank files at K&gt;1.</p></details>
<h4>Checkpointing turns memory savings into faster execution</h4>
{cp_table}
<figure>{embedded_plot('checkpoint-tradeoff','Nils H64 checkpoint trade-off: lower step time requires more memory at fixed GPU count')}<figcaption>Nils slide 7, H64 and Mt-KaHyPar with cached input. Each on/off pair keeps GPU count fixed. Off cases have one launch. Memory points and milliseconds are rounded slide values.</figcaption></figure>
<p>Disabling checkpointing reduces step time by <strong>33.9% on two GPUs</strong> and <strong>40.9% on four</strong>, while peak memory grows by about <strong>3.6×</strong>. His four-GPU, checkpoint-off case takes <strong>320 ms and 3.11 GiB/GPU</strong>. Its reported <strong>4.09×</strong> gain over one GPU combines distribution with avoided recomputation. It does not demonstrate superlinear scaling under an unchanged checkpoint policy. The deck reports that an earlier one-GPU/off run did not fit; it supplies no matched one-GPU/off timing.</p>
<h4>Memory scaling is the strongest common result</h4>
{memory_table}
<figure>{embedded_plot('memory-scaling','Per-GPU memory normalized to each experiment own one-GPU baseline')}<figcaption>Both checkpoint-on experiments approach inverse-GPU memory scaling: about 72% less per device at four GPUs. Absolute memory differences also reflect graph size and buffer lifetimes, so they do not isolate the communication library’s memory cost.</figcaption></figure>
<p>At four GPUs, Nils can spend the saved capacity on removing checkpointing: his <strong>3.11 GiB/GPU without checkpointing</strong> is close to his one-GPU <strong>3.08 GiB with checkpointing</strong>. This is a concrete capacity-to-speed trade-off. A separate wider-model sweep can spend the same capacity on model size instead.</p>
<details><summary>Partition evidence and the limits of the communication claim</summary>{partition_table}<p>The deck’s counts differ by about <strong>71×</strong> between chunks and Mt-KaHyPar. These are halo counts, not measured training-time ratios. The slides do not give the corresponding chunk-runtime comparison.</p><p>For four-GPU METIS, weighting reduces grid-node max/mean imbalance from <strong>1.336 to 1.016</strong>, while mesh-0 imbalance rises from 1.007 to 1.082 and mesh-1 from 1.009 to 1.066. Weighted METIS/Mt-KaHyPar give a reported <strong>2–4% step-time gain on two GPUs</strong>; the deck reports no significant difference on four under its CI criterion. Optimizing a structural count helps define candidates, but runtime remains the decision metric.</p><p>Nils’s framework uses indexed all-to-all; ours posts neighbour P2P operations. Both send selected feature rows and return their gradients. Neither implementation deliberately overlaps local edge compute with halo exchange. NCCL kernel durations can include rank waiting, and his explicit halo timers cover forward/replay, rather than the complete reverse-gradient exchange.</p></details>
<h4>What the combined evidence supports</h4>
<ul><li><strong>Demonstrated:</strong> both implementations reduce per-GPU memory substantially; four GPUs can unlock execution choices that one GPU cannot fit.</li><li><strong>Demonstrated within Nils’s setup:</strong> four-GPU scaling improves with width; H128 reaches 3.42× over one GPU and H256 reaches 1.93× from two to four. Wider cases have only one launch and do not assess forecast accuracy.</li><li><strong>Demonstrated within our setup:</strong> MP2 is faster than MP4 at H64, and excluding visible input does not remove that result.</li><li><strong>Still a hypothesis:</strong> which part of the gap comes from transport, deterministic kernels, graph/ownership, CPU scheduling or the adapter. The slides do not isolate these causes.</li></ul>
<p><a href="{base}README_E.md">Pinned PhysicsNeMo reproduction settings</a> and <a href="{base}dphpc/pnemo/report.py#L735">its reporting method and limits</a>. PPTX SHA256: <code>{data['source']['sha256']}</code>. The extracted slide text, image hashes, rounded observations and derived calculations are preserved in the comparison’s evidence JSON.</p>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pptx", type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_data(args.pptx.resolve())
    build_plots(data)
    print(json.dumps({"evidence": str(OUT / "comparison-data.json"),
                      "max_model_speedups": [r["model_speedup"] for r in data["ours"]],
                      "nils_checkpoint_speedups": [r["speedup"] for r in data["nils"]["checkpoint"]]}, indent=2))


if __name__ == "__main__":
    main()
