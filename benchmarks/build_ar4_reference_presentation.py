"""Extend the existing presentation with the verified original/MP AR4 study.

The established theme and navigation are reused. The earlier AR1 deck is
preserved as a labeled historical appendix, rather than presenting its old
input-bound conclusion as a finding about the new AR4 experiment.
"""

import base64
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO.parent / "njord-message-passing-website"
OUT = REPO / "benchmark_results/model-parallel-real/analysis/ar4-original-vs-mp-4981-4944-4949-4952"
DECK = SITE / "benchmark-analysis-presentation.html"
HISTORY = SITE / "benchmark-analysis-ar1-history.html"


def main():
    if not HISTORY.exists():
        old = DECK.read_text()
        old = old.replace("<title>Njord Multi-GPU Benchmark Analysis</title>",
                          "<title>Historical AR1 Benchmark Analysis — 18 November</title>")
        old = old.replace("Njord · distributed GraphEFM", "Historical AR1 · 18 November")
        old = old.replace("What the multi-GPU results", "What the historical AR1 results", 1)
        old = old.replace("<div class=\"keyboard-hint\">", "<div class=\"keyboard-hint\"><a style=\"color:inherit\" href=\"benchmark-analysis-presentation.html\">Current AR4 comparison</a> · ", 1)
        HISTORY.write_text(old)
    old = HISTORY.read_text()
    css = re.search(r"<style>(.*?)</style>", old, re.S).group(1)
    js = re.search(r"<script>(.*?)</script>", old, re.S).group(1)
    js = js.replace("behavior: smooth ? 'smooth' : 'auto'",
        "behavior: smooth && !window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'smooth' : 'auto'")
    analysis = json.loads((OUT / "comparison.json").read_text())
    rows = analysis["rows"]
    original, one, two, four = rows

    def esc(value):
        return html.escape(str(value), quote=True)

    def cards(items, columns=3):
        return f'<div class="analysis-cards cols-{columns}">' + "".join(
            f'<article class="card {kind}"><span class="card-kicker">{kicker}</span>'
            f'<h3>{title}</h3><p>{text}</p></article>'
            for kind, kicker, title, text in items) + '</div>'

    def table(headers, body):
        return '<table class="data-table analysis-table"><thead><tr>' + ''.join(
            f'<th scope="col">{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
            '<tr>' + ''.join(f'<td>{value}</td>' for value in row) + '</tr>' for row in body) + '</tbody></table>'

    def plot(name, alt):
        content = base64.b64encode((OUT / "plots" / f"{name}.png").read_bytes()).decode()
        return f'<figure class="analysis-plot"><img src="data:image/png;base64,{content}" alt="{esc(alt)}"></figure>'

    slides = []

    def slide(label, title, body, note="", paper=False, extra=""):
        slides.append(f'<section class="slide {"paper-slide" if paper else ""} {extra}" '
            f'data-notes="{esc(note)}"><div class="slide-content"><div class="topline">'
            f'<span class="section-label">{label}</span><span class="run-tag">AR4 · H64 · B2</span></div>'
            f'<h2>{title}</h2>{body}</div><span class="slide-number"></span></section>')

    slides.append('''<section class="slide analysis-hero" data-notes="Lead with the measured outcomes. Two speedup denominators are deliberately shown. All four cases are one independent run each.">
      <div class="slide-content"><span class="section-label">Njord · original versus distributed</span>
      <h1>Two GPUs are fastest.<br><span class="accent">Four use less memory.</span></h1>
      <p class="lede">The original GraphEFM reference now completes the comparison. Same scientific model, four forecast steps, checkpointing, eight workers per rank.</p>
      <div class="analysis-kpis"><article class="metric-card"><span>2 GPUs vs original</span><strong>1.59×</strong><p>Measured throughput</p></article>
      <article class="metric-card"><span>2 GPUs vs MP 1</span><strong>1.35×</strong><p>Internal strong scaling</p></article>
      <article class="metric-card"><span>4-GPU memory</span><strong>−72%</strong><p>Per GPU vs original</p></article></div>
      <p class="fine">22 November 2026 · Jobs 4981 / 4944 / 4949 / 4952 · GTX 1080 Ti · 10 warmup + 50 measured steps per case</p>
      </div><span class="slide-number"></span></section>''')

    slide("01 · Scope", "The full scientific workload stays the same", cards([
        ("good", "Model", "645,020 parameters", "H64 throughout; two decoder processor layers; one layer in each encoder. The model was not made smaller for distribution."),
        ("good", "Training", "Four autoregressive steps", "Two input states; each forecast feeds the next. Gradients pass through the rollout. WMSE + KL, beta 1; CRPS weight 0."),
        ("info", "Data", "42,348 ocean nodes", "8,456 fine + 2,097 coarse mesh nodes. Global batch two at every GPU count; FP32 and identical AdamW settings."),
    ]) + '<div class="forecast-chain"><span>History: 2 states</span><b>→</b><span>t + 1</span><b>→</b><span>t + 2</span><b>→</b><span>t + 3</span><b>→</b><span>t + 4</span></div>',
        "This is the complex AR4 ELBO workload. CRPS fine-tuning and ensemble inference are separate workloads; none is claimed measured here.")

    slide("02 · Controls", "Comparable model; explicit execution differences", table(
        ["Case", "Input", "Workers total", "Checkpoint replay"], [
            ["Original 1", "Full-grid Zarr", "8", "Early stop on"],
            ["MP 1", "Ocean shard", "8", "Full replay"],
            ["MP 2", "Ocean shards", "16", "Full replay"],
            ["MP 4", "Ocean shards", "32", "Full replay"],
        ]) + '<p>All runs: <strong>8 workers/rank, prefetch 2/worker</strong>, same GPU model and physical node, Python/PyTorch/CUDA, shuffle seed, optimizer and measured sample count.</p><p class="fine">Original source: frozen global-model 76b30d7. All 13 checked original model/loader files match in both benchmark commits. The original-to-MP comparison includes the differences above.</p>',
        "Identical model semantics do not mean identical input layout, tensor execution or checkpoint replay policy. A standalone harness times both paths; this is not the complete Lightning lifecycle.")

    slide("03 · Correctness", "The measured loss trajectory is preserved", cards([
        ("good", "Original ↔ MP 1", "50 of 50 exactly equal", "The scalar loss equals the original at every measured optimizer step; mean loss is 15,698,777.445 in both cases."),
        ("good", "Original ↔ MP 2 / 4", "Below 2.2 × 10⁻⁷", "Maximum per-step relative loss difference. Same first-batch fingerprint and matching preflight loss."),
        ("warn", "Scope of evidence", "Numerical training check", "This does not establish forecast skill, long-run convergence or equality of every GPU parameter/gradient element."),
    ]), "Previous small-graph NCCL tests compare forecasts, gradients and AdamW; a full-H64 CPU test exists. These 50 loss records are additional evidence, not a replacement.", paper=True)

    slide("04 · Overview", "Throughput and memory answer different questions", plot("01-overview", "Four cases: original one GPU, distributed one, two and four GPUs; two GPUs fastest, four smallest per-device memory")
        + '<p class="fine">One independent run per case. Peak allocated memory includes all live buffers. Full-resolution PNG and vector PDF figures accompany the report.</p>')

    slide("05 · Numbers", "Keep both speedup references visible", table(
        ["Case", "Samples/s", "vs original", "vs MP 1", "GiB/GPU"],
        [[name, f"{row['observed_samples_per_second']:.3f}", f"{row['speedup_vs_original']:.3f}×",
          '—' if i == 0 else f"{row['speedup_vs_mp1']:.3f}×", f"{row['peak_allocated_gib_per_gpu']:.3f}"]
         for i, (name, row) in enumerate(zip(("Original 1", "MP 1", "MP 2", "MP 4"), rows))])
        + '<p><strong>MP 2:</strong> 67.6% internal parallel efficiency. <strong>MP 4:</strong> 31.7%. Four GPUs deliver 6.2% fewer samples/s than two GPUs in this series.</p>',
        "Throughput counts 100 measured samples over the full measured wall window. Internal efficiency uses MP1 as denominator, never the slower original.")

    slide("06 · Timing", "The model intervals now dominate", plot("02-critical-rank-phases", "Additive timing stack based on a single critical rank per measured step")
        + '<p class="fine">Forward includes halos. Backward includes checkpoint recomputation and reverse halos. The weight-gradient reduction is a separate phase.</p>',
        "The JSON summary maximizes each phase over ranks independently. This stack instead selects the rank with maximum end-to-end time for each step, so its components genuinely add to the total.")

    slide("07 · Phase means", "Backward is the largest measured phase", table(
        ["Mean ms", "Original 1", "MP 1", "MP 2", "MP 4"],
        [[name, *[f"{r['published_phase_mean_ms'][key]:.1f}" for r in rows]] for name, key in (
            ("Input", "data_distribution_ms"), ("Forward", "forward_ms"),
            ("Backward", "backward_ms"), ("Weight sync", "gradient_sync_ms"),
            ("Model step", "step_ms"), ("End-to-end", "end_to_end_ms"))])
        + '<p class="fine">These reproduce the published summary means. Multi-rank phase maxima can come from different ranks; do not add them. Model step and end-to-end are nested totals.</p>')

    slide("08 · Original → MP 1", "A faster run is not an isolated halo speedup", cards([
        ("good", "Observed", "+17.5% throughput", "Mean end-to-end: 1,877.7 → 1,599.9 ms. Input improves by 58.5 ms; the model-step interval by 219.3 ms."),
        ("info", "Interpretation", "One GPU has no remote halo", "Input layout, checkpoint policy and execution structure differ. The measured gain belongs to that bundle of changes."),
        ("warn", "Diagnostic clue", "AdamW: 17.95 → 0.63 ms", "Same optimizer and parameter count. CPU launch gaps or background work may affect the intervals; trace before attributing a cause."),
    ]), "The original log warns about 8 workers with a suggested CPU maximum of 2. This supports investigating CPU pressure; it does not quantify its contribution.")

    slide("09 · Two → four", "Most of the extra time is inside backward", plot("03-phase-deltas", "Critical-rank phase changes: one-GPU refactoring saves time; two-to-four GPUs increases backward and forward intervals")
        + '<p><strong>Two → four:</strong> +63.0 ms backward, +11.9 ms forward, +7.2 ms weight sync; input saves 8.0 ms. Net: <strong>+78.1 ms per global batch</strong>.</p>',
        "The location of the penalty is measured. Halo latency, replay cost, small kernels, CPU starvation and boundary imbalance are candidates, not yet isolated causes.")

    slide("10 · Input", "The old 90% data conclusion does not apply", plot("07-input-diagnostics", "Visible input fraction falls from 6.3 percent original to 2.3 percent on four GPUs; input substage residual remains")
        + '<p class="fine">The earlier AR1 experiment used a different loader policy. Both forecast length and input setup changed; AR4 alone cannot be credited for the lower fraction.</p>')

    slide("11 · Workers and prefetch", "Low queue wait does not mean free decoding", cards([
        ("info", "What is measured", "About 6–9 ms queue wait", "Workers can prepare later batches during model execution. This foreground wait excludes much background read/decode service time."),
        ("warn", "What is missing", "5–40 ms input remainder", "The outer input timer also spans return/cleanup boundaries. Its remainder is not measured as disk I/O or halo traffic."),
        ("good", "Decision", "Freeze 8 / prefetch 2 first", "Removing only MP2's visible input, holding model intervals fixed, offers about 3.2%. CPU contention could affect model time too."),
    ]), "Do not sweep more workers just because a small data timer exists. Use a bounded cached-batch diagnostic to test whether background CPU work affects model launch timing.")

    slide("12 · Stability", "The ordering persists across all five blocks", plot("05-step-stability", "Raw and smoothed timing traces with consecutive ten-step block means")
        + '<p class="fine">Original E2E p95: 2.549 s; MP1: 1.862 s; MP2: 1.476 s; MP4: 1.563 s. Fifty steps within one job are not fifty independent replications.</p>',
        "Original within-run CV is 15.4%, versus MP1 4.9%, MP2 9.6%, MP4 10.3%. Repeat independent jobs before assigning a causal algorithmic improvement.")

    slide("13 · Loss evidence", "All four measured training curves overlap", plot("06-loss-agreement", "Measured loss curves overlap; relative differences remain at roundoff scale")
        + '<p class="fine">This repeated physical-data proxy measures execution, not forecast skill. Zero error is drawn at 10⁻¹² only to make a logarithmic chart possible.</p>')

    slide("14 · Resource trade-offs", "Choose wall time, capacity or GPU time explicitly", plot("04-scaling-and-resource-cost", "Two speedup denominators and allocated GPU-seconds per sample")
        + '<p class="fine">MP1 minimizes GPU-seconds/sample. MP2 minimizes elapsed training-step time. MP4 minimizes per-device peak memory. GPU-seconds here exclude startup and are not the course billing formula.</p>')

    slide("15 · Memory caveat", "Align buffer lifetime before a model-only memory claim", cards([
        ("good", "Measured", "3.408 → 0.965 GiB/GPU", "Original to four GPUs: 71.7% lower peak allocated memory. The capacity benefit is substantial despite limited H64 scaling."),
        ("warn", "Harness difference", "A retained preflight batch", "The MP runner keeps its first GPU batch throughout measurement. The original replaces it. One global batch is about 93.1 MiB."),
        ("info", "Next reference", "Match what stays alive", "MP1 is 89.4 MiB above the original peak. The retained batch can influence that gap; do not subtract it and claim a corrected peak."),
    ]), "Peak memory includes buffers and tensors, not only model parameters. This is source inspection, not a measured allocation attribution.", paper=True)

    slide("16 · Partition evidence", "Equal grid counts are not equal boundary cost", plot("08-partition-footprint", "Four ranks have balanced node and edge counts but uneven halo entries")
        + '<p class="fine">Halo entries are static remote-sender counts within relations, not measured bytes or latency. Similar elapsed rank times can include waiting for another rank.</p>')

    slide("17 · Immediate experiments", "Separate the causes before changing the model", table(
        ["Control", "Model", "Input", "Early stop"], [
            ["A", "Original", "Full grid", "On"],
            ["B", "Original", "One-rank shard", "On"],
            ["C", "Original", "One-rank shard", "Off"],
            ["D", "MP 1", "One-rank shard", "Off"],
        ]) + '<p>Keep H64/L2/B2/AR4 and 8/P2 fixed. Align preflight lifetimes and record actual CPU affinity. Repeat selected cases independently, rotating their order.</p>',
        "These are planned controls, not results already measured. A→B isolates input layout, B→C checkpoint policy, C→D execution structure. Match CPUs and sample/latent sequence.")

    slide("18 · Profiling plan", "Measure the two-versus-four critical timeline", cards([
        ("info", "First", "CPU + CUDA trace per rank", "Separate useful kernels, launch gaps, replay, pack/transfer/wait and gradient reduction. Use short traces beside an untraced timing reference."),
        ("info", "Then", "Bounded compute-only control", "Reuse the same prepared batches after stopping loader work. Test whether background decoding changes the model intervals."),
        ("good", "Exit condition", "Attribute the +78 ms", "Choose CPU placement, buffer reuse, fusion or halo overlap only after the dominant contribution is measured."),
    ]), "Current grouped P2P can transfer among peers concurrently, but local MLP work is not deliberately scheduled before consuming halos. CPU wait duration alone is not network latency.")

    slide("19 · Team integration", "Use the complementary work; resolve the overlap", cards([
        ("warn", "Alex", "Alternative halo runtime", "Mesh processing is sharded; grid/G2M/M2G remain replicated. Align interfaces. Archived overlap trials ran out of GPU memory."),
        ("info", "Lassi · partitioning", "External owner arrays", "Port the owner-file interface to current AR4. Bind graph order and partition hash to regenerated matching data shards."),
        ("info", "Nils + Lassi · profiling", "Reuse attribution methods", "Adapt stage/replay hooks to the distributed path. Their other hardware and workloads prevent direct timing comparisons."),
    ]), "Branch evidence is in TEAM_BRANCH_REVIEW_2026-11-22.md. No team branches were merged and no new cluster jobs were launched by this analysis.")

    slide("20 · Decision", "Use two GPUs for H64 speed; measure what limits four", cards([
        ("good", "Current choice", "Two GPUs for wall time", "Best observed throughput, preserved loss trajectory and about half the per-device memory. Keep the one-GPU case as the scaling denominator."),
        ("info", "Next step", "Controls + targeted traces", "Establish independent repetitions and align input/checkpoint/buffer policy. Then profile and integrate the team's partitions."),
        ("warn", "Later", "Width, depth and batch sweeps", "Vary one factor at a time after attribution. Bigger models characterize a new regime; they do not explain this result."),
    ]), "This is an evidence-based choice for this H64 AR4 series, not a universal assertion that two GPUs are best or multi-GPU does not scale.", paper=True)

    slide("21 · Evidence and appendix", "The analysis is reproducible", '<p>Four immutable result JSONs, checked source identities, raw rank records, full numerical checks and both speedup references.</p>'
        + '<div class="analysis-links"><a href="analysis.md">Detailed report</a><a href="comparison.json">Machine-readable comparison</a><a href="metrics.csv">All phase means (CSV)</a><a href="critical-rank-phases.csv">Additive phases (CSV)</a><a href="HISTORY_LINK">Historical AR1 presentation</a></div>'
        + '<p class="fine">Scientific plots are available as PNG and vector PDF. Navigation: arrow keys / space, Home and End. Earlier AR1 input-bound conclusions remain in the labeled historical appendix.</p>',
        "One run per case; no claim of independent-repetition confidence intervals. Input mode and checkpoint policy differ intentionally in the original comparison.")

    extra_css = '''
    /* Current AR4 comparison, retaining the original forest/paper theme. */
    html { scrollbar-width:none; }
    body { scroll-snap-type:y mandatory; }
    html::-webkit-scrollbar { display:none; }
    :root { --title-size: clamp(2rem, 4.6vw, 4.5rem); --h2-size: clamp(1.55rem, 3.05vw, 3rem); --body-size: clamp(.9rem, 1.17vw, 1.1rem); --small-size: clamp(.72rem, .88vw, .86rem); --slide-padding: clamp(1.1rem, 3.3vw, 3.5rem); }
    h2 { max-width: 32ch; line-height: 1.06; }
    .analysis-hero h1 { max-width: 23ch; }
    .analysis-hero .lede { max-width: 76ch; }
    .analysis-kpis, .analysis-cards { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:clamp(.65rem,1.4vw,1.15rem); min-height:0; }
    .analysis-kpis .metric-card strong { font-size:clamp(1.9rem,4vw,3.5rem); }
    .analysis-cards .card { padding:clamp(.9rem,1.7vw,1.4rem); }
    .analysis-cards .card p { font-size:1em; line-height:1.43; }
    .analysis-cards .card h3 { font-size:clamp(1.05rem,1.7vw,1.45rem); }
    .paper-slide .card-kicker { color:#53645f; }
    .analysis-table th, .analysis-table td { padding:clamp(.48rem,1.05vw,.85rem); }
    .analysis-table tbody tr:nth-child(3) { background:rgba(101,214,199,.07); }
    .analysis-table th { white-space:normal; }
    .analysis-plot { flex:0 1 auto; min-height:0; max-height:58vh; border-radius:16px; background:white; padding:.3rem; display:grid; place-items:center; }
    .analysis-plot img { width:auto; max-width:100%; max-height:54vh; object-fit:contain; display:block; }
    .forecast-chain { display:flex; gap:.6rem; align-items:center; justify-content:center; flex-wrap:wrap; }
    .forecast-chain span { padding:.6rem .8rem; border:1px solid var(--line); border-radius:9px; color:var(--teal); }
    .forecast-chain b { color:var(--orange); }
    .analysis-links { display:flex; flex-wrap:wrap; gap:.8rem; }
    .analysis-links a { border:1px solid var(--line); color:var(--teal); text-decoration:none; border-radius:9px; padding:.8rem 1rem; }
    a:focus-visible, button:focus-visible { outline:3px solid var(--orange); outline-offset:4px; }
    @media(max-width:760px) {
      .analysis-cards { grid-template-columns:1fr; gap:.6rem; }
      .analysis-cards .card { padding:.65rem .85rem; }
      .analysis-cards .card h3 { font-size:1.05rem; margin:.25rem 0; }
      .analysis-cards .card p { font-size:.86rem; line-height:1.33; }
      .analysis-kpis { gap:.45rem; }
      .analysis-kpis .metric-card { padding:.7rem .55rem; }
      .analysis-kpis .metric-card span { font-size:.65rem; letter-spacing:0; }
      .analysis-kpis .metric-card p { font-size:.7rem; }
      .analysis-table { font-size:.78rem; }
      .analysis-table th { font-size:.65rem; letter-spacing:0; }
      .analysis-table th, .analysis-table td { padding:.55rem .25rem; }
      .forecast-chain { gap:.3rem; font-size:.76rem; }
      .forecast-chain span { padding:.4rem .45rem; }
      .analysis-links { gap:.55rem; }
      .analysis-links a { padding:.65rem .8rem; font-size:.85rem; }
      h2 { max-width:27ch; }
    }
    @media(max-height:700px) {
      :root { --slide-padding:clamp(.85rem,2.3vw,1.9rem); --gap:.65rem; }
      .analysis-plot img { max-height:49vh; }
    }
    @media(max-height:600px) {
      .analysis-cards { grid-template-columns:repeat(3,minmax(0,1fr)); gap:.6rem; }
      .analysis-cards .card { padding:.65rem; }
      .analysis-cards .card p { font-size:.78rem; line-height:1.3; }
      .analysis-cards .card h3 { font-size:.95rem; }
      .analysis-plot img { max-height:45vh; }
    }
    @media(max-height:500px) {
      :root { --slide-padding:.65rem 1rem; --gap:.45rem; --body-size:.8rem; --small-size:.64rem; --title-size:clamp(1.6rem,4.2vw,2.6rem); --h2-size:clamp(1.3rem,3vw,1.75rem); }
      .analysis-kpis .metric-card { padding:.4rem .65rem; }
      .analysis-kpis .metric-card strong { margin:.05rem 0; font-size:1.8rem; }
      .analysis-kpis .metric-card p { display:none; }
      .analysis-cards .card p { font-size:.72rem; }
      .analysis-cards .card h3 { font-size:.87rem; }
      .analysis-table th, .analysis-table td { padding:.3rem; }
      .analysis-plot img { max-height:40vh; }
    }
    @media(prefers-reduced-motion:reduce) { html { scroll-behavior:auto; } }
    '''
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Verified original GraphEFM versus 1/2/4 GPU distributed AR4 benchmark, numerical agreement, timing attribution and next experiments."><title>Njord AR4 — Original and Multi-GPU Comparison</title><style>' + css + extra_css + '</style></head><body data-transition="fade"><div class="progress-bar" aria-hidden="true"></div><nav class="nav-dots" aria-label="Slide navigation"></nav><div class="keyboard-hint">← → / space · Home · End</div><main>' + '\n'.join(slides) + '</main><script>' + js + '</script></body></html>'
    local = document.replace("HISTORY_LINK", "../../../../../njord-message-passing-website/benchmark-analysis-ar1-history.html")
    (OUT / "presentation.html").write_text(local)
    prefix = '../baltic-sea-forecasting/' + str(OUT.relative_to(REPO)) + '/'
    site_document = document.replace('href="HISTORY_LINK"', 'href="benchmark-analysis-ar1-history.html"')
    for file in ("analysis.md", "comparison.json", "metrics.csv", "critical-rank-phases.csv"):
        site_document = site_document.replace(f'href="{file}"', f'href="{prefix}{file}"')
    DECK.write_text(site_document)
    print(json.dumps({"slides": len(slides), "presentation": str(DECK),
                      "standalone_copy": str(OUT / "presentation.html"),
                      "historical_appendix": str(HISTORY)}, indent=2))


if __name__ == "__main__":
    main()
