"""Out-of-sample test of the step-time model: predicted vs measured mean step
time. Usage: python oos_fig.py <results_dir> <out.png> <fit_jobs> <test_job>"""
import glob
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

root, out = sys.argv[1:3]
fit_jobs, test_job = sys.argv[3].split(","), sys.argv[4]
OLIVE, GREY, INK, MUTED, GRID = "#4f5b1a", "#9a9a8e", "#1f1f1c", "#6b6b63", "#e4e4dc"

# Cost model, k = 4, global_hierarchical_graph, hidden-256 cost model
IMB = {"lon_bands": 1.025, "geographic": 1.028, "nils_weighted_chunks": 1.033,
       "nils_weighted_metis": 1.067, "nils_weighted_rcb": 1.077,
       "nils_weighted_mtkahypar_km1": 1.090,
       "test_noise01": 1.024, "test_noise03": 1.027, "test_noise10": 1.021,
       "test_noise30": 1.014, "test_skew10": 1.102, "test_skew20": 1.201,
       "test_skew10_noise03": 1.097, "test_random": 1.029}
HALO = {"lon_bands": 2090, "geographic": 3561, "nils_weighted_chunks": 69234,
        "nils_weighted_metis": 1070, "nils_weighted_rcb": 1845,
        "nils_weighted_mtkahypar_km1": 945,
        "test_noise01": 3414, "test_noise03": 6780, "test_noise10": 14384,
        "test_noise30": 33490, "test_skew10": 2232, "test_skew20": 2296,
        "test_skew10_noise03": 6349, "test_random": 64218}
NAMES = {"lon_bands": "Longitude bands", "geographic": "Per-layer split",
         "nils_weighted_chunks": "Node-ID chunks",
         "nils_weighted_metis": "METIS on G0",
         "nils_weighted_rcb": "RCB on G0",
         "nils_weighted_mtkahypar_km1": "Mt-KaHyPar on G0",
         "test_noise01": "noise 1%", "test_noise03": "noise 3%",
         "test_noise10": "noise 10%", "test_noise30": "noise 30%",
         "test_skew10": "skew 10%", "test_skew20": "skew 20%",
         "test_skew10_noise03": "skew 10% + noise 3%",
         "test_random": "random"}


def load(jobs):
    steps, peaks = {}, {}
    for job in jobs:
        for f in glob.glob(f"{root}/{job}/h256-4gpu-*-rep*.json"):
            k = f.split("h256-4gpu-")[1].rsplit("-rep", 1)[0]
            d = json.load(open(f))
            if d.get("status") != "completed":
                continue
            steps.setdefault(k, []).extend(r["compute_ms"] for r in d["records"])
            peaks.setdefault(k, []).append(
                d["summary"]["peak_memory_allocated_bytes_per_gpu"] / 2**30)
    return ({k: np.mean(v) for k, v in steps.items()},
            {k: np.mean(v) for k, v in peaks.items()})


fit_t, _ = load(fit_jobs)
test_t, test_m = load([test_job])
keys = [k for k in IMB if k in fit_t and not k.startswith("test_")]
X = np.array([[1, IMB[k], HALO[k]] for k in keys], float)
coef, *_ = np.linalg.lstsq(X, np.array([fit_t[k] for k in keys]), rcond=None)


def pred(k):
    return coef @ [1, IMB[k], HALO[k]]


# Day-to-day offset: lon_bands in the test job vs the fitting jobs
offset = test_t["lon_bands"] - fit_t["lon_bands"] if "lon_bands" in test_t else 0
print(f"model: {coef[0]:.2f} + {coef[1]:.2f} x imbalance + "
      f"{coef[2] * 1e3:.4f} us x halo; lon_bands offset {offset:+.2f} ms")
rows = []
for k in IMB:
    if k.startswith("test_") and k in test_t:
        p, m = pred(k), test_t[k]
        rows.append((k, p, m, m - p, m - offset - p))
        print(f"{NAMES[k]:22s} pred {p:6.1f}  meas {m:6.1f}  err {m - p:+5.1f}"
              f"  err (offset-corrected) {m - offset - p:+5.1f}  "
              f"peak {test_m[k]:.2f} GiB")
err = np.array([r[3] for r in rows])
err_c = np.array([r[4] for r in rows])
print(f"RMS error {np.sqrt(np.mean(err ** 2)):.2f} ms, offset-corrected "
      f"{np.sqrt(np.mean(err_c ** 2)):.2f} ms, max |err| {np.abs(err).max():.1f}")

# ---- figure ---------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
fig = plt.figure(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("white")
fig.text(0.045, 0.93, "TITLE", fontsize=30, weight="bold", color=INK)
fig.text(0.045, 0.875,
         r"Predicted by $\mathit{time}$ = "
         f"{coef[0]:.1f} ms + {coef[1]:.1f} ms × " r"$\mathit{imbalance}$"
         f" + {coef[2] * 1e3:.2f} µs × " r"$\mathit{traffic}$"
         ", fitted on the first six strategies only.",
         fontsize=17, color=MUTED)
ax = fig.add_axes([0.09, 0.17, 0.86, 0.64])
lo, hi = 238, 290
ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.5, zorder=1)
ax.fill_between([lo, hi], [lo - 5, hi - 5], [lo + 5, hi + 5], color=GRID,
                alpha=0.6, zorder=0, lw=0)
ax.text(hi - 1, hi - 6.5, "±5 ms", color=MUTED, fontsize=13, ha="right")
for k in keys:
    ax.scatter(pred(k), fit_t[k], s=90, color=GREY, edgecolors="white",
               linewidths=1.5, zorder=2)
for k, p, m, e, _ in rows:
    ax.scatter(p, m, s=150, color=OLIVE, edgecolors="white", linewidths=2,
               zorder=3)
    ax.annotate(f"{NAMES[k]} ({e:+.1f})", (p, m), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=13,
                color=INK)
ax.scatter([], [], s=90, color=GREY, label="fitted (6 strategies)")
ax.scatter([], [], s=150, color=OLIVE, label="new partitions, not used in the fit")
ax.legend(loc="upper left", frameon=False, fontsize=15)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Predicted step time (ms)", color=MUTED, fontsize=15)
ax.set_ylabel("Measured mean step time (ms)", color=MUTED, fontsize=15)
ax.grid(color=GRID, lw=1)
ax.set_axisbelow(True)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"):
    ax.spines[s_].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=13)
fig.text(0.045, 0.075,
         f"Numbers in brackets: measured minus predicted (ms). Out-of-sample RMS "
         f"error {np.sqrt(np.mean(err ** 2)):.1f} ms over {len(rows)} partitions.",
         fontsize=12.5, color=MUTED)
fig.text(0.045, 0.045,
         "1° icosahedral hierarchical graph, hidden 256, batch 2, 4 × GTX 1080 Ti. "
         f"Fit: jobs {', '.join(fit_jobs)}; test: job {test_job}.",
         fontsize=12.5, color=MUTED)
fig.savefig(out, dpi=150, facecolor="white")
