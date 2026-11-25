# Cost model at 0.25°, with three mesh levels, and on Nils's partitions

Cost-model results that need no GPU: they follow from the graph and a cost
model measured once on CPU. Halo and edge counts from this cost model equal
the ones the `message-passing` implementation records on GPUs (jobs
4311/4312/4465, zero differences); time and memory predictions were checked
in job 4467 ([README](README.md)).

## Cost models measured (`cost_models/`)

All with the realistic baseline: GraphEFM, 2 processor layers, hidden 64
everywhere (or 256), batch 1, one forecast step, CPU, branch
`lassi-balance-metrics`, `measure_weights.py`.

| File | Graph | Stored values per step | Notes |
| --- | --- | --- | --- |
| `../../../neural_lam/graph_viewer/weights/realistic_baseline.json` (viewer branch) | `global_cluster_graph` (2 mesh levels) | 427 M | the model used in job 4467 |
| `cluster_graph_h256.json` | same, hidden 256 | 1,670 M (3.91×) | values per item within 0–13% of hidden 64: memory is linear in hidden size |
| `cluster_3level_h64.json` | `global_cluster_3level`: 1° cluster graph rebuilt with 3 mesh levels (8,462 / 2,106 / 520), same construction as the 0.25° graph | 447 M | the middle level's m2m edges are used 6× per step (u = x = 6) |
| `hierarchical_graph_h64.json` | `global_hierarchical_graph` (Nils's graph, 7,193 / 1,812) | 391 M | |

The 3-level graph was built with
`create_global_graph --type cluster --levels 3 --grid_to_first_mesh_refinement 5 --mesh_refinement_factor 4 --connect_disconnected`.
Per-item values depend on the model, not the resolution, so the 3-level model
is used for the 0.25° graph.

## 0.25° (`global_cluster_analysis`)

675,219 grid nodes; mesh 33,777 / 8,409 / 2,088; 3.5 M edges. Node order
checked: every g2m edge joins points less than 1.9° apart (median 0.48°).
Partitioning takes 0.1 s (geometric), 3–5 s (Mt-KaHyPar) and 1.3–1.6 s
(METIS). Script: [`score_any.py`](../../benchmarks/partition_candidates/score_any.py).

Stored activations per sample: **71.6 M values per hidden unit, 17.1 GiB at
hidden 64** (fp32, one forecast step, 2 processor layers).

| GPUs | Strategy | Busiest GPU's halo traffic | Synchronised | Slowdown S | Memory share | Worst layer imbalance | d_max, batch 1 | d_max, batch 2 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | — | 0 | 0 | 1 | 100% | — | 39 | 19 |
| 2 | geographic (per layer) | 25,746 | 27,853 | 1.015 | 50.29% | 1.000 | 77 | 39 |
| 2 | **lon_bands** | 9,412 | 9,430 | **1.002** | **50.01%** | 1.008 (mesh 1) | 78 | 39 |
| 2 | RCB (shared borders) | 6,920 | 6,978 | 1.024 | 50.01% | 1.054 (mesh 0) | 78 | 39 |
| 2 | Mt-KaHyPar | **1,691** | **1,720** | 1.058 | 51.47% | 1.072 (mesh 1) | 75 | 38 |
| 2 | METIS (per-layer constraints) | 2,099 | 2,307 | 1.033 | 50.99% | 1.120 (mesh 1) | 76 | 38 |
| 4 | geographic (per layer) | 32,496 | 33,679 | 1.033 | 25.43% | 1.000 | 153 | 76 |
| 4 | **lon_bands** | 8,868 | 9,961 | **1.010** | **25.04%** | 1.018 (mesh 1) | 155 | 77 |
| 4 | RCB (shared borders) | 8,091 | 8,533 | 1.045 | 25.02% | 1.072 (mesh 0) | 155 | 78 |
| 4 | Mt-KaHyPar | **3,928** | **3,956** | 1.084 | 25.65% | 1.109 (mesh 1) | 151 | 76 |
| 4 | METIS (per-layer constraints) | 4,083 | 4,339 | 1.081 | 25.94% | 1.300 (mesh 1) | 150 | 75 |

d_max: largest hidden size whose activations fit 10.5 GiB per GPU with
0.16 GiB fixed, fp32, one forecast step in memory (checkpointing between
steps), rank-local data. *Estimate*; it leaves out optimizer state (small
at these sizes) and broadcast-mode replication.

Findings:

- **Memory, not partitioning, limits the 0.25° model.** With at most 4 GPUs,
  the largest hidden size is ≈ 150 at batch 1 (paper model: hidden 256 and 6
  processor layers). Reaching 256 needs ≈ 7 GPUs, or memory reductions:
  bf16 (≈ 2×), checkpointing inside a forecast step, or keeping fewer
  activations per edge.
- **lon_bands is again the best balanced split** (S 1.002 / 1.010, memory share
  closest to ideal) and has 2.7–3.7× less halo traffic than the per-layer
  geographic split, whose misaligned grid and mesh borders become expensive at
  high resolution.
- **Scaling of halo traffic with resolution.** From 1° to 0.25° the grid grows
  16×; for compact parts the halo should grow √16 = 4×. Busiest-GPU traffic
  grows 3.8–5.5× for lon_bands, RCB and Mt-KaHyPar, but 9.7–10.4× for the
  per-layer geographic split. Compute per GPU grows ≈ 16× (grid side), so
  communication relative to compute falls ≈ 4× for the consistent splits.

## Nils's partitions (`nils-pnemo`, `dphpc/pnemo/owners`)

Scored on `global_hierarchical_graph` with `hierarchical_graph_h64.json`.
Script: [`score_nils.py`](../../benchmarks/partition_candidates/score_nils.py).

| GPUs | Partition | Busiest GPU's halo traffic | Synchronised | S | Memory share | Grid / mesh 0 / mesh 1 node imbalance |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | Nils unweighted METIS | 422 | 465 | 1.066 | 53.16% | 1.126 / 1.001 / 1.003 |
| 2 | Nils unweighted Mt-KaHyPar | 371 | 380 | 1.070 | 53.52% | 1.138 / 1.001 / 1.006 |
| 2 | Nils unweighted RCB | 1,298 | 1,337 | 1.055 | 52.75% | 1.107 / 1.000 / 1.006 |
| 2 | Nils weighted chunks | 61,869 | 63,846 | 1.007 | 51.03% | 1.000 / 1.000 / 1.000 |
| 2 | Nils weighted METIS | 652 | 694 | 1.025 | 50.97% | 1.005 / 1.033 / 1.033 |
| 2 | Nils weighted Mt-KaHyPar | 618 | 641 | 1.031 | 51.41% | 1.015 / 1.034 / 1.039 |
| 2 | Nils weighted RCB | 1,515 | 1,563 | 1.043 | 51.83% | 1.000 / 1.068 / 1.068 |
| 2 | lon_bands | 2,070 | 2,088 | 1.007 | 50.05% | 1.005 / 1.010 / 1.007 |
| 2 | geographic (per layer) | 2,948 | 3,272 | 1.018 | 50.37% | 1.000 / 1.000 / 1.000 |
| 4 | Nils unweighted METIS | 950 | 950 | 1.177 | 29.42% | 1.336 / 1.007 / 1.009 |
| 4 | Nils unweighted Mt-KaHyPar | 921 | 936 | 1.175 | 29.16% | 1.327 / 1.012 / 1.020 |
| 4 | Nils unweighted RCB | 1,601 | 1,702 | 1.138 | 28.44% | 1.255 / 1.000 / 1.011 |
| 4 | Nils weighted chunks | 69,234 | 71,710 | 1.033 | 26.20% | 1.000 / 1.000 / 1.000 |
| 4 | Nils weighted METIS | 1,070 | 1,125 | 1.067 | 26.33% | 1.016 / 1.082 / 1.066 |
| 4 | Nils weighted Mt-KaHyPar | 945 | 1,004 | 1.090 | 26.96% | 1.015 / 1.120 / 1.110 |
| 4 | Nils weighted RCB | 1,845 | 1,928 | 1.077 | 26.62% | 1.000 / 1.098 / 1.102 |
| 4 | lon_bands | 2,114 | 2,179 | 1.025 | 25.05% | 1.032 / 1.023 / 1.031 |
| 4 | geographic (per layer) | 3,561 | 3,727 | 1.028 | 25.21% | 1.000 / 1.000 / 1.000 |

- Weighting by grid nodes per mesh node fixes most of the grid imbalance of
  the unweighted graph partitions (S 1.07 → 1.03 at K = 2) while keeping 3×
  less halo traffic than lon_bands.
- **Chunks (PhysicsNeMo's default node-ID ranges) has 30–170× the halo
  traffic of every other partition with near-perfect balance.** Its measured times are
  the cleanest available test of whether halo volume matters at this scale.
- Nils's report states that RCB is faster than Mt-KaHyPar at K = 2 (hidden 64).
  The cost model rates his weighted Mt-KaHyPar slightly better (S 1.031 vs
  1.043). At hidden 64, S is known not to predict times (job 4467); his raw
  measurements are needed to compare properly.
