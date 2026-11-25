# Slide figures and partition-file scripts

Scripts behind the partition-comparison slides (2026-11-25). Each writes a
16:9 PNG; run them from the repository root with the project venv.

| Script | Figure | Arguments |
| --- | --- | --- |
| `box_fig.py` | Training step time and peak memory by partitioning strategy | `<results_dir> <out.png> <jobs>` |
| `balance_fig.py` | Imbalance alone doesn't predict training time | same |
| `halo_fig.py` | Halo traffic alone doesn't predict training time | same |
| `model_fig.py` | Fitted model, 2D and 3D (imbalance + halo traffic) | `<results_dir> <out_2d.png> <out_3d.png> <jobs>` |
| `two_slides.py` | Formula slides: imbalance, halo traffic | `<imbalance.png> <halo.png>` |
| `formula_fig.py` | Combined formula slide | `<out.png> [font size]` |
| `table_fig_025_h256.py` | Cost-model table, 0.25° graph, hidden 256 | `<out.png>` |

`<results_dir>` is `benchmark_results/partition-comparison`, `<jobs>` e.g.
`6088,6104,6116`. Imbalance and halo values in the scatter and model figures
are the cost-model numbers for `global_hierarchical_graph`, 4 GPUs (see
`COST_MODEL_EXTENSIONS.md`).

Partition files and checks (need the viewer from branch
`lassi-balance-metrics` on `PYTHONPATH`):

- `make_hier_files.py`: Nils's owner files converted to the loader format,
  plus lon_bands from the hidden-256 cost model
  (`benchmark_results/partition-comparison/partitions_hier/`).
- `check_hier.py`: loads partition files through
  `build_partition_from_graph_efm` and compares owners and halo traffic with
  the cost model.
- `score025_table.py`: cost-model numbers on the 0.25° graph (weighted METIS
  on mesh 0, lon_bands, per-layer split).
