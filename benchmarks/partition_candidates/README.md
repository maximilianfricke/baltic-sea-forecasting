# Partition candidate generation

The scripts that produced the owner files in
[`benchmark_results/partition-comparison/partitions`](../../benchmark_results/partition-comparison/partitions).
They were run as one-off experiment scripts; they are kept for
reproducibility, not as a library.

The graph export and scoring use the cost model in `neural_lam/graph_viewer`
(branch `lassi-balance-metrics`). The partitioners run in a separate
environment, so the project environment needs none of their packages:
numpy, `mtkahypar==1.7.1`, `kahip==3.28` and `pymetis==2025.2.2` (on Windows,
pymetis came from the conda-forge `py312` build because PyPI has no Windows
wheel).

| Step | Script | Environment |
| --- | --- | --- |
| 1. Export graph, node weights, uses and exchanges | `export_graph.py <dir>` | project, with the graph viewer |
| 2. Geometric, Mt-KaHyPar and KaHIP partitions | `run_partitioners.py <dir> <graph_viewer dir>` | partitioner env |
| 3. METIS (intended: one constraint per layer; pymetis actually uses one weight per node, the first n entries; see `benchmark_results/partition-comparison/README.md`) | `run_metis.py` / `run_metis_rb.py <dir> <pymetis path>` | partitioner env |
| 4. Write owner files (best seed by synchronised halo traffic) | `make_partition_files.py <dir> <dir holding partition.py>` | partitioner env |
| 5. Check files through `build_partition_from_graph_efm` | `check_harness.py <dir>` | project |
| 6. Summarise results against predictions | `analyse.py <results dir> <dir>` | project, with the graph viewer |

Owner file format (read by `load_partition_file` in
`neural_lam/models/distributed_graph_efm.py`): `world_size`, `grid` (owner of
every grid node in g2m-sender order), `mesh_0`, `mesh_1`, … and optionally
`name`.
