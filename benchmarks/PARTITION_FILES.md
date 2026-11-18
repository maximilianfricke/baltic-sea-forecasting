# Externally computed partitions

`--partition-strategy file --partition-file OWNERS.npz` runs the model-parallel
benchmark with node owners computed elsewhere (graph viewer, METIS,
Mt-KaHyPar, …) instead of the built-in geographic split. This is item 1 of
the integration list in `TEAM_BRANCH_REVIEW_2026-11-22.md`, ported onto the
AR4 implementation without changing its rollout, checkpoint or loss code.

## File format

| Key | Content |
| --- | --- |
| `world_size` | number of ranks the owners are for |
| `grid` | owner of every full grid node, in g2m-sender order |
| `mesh_0`, `mesh_1`, … | owner of every node of each mesh level |
| `name` (optional) | label recorded in the results |
| `graph_sha256` (optional) | `graph_fingerprint` of the graph the owners were made for |

`load_partition_file` checks the rank count, every array's length, the rank
range, and that every rank owns at least one node of every layer. When the
file has `graph_sha256`, it must equal `graph_fingerprint_from_graph_efm` of
the loaded model: owner arrays of the right length for a different graph or
node order are rejected. The fingerprint hashes the layer-local edge indices
(g2m, m2g, m2m per level, up per level, down per level) with their shapes, so
a generator working from the saved graph files computes the same value as
the model (checked on `global_cluster_graph`: `e0d53717…` both ways).

Results record the file's path, name, SHA-256, `graph_sha256` and whether
the graph was verified.

## Data mode

The rank-local Zarr shards encode the geographic owner map, so file
partitions are refused with `--data-mode rank-local`; use
`rank-zero-broadcast`. Matching shards per partition (review item 3) are not
implemented yet.

## Earlier results

`lassi-partition-benchmark` holds a one-node comparison (job 4467, AR1,
broadcast) of geographic, longitude-band, Mt-KaHyPar and METIS owners at
hidden 64/256 on 2/4 GPUs, in `benchmark_results/partition-comparison/`.
