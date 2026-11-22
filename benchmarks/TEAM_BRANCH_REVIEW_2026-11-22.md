# Team branch review — 22 November 2026

This is a review of fetched Git branches and their tracked code/reports, not
a claim about unpushed local work or who is currently working on each topic.
No team branches were merged. Links below identify the inspected commits.

## Snapshot and responsibilities visible in the code

| Branch | Inspected head | Latest commit | Work present | Relation to message-passing |
| --- | --- | --- | --- | --- |
| `origin/alex` | `9f058a3` | 18 Nov | Mesh graph parallelism, forward/backward halos, Lightning strategy, optional overlap, tests and GPU benchmark archive | Direct overlap with our communication work; different partition boundary |
| `origin/lassi-partition-benchmark` | `2539d36` | 18 Nov | External owner arrays in NPZ files, validation, comparison job | Intended integration point for team-generated partitions |
| `origin/lassi-graph-viewer` | `f3561a2` | 4 Nov | Interactive graph viewer, weighted geographic cuts and communication/memory cost model | Helps assess partition balance and halo volume; merged into `origin/main` |
| `origin/lassi-profiling` | `883c48e` | 28 Oct | GPU trace attribution by model component and forward/recompute/backward; memory attribution | Useful for measuring where distributed execution spends time |
| `origin/nils` | `9d858b6` | 28 Oct | Width/depth capacity sweeps and model-stage profiler, figures and reports | Useful profiling methods; different GPU and graph prevent direct timing comparisons |
| `origin/max` | `51ae805` | 28 Oct | Benchmark framework and archived original-model measurements | Historical workload reference; not a fresh same-run reference measurement |

## Alex: a second communication implementation

Source: [implementation notes](https://github.com/maximilianfricke/baltic-sea-forecasting/blob/9f058a3/benchmarks/graph_parallel.md)
and [benchmark report](https://github.com/maximilianfricke/baltic-sea-forecasting/blob/9f058a3/benchmark_archive/graph-parallel/report.md).

The branch adds `neural_lam/graph_parallel.py`, a Lightning strategy, graph
partition artifacts and model integration. It shards mesh processing in the
prior, posterior and decoder. Receiver-owned edges use halo exchange, with
custom autograd handling the reverse traffic. Checkpoint recomputation
executes the complete communication schedule, as in our implementation.

The main architectural difference is that grid computation, G2M, M2G,
predictions and loss remain replicated. Boundary mesh states are rebuilt
for those replicated operations. Our path partitions these parts as well
and maintains rank-local autoregressive feedback. These are alternative
execution boundaries, not two halves that can simply be merged together.

The standard runtime has correctness evidence in the branch. Its integration
also covers Lightning checkpointing/resume, gradient clipping and rank-zero
output handling, which our narrow benchmark does not attempt to reproduce.
It supports CRPS, while our present benchmark intentionally measures the
ELBO-only workload with CRPS weight zero.

Its reported H64/L2/B2/AR4/checkpointed GTX 1080 Ti experiment uses **zero
loader workers**, 20 warmup and 200 measured steps with three repetitions.
Median throughput is 0.588 / 0.521 / 0.495 samples/s for 1/2/4 GPUs, and peak
memory is about 3.41 / 2.80 / 2.43 GiB. These figures show poor scaling within
that experiment; their absolute times cannot be used as a controlled
comparison with our eight-worker, rank-local input experiment.

The overlap option schedules communication alongside local edge work, but
the archived 2- and 4-GPU overlap trials ran out of CUDA memory. Treat it as
an experiment to inspect and budget for, not an established improvement.
Capacity measurements include a width-256 case that only completed on four
GPUs, illustrating the memory-capacity motivation even when H64 scaling is
weak. No overlap or runtime implementation was imported into our branch.

## Lassi: externally supplied partitions and their evaluation

Source: [comparison job](https://github.com/maximilianfricke/baltic-sea-forecasting/blob/2539d36/jobs/partition_comparison.sbatch)
and [partition loader](https://github.com/maximilianfricke/baltic-sea-forecasting/blob/2539d36/neural_lam/models/distributed_graph_efm.py).

The new branch extends an older version of our distributed benchmark with
`--partition-strategy file --partition-file ...`. Its NPZ schema contains
`world_size`, `grid`, `mesh_0`, `mesh_1`, and optionally `name`. Owner arrays
map every global node to a rank. Validation checks array sizes, owner ranges
and nonempty ownership on each level; file hashes are included in metadata.

The job is set up to compare geographic, longitude-band, METIS and
Mt-KaHyPar candidate artifacts at H64/H256 on two/four GPUs. It does not
contain tracked candidate NPZ files or archived comparison measurements at
the inspected head. The presence of a candidate name in the job is not
evidence that the corresponding partition has already been produced or run.

The comparison branch uses **AR1** and rank-zero batch broadcasting. It
explicitly rejects external partitions with rank-local loading: our current
Zarr shards encode a particular geographic owner map and cannot be reused
with arbitrary owners. Its code also fixes `PYTHONPATH` so jobs import the
intended checkout rather than another editable installation.

After the original reference is measured, the integration should:

1. Port the external-owner loader and validation into the current AR4
   implementation, without reverting its rollout, checkpoint or loss code.
2. Validate exact graph/node order as well as shapes and rank counts. A
   length-correct owner array for a different graph is still invalid.
3. Bind every generated data shard and manifest to the partition identity;
   generate matching shards before any end-to-end comparison.
4. Keep the initial comparison at H64/L2/B2/AR4 and the established loader
   settings. Compare per-relation unique halos, byte volume, local edge
   work, memory and timing before changing model size or input policy.

The separate [graph viewer](https://github.com/maximilianfricke/baltic-sea-forecasting/blob/f3561a2/neural_lam/graph_viewer/README.md)
already provides longitude bands and weighted recursive coordinate
bisection. Its cost model counts distinct remote sender states per relation,
weighted by how often that relation is used; it does not equate halo volume
with cut-edge count. Its viewer interface does not yet load arbitrary fixed
METIS-style owner arrays. Estimated memory is a relative balancing model,
not a promise of absolute GPU memory usage.

## Nils and Lassi: attribution and capacity profiling

Sources: [Nils's baseline report](https://github.com/maximilianfricke/baltic-sea-forecasting/blob/9d858b6/BASELINE_REPORT.md),
[stage profiler](https://github.com/maximilianfricke/baltic-sea-forecasting/blob/9d858b6/neural_lam/profile_stages.py),
and [Lassi's trace attribution](https://github.com/maximilianfricke/baltic-sea-forecasting/blob/883c48e/benchmarks/attribute_profile.py).

Nils measures memory limits across width/depth settings and separates
embeddings, G2M, mesh processing and M2G. His report uses an RTX 5060 Ti,
a different hierarchical graph, batch size one and AR1, so its timings and
capacity limits are not transferable to our 1080 Ti / cluster-graph AR4
experiment. It shows mesh work becoming a substantial part of the forward
pass as model complexity increases. The measured training-step times exclude
the loader and host-to-device transfer.

Lassi's trace tool distinguishes original forward kernels, checkpoint
recomputation and true backward kernels using profiler regions and autograd
links. This is particularly relevant now that our data wait is small and
backward dominates. Nils's stage hooks wrap the original model path; they
must be adapted to our distributed training wrapper and local embedding
methods. Neither profiler can simply be pointed at a new execution path
without checking which regions it actually captures.

For the next profiling experiment, preserve a timed run without tracing and
use a separate short traced run. Measure G2M, same-level and up/down relations,
M2G, checkpoint replay, halo transfers, CPU launch/wait gaps and parameter
gradient reduction. CUDA-event intervals can include launch starvation or
waiting; they do not by themselves prove that all elapsed time is GPU math.

## Upstream source drift and immediate order of work

Our fork's `global-model` remains at `76b30d7`, the source of the current
controlled experiment. The supervisor repository's `upstream/global-model`
has advanced to `288164d` (23 October). Its newer changes include flow-model
work and shared training/data/RNG changes; the GraphEFM model file itself is
unchanged. This review does not merge that newer source into the experiment.

The immediate next measurement is the unchanged original one-GPU GraphEFM
reference described in `MODEL_PARALLEL_MAX_BASELINE.md`. Then:

1. Compare original one GPU against distributed one GPU, separating model
   time from the different original/sharded input layouts and recording
   their checkpoint replay-policy difference.
2. Explain distributed 1/2/4 scaling using the existing controlled runs and
   the new reference. Update the analysis/presentation with measured evidence.
3. Choose targeted profiling based on that comparison. Align the two
   communication implementations before investing in duplicate changes.
4. Connect externally supplied partitions to our current runtime and their
   matching data shards. Change one factor at a time in subsequent runs.

Loader tuning, bigger hidden sizes and experimental overlap are deferred
until this comparison establishes where the current execution loses time.
