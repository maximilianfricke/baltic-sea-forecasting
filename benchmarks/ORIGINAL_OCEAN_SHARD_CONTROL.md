# Control B: original GraphEFM with the existing one-GPU ocean shard

This changes the dynamic input store of original reference job 4981. It
does not use the distributed model, partition the graph, change weights or
equations, or change checkpoint replay. The original 13 source-file checks
remain active. Original checkpoint early stopping stays enabled.

| Setting | Control B |
| --- | --- |
| Model | Original GraphEFM, H64/L2/B2/AR4, FP32, 645,020 parameters |
| Input | Existing `geographic_rcb_ws1/rank_0` ocean and atmosphere stores |
| Scientific settings | Same original masks, clamping, normalization and WMSE + KL |
| Loader | Eight persistent spawn workers, prefetch two, seed 42 |
| Checkpointing | Original complete-step non-reentrant checkpoint; early stop on |
| Timing | Ten warmup and fifty measured optimizer steps; actual sample counts |
| Reference | Original full-grid job 4981, first batch and measured loss trajectory |

The model still loads its graph, statistics, masks and static features from
the original configuration. Dynamic samples use the existing shard through
the original WeatherDataset. Ocean and atmosphere node IDs must equal the
complete original ocean-node order; a same-length permutation is rejected.
No dataset is regenerated and no source store is modified.

The job copies the same original source directory as job 4981 to `$TMPDIR`,
then also copies the existing one-GPU shard layout. It records source-copy,
shard-copy and total staging times separately from measured training. The
additional startup copy must be included when comparing total job costs,
rather than only the steady-state input speed. CPU affinity and Slurm CPU
allocation fields are recorded for the new run.

On the student cluster:

```bash
cd /home/mfricke/baltic-sea-forecasting
git pull --ff-only origin message-passing
sbatch jobs/original_graph_efm_ocean_shard_benchmark.sbatch
```

The wrapper fixes the ocean-shard mode and reuses the original reference
job. It requests one GTX 1080 Ti through `dphpc`, 24 GB RAM and a one-hour
wall-time limit. The usual full-grid job continues to default to full-grid
input. Existing `NJORD_NUM_WORKERS`/prefetch overrides do not change this
fixed 8/P2 reference. The normal Python/CUDA environment and known data
paths are reused:

```text
/scratch/users/mfricke/baltic-sea-forecasting-data/physical_repeat_16
/scratch/users/mfricke/baltic-sea-forecasting-data/partitioned/physical_repeat_16/geographic_rcb_ws1
```

Outputs (replace JOBID with the submitted job number):

```text
benchmark_results/model-parallel-real/slurm-original-ocean-shard-JOBID.out
benchmark_results/model-parallel-real/original-graph-efm-ar4-ocean-shard-1gpu-JOBID.json
```

Before timing, the first-batch fingerprint and preflight loss must match
4981. After timing, equal-length measured windows must also have matching
per-step losses; a failure stops before writing a completed result JSON.
The recorded `preflight.loss_trajectory_check` includes the maximum relative
loss difference and whether all scalar losses were exactly equal. A short
smoke with a different warmup/measurement window records that the full loss
trajectory comparison was skipped; the initial batch/preflight check remains.

Compare A (4981, original/full-grid) against B (this job, original/ocean
shard) first. This measures input-layout/pipeline effects with the original
model and checkpoint policy held fixed. Comparing B against MP1 still
requires accounting for MP's different checkpoint replay policy and retained
preflight batch; it is not an isolated communication-overhead measurement.

Local checks:

```bash
.venv/bin/python -m pytest --noconftest \
  tests/test_original_graph_efm_benchmark.py \
  tests/test_model_parallel_real_benchmark.py -q
bash -n jobs/original_graph_efm_benchmark.sbatch
bash -n jobs/original_graph_efm_ocean_shard_benchmark.sbatch
```

When local physical data is available, the integration test builds both
original H64 models, compares their complete initial state dictionaries,
loads two shuffled batches per input mode through eight workers/P2, compares
all nonempty tensors exactly, then checks identical AR4 likelihood, KL and
loss on CPU. This verifies functionality; GPU performance is measured only
by the user's cluster job.

Verification on 23 November 2026: all twelve reference/configuration tests
passed, including the physical eight-worker integration and exact CPU AR4
loss comparison. Both job scripts passed `bash -n`. No cluster job was
submitted during preparation.
