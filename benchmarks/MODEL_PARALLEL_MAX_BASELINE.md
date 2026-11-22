# GraphEFM max-baseline model-parallel experiment

The experiment reproduces the effective model configuration of the `max`
branch's archived `1614-hidden-h64-ckpt-on` run. One, two and four GPUs solve
the same global problem using spatial message passing, including backward
halo exchange. This suite does not tune data loading. The subsequent
reference decision uses eight workers/rank from the original global-model
AR4 scripts instead of the three workers in our archived max benchmark.
Both used prefetch two.

## Fixed configuration

| Setting | Value |
| --- | --- |
| Graph | global_cluster_graph: 42,348 grid / 8,456 + 2,097 mesh nodes |
| Hidden dimensions (grid, edges, mesh, latent) | 64 |
| Hidden layers per MLP | 1 |
| Decoder processor layers | 2 |
| Prior / posterior processor layers | 1 / 1 |
| Input states / autoregressive forecast steps | 2 / 4 |
| Global batch size | 2, shared temporal sample indices on all ranks |
| Parameters per rank | 645,020, replicated weights with summed gradients |
| Precision | FP32 |
| Loss | WMSE reconstruction + KL, beta 1, CRPS weight 0 |
| Loss reduction | Sum over spatial points, features and four times; mean over batch |
| Prior / posterior | Learned isotropic / learned diagonal Gaussian |
| Observation noise / output standard deviation | Disabled / per-variable constant |
| Output constraints / density feedback | Original config and GraphEFM methods |
| Optimizer | AdamW, LR 0.001, betas (0.9, 0.95), weight decay 0.1 |
| Gradient checkpointing | Enabled around each complete forecast step |
| Loader workers | 8 per rank, as in the original global-model AR4 scripts |
| Prefetch factor | 2 per worker (max used the PyTorch default) |
| Worker start / persistence | spawn / persistent |
| Shuffle / drop last / pin memory | True / False / False |
| Seed | 42; explicit identical shuffle generator on every rank |
| Full measurement | 10 warmup + 50 measured optimizer steps |

The old CLI recorded `grad_checkpointing=false`, but ARModel automatically
enabled it for AR > 1. Its effective model metadata records `true`. The
distributed suite enables it explicitly.

## Training implementation

`neural_lam/models/distributed_graph_efm_training.py` contains the reusable
training module. It uses the original embedders, loss weights, output
clamping, density threshold and feedback semantics. Each forecast is fed
back into the next step without detaching. Every graph relation exchanges
its required halo features again at every step; full predictions are never
gathered onto every GPU. Rank-local likelihood and KL contributions are
summed, and replicated parameter gradients are summed before AdamW.

Posterior noise is generated in global top-mesh order and then selected for
each rank, preserving the same random scenario across GPU counts. Each
checkpoint receives its noise as an input, so backward replay cannot advance
the explicit generator. Non-reentrant checkpoint replay disables early stop:
all ranks must execute the entire communication schedule during recomputation.
The original GraphEFM, WeatherDataset and datastore implementations are unchanged.

Existing world-size 1/2/4 spatial Zarr shards can be reused. AR4 changes the
temporal windows read from those arrays; the shards do not need rebuilding.
With this dataset AR4 has 473 training samples, or 237 batches including the
last one-sample batch. Throughput counts actual samples, including that batch.

The new 1-GPU distributed run is the speedup reference. Historical max times
are context, rather than an automatically controlled end-to-end comparison:
the old run had four allocated CPUs and a full-grid input store. The new suite
retains the existing rank-local shards and course GPU/CPU allocations. Its
explicit shared shuffle stream also need not reproduce the historical run's
global-RNG-dependent shuffle order. All new 1/2/4-GPU runs use the same stream.

## Local verification

Run from the repository root:

```bash
.venv/bin/python -m pytest --noconftest \
  tests/test_distributed_graph_efm_rollout.py \
  tests/test_distributed_graph_efm.py \
  tests/test_distributed_message_passing.py \
  tests/test_model_parallel_real_benchmark.py -q
```

The AR4 cases compare every forecast, total loss, all parameter gradients and
two AdamW updates against the original `GraphEFM.training_step` on 1/2/4 Gloo
ranks, with checkpointing both enabled and disabled. A separate optional test
uses the complete physical H64/L2/B2/AR4 graph when local data is present,
including the real masks, clamping, density channel, loss and gradients.
CPU process tests verify correctness, not GPU performance.

Local verification on 2026-11-22: all 26 tests in this command passed,
including the physical full-graph test. CUDA/NCCL remains to be checked on
the cluster; the smoke jobs perform that check automatically.

## Sequential cluster runs

Use `/home/mfricke/baltic-sea-forecasting` with its existing `.venv`. Original
data/config/graph live under
`/scratch/users/mfricke/baltic-sea-forecasting-data/physical_repeat_16`;
the shard layouts live under
`/scratch/users/mfricke/baltic-sea-forecasting-data/partitioned/physical_repeat_16`.
Jobs stage only the chosen layout to `$TMPDIR` before measurement.
Smoke and full jobs request a one-hour wall-time limit by default; this is
a scheduling limit, not the measured training duration.

First clear old shell tuning overrides and submit one smoke job:

```bash
cd /home/mfricke/baltic-sea-forecasting
unset NJORD_NUM_WORKERS NJORD_PREFETCH_FACTOR NJORD_WARMUP_STEPS NJORD_MEASUREMENT_STEPS
bash jobs/submit_model_parallel_real.sh 1 smoke
```

Each smoke job first runs the AR4 equivalence test on CUDA/NCCL. If it passes,
the job runs one physical-data optimizer step with the complete model and the
fixed eight-worker/prefetch-two input setup. A failed check stops the job.
Review both the Slurm exit code and the result JSON before submitting the
next GPU count:

```bash
bash jobs/submit_model_parallel_real.sh 2 smoke
bash jobs/submit_model_parallel_real.sh 4 smoke
```

Submit these commands individually after the previous job passes. Then do
the full jobs individually in the same order:

```bash
bash jobs/submit_model_parallel_real.sh 1 full
bash jobs/submit_model_parallel_real.sh 2 full
bash jobs/submit_model_parallel_real.sh 4 full
```

The new result names contain `max-ar4-ckpt` and cannot overwrite earlier AR1
results. Smoke equivalence evidence is saved as
`benchmark_results/model-parallel-real/ar4-equivalence-Ngpu-JOBID.json`.
Physical results are saved as
`real-mp-max-ar4-ckpt-full-rank-local-Ngpu-JOBID.json` in the same directory.

Compare the three new full JSON files using
`benchmarks/compare_model_parallel_real.py`. It checks AR length, effective
checkpointing, full model and optimizer settings, loader policy, global batch
fingerprint and preflight loss. JSON files retain raw per-rank timings,
startup/staging time, forward, backward (including checkpoint replay),
gradient synchronization, optimizer, input phase, rank wait, loss, throughput
and memory. Forward and backward include halo communication; this suite
does not claim to isolate communication as a separate timer.

GPU/NCCL and performance results are pending execution on the cluster.

## Real-data local loader integration smoke

The additional local runner tests the complete H64/L2/B2/AR4 model with
checkpointing, rank-local physical Zarr shards, eight persistent spawn
workers per rank and prefetch factor two. It executes two optimizer steps
on 1/2/4 CPU/Gloo ranks and verifies identical global batch fingerprints
and close training losses across the layouts. It is a functional check;
its CPU timings must not be used as GPU strong-scaling measurements.

```bash
.venv/bin/python -m benchmarks.local_ar4_loader_smoke \
  --world-sizes 1 2 4 \
  --output-dir benchmark_results/model-parallel-real/local-ar4-workers8-2026-11-22
```

Individual rank counts can also be run sequentially. To compare their
saved JSONs afterwards, use the same command with `--check-only`.

On 2026-11-22, the real-data smoke passed for 1/2/4 CPU ranks with
8/16/32 loader workers in total. All global batch fingerprints matched.
The first-step losses were 1,102,355,200 / 1,102,355,200 / 1,102,355,328;
the second-step losses were 290,406,464 for every layout. Forward,
checkpointed backward, finite synchronized gradients, AdamW and worker
shutdown completed. Evidence is in
`benchmark_results/model-parallel-real/local-ar4-workers8-2026-11-22/`.
