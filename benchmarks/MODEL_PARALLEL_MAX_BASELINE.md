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
including the physical full-graph test. Subsequent cluster smoke jobs
4941/4948/4950 also passed the CUDA/NCCL equivalence checks and the physical
optimizer step on 1/2/4 GPUs.

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

GPU/NCCL checks and the first controlled AR4 performance runs have now
completed. The archived 1/2/4-GPU results (4944/4949/4952) are under
`benchmark_results/model-parallel-real/full/ar4-workers8-prefetch2-4944-4949-4952/`.
These compare the distributed implementation across GPU counts; they do not
yet measure its overhead against the original, unpartitioned implementation.

## Original unpartitioned one-GPU reference

`benchmarks/original_graph_efm.py` calls the unchanged original
`GraphEFM.training_step`, including its original checkpointed rollout and
loss. It captures the logged ELBO components without using Lightning's
training loop, and reuses the same CUDA timers and AdamW step as the
distributed benchmark. It does not partition the graph or use halo exchange.

The reference freezes our experimental starting point,
`origin/global-model` commit `76b30d7`, and verifies the Git blob identities
of 13 original model, message-passing, loss and data-loader files before
running. `upstream/global-model` has newer changes, but merging those would
change the experiment's source baseline.

Both paths use H64/L2/B2/AR4/FP32, the learned prior, KL beta 1, checkpointing,
AdamW at 0.001, seed 42, eight persistent spawn workers, prefetch two, and
the same explicit shuffled sample order. The reference preserves the original
default CUDA RNG and seeds it to the scenario stream used by the distributed
benchmark. Its preflight must match the archived 1-GPU batch fingerprint and
initial loss or the job stops before collecting measurements.
The comparison canonicalizes the recorded shape of unused zero-element
inputs, because the original loader retains a rectangular grid extent in
its empty boundary placeholder. Actual tensors are not changed; their raw
shapes are also saved.

Two execution differences are retained and recorded:

- The original WeatherDataset reads the full rectangular Zarr store and
  applies the ocean masks. The distributed runs read prebuilt ocean shards.
  End-to-end comparisons therefore include input layout differences; report
  the model-compute and input phases separately.
- Original non-reentrant checkpointing retains early stopping. Distributed
  replay disables early stopping to execute a consistent collective schedule.
  Backward timing includes the cost of that required replay policy as well
  as the distributed implementation and communication.

The reference job stages the original source directory, including config,
statistics and graph, into its job-local `$TMPDIR`. This startup copy is
recorded separately from the measured training window. It requests the same
1080 Ti model and course allocation as the earlier single-GPU runs; the
cluster applies its normal CPU allocation. Eight loader workers do not imply
eight allocated CPU cores.

Local verification:

```bash
.venv/bin/python -m pytest --noconftest \
  tests/test_original_graph_efm_benchmark.py \
  tests/test_model_parallel_real_benchmark.py -q
bash -n jobs/original_graph_efm_benchmark.sbatch
```

All nine tests in this reference/configuration check passed locally on
22 November, including the actual full-grid loader with eight spawn workers,
prefetch two and a matching archived first-batch fingerprint. The logging
adapter preserves the original checkpointed AR4 loss and gradients in the
equivalence test. Original CUDA reference job **4981** subsequently completed
with matching batch/preflight checks and exactly equal scalar losses to MP1
at all 50 measured steps. Its archive is
`benchmark_results/model-parallel-real/baseline/original-ar4-workers8-prefetch2-4981/`.
The full original-versus-MP 1/2/4 comparison, source identities, raw-derived
metrics, report and presentation are in
`benchmark_results/model-parallel-real/analysis/ar4-original-vs-mp-4981-4944-4949-4952/`.
Regenerate the analysis with
`.venv/bin/python -m benchmarks.analyze_ar4_reference_comparison`.

Submit on the student cluster, from the repository root:

```bash
cd /home/mfricke/baltic-sea-forecasting
git pull --ff-only origin message-passing
NJORD_WARMUP_STEPS=10 NJORD_MEASUREMENT_STEPS=50 \
  sbatch jobs/original_graph_efm_benchmark.sbatch
```

The job performs ten warmup and fifty measured optimizer steps, matching
4944/4949/4952. Every optimizer step unrolls four forecast steps. This is a
performance window, not a full training epoch or a scientific accuracy study.
The result includes individual measured records, phase distributions,
observed samples/second, allocated/reserved peak GPU memory, loss components,
hardware, source identities, original dataset metadata and staging time.
Output files are:

```text
benchmark_results/model-parallel-real/slurm-original-ar4-JOBID.out
benchmark_results/model-parallel-real/original-graph-efm-ar4-1gpu-JOBID.json
```

The existing three-way comparison command is deliberately specific to
distributed results and cannot accept this new reference as a substitute.
After the reference completes, analyze original 1 GPU versus distributed
1 GPU, then the already controlled distributed 1/2/4-GPU scaling. Check
hardware, batch identity and the measured loss trajectory before interpreting
timing differences. No new loader tuning or partition changes are needed
for this reference run.

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
