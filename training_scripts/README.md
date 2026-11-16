# Training scripts

Everything here is submitted to Slurm through `train.sh`. There are three layers:

```
launch_sweep.sh          queues one Slurm job per configuration
  └─ train.sh            the sbatch job: 1 node, 8 GPUs, 72h, loads the conda env
       └─ seacast_pretrain.sh / seacast_finetune.sh / ...
                         the actual `python -m neural_lam.train_model` command
```

Each layer forwards its arguments to the next one, and the training scripts append
those arguments **after** their own defaults. `argparse` keeps the last occurrence
of a flag, so anything in a training script can be overridden from the command
line without editing the file.

Run all commands from the repository root.

## Running a single job

```sh
# the default (seacast_pretrain.sh, unmodified)
sbatch training_scripts/train.sh

# a specific script
sbatch training_scripts/train.sh training_scripts/seacast/seacast_pretrain.sh

# a specific script with overrides
sbatch training_scripts/train.sh training_scripts/seacast/seacast_pretrain.sh \
    --hidden_dim 512 --graph global_cluster_1_deg_10_refinement_4_levels
```

Logs go to `slurm_logs/<jobid>_<jobname>.out`. The first line of the log echoes the
resolved command, so you can always see which settings a job actually ran with.

## Two-GPU distributed GraphEFM smoke

Before running full model-parallel benchmarks, validate the receiver-owned
partitioning and differentiable halo exchange on two CUDA devices:

```sh
sbatch training_scripts/distributed_cuda_smoke.sbatch
```

The job runs two NCCL checks. The first validates one distributed message-
passing layer, including halo gradients and parameter-gradient reduction. The
second validates the complete hierarchical distributed GraphEFM core (prior,
posterior, Grid-to-Mesh, mesh hierarchy and Mesh-to-Grid) against the
unpartitioned reference computation. A successful log contains both:

```text
Two-GPU NCCL distributed message-passing smoke: PASS
Two-GPU hierarchical GraphEFM core smoke: PASS
```

This is a correctness smoke, not yet an end-to-end Lightning training or
performance benchmark. It writes two JSON files to `benchmark_results/` with
the commit, environment, correctness errors, per-rank memory and diagnostic
timings. Both JSON files explicitly set
`"comparable_to_training_baseline": false`, because the smoke uses synthetic
toy graphs rather than the real benchmark workload.

## Real-data 1-vs-2-GPU model-parallel benchmark

After the synthetic smoke passes, run the controlled strong-scaling pair on
the physical 1-degree Njord Zarr data and the real `global_cluster_graph`:

```sh
bash jobs/submit_model_parallel_real.sh 1
# Wait for completion and inspect the JSON before submitting the next job.
bash jobs/submit_model_parallel_real.sh 2
```

Both cases use the same global batch size (2), initial seed, GraphEFM model
(hidden dimension 64, two processor layers), FP32 precision and one-step ELBO
training workload. Rank zero loads every batch once and broadcasts the same
physical sample to the other model-parallel ranks. The measurement includes
the real embedding frontend, partitioned prior/posterior/decoder, likelihood
and KL loss, backward halo exchange, summed shared-parameter gradients and an
AdamW step.

The submission script currently defaults to a diagnostic calibration run:
zero data-loader workers, one warm-up step, one measured step, progress output
for every phase and a 60-minute limit. This preserves the real per-step
workload while exposing whether initialization, data loading, forward or
backward computation is responsible for a slow run. Once calibration succeeds,
submit a longer quiet measurement with identical settings for both GPU counts:

```sh
NJORD_NUM_WORKERS=2 \
NJORD_WARMUP_STEPS=5 \
NJORD_MEASUREMENT_STEPS=20 \
NJORD_PROGRESS_INTERVAL=0 \
bash jobs/submit_model_parallel_real.sh 1
```

The default dataset location is:

```text
/scratch/users/$USER/baltic-sea-forecasting-data/physical_repeat_16
```

Override it at submission time with `NJORD_DATA_ROOT=/path/to/data`. The
physical-repeat dataset retains real Njord fields, graph geometry and Zarr I/O
but repeats the available January sequence to create a stable performance
window; it must not be described as additional scientific observations.

Results are written under `benchmark_results/model-parallel-real/`. The JSON
records per-rank partitions, environment, batch fingerprint, preflight loss,
per-step data/forward/backward/gradient-sync/optimizer times, memory and
throughput. Compare the finished pair with:

```sh
python benchmarks/compare_model_parallel_real.py \
  --one-gpu benchmark_results/model-parallel-real/real-mp-1gpu-JOBID.json \
  --two-gpu benchmark_results/model-parallel-real/real-mp-2gpu-JOBID.json \
  --output benchmark_results/model-parallel-real/comparison-1-vs-2.json
```

The comparison checks that workload, GPU model and first physical batch match,
then reports observed wall-time speedup, parallel efficiency, phase times,
memory and numerical loss equivalence. It does not call a timing result an
accuracy result; forecast accuracy requires separate validation rollouts.

In this first stage the embedding frontend is replicated and the GNN core is
partitioned. Autoregressive rollouts longer than one step are intentionally
rejected until differentiable full-grid reconstruction across ranks is
implemented.

## Running a sweep

Edit the `RUNS` array at the top of [`launch_sweep.sh`](launch_sweep.sh) — one line
per job, containing the flags for that run:

```bash
RUNS=(
    "--hidden_dim 256 --graph global_cluster_1_deg_20_refinement_4_levels"
    "--hidden_dim 256 --graph global_cluster_1_deg_15_refinement_4_levels"
    "--hidden_dim 512 --graph global_cluster_1_deg_20_refinement_4_levels --batch_size 2"
)
```

Then queue them all:

```sh
bash training_scripts/launch_sweep.sh
```

Options:

| Command | Effect |
| --- | --- |
| `DRY_RUN=1 bash training_scripts/launch_sweep.sh` | Print the `sbatch` commands without submitting. Do this first. |
| `DEPEND=15131 bash training_scripts/launch_sweep.sh` | Start every job only after job 15131 finishes (`afterany`). |
| `bash training_scripts/launch_sweep.sh <script>` | Sweep a different training script than `seacast_pretrain.sh`. |

Each job is named `SeaCast_h<hidden_dim>_<short graph>` (e.g. `SeaCast_h256_1_deg_20`),
so the runs land in separate log files. `squeue -u $USER` shows the queue.

The `RUNS` lines are not limited to `--hidden_dim` and `--graph` — any flag of
`neural_lam.train_model` can be added, e.g. `--batch_size`, `--lr`,
`--processor_layers`, `--load`.

## SeaCast pretraining

[`seacast/seacast_pretrain.sh`](seacast/seacast_pretrain.sh) — `graph_det` on the 1°
data (`data/global_ocean_1_density.yaml`), `hidden_dim 256`, `processor_layers 3`,
`batch_size 3`, muon optimizer, `--scheduler deterministic`.

```sh
# single run with the defaults
sbatch training_scripts/train.sh training_scripts/seacast/seacast_pretrain.sh

# sweep hidden_dim / graph
DRY_RUN=1 bash training_scripts/launch_sweep.sh   # check
bash training_scripts/launch_sweep.sh             # submit
```

Graphs available at 1° (`data/graphs/`):

```
global_cluster_1_deg_20_refinement_4_levels
global_cluster_1_deg_15_refinement_4_levels
global_cluster_1_deg_10_refinement_4_levels
```

Checkpoints are written to `saved_models/train-graph_det-<layers>x<hidden_dim>-<date>-<id>/`
as `min_val_loss.ckpt` and `last.ckpt`.

## SeaCast finetuning

[`seacast/seacast_finetune.sh`](seacast/seacast_finetune.sh) — the same model at
**1/4°** (`data/global_ocean_1_4.yaml`), with `--scheduler finetune_deterministic`
and `--load` pointing at a pretrained checkpoint.

Two things differ from pretraining and are easy to get wrong:

1. The graph must be a **`1_4`** graph, matching the 1/4° config.
2. `--load` in the script is the literal placeholder `CHECKPOINT`. Always pass a real
   checkpoint path, either by editing the script or by overriding it on the command
   line (the override wins).

```sh
sbatch training_scripts/train.sh training_scripts/seacast/seacast_finetune.sh \
    --load saved_models/train-graph_det-3x256-08_04_23-0203/min_val_loss.ckpt
```

To sweep finetuning runs, put the matching graphs and checkpoints in `RUNS` and pass
the finetune script to the launcher:

```bash
RUNS=(
    "--hidden_dim 256 --graph global_cluster_1_4_deg_20_refinement_4_levels --load saved_models/<pretrain_run_a>/min_val_loss.ckpt"
    "--hidden_dim 256 --graph global_cluster_1_4_deg_15_refinement_4_levels --load saved_models/<pretrain_run_b>/min_val_loss.ckpt"
)
```

```sh
bash training_scripts/launch_sweep.sh training_scripts/seacast/seacast_finetune.sh
```

Use `DEPEND=<pretrain_jobid>` to queue the finetuning behind a pretraining job that is
still running.

Graphs available at 1/4°:

```
global_cluster_1_4_deg_20_refinement_4_levels
global_cluster_1_4_deg_15_refinement_4_levels
global_cluster_1_4_deg_10_refinement_4_levels
```

## Notes

- `train.sh` requests 8 GPUs on 1 node for 72 hours. Change the `#SBATCH` header there
  if a run needs different resources — it applies to every job in a sweep.
- W&B run names are `train-<model>-<processor_layers>x<hidden_dim>-<date>-<id>` and do
  **not** contain the graph. When comparing runs that share a `hidden_dim`, tell them
  apart by the Slurm job name or the graph recorded in the W&B config.
- Raising `hidden_dim` may need a smaller `batch_size` to fit in memory; add
  `--batch_size 2` to that run's line in `RUNS`.
- [`debug.sh`](debug.sh) runs training directly (no `srun`, `wandb off`) for quick
  interactive checks on an allocated node.
