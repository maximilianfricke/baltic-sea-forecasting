# NJORD baseline benchmark

This benchmark measures the unmodified single-GPU training path before any
parallel implementation is introduced. Official comparisons must use the same
GPU type, dataset, model configuration, precision, software environment, and
measurement protocol.

## Protocol

- Dataset: processed 1-degree global ocean data.
- Model: `graph_efm` with the existing interaction-network graph layers.
- GPU: one NVIDIA GTX 1080 Ti. This type is used because the student cluster
  permits later jobs with up to four identical GPUs.
- Precision: FP32 for the first reference result.
- Workload: batch size 1 and one autoregressive training step.
- Timing: 5 warm-up steps followed by 30 measured optimizer steps.
- Excluded from timing: validation, plotting, external experiment logging, and
  checkpoint writes.
- Reported: GPU phase timings; compute and end-to-end wall time; data-wait
  time; both compute-only and end-to-end throughput; allocated/reserved/peak
  GPU memory; CPU RSS; parameter, node and edge counts; Git revision; device
  and software metadata.

Run every reported configuration at least three times. Report the median of
the run means and retain every JSON file. Do not compare a local CPU or Apple
GPU run with a cluster GPU result.

## Cluster execution

The authoritative reference is the ETH D-INFK
[Student Cluster documentation](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster).
Log in with `ssh student-cluster.inf.ethz.ch`, then run `courses`, `space`, and
`limits` before submitting anything. `courses` shows the account tag and
remaining compute budget; `space` shows the writable course/team paths; and
`limits` shows current login-node restrictions. Login nodes are only for
setup, building, and submission, never for benchmark computation.

Keep the repository, environment, 51-GiB dataset, and results in the course or
team path reported by `space`, normally below `/work/courses`. Home has only
20 GiB. Personal scratch has 100 GiB but is not backed up. A job-local
`$TMPDIR` is fast SSD storage but is limited to 40 GiB and deleted after the
job, so the complete current dataset cannot be staged there.

Use Git to obtain the code. Transfer the dataset through `scp`/SFTP to
`student-cluster.inf.ethz.ch`, or mount `smb://student-files.inf.ethz.ch` on
macOS and select the `work` share. Do not copy the dataset into the home share.

Create the output directory before submission because Slurm opens the log file
before the job script starts. Replace `<course-tag>` with the DPHPC account tag:

```bash
cd /path/to/baltic-sea-forecasting
mkdir -p benchmark_results
sbatch --account=<course-tag> jobs/benchmark_1080ti.sbatch
```

The virtual environment defaults to `.venv`. Set `NJORD_VENV` when the cluster
environment lives elsewhere. The main experiment settings can be changed
without editing the job script:

```bash
HIDDEN_DIM=128 PROCESSOR_LAYERS=3 GRAD_CHECKPOINTING=1 \
  sbatch --account=<course-tag> jobs/benchmark_1080ti.sbatch
```

Use the same command with `GRAD_CHECKPOINTING=0` for the direct comparison.
Increase `HIDDEN_DIM` until the largest stable single-GPU configuration and
the first out-of-memory configuration are known. Record failures from the
Slurm log as capacity limits; do not silently change batch size or precision.
The GTX 1080 Ti requires a PyTorch build using a CUDA 12.x runtime; verify the
`cuda_runtime` and compute capability stored in each JSON result. In
particular, CUDA 13 no longer supports the GTX 1080 Ti, so all supplied jobs
initialize the module system and select `cuda/12.6.3` explicitly. The
prebuilt PyTorch CUDA 12.6 wheels retain Pascal (`sm_61`) support required by
the GTX 1080 Ti; the CUDA 12.9 PyTorch wheels do not.

The official GTX 1080 Ti job limits are 1--4 GPUs, 2 CPU cores, 24 GiB host
RAM, and 40 GiB job-local temporary storage. The scripts stay inside those
limits. Multi-GPU DDP ranks share the two allocated CPU cores using Slurm
overcommit and therefore use `num_workers=0`; this intentionally prevents
hidden CPU over-allocation from contaminating the GPU scaling comparison.

GTX 1080 Ti nodes normally use a 250 W GPU power limit, but the cluster may
reduce this progressively to 125 W when a rack is heavily loaded. Every job
therefore records power and clocks. Keep all three repeats, report the observed
power limits, and do not attribute a throttled run to the parallel algorithm.

## Local smoke test

This checks that the harness works and can provide local development numbers;
it is not an official cluster performance result. Apple Silicon automatically
uses MPS. PyTorch-Geometric scatter operations are non-deterministic on MPS,
so strict determinism remains enabled only for the official CUDA baseline.

```bash
python -m neural_lam.train_model \
  --config_path data/global_ocean_1.yaml \
  --model graph_efm --graph global_cluster_graph \
  --hidden_dim 4 --hidden_dim_grid 4 --hidden_dim_mesh_nodes 4 \
  --hidden_dim_edge 4 --processor_layers 1 --batch_size 1 \
  --precision 32 --epochs 1 --ar_steps_train 1 --ar_steps_eval 1 \
  --num_workers 0 --devices auto --n_example_pred 0 \
  --benchmark_warmup_steps 1 --benchmark_steps 2 \
  --benchmark_output benchmark_results/local-smoke.json
```

The JSON output is the source of truth for later plots and speedup tables.
Strong scaling should keep the model and problem fixed. Capacity/weak-scaling
experiments may increase hidden dimension, but must be reported separately.
CUDA peak memory comes from the allocator's resettable peak counter. MPS has no
equivalent counter, so its reported peak is the largest allocation observed at
a synchronized forward/backward boundary; driver memory also includes caching.

Every supplied Slurm benchmark rebuilds
`benchmark_results/all-benchmarks.csv` after it finishes. The CSV contains one
row per raw benchmark JSON across the baseline, single-GPU sweeps, scaling, and
profiler runs. Raw JSON files remain the source of truth and are never replaced
by the summary.

Rebuild the combined table manually at any time with:

```bash
bash jobs/update_benchmark_summary.sh benchmark_results
```

## Complete single-GPU experiment matrix

The sweep varies one factor at a time around the controlled GraphEFM reference
(`hidden_dim=64`, three processor layers, batch size one, one autoregressive
step, FP32). It covers hidden dimensions, checkpointing, DataLoader workers,
processor depth, batch size, autoregressive unroll, mixed precision, inference
ensemble size, and inference horizon.

Run it locally for exploratory MPS numbers:

```bash
python benchmarks/run_local_sweep.py \
  --group all \
  --output-dir benchmark_results/local-sweep \
  --warmup 3 --steps 10
```

Run the official matrix three times on identical cluster GPUs:

```bash
bash jobs/submit_single_gpu_sweep.sh <course-tag>
```

The single-GPU limits are established by increasing hidden dimension and batch
size until the first out-of-memory failure. Preserve the Slurm log for failed
points; an OOM boundary is a result, not a configuration to silently omit.

## Multi-GPU scaling

The distributed suite measures both training and inference on 1, 2, and 4
identical GPUs, with three repeats. The cluster permits only one running GPU
job and two queued jobs for a user/team, so each GPU count is one bundled job.
Submit the next count after the preceding job has completed:

```bash
bash jobs/submit_scaling_sweep.sh <course-tag> 1
squeue
# after completion:
bash jobs/submit_scaling_sweep.sh <course-tag> 2
# after completion:
bash jobs/submit_scaling_sweep.sh <course-tag> 4
```

- Strong scaling holds the global batch at four and divides it across GPUs.
- Weak scaling holds the local batch at one, so total work grows with GPUs.
- The JSON uses the slowest rank for each measured step and retains every
  rank's raw records for load-imbalance inspection.
- Training records an analytical DDP gradient payload and ring-all-reduce wire
  volume. The profiler trace is the source of measured NCCL duration.
- Inference scaling is replicated data parallelism. It has almost no per-step
  model communication and therefore isolates scheduling and input-pipeline
  overhead from training gradient synchronization.

This is the data-parallel reference, not the later graph/tensor-parallel
implementation. A graph-partitioned implementation should use the identical
protocol and additionally report halo nodes/edges, neighbor GPU count, bytes
sent and received per message-passing layer, compute imbalance, memory
imbalance, and useful-compute/communication overlap.

## Operator and communication profiling

Profiler runs are deliberately separate from timing runs:

```bash
sbatch --account=<course-tag> jobs/profile_1080ti.sbatch
sbatch --account=<course-tag> jobs/profile_multi_gpu.sbatch
```

Do not queue these while a benchmark bundle is still running. Check `squeue`
and cancel an incorrect job immediately with `scancel <job-id>` to preserve
the team's course budget.

The trace contains semantic GraphEFM regions for embeddings, encoder, prior,
decoder, and loss plus PyTorch operator, memory, shape, FLOP, CUDA, and NCCL
events. Inspect the Chrome trace for the critical path, kernel gaps, scatter
operations, all-reduce duration, and communication/compute overlap. Do not use
profiled step time as the headline performance number.

## Plots and report

Create PNG/PDF figures, a flat CSV, and a Markdown interpretation from any set
of completed result files:

```bash
python benchmarks/plot_results.py benchmark_results \
  --output-dir benchmark_results/plots
python benchmarks/analyze_results.py benchmark_results \
  --output benchmark_results/plots/analysis.md
```

Generated figures cover time breakdown and variance, hidden dimension,
processor depth, batch size, autoregressive steps, input workers, precision,
checkpointing, inference ensemble/horizon, estimated gradient payload, and
training/inference strong and weak scaling when distributed data are present.

Every Slurm job also samples GPU utilization, memory, power, temperature, and
clocks at 200 ms. Convert those logs into per-GPU plots and a summary including
integrated energy:

```bash
python benchmarks/plot_telemetry.py benchmark_results \
  --output-dir benchmark_results/plots/telemetry
```
