# Controlled P2P versus PhysicsNeMo comparison

This benchmark changes only the remote halo transport inside **our** distributed
GraphEFM implementation. It does not substitute Nils's graph, rank `.pt` files,
model adapter or edge ordering. It is an AR4 training benchmark, not inference
or convergence testing. Original model and production benchmark files remain
unchanged.

## The differences in the earlier experiments

| Difference | Our archived runs | Nils's supplied experiment | Why it matters |
|---|---|---|---|
| Graph | 8,456 / 2,097 mesh nodes | 7,193 / 1,812 | Different graph computation and latent size |
| Ownership | Independently balanced geographic splits per layer | Weighted G0 partition; other owners inherit nearest G0 | Different edges, rank work and remote senders |
| Graph layout | Original edge order, owned senders followed by halos | CSC ordering and a compact list of required senders | Memory access and packing may differ beyond transport |
| Numerical policy | Deterministic algorithms on | Off for timing | Different CUDA aggregation/backward algorithms may be selected |
| Input | Live Zarr, 8 workers/rank, unpinned buffers | Main timing cycles 4 CPU batches; loader stopped; live cases use 1 worker/rank and pinned buffers | Exposed waits and background CPU contention both change |
| Storage | Rank-local Zarr, staged in TMPDIR | Memory-mapped raw `.pt` shards | Different preparation/I/O cost |
| One-GPU path | Our MP implementation with ocean shard | Original GraphEFM | Different scaling denominator |
| Checkpoint replay | Full replay at 1/2/4 GPUs | Early stopping at 1 GPU, full replay at 2/4 | Different recomputation cost; this makes Nils's one-GPU reference cheaper |
| Samples/statistic | Shuffled repeated-month windows; wall throughput and phase means | Four cycled batches; model-step medians | Different sample sequence and sensitivity to outliers |
| Repetition | One archived job per case | Three launches for main H64 cached cases; one for wider/off cases | Different evidence about repeatability |
| Transport | Neighbour P2P plus custom reverse-gradient exchange | PhysicsNeMo indexed all-to-all plus its autograd | The factor isolated by the new benchmark |

The 4.09x result in Nils's slides additionally turns checkpointing off. That
policy change does not explain his separate checkpoint-on 2.42x H64 curve.
Removing our visible input wait alone cannot explain the whole gap: the model
interval still scales only 1.251x at four GPUs. Active CPU workers may still
inflate that interval through scheduling/launch contention. No controlled
measurement has yet apportioned the gap among these causes.

## Fixed controls in this benchmark

- Our `global_cluster_graph`: 42,348 ocean points, meshes 8,456 / 2,097.
- Our geographic RCB ownership and matching existing Zarr shards. The balanced
  geographic split is used, not the old fixed -5 degree experiment.
- H64, decoder L2, encoder/prior L1, AR4, global batch 2, FP32, WMSE+KL,
  AdamW learning rate .001 / betas .9,.95 / weight decay .1.
- Deterministic algorithms on, full AR-step checkpoint replay on every GPU count.
- Four CPU batches: dataset sample indices 0..7, in order. Each rank prepares
  its cache in the main process with **0 workers**, for 1/2/4 GPUs alike.
  Prefetch is disabled and recorded as `null`; buffers remain unpinned.
  Both candidates reuse those exact CPU tensors. Only one batch is copied to
  a GPU for a step. No loader processes are started.
- SHA256 checks on the graph, initial state, local batch tensors, node/edge IDs,
  halo IDs and packing indices. Rank data IDs must match graph ownership.
- The same local sender/halo layout and the same original MLP/aggregation calls.
  NVIDIA's exact vendored 2.3.0 `indexed_all_to_all_v` and backward are used for
  the PN candidate. No full PhysicsNeMo installation is needed.
- Both use the existing no-traffic local path on one GPU or for a relation with
  zero global remote traffic. PN does not perform extra self-feature selection
  in this test. Nils's complete adapter/layout is a separate future comparison.

Each pair first checks predictions, loss, synchronized gradients and **two full
unmasked AdamW updates**. It aborts before accepting timings on failure. Reset
weights, buffers, optimizer and latent RNG between timed candidates. Three fresh
torchrun launches give order P2P/PN, PN/P2P, P2P/PN. Each full candidate has 10
warmup + 50 measured optimizer steps. A smoke has one pair and one measured
step, so its rate is not a steady-state benchmark.

Smoke job 6107 used the earlier eight-worker cache preparation. Keep that
historical functional result, but use the new zero-worker policy consistently
for all full runs. Existing benchmark/model files and scientific data are
unchanged. The controls hash prevents pooling repeats with different policies.

The headline is **cached model wall-step time**, starting after copy and rank
readiness. It includes training, reverse/replayed communication, parameter
gradient synchronization, optimizer and metric reduction. CPU preparation,
H2D, barriers before timing, startup and file staging are outside this named
timer. H2D is separately recorded. Do not compare this rate directly with our
old application wall throughput. All rank records and critical-rank phases are
saved. The summary uses a ratio of medians of per-launch step medians and a
paired-launch bootstrap. Three independent pairs offer limited uncertainty
resolution, and the smoke has no independent-run confidence interval.

## Local verification

```bash
cd /Users/maximilian/ETH/Master/4_semester/DPHPC/baltic-sea-forecasting
MPLCONFIGDIR=/private/tmp/njord-matplotlib-cache MLFLOW_DISABLE_AGENT_HINT=1 \
  .venv/bin/python -m pytest --noconftest -q \
  tests/test_halo_transport_comparison.py tests/test_model_parallel_real_benchmark.py
bash -n jobs/submit_halo_transport_comparison.sh
bash -n jobs/halo_transport_comparison.sbatch
```

`--noconftest` follows the existing benchmark test workflow: legacy global
fixtures import the removed `npyfilesmeps` datastore and attempt unrelated data
downloads. The local tests use 1/2/4 Gloo processes, a small original AR4 model,
checkpointing on/off, unequal exchange sizes and duplicate sender use. They
also check our actual H64 graph and exact rank-cache sample identity.

**Local PN transport is explicitly emulated:** Gloo does not support the
library's unequal-list all-to-all. A test-only shim replaces that lowest call
with CPU P2P. The original PN packing and gradient code execute unchanged.
There is no CUDA/MPS device available to this session. Native CUDA/NCCL
correctness and timings remain the cluster smoke gate.

## Get the new files to the cluster

No commit/push is required by these instructions. The benchmark records source
hashes and the current git state, including an uncommitted worktree.

First on the **cluster**, update the existing branch, without merging Nils:

```bash
cd /home/mfricke/baltic-sea-forecasting
git pull --ff-only origin message-passing
```

Then on the **Mac**, copy only this new benchmark code:

```bash
cd /Users/maximilian/ETH/Master/4_semester/DPHPC/baltic-sea-forecasting
rsync -av --relative --exclude='__pycache__/' --exclude='*.pyc' \
  benchmarks/compare_halo_transports.py \
  benchmarks/halo_transport_backends.py \
  benchmarks/vendor/ \
  benchmarks/HALO_TRANSPORT_COMPARISON.md \
  jobs/submit_halo_transport_comparison.sh \
  jobs/halo_transport_comparison.sbatch \
  tests/test_halo_transport_comparison.py \
  mfricke@student-cluster1.inf.ethz.ch:/home/mfricke/baltic-sea-forecasting/
```

The old datasets already uploaded are sufficient. No Nils `.pt` shards, NPZs,
new graph or new Python package are needed.

## Submit sequentially on the cluster

Preflight on the login node only checks imports/files/syntax:

```bash
cd /home/mfricke/baltic-sea-forecasting
.venv/bin/python --version
bash -n jobs/submit_halo_transport_comparison.sh
bash -n jobs/halo_transport_comparison.sbatch
.venv/bin/python -m benchmarks.compare_halo_transports --help
ls -lh /scratch/users/mfricke/baltic-sea-forecasting-data/physical_repeat_16/graphs/global_cluster_graph/mesh_features.pt
ls /scratch/users/mfricke/baltic-sea-forecasting-data/partitioned/physical_repeat_16/geographic_rcb_ws{1,2,4}/manifest.json
bash jobs/submit_halo_transport_comparison.sh 1 smoke
```

Watch the submitted job, replacing `JOBID` with its printed number:

```bash
JOB=JOBID
squeue -j "$JOB"
tail -n 60 -f "benchmark_results/halo-transport/slurm-smoke-1g-${JOB}.out"
```

After it ends, Ctrl-C exits tail. Check state and the complete pair:

```bash
sacct -j "$JOB" --format=JobID,State,Elapsed,AllocTRES,ExitCode
.venv/bin/python -m benchmarks.compare_halo_transports summarize \
  --directory "benchmark_results/halo-transport/smoke-1gpu-${JOB}" --repeats 1
```

Require `COMPLETED`, exit 0:0, `all_equivalence_checks_passed: true`, and both
case records. The one-GPU pair is a local-path baseline; it does not exercise
network exchange. Then submit **one command at a time**, inspect completion
and results, and proceed:

```bash
bash jobs/submit_halo_transport_comparison.sh 1 full
# After 1 full completes and is checked:
bash jobs/submit_halo_transport_comparison.sh 2 smoke
# After the native 2-GPU equivalence gate passes:
bash jobs/submit_halo_transport_comparison.sh 2 full
# After 2 full completes and is checked:
bash jobs/submit_halo_transport_comparison.sh 4 smoke
# After the native 4-GPU equivalence gate passes:
bash jobs/submit_halo_transport_comparison.sh 4 full
```

For each later job, set the printed job number, GPU count and phase:

```bash
JOB=JOBID
GPUS=2
PHASE=full
REPEATS=3
tail -n 60 -f "benchmark_results/halo-transport/slurm-${PHASE}-${GPUS}g-${JOB}.out"
# After it ends, Ctrl-C, then:
sacct -j "$JOB" --format=JobID,State,Elapsed,AllocTRES,ExitCode
.venv/bin/python -m benchmarks.compare_halo_transports summarize \
  --directory "benchmark_results/halo-transport/${PHASE}-${GPUS}gpu-${JOB}" \
  --repeats "$REPEATS"
```

Use `REPEATS=1` for smoke, 3 for full. Every job writes a unique directory with
staging time, GPU UUIDs/topology, Slurm allocation, package versions, per-repeat
validation/control hashes, raw per-rank steps and `comparison.json`.
`physicsnemo_speedup_over_p2p > 1` favors PN; below 1 favors P2P.

Accounts follow your existing setup: `dphpc` for one GPU, `dphpc_4gpu` for 2/4.
Jobs request GTX 1080 Ti, one node and 24 GiB host memory, without overriding
the course wrapper's CPU allocation. Smoke requests 15 minutes, full one hour.
These are requested limits, not measured duration estimates. Check `courses`
for current course-specific budgets. Staging uses the existing rank layout
once per job in shared job-local `$TMPDIR`; no data-intensive work runs on the
login node.

ETH documents GPU-count/model requests and job-local temporary storage here:
https://isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs

After the three GPU counts complete, copy this results tree to the Mac:

```bash
cd /Users/maximilian/ETH/Master/4_semester/DPHPC/baltic-sea-forecasting
rsync -av --partial \
  mfricke@student-cluster1.inf.ethz.ch:/home/mfricke/baltic-sea-forecasting/benchmark_results/halo-transport/ \
  benchmark_results/halo-transport/
```
