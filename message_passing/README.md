# First P2P implementation on global-model

Start from **global-model@76b30d7**, keep every original tracked file unchanged,
and add only our spatial P2P message passing. Measure this version first.
Add a workaround or optimization only after a new run demonstrates its need.

The previous experiments remain on `archive/message-passing-2026-11-26`.
The superseded rebuild with CPU assignment is saved by the tag
`archive/message-passing-with-affinity-2026-11-26`.

## What is added

- `partition.py`: accept node owners, assign each edge to its receiver, and
  construct unique remote-sender lists while retaining original edge order.
- `transport.py`: our `torch.distributed` P2P sends/receives and reverse-gradient
  accumulation. No PhysicsNeMo dependency or comparison machinery.
- `model.py`: wrap the original GNN layers. Their MLPs, aggregation, encoders,
  decoder, prior/posterior, loss, AR feedback and checkpoint implementation
  remain the original code. Globally indexed latent noise preserves the
  reference samples; shared parameter gradients are summed across ranks.
- `data.py`: original live WeatherDataset input and a reproducible shared sample
  order. Each rank selects its owned nodes on the CPU before copying to its GPU.
- `validation.py`: compare predictions, loss, gradients, two complete unmasked
  AdamW updates and prior-only inference directly against the original model.
- `run.py`: one P2P run, raw per-rank observations and phase timings.

The geographic split is a development default; `--owners` accepts an external
ownership file. The final partitioning is the other team members' work.

## What this first version deliberately leaves ordinary

There is **no manual CPU binding**, no affinity restoration, no per-rank CPU
shares, no affinity-specific launcher and no rejection because a CPU mask
changed. CPU masks and PyTorch thread counts are only observed. Slurm and the
standard torchrun launcher supply their normal inherited process settings.
There is no override of PyTorch's original checkpoint early-stop behaviour.

Input comes from the **original full Zarr store**, using the unchanged original
WeatherDataset. There is no prepared ocean-only store, rank-specific Zarr,
preloaded batch cache, added data staging or data-loader tuning. Each rank reads
the same samples and selects its own spatial nodes. This intentionally retains
redundant input work as a simple initial control; it does not claim scalable I/O.

The selected measured reference (original-model job 4981) used **8 workers**.
We retain 8 per rank and DataLoader's default prefetch 2, spawn processes and
persistent workers. These are the chosen reference settings, not a claim that
the generic global-model CLI default is 8 (it is 4). At 2/4 ranks the total
worker count becomes 16/32; that cost is measured rather than corrected upfront.
`--workers 0` is available for local CPU correctness tests.

## Model and measurements

H64 for grid/mesh/edges/latents, decoder L2, prior/posterior L1, global batch 2,
AR4, FP32, WMSE+KL and checkpointing. Optimizer: AdamW .001, betas .9/.95,
weight decay .1. Original deterministic and Normal-validation policies stay on.
On one GPU all graph nodes are local and the P2P transport returns them directly.

`data_read_ms` is the exposed DataLoader wait, not all background worker work.
`partition_select_ms` measures selection of owned points; `host_to_device_ms`
measures copying and completion; `rank_wait_ms` measures readiness waiting.
The model interval includes forward/backward halo communication, original
checkpoint replay, parameter-gradient synchronization, optimizer and loss
reduction. Model phase sums use the same critical rank per step; application
throughput uses the maximum rank end-to-end interval. Every rank's raw records
and CPU observations are retained. A slow run is still recorded for diagnosis.

## Verify locally

```bash
MPLCONFIGDIR=/private/tmp/njord-matplotlib-cache MLFLOW_DISABLE_AGENT_HINT=1 \
  .venv/bin/python -m pytest --noconftest -q tests/test_message_passing.py
```

The isolated suite avoids legacy fixtures for a removed datastore. It tests
P2P on 1/2/4 Gloo ranks, checkpoint on/off, unequal and send-only halo traffic,
and H64 original-data configuration. CPU timings are explicitly not GPU
performance evidence. Native H64 CUDA/NCCL needs the first cluster test.

## First cluster run

Only the user submits cluster jobs. From an updated checkout:

```bash
bash jobs/submit_message_passing.sh 1 smoke
```

Inspect the result before proceeding to full and then 2/4 GPUs. Full is one
launch with 10 warmup + 50 measured steps; smoke is one measured step. Existing
original data under `/scratch/users/mfricke/baltic-sea-forecasting-data/` are
sufficient. Outputs go to `benchmark_results/message-passing/PHASE-Kgpu-JOB/`.
This branch was rebuilt, so an old checkout needs an explicit update after
preserving local changes; ordinary fast-forward pull cannot cross the rewrite.
