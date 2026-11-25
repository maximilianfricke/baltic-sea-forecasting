# Message passing on the original global model

This is a clean rebuild on **global-model@76b30d7**. Every file already tracked
by that reference is unchanged. The previous implementation, results, plots and
documents are preserved at
[`archive/message-passing-2026-11-26`](https://github.com/maximilianfricke/baltic-sea-forecasting/tree/archive/message-passing-2026-11-26)
(snapshot `660744a`). The accompanying `LEARNINGS.md` records the evidence and
remaining uncertainty that informed this rebuild.

## What executes

The original `GraphEFM.training_step` still executes the complete model:

1. Original per-grid input and target-conditioned embeddings.
2. Posterior grid-to-mesh, same-level processing and upward propagation.
3. Latent sampling at the top mesh, with the original global random sequence.
4. Original decoder propagation within/between meshes, down to mesh 0 and grid.
5. Original prior encoder and KL, original masked likelihood and AR feedback.
6. The same computation for all **four autoregressive steps**, including
   full checkpoint replay during backward.

`model.LocalLayer` wraps each original GNN layer. It fetches the remote sender
representations, then calls that original layer's own MLPs and aggregation.
The adapter does not reimplement encoders, decoder, training loss or AR rollout.
Nils's original-layer adapter at `nils-pnemo@69939c9` is an acknowledged design
reference; his branch is not modified or merged.

## Small, explicit components

| File | Responsibility |
|---|---|
| `partition.py` | Validate ownership, assign edges to receivers, preserve original edge order, build unique halo send lists |
| `geographic.py` | Existing balanced geographic development partition; final owners can be supplied externally |
| `transport.py` | P2P or the exact vendored PhysicsNeMo indexed exchange, with reverse gradients |
| `model.py` | Slice static graph tensors and adapt original layers; global noise IDs; sum shared parameter gradients |
| `workload.py`, `data.py` | Construct the original model/dataset and validate the existing shards against ownership |
| `validation.py` | Compare the distributed model directly against original AR4 training and prior-only inference |
| `launch.py` | CPU placement before importing torch; observation and main-thread affinity checks |
| `run.py`, `report.py` | One benchmark runner, raw all-rank records and paired-launch summaries |

There are no import-time process/device/CPU mutations and no global transport
monkey patches. One transport object is explicitly shared by the adapted layers.

## Ownership and communication

Each node has exactly one owner. Each edge lives on its receiver's rank; all
messages needed by that receiver are therefore aggregated locally. A remote
sender needed by several local edges is fetched once per relation call.
The local source layout is owned sources followed by halos ordered by peer and
global ID. Both transports use that same layout and original edge ordering.
Backward sends halo gradients to the original owner and adds contributions.
After backward, replicated model-parameter gradients are summed across ranks.
There is no division by GPU count: this is spatial model parallelism and the
original loss already takes its batch mean while summing spatial contributions.

The default owner maps reproduce the existing 1/2/4-rank geographic shards.
To provide another partition, pass `--owners owners.pt`, saved as:

```python
torch.save({"graph_sha256": graph_fingerprint(model), "world_size": K,
            "owners": {"grid": grid_owners, "mesh_0": mesh0_owners,
                       "mesh_1": mesh1_owners}}, "owners.pt")
```

Each owner array is a CPU int64 vector in original node order. Empty node sets
on a rank are rejected. Shard input must have those exact node IDs. The
`--data-source original` control supports other owner maps without rewriting
the source Zarr: each rank reads an original batch and selects its nodes.
That mode intentionally repeats input work and must be labelled separately.

## Fixed benchmark reference

Defaults: H64 for grid/mesh/edges/latents, decoder L2, prior/posterior L1, batch 2
globally, AR4, FP32, learned isotropic prior, WMSE+KL (beta 1), AdamW .001 /
betas .9,.95 / weight decay .1. Deterministic algorithms and original Normal
argument validation stay enabled. Full checkpoint replay is used at every GPU
count. The historical one-GPU Lightning early-stop policy is a separate factor;
do not silently mix its timing with this controlled replay policy.

Each backend must first match the **original model**, using the first physical
batch: input values, four predictions, loss, all gradients, **two unmasked
AdamW updates**, and prior-only inference. A failed check aborts the job before
timing is accepted. Weights, optimizer and latent seeds are reset per candidate.

The default timing input is four CPU batches containing samples 0..7. They are
prepared with **zero loader workers**, with effective prefetch `null`, and no
pinned buffers. Preparation is outside model timing. No workers remain active
while cached cases run. Live loading is a separate `--input-mode live` option;
worker/prefetch experiments are deliberately not part of the initial comparison.

Full jobs use four fresh paired launches with orders P2P/PN, PN/P2P, P2P/PN,
PN/P2P. Each case has 10 warmup and 50 measured optimizer steps, not 100 epochs.
A smoke has one measured step and is only a functional check.

## CPU placement

Slurm allocation and a rank's actual CPU mask are recorded separately. The
launcher discovers the already granted CPUs inside an explicitly bound `srun`
step. The default `disjoint` policy assigns each rank a distinct share before
torch imports. Set `MP_CPU_POLICY=shared` to test all ranks using the complete
verified allocation. No CPU IDs are hard-coded or taken from outside that mask.

The runner records masks after import/thread setup/CUDA/NCCL's first collective.
It restores the main thread to the verified rank mask once after initialization.
Helper-thread masks are recorded, not repeatedly rewritten. A main-thread mask
change during measurement rejects the result. This retains a controlled start
while avoiding the old repeated blanket repair machinery.

Historical evidence demonstrates that placement mattered, not that disjoint
pairs are universally required. The old narrowing's exact origin and the
remaining small PhysicsNeMo advantage still need isolated native tests.

## Timers and result interpretation

- `cache_prepare_ms` and `setup_ms`: preparation outside the step benchmark.
- `input_wait_ms`: exposed wait for a batch, including loader processing when
  live. With workers it is not the total background decompression work.
- `h2d_ms`: copy to the current GPU and wait for completion.
- `readiness_ms`: rank barrier after copying, outside model timing.
- `model_wall_ms`: forward, backward/checkpoint replay, halo communication,
  shared-gradient sum, optimizer and metric reduction. Includes CPU dispatch
  delays, not only GPU arithmetic. Each phase is synchronized consistently.
- `application_wall_ms`: maximum rank end-to-end step interval, including input,
  copy and readiness. Use this for live-input throughput, not the model interval.

Raw records retain every rank. Additive model phases come from the **same rank
that has the largest model wall time for that step**. Application and model
critical ranks can differ. A model-critical rank's `end_to_end_ms` is retained
as diagnostic data; the application summary uses `application_wall_ms`.

Hashes cover original sources, adapter sources, graph, ownership, initial state,
cached batches and comparable CPU/runtime settings. The summary rejects changes
between launches. Its headline transport ratio is the ratio of median launch
medians; individual paired ratios and a paired-launch bootstrap are also saved.
The single-GPU path has no remote exchange and cannot establish a library win.
CPU correctness runs carry `performance_valid: false`.

## Local verification

```bash
MPLCONFIGDIR=/private/tmp/njord-matplotlib-cache MLFLOW_DISABLE_AGENT_HINT=1 \
  .venv/bin/python -m pytest --noconftest -q tests/test_message_passing.py
bash -n jobs/message_passing.sbatch
bash -n jobs/submit_message_passing.sh
```

The legacy global test fixtures refer to a removed datastore and download
unrelated data; `--noconftest` isolates this suite. Tests exercise 1/2/4 Gloo
ranks, checkpoint on/off and real H64 input/shard identity. For CPU tests only,
PhysicsNeMo's unequal-list all-to-all is emulated using P2P because Gloo does not
support that NCCL operation. Native CUDA/NCCL remains a cluster gate.

## Cluster use

This branch history was deliberately rebuilt. **An existing old checkout needs
an explicit branch update after saving any local changes/results. Ordinary
fast-forward pull cannot switch it to the rebuilt history.** The archive branch
retains the old committed results. No cluster commands were executed by Codex.

From the updated `/home/mfricke/baltic-sea-forecasting`:

```bash
bash jobs/submit_message_passing.sh 1 smoke
# Inspect its numerical gate and result, then:
bash jobs/submit_message_passing.sh 1 full
# Proceed sequentially to 2, then 4, checking each smoke before full.
```

The jobs use the existing original store and geographic shards below
`/scratch/users/mfricke/baltic-sea-forecasting-data/`. They stage the source and
the requested shard layout into job-local `$TMPDIR` once, record staging time,
and write results to `benchmark_results/message-passing/PHASE-Kgpu-JOB/`.
The wrapper requests dphpc for one GPU and dphpc_4gpu for two/four GPUs; the
course wrapper supplies the CPUs. The bound step reuses that allocation.
Four-launch full jobs request one hour; smoke jobs request fifteen minutes.

```bash
.venv/bin/python -m message_passing.report \
  --directory benchmark_results/message-passing/full-4gpu-JOB --repeats 4
```

The two policies for the same GPU count can be submitted separately with
`MP_CPU_POLICY=shared` and `MP_CPU_POLICY=disjoint`. Keep them as separate
experiments; neither their controls nor results should be pooled.
