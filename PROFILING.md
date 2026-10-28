Baseline: Njord = `graph_efm`, one GPU, batch size 1, AdamW, FP32,
one forecast step, gradient checkpointing, KL weight 0.1, no CRPS phase.
The script explicitly selects the supplied hierarchical graph.

```bash
sbatch profile.sh
# Optional online W&B logging (uses your existing login):
WANDB_MODE=online sbatch profile.sh
# Refine the interval between the largest success and first OOM:
WIDTHS="160 192 224" sbatch profile.sh
# Depth scaling: hold width fixed, submit one job per depth:
WIDTHS=64 sbatch profile.sh --processor_layers 6
WIDTHS=64 sbatch profile.sh --processor_layers 12
```

Results: `output/profile-JOBID/results.csv`, plus a log per width.
`step_ms` is synchronized wall time for forward + loss + backward (including
checkpoint recomputation) + optimizer, averaged over 10 batches after 3 warmup
batches. It excludes loading and transferring batches, validation, saving,
and startup. Batches can span epochs on this small dataset.
Peak VRAM includes warmup and Adam state initialization. `allocated` counts
PyTorch tensors; `reserved` includes its caching allocator. Neither includes
all CUDA context/library memory. OOM peaks exclude the unsuccessful request.
W&B receives the same successful-run metrics under `profile/`.

For your report, plot peak allocated GiB and step_ms against width, then repeat
for depth with width fixed. Report the largest successful value and first OOM
as a bracket, with GPU, precision, graph, batch size and training phase fixed.
For graph-size scaling, repeat with other compatible graph files and count
their nodes/edges; the supplied two graphs also differ in topology.
The 0.25° directory currently lacks the training YAML, ocean store and graphs;
complete it before treating it as a runnable second resolution.

Code map: `train_model.py` builds data/model/trainer; `weather_dataset.py`
loads sequences; `graph_efm.py::training_step` computes the loss;
`ar_model.py::unroll_ckpt_func` checkpoints a complete forecast step;
`interaction_net.py` implements message passing. Checkpointing recomputes
activations in backward, but one step's working memory must still fit.

Inside an allocated GPU shell, activate the environment as in `profile.sh`.
Capture a short representative run with Nsight Systems:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
  --capture-range=nvtx --nvtx-capture=profile --capture-range-end=stop \
  -o output/baseline \
  python -m neural_lam.train_model \
  --config_path sample_data/global_ocean_1.yaml \
  --model graph_efm --graph global_hierarchical_graph \
  --hidden_dim 64 --batch_size 1 --num_workers 2 \
  --kl_beta 0.1 --precision 32-true \
  --profile_warmup 3 --profile_steps 3
```

The NVTX `profile` range covers measured batches. Use Nsight Compute only on
selected expensive kernels from that trace; its replay overhead distorts timing.
Use runs without Nsight for the scaling CSV.

## Forward, backward, and graph stages

```bash
sbatch profile_stages.sh
# Change the representative model if needed:
sbatch profile_stages.sh --hidden_dim 256 --processor_layers 6
# Optional Nsight capture with the same stage NVTX annotations:
NSYS=1 sbatch profile_stages.sh
```

This short job produces `output/stages-JOBID/` containing:

- `results.csv`: the existing runtime and memory summary.
- `results.phases.csv`: CUDA-event times per measured step for forward + loss,
  backward, optimizer update, and the whole step.
- `results.stages.csv`: timings for embeddings, grid-to-mesh, mesh processing
  (same-level and up/down hierarchy), and mesh-to-grid, with module names to
  distinguish the encoder, prior, and decoder.
- `figures/`: phase, stage, and per-step variability figures, in PNG and PDF.
  Bars show means and sample standard deviations across measured steps.
- `train.log`, and optionally `trace.nsys-rep`.

Timings use CUDA events on the current stream, resolved after the existing
batch-end synchronization. They measure elapsed intervals, including CPU launch
gaps, rather than summed kernel execution time. Instrumentation adds overhead;
keep these runs separate from the original scaling baseline. No synchronization
is added between individual graph stages.

**Backward includes checkpoint recomputation.** The stage figure distinguishes
the original forward from the forward operations replayed inside backward.
Replay can stop partway through a module when its last needed tensor is saved.
Stage timings do **not** include that stage's autograd gradient kernels, which
are included in the total backward time. They must not be presented as a
complete per-stage backward breakdown. Embeddings and message-passing stages
also exclude projections, distribution operations, and loss bookkeeping outside
the selected scopes; their sums need not equal the full phase times.

Optimizer timing starts after the forward/backward closure, avoiding counting
that closure twice. The whole-step bar includes framework work outside the
three named phases. W&B receives means under `timing/`, including
`timing/forward_ms`, `timing/backward_ms`, and `timing/recompute/g2m_ms`.

To regenerate figures:

```bash
python scripts/plot_stage_profile.py output/stages-JOBID/results.csv
```

For figures linked from a report on GitHub, export to a versioned directory:

```bash
python scripts/plot_stage_profile.py output/stages-JOBID/results.csv \
  --output-dir figures/stages-JOBID
```

Commit the exported figures along with the report. Selected CSV snapshots used
by `BASELINE_REPORT.md` are stored in `reports/baseline/`. Working files under
`output/` remain ignored; report links should use the versioned copies.

For optional Nsight, the job uses `/usr/local/bin/nsys` to avoid the older binary
selected by the CUDA module. Set `NSYS_BIN` if the GPU node uses a different
compatible installation. The script prints its version before capture.
