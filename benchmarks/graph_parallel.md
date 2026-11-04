# GraphEFM graph parallel training

Mesh processing in the variational encoder, learned prior, and decoder is
partitioned by receiver ownership. Grid work, g2m/m2g, output losses, and model
weights remain replicated. The GPUs process one logical batch together;
adding GPUs does **not** increase the global batch size. This first version
supports one host and hierarchical GraphEFM graphs.
FP16 with dynamic gradient scaling is rejected: overflow handling would need
additional synchronization. Use FP32 for the initial target hardware.

## Launch and correctness

Use the existing Linux CUDA 12.6 environment and FP32 for GTX 1080 Ti. With
two GPUs allocated on one host, append these options to the normal training
command:

```bash
python -m neural_lam.train_model \
    --config_path /path/to/config.yaml --model graph_efm --graph cluster \
    --devices 0 1 --graph_parallel_size 2 --precision 32
```

Lightning launches one process per device. An externally launched
`torchrun --standalone --nproc-per-node=2 -m neural_lam.train_model ...`
also works. For Slurm allocate one task per GPU on **one node**, use
`srun python -m neural_lam.train_model ... --devices auto`, and set the
graph-parallel size to the allocation's GPU count. Graph parallelism cannot
be combined with data parallelism or `--train_distr_crps` in this version.

The default completes halo exchange before computing messages. Append
`--graph_parallel_overlap` to overlap both forward halo exchange and backward
halo gradient return with the independent local edge MLP work. Benchmark both
modes: smaller kernels and extra autograd dispatch may outweigh the overlap.
Both modes reuse communication buffers. Overlap supports first-order training;
higher-order derivatives and `torch.compile` are outside this milestone.

```bash
python -m pytest --noconftest tests/test_graph_parallel.py -q
```

The CPU/Gloo tests check FP64 layer outputs, node/edge/weight gradients,
GraphEFM likelihood/KL/CRPS, AdamW updates, empty partitions, zero processor
layers, constant priors, checkpointed autoregression, both communication modes,
and Lightning clipping/checkpoint resume. `--noconftest` bypasses the existing
shared test fixture, which imports a datastore module absent from this checkout.
The same suite includes an FP32 CUDA/NCCL test, automatically skipped unless
two GPUs and NCCL are available, using `rtol=1e-4`, `atol=1e-5`.

## Partition handoff

No partition file is required: deterministic recursive coordinate bisection
assigns owners independently at each level. This temporary partitioner balances
node counts, and can produce uneven edge work or large halos.

Your teammate's partitioner should provide one CPU `torch.int64` vector per
level, where `owners[level][node_id]` is a rank in `[0, size)`. Node IDs are the
existing IDs after graph loading. On a newly constructed CPU GraphEFM, before
sharding or moving it to a GPU:

```python
from neural_lam.graph_parallel import make_mesh_partition

partition = make_mesh_partition(model, size=2, owners=owners)
partition.save("partition-2.pt")
```

Pass `--graph_partition_path partition-2.pt`. The artifact includes a fingerprint
of mesh coordinates, topology, edge features, and ordering. Training validates
the fingerprint and identical ownership on every rank. Edges belong to their
receiver's rank and retain their original relative order. Replacing the owner
vectors requires no changes to communication code.

Weights retain the existing checkpoint names and shapes. Ordinary single-GPU
checkpoints can initialize graph-parallel training. Graph-parallel checkpoints
also store partition identity and CPU/CUDA RNG state; resuming graph-parallel
training requires the same partition. They can also be loaded on one GPU.
Checkpoint files and external artifacts are written only by rank zero.

## Scaling measurements

First validate the actual transport and record topology and library versions:

```bash
torchrun --standalone --nproc-per-node=2 -m benchmarks.graph_parallel_probe \
    --output benchmark_results/graph-parallel/transport-2.json
```

Run matched 1-, 2-, and 4-GPU jobs sequentially where allocations permit.
Use graph-parallel size 1 for the existing baseline and size 2/4 for sharding.
Keep the same data, batch size, model dimensions, processor layers, AR steps,
precision, checkpointing, and warmup/measurement counts. Follow
`benchmarks/README.md` for the existing baseline workload and aggregate/phase
timing distinction. The existing `--benchmark_output`, `--benchmark_steps`,
and `--benchmark_timing_mode` options also apply to graph-parallel runs.

Benchmark JSON includes the slowest-rank training time, correct logical batch
throughput, maximum GPU memory, and per-rank partition metadata: owned nodes,
owned edges, halo/send counts, and cumulative forward/backward sent bytes.
Byte counters include warmup and checkpoint recomputation; they are not
per-step wire-volume measurements. Use the per-rank counts to assess imbalance.
Phase mode additionally records cumulative halo communication windows and
gradient synchronization time. Overlapped halo windows include the local MLP
work performed while communication is in flight. Aggregate mode omits these
events so they do not affect headline measurements.

Then repeat with increasing mesh-heavy model sizes until the one-GPU baseline
runs out of memory, retaining failures as evidence. Replicated grid/g2m/m2g
activations and the full bottom-mesh tensors at those boundaries remain a
memory floor. Capacity and speedup require measurements on the target GPUs;
CPU numerical checks cannot establish either result.
