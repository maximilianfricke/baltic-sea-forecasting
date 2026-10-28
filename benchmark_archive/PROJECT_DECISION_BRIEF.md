# NJORD parallelisation decision brief

Last updated: 2026-10-28

## Decision in one paragraph

The first implementation target should be the existing Interaction-Network
model, not a new Graph Transformer. Establish the single-GPU capacity curve
and the exact cost of gradient checkpointing first, then distribute one
prediction step across GPUs. The most relevant parallel work is therefore a
combination of graph partitioning, halo exchange/data placement, and tensor
parallelism of the shared edge/node MLPs in grid-to-mesh, mesh processing, and
mesh-to-grid. Transformer work remains a possible later extension, after the
distributed runtime and measurement harness are stable.

## What is already measured

The comparable realistic baseline is a GTX 1080 Ti run with hidden dimension
64, two processor layers, batch size 2, four autoregressive steps, FP32, and
automatic gradient checkpointing. Job 1453 measured 200 steps after warm-up.

| Metric | Result | Interpretation |
|---|---:|---|
| Forward mean | 453.56 ms | 33.9% of measured compute step |
| Backward mean | 884.11 ms | 66.1% of measured compute step |
| Optimizer/hooks mean | 0.65 ms | negligible at 0.05% |
| Compute step mean | 1338.32 ms | stable single-GPU compute baseline |
| Compute throughput | 1.494 samples/s | excludes input waiting |
| End-to-end throughput | 1.068 samples/s | pre-fix, affected by tiny epochs |
| Peak allocated GPU memory | 3.41 GiB | current small model is not near OOM |

The old end-to-end input result is not a steady-state storage bottleneck. The
public training subset has eight valid samples at this unroll length, hence
four batches at batch size two. The observed two-second wait every fourth
step was the DataLoader restarting the tiny epoch. The official jobs now use
a separate 2.2-GB physical-repeat I/O proxy with 479 training time steps. This
avoids the tiny epoch without changing the normal dataset implementation.

The graph in the archived baseline contains 42,348 grid nodes and 10,553 mesh
nodes. Its edge sets include 70,640 grid-to-mesh, 60,186 mesh-to-mesh, 126,107
mesh-to-grid, and 8,456 edges in each hierarchical up/down direction. These
counts explain why placement and boundary edges matter, but they do not by
themselves identify which model component dominates runtime.

## Ranked bottlenecks and opportunities

1. **Memory capacity at project scale.** The project brief explicitly asks
   for the largest graph/hidden dimension that fits in GPU memory. The current
   3.41 GiB result only proves that the small 1-degree, hidden-64 case fits; it
   does not expose the intended limit. Hidden-dimension capacity probes are
   therefore the first missing measurement.
2. **Backward and checkpoint recomputation.** Backward accounts for about two
   thirds of measured compute while checkpointing is active. A controlled
   checkpoint-on/off pair is needed to separate normal backward work from
   recomputation and quantify the memory/runtime trade-off.
3. **Repeated graph message passing.** Grid-to-mesh, hierarchical mesh
   processing, mesh-to-grid, and every autoregressive step repeat edge/node
   work. Partitioning must preserve exact reductions while exchanging remote
   neighbor features through halo nodes.
4. **Communication and data placement.** A useful partition must jointly
   minimize cut edges/communication partners and balance node, edge, memory,
   and compute load. The shared MLP weights are not a reason to communicate
   between graph partitions during the forward pass; each rank can hold the
   same weights. Training additionally requires gradient synchronization, or
   a tensor-parallel scheme that reduces partial MLP results.
5. **Component-level runtime is still unknown.** Static edge counts do not
   prove whether the encoder, prior, decoder, or a specific graph transition
   dominates. A short semantic PyTorch profile is required before optimizing
   an individual component.

## Recommended five-person split

1. **Baseline, profiling, and capacity owner:** maintain reproducible runs,
   checkpoint trade-off, capacity limits, reports, and regression checks.
2. **Graph partitioning owner:** quantify halo nodes, cut edges, communication
   partners, memory balance, and compute balance for the fixed ocean graph.
3. **Distributed graph runtime owner:** implement feature exchange and exact
   aggregation for grid-to-mesh, mesh processor, and mesh-to-grid; investigate
   overlap of communication with independent local computation.
4. **Scaling and inference owner:** build strong/weak-scaling experiments for
   training and inference and separate latency, throughput, startup, and
   communication measurements.
5. **Tensor/data parallelism owner:** test channel/MLP sharding and input/output
   placement across GPUs. Attention-head distribution and a sparse Graph
   Transformer are optional later adjustments, not the first dependency.

## Immediate experiment-to-decision map

| Experiment | Decision it unlocks |
|---|---|
| Physical-dataset validation | whether input waiting is still material |
| h64 checkpoint on vs off | exact memory saved per recomputation cost |
| h128/h192/h256 capacity probes | scaling curve and first OOM boundary |
| Semantic component profile | which of encoder/prior/decoder to optimize first |
| 1/2/4-GPU exactness + timing | communication overhead and viable split |

The decision suite intentionally runs only the first three rows. Failed
capacity probes are retained as evidence rather than silently discarded. Raw
JSON is the source of truth; the CSV, Markdown report, and plots in this
archive are derived and reproducible.

## Evidence boundary

- **Measured:** the timing and memory figures above, graph sizes, and the
  periodic four-batch epoch behavior in job 1453.
- **Project requirements:** single-GPU checkpointed baseline; parallel
  grid-to-mesh, mesh processor, and mesh-to-grid; graph splitting; data
  movement; training/inference scaling; communication measurement.
- **Inference:** capacity, checkpoint recomputation, graph communication, and
  repeated message passing are the leading targets. Their final ranking can
  change after the decision suite and semantic profile.
- **Not yet claimed:** multi-GPU speedup, communication volume measured on the
  cluster, or a component-level hotspot. Those require additional runs.
