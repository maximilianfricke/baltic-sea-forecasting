# Message-passing archive and rebuild contract

Archived on 26 November 2026 before rebuilding `message-passing` from
`global-model@76b30d778382939963219142053fcf6f2d54c54e`.
The previous benchmark archive ends at `8554302`; this snapshot also preserves
the subsequently generated reports, plots, source scripts and final documents.
Historical document-rendering intermediates are preserved locally outside the
working repository rather than added to Git.

## Evidence to retain

- AR1 was useful for development and I/O diagnosis, but is not our representative
  model comparison. The reference experiment uses AR4, H64, decoder L2,
  prior/posterior L1, FP32, WMSE+KL, AdamW and checkpointing.
- Original full-grid input and ocean-only rank-local Zarr are different data
  paths. Ocean filtering and pre-sharding improve I/O independently of GPU
  communication. Report their preparation costs and avoid attributing this
  improvement to message passing.
- The physical test store repeats the same month sixteen times. This increases
  storage/workload length, not independent meteorological diversity.
- Slurm CPU allocation is not the same as the actual affinity of each rank or
  helper thread. Job 6121 allocated eight CPUs but all four rank main threads
  reported the same two-CPU mask. Record both allocation and actual masks.
- Under corrected CPU placement, full four-GPU job 6131 passed all equivalence
  and CPU checks. P2P launch-median aggregate fell from 1974.17 to 759.45 ms;
  PhysicsNeMo fell from 1805.69 to 734.87 ms. This before/after bundles launch
  changes and does not isolate the necessity of disjoint CPU pairs.
- The remaining PhysicsNeMo throughput advantage was 3.34% by the published
  ratio of launch-median aggregates. Individual paired gains were 1.51%, 10.11%
  and 1.89%. Three launches and the ordering imbalance limit the conclusion.
- In 6131 the only CPU-mask changes were the initial per-rank assignments.
  No later repair was required. Avoid repeated blanket helper-thread repinning
  without evidence that it is necessary.
- NCCL 2.29.3 source provides a plausible affinity-narrowing mechanism. The
  exact origin of the old restriction is not proven by our historical traces.
  Observe imports, CUDA, process-group initialization and the first collective
  before deciding on a permanent workaround.
- Nils's original-layer adapter is a useful architecture reference. His slides
  differ in graph, partition/layout, determinism, distribution validation,
  checkpoint replay, batching and timing scope. PhysicsNeMo alone does not
  explain all differences. His raw CPU traces are not in the branch.

## Required structure for the rebuild

1. Keep the original `neural_lam` scientific model and loader files unchanged.
   Use the original `GraphEFM.training_step`, prior/posterior, AR feedback,
   loss and checkpointing rather than maintaining a second training algorithm.
2. Place partition plans, transport, original-layer adapters, data preparation
   and benchmark/launch concerns in a separate package. No affinity mutation
   or process initialization during module import.
3. Accept explicit ownership arrays. A balanced geographic partition is a
   development default; final partitioning belongs to the other team members.
   Preserve global node/edge order and validate every owner and relation.
4. One halo interface supports P2P and the exact vendored PhysicsNeMo primitive.
   Both use identical local graph layouts and the same original layer MLPs.
   Different layouts must be tested as an additional factor.
5. Own edges at the receiver. Exchange the needed remote sender features once
   per unique node and relation; accumulate reverse halo gradients. Sum shared
   parameter gradients without an extra world-size division. Preserve global
   loss normalization and globally indexed stochastic latent samples.
6. Validate against the original model, not only against the other transport:
   AR4 predictions, scalar loss, gradients, two unmasked AdamW updates,
   checkpoint replay and prior-only inference, on 1/2/4 ranks.
7. First isolate communication with identical cached CPU batches. Record live
   loading as a separate mode, preserving original normalization and sample
   identity. Worker/prefetch changes are separate experiments.
8. Record all-rank CPU/hardware/software information, source and input hashes,
   setup/H2D/model wall intervals and raw per-rank records. For additive phases,
   use one critical rank per step rather than summing unrelated rank maxima.
9. Fail clearly on unsupported model/data options or inconsistent ownership.
   Never silently accept missing gradients, invalid masks or changed controls.

## Retained evidence

- `benchmark_results/halo-transport/full-{1gpu-6114,2gpu-6117,4gpu-6121,4gpu-6131}`
- `benchmark_results/halo-transport/analysis/cpu-transport-2026-11-25/analysis.html`
- `docs/njord-guide.html`: data paths, model flow, historical comparisons and
  CPU-placement findings, including the remaining uncertainty.
- `benchmarks/analyze_halo_cpu_results.py`: reproducible raw-result analysis.

## Reference commits

- Original: `76b30d778382939963219142053fcf6f2d54c54e` (`global-model`).
- Nils reference: `69939c991289a5f5901fbd24ef64526a56199716` (`nils-pnemo`).
- Archived implementation/results before report snapshot: `8554302`.

Only `message-passing` is rebuilt. The original and team branches are read-only
references. The archive branch must remain available before rewriting the
working branch history.
