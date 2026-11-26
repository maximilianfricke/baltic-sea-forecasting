# Learnings guide checks; they do not mandate workarounds

The first implementation is only P2P on unchanged `global-model@76b30d7`.
Run it, inspect correctness and timing, and change one demonstrated problem
at a time. The previous hardware-specific remedies are not defaults.

- **Use the real model.** AR1 helped development but was too simple for the
  main scaling comparison. Retain AR4 and the chosen H64/B2 reference.
- **Validate against the original.** Compare outputs, loss, gradients and
  two unmasked AdamW updates. Keep original training/feedback/loss code.
- **Spatial ranks share examples.** Partition nodes, not sample IDs. Sum
  parameter gradients rather than averaging away spatial contributions.
- **Keep randomness comparable.** Sample noise using global latent-node IDs
  so partitions do not change which random values belong to each node.
- **Observe CPU state first.** Job 6121 had eight allocated CPUs while four
  main threads reported the same two-CPU mask. A changed launch setup improved
  both transports substantially, but fixed disjoint pairs were not isolated
  as necessary. The old root cause is not proven. No affinity writes now.
- **Use Nils as a reference, not proof of a missing fix.** His successful
  code does not assign CPU pairs. Start without that assignment here too.
  His graph, ownership, numerical policies and timing scope also differ.
- **Separate I/O from communication.** Original full Zarr, ocean filtering,
  pre-sharding and caching are different input treatments. This first run
  retains the original live input and reports input and model times separately.
- **Preserve evidence.** Record all-rank timings, source/graph identity,
  numerical checks and read-only CPU snapshots. Investigate a slow result
  before introducing another policy, library or optimization.

All historical analyses remain on `archive/message-passing-2026-11-26`.
The superseded affinity-based rebuild is preserved by the tag
`archive/message-passing-with-affinity-2026-11-26`. No original/team branch is
modified by this rebuild. No cluster jobs are submitted automatically.
