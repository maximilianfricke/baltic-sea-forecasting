# Benchmark analysis

This report is generated from completed benchmark JSON files. Local CPU results are exploratory and must not be compared numerically with the official CUDA cluster baseline.

## Experiment scope

- Accelerator: NVIDIA GeForce GTX 1080 Ti
- Model: GraphEFM
- Grid nodes: 42,348
- Mesh nodes: 10,553
- Trainable parameters: 645,020

## Canonical realistic baseline

- Mean compute step: 1369.43 ms
- Mean data wait: 37.16 ms
- End-to-end throughput: 1.42 samples/s
- Peak allocated accelerator memory: 3.41 GiB

## Single-GPU controlled sweeps

### Hidden dimension and checkpointing

| Setting | Checkpointing | Step [ms] | End-to-end [ms] | Samples/s | Peak GiB |
|---:|:---:|---:|---:|---:|---:|
| 64 | on | 1369.43 | 1406.77 | 1.42 | 3.41 |
| 128 | on | 2314.21 | 2354.51 | 0.85 | 6.44 |
| 192 | on | 3360.58 | 3393.23 | 0.59 | 9.49 |

## Local observations


## Multi-GPU status

No multi-GPU result is present yet. Speedup, efficiency, communication time, and per-rank imbalance must be measured after the distributed implementation is available.

## Measurement notes

- CUDA timings use accelerator events and resettable allocator peaks.
- CPU smoke tests use wall clock and are validation only.
- Distributed communication is not yet measured and no multi-GPU performance claim is made.
- Current official runs use the physical-repeat I/O proxy. It has independent Zarr chunks and a long epoch but repeats the same scientific values, so it is an I/O proxy rather than new training data.
- Report medians across at least three independent cluster runs.
