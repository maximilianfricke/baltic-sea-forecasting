# Benchmark analysis

This report is generated from completed benchmark JSON files. Local Apple-MPS results are exploratory and must not be compared numerically with the official CUDA cluster baseline.

## Experiment scope

- Accelerator: Apple MPS
- Model: GraphEFM
- Grid nodes: 42,348
- Mesh nodes: 10,553
- Trainable parameters: 732,572

## Canonical local baseline

- Mean compute step: 175.77 ms
- Mean data wait: 20.69 ms
- End-to-end throughput: 5.09 samples/s
- Peak allocated accelerator memory: 1.86 GiB

## Gradient checkpointing

Across 5 controlled pairs, median compute overhead is 40.1%. Median observed allocated-memory reduction is 96.6%.

## Single-GPU controlled sweeps

### Hidden dimension and checkpointing

| Setting | Checkpointing | Step [ms] | End-to-end [ms] | Samples/s | Peak GiB |
|---:|:---:|---:|---:|---:|---:|
| 16 | off | 79.65 | 99.76 | 10.02 | 0.54 |
| 16 | on | 111.24 | 131.46 | 7.61 | 0.06 |
| 32 | off | 105.61 | 125.80 | 7.95 | 0.98 |
| 32 | on | 146.12 | 167.77 | 5.96 | 0.06 |
| 64 | off | 175.77 | 196.47 | 5.09 | 1.86 |
| 64 | on | 249.69 | 270.33 | 3.70 | 0.06 |
| 128 | off | 342.95 | 365.10 | 2.74 | 3.62 |
| 128 | on | 486.52 | 506.84 | 1.97 | 0.09 |
| 256 | off | 790.07 | 811.56 | 1.23 | 7.21 |
| 256 | on | 1107.10 | 1127.88 | 0.89 | 0.21 |

### Input workers

| Setting | Checkpointing | Step [ms] | End-to-end [ms] | Samples/s | Peak GiB |
|---:|:---:|---:|---:|---:|---:|
| 0 | off | 178.15 | 308.44 | 3.24 | 1.86 |
| 1 | off | 173.50 | 190.51 | 5.25 | 1.86 |
| 2 | off | 176.04 | 194.47 | 5.14 | 1.86 |
| 4 | off | 175.77 | 196.47 | 5.09 | 1.86 |
| 8 | off | 175.60 | 199.98 | 5.00 | 1.86 |

### Processor depth

| Setting | Checkpointing | Step [ms] | End-to-end [ms] | Samples/s | Peak GiB |
|---:|:---:|---:|---:|---:|---:|
| 1 | off | 137.73 | 158.52 | 6.31 | 1.49 |
| 3 | off | 175.77 | 196.47 | 5.09 | 1.86 |
| 6 | off | 226.93 | 247.02 | 4.05 | 2.41 |

### Batch size

| Setting | Checkpointing | Step [ms] | End-to-end [ms] | Samples/s | Peak GiB |
|---:|:---:|---:|---:|---:|---:|
| 1 | off | 175.77 | 196.47 | 5.09 | 1.86 |
| 2 | off | 377.89 | 448.84 | 4.46 | 3.47 |
| 4 | off | 743.01 | 994.17 | 4.02 | 6.69 |

### Autoregressive training unroll

| Setting | Checkpointing | Step [ms] | End-to-end [ms] | Samples/s | Peak GiB |
|---:|:---:|---:|---:|---:|---:|
| 1 | on | 249.69 | 270.33 | 3.70 | 0.06 |
| 2 | on | 472.33 | 495.72 | 2.02 | 0.08 |
| 4 | on | 931.31 | 959.72 | 1.04 | 0.12 |

### Numerical precision

| Setting | Checkpointing | Step [ms] | End-to-end [ms] | Samples/s | Peak GiB |
|---:|:---:|---:|---:|---:|---:|
| 16-mixed | off | 158.54 | 179.52 | 5.57 | 1.14 |
| 32 | off | 175.77 | 196.47 | 5.09 | 1.86 |

## Local observations

- The fastest input configuration used 1 worker(s), at 190.51 ms end-to-end.
- Mixed FP16 changed compute time by -9.8% relative to FP32.
- Increasing hidden dimension from 16 to 256 increased compute time by 9.9x.

## Inference

| Configuration | Horizon | Ensemble | Step [ms] | Forecast members/s |
|---|---:|---:|---:|---:|
| inference-ensemble-1 | 1 | 1 | 58.07 | 16.44 |
| inference-ensemble-10 | 1 | 10 | 644.60 | 15.46 |
| inference-ensemble-5 | 1 | 5 | 324.85 | 15.29 |
| inference-horizon-2 | 2 | 1 | 113.89 | 8.54 |
| inference-horizon-4 | 4 | 1 | 222.28 | 4.43 |

## Multi-GPU status

No multi-GPU result is present yet. Cluster scripts are ready, but speedup, efficiency, NCCL time, and per-rank imbalance must be measured on the target cluster before drawing conclusions.

## Measurement notes

- CUDA timings use accelerator events and resettable allocator peaks.
- MPS timings use synchronized wall clock. Its memory value is the largest sample at a synchronized phase boundary, not an exact transient peak.
- Profiler runs are kept separate because tracing changes runtime.
- DDP communication volume is estimated from trainable gradient bytes; NCCL duration is obtained from the multi-GPU profiler trace.
- Report medians across at least three independent cluster runs.
