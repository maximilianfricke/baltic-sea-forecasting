# Final real-data model-parallel benchmark

| Metric | 1 GPU | 2 GPUs | Comparison |
|---|---:|---:|---:|
| Data distribution | 1817.115 ms | 1528.470 ms | 1.189x |
| Forward | 119.486 ms | 73.601 ms | 1.623x |
| Backward | 102.185 ms | 64.886 ms | 1.575x |
| Gradient synchronization | 0.004 ms | 0.777 ms | +0.773 ms overhead |
| Compute | 221.911 ms | 138.821 ms | 1.599x |
| End-to-end | 2039.135 ms | 1667.377 ms | 1.223x |
| Observed throughput | 0.980 samples/s | 1.198 samples/s | 1.223x |
| Compute throughput | 9.013 samples/s | 14.407 samples/s | 1.599x |
| Peak memory per GPU | 3.553 GiB | 2.135 GiB | 39.9% less |

- Compute parallel efficiency: 79.9%
- End-to-end parallel efficiency: 61.1%
- Mean-loss relative difference: 2.717e-08
- Data timing is cache-sensitive; numerical equivalence is not a forecast-accuracy evaluation.
