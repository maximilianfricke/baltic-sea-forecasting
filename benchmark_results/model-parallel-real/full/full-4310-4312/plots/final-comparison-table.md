# Final real-data model-parallel benchmark

Controlled strong-scaling comparison on GTX 1080 Ti GPUs with 238 measured batches, five warm-up batches, and a constant global batch size of two.

| Metric | 1 GPU | 2 GPUs | 4 GPUs | 1→2 | 1→4 |
|---|---:|---:|---:|---:|---:|
| Data read (mean) | 1407.557 ms | 1222.840 ms | 1104.780 ms | 1.151× | 1.274× |
| Forward (mean) | 122.269 ms | 95.501 ms | 101.522 ms | 1.280× | 1.204× |
| Forward (median) | 122.331 ms | 85.674 ms | 77.301 ms | 1.428× | 1.583× |
| Backward (mean) | 101.496 ms | 58.807 ms | 42.971 ms | 1.726× | 2.362× |
| Gradient synchronization (mean) | 0.002 ms | 0.870 ms | 8.458 ms | +0.868 ms | +8.456 ms |
| Metric reduction (mean) | 0.126 ms | 0.213 ms | 6.731 ms | +0.087 ms | +6.605 ms |
| Optimizer (mean) | 0.238 ms | 0.242 ms | 5.590 ms | +0.004 ms | +5.351 ms |
| Model step (mean) | 224.132 ms | 155.390 ms | 157.708 ms | 1.442× | 1.421× |
| Model step (median) | 224.279 ms | 145.455 ms | 134.183 ms | 1.542× | 1.671× |
| End-to-end (mean) | 1635.099 ms | 1354.546 ms | 1202.186 ms | 1.207× | 1.360× |
| End-to-end (median) | 1617.879 ms | 1342.055 ms | 1178.092 ms | 1.206× | 1.373× |
| Observed throughput | 1.221 samples/s | 1.474 samples/s | 1.660 samples/s | 1.207× | 1.360× |
| Peak memory per GPU | 3.341 GiB | 1.700 GiB | 0.894 GiB | 49.1% less | 73.2% less |
| Data staging | 3462.000 ms | 5223.000 ms | 8104.000 ms | +1761.000 ms | +4642.000 ms |

## Key findings

- End-to-end speedup is 1.207× on two GPUs and 1.360× on four GPUs.
- End-to-end parallel efficiency falls to 60.3% and 34.0%.
- Per-GPU peak memory falls by 49.1% on two GPUs and 73.2% on four GPUs.
- Data-read speedup reaches only 1.274× at four GPUs, so input latency dominates end-to-end scaling.
- Four-GPU gradient synchronization averages 8.458 ms; forward-pass jitter further limits compute scaling.
- Owned grid nodes are exactly balanced, but total halo nodes range from 404 to 1262 across four ranks (3.12× imbalance).
- Preflight-loss relative differences are 9.761e-08 for two GPUs and 0.000e+00 for four GPUs.
- Numerical equivalence checks one-step training consistency, not forecast accuracy; forecast accuracy requires equal validation rollouts.
- Multi-rank phase maxima are reported independently and should not be summed; end-to-end time is the measured critical-path total.
