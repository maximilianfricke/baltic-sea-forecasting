# Final real-data model-parallel benchmark

Controlled strong-scaling comparison on GTX 1080 Ti GPUs with 238 measured batches, five warm-up batches, and a constant global batch size of two.

| Metric | 1 GPU | 2 GPUs | 4 GPUs | 1→2 | 1→4 |
|---|---:|---:|---:|---:|---:|
| Data read (mean) | 479.983 ms | 502.418 ms | 2160.052 ms | 0.955× | 0.222× |
| Data rank wait (mean) | 0.000 ms | 435.956 ms | 2115.598 ms | +435.956 ms | +2115.598 ms |
| Forward (mean) | 121.984 ms | 82.527 ms | 344.344 ms | 1.478× | 0.354× |
| Forward (median) | 121.192 ms | 78.856 ms | 343.114 ms | 1.537× | 0.353× |
| Backward (mean) | 96.749 ms | 67.516 ms | 365.727 ms | 1.433× | 0.265× |
| Gradient synchronization (mean) | 0.002 ms | 9.180 ms | 71.923 ms | +9.178 ms | +71.921 ms |
| Metric reduction (mean) | 1.434 ms | 14.701 ms | 76.706 ms | +13.267 ms | +75.272 ms |
| Optimizer (mean) | 0.364 ms | 7.392 ms | 55.435 ms | +7.028 ms | +55.071 ms |
| Model step (mean) | 220.534 ms | 167.320 ms | 835.858 ms | 1.318× | 0.264× |
| Model step (median) | 219.421 ms | 166.464 ms | 840.044 ms | 1.318× | 0.261× |
| End-to-end (mean) | 706.403 ms | 673.275 ms | 3004.248 ms | 1.049× | 0.235× |
| End-to-end (median) | 736.851 ms | 632.122 ms | 2900.028 ms | 1.166× | 0.254× |
| Observed throughput | 2.814 samples/s | 2.960 samples/s | 0.667 samples/s | 1.052× | 0.237× |
| Peak memory per GPU | 3.341 GiB | 1.700 GiB | 0.894 GiB | 49.1% less | 73.2% less |
| Data staging | 8454.000 ms | 5641.000 ms | 8949.000 ms | -2813.000 ms | +495.000 ms |

## Key findings

- End-to-end speedup is 1.049× on two GPUs and 0.235× on four GPUs.
- End-to-end parallel efficiency falls to 52.6% and 5.9%.
- Per-GPU peak memory falls by 49.1% on two GPUs and 73.2% on four GPUs.
- Data-read speedup reaches only 0.222× at four GPUs, so input latency dominates end-to-end scaling.
- Four-GPU gradient synchronization averages 71.923 ms; forward-pass jitter further limits compute scaling.
- Owned grid nodes are exactly balanced, but total halo nodes range from 404 to 1262 across four ranks (3.12× imbalance).
- Preflight-loss relative differences are 9.761e-08 for two GPUs and 0.000e+00 for four GPUs.
- Numerical equivalence checks one-step training consistency, not forecast accuracy; forecast accuracy requires equal validation rollouts.
- Multi-rank phase maxima are reported independently and should not be summed; end-to-end time is the measured critical-path total.
