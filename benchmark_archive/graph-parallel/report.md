# Njord graph-parallel benchmark

GTX 1080 Ti, FP32, hidden dimension 64, two processor layers, four AR steps, logical batch size two, activation checkpointing, and zero DataLoader workers per rank. Each timed job requested 20 warmup steps and 200 measured steps.

CPU/Gloo and two- and four-GPU NCCL equivalence passed. [CPU log](cpu-correctness.log), [two-GPU test](validate-g2-blocking-h64-r3-4386.log), [four-GPU test](validate-g4-blocking-h64-r3-4387.log).

The pairwise 16 MiB NCCL probe measured 21.73 GB/s duplex on the two-GPU allocation and 13.15–22.08 GB/s across pairs on the four-GPU allocation. [Two-GPU probe](validate-g2-blocking-h64-r3-4386.transport.json), [four-GPU probe](validate-g4-blocking-h64-r3-4387.transport.json).

| GPUs | Mode | Examples/s, median (range) | Speedup | Peak allocated GiB | Outcome | Jobs |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | blocking | 0.588 (0.571–0.590) | 1.000× | 3.41 | 3/3 completed | [4395](timed-g1-blocking-h64-r1-4395.json), [4403](timed-g1-blocking-h64-r2-4403.json), [4413](timed-g1-blocking-h64-r3-4413.json) |
| 2 | blocking | 0.521 (0.519–0.543) | 0.886× | 2.80 | 3/3 completed | [4396](timed-g2-blocking-h64-r1-4396.json), [4404](timed-g2-blocking-h64-r2-4404.json), [4416](timed-g2-blocking-h64-r3-4416.json) |
| 2 | overlap | — | — | — | 0/3 completed, 3 CUDA OOM | [4400](timed-g2-overlap-h64-r1-4400.json), [4409](timed-g2-overlap-h64-r2-4409.json), [4418](timed-g2-overlap-h64-r3-4418.json) |
| 4 | blocking | 0.495 (0.495–0.506) | 0.841× | 2.43 | 3/3 completed | [4401](timed-g4-blocking-h64-r1-4401.json), [4410](timed-g4-blocking-h64-r2-4410.json), [4419](timed-g4-blocking-h64-r3-4419.json) |
| 4 | overlap | — | — | — | 0/3 completed, 3 CUDA OOM | [4402](timed-g4-overlap-h64-r1-4402.json), [4412](timed-g4-overlap-h64-r2-4412.json), [4423](timed-g4-overlap-h64-r3-4423.json) |

Peak allocation is the maximum across all ranks and completed repetitions.

## Capacity

- Hidden dimension 128 on 1 GPU, completed at 6.44 GiB peak allocated per GPU. [4424](capacity-g1-blocking-h128-r1-4424.json)
- Hidden dimension 192 on 1 GPU, completed at 9.49 GiB peak allocated per GPU. [4425](capacity-g1-blocking-h192-r1-4425.json)
- Hidden dimension 256 on 1 GPU, CUDA OOM. [4426](capacity-g1-blocking-h256-r1-4426.json)
- Hidden dimension 256 on 2 GPUs, CUDA OOM. [4427](capacity-g2-blocking-h256-r1-4427.json)
- Hidden dimension 256 on 4 GPUs, completed at 8.73 GiB peak allocated per GPU. [4428](capacity-g4-blocking-h256-r1-4428.json)
