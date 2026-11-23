# Njord graph-parallel overlap checkpoint fix

November 18, 2026 · GTX 1080 Ti · PyTorch 2.14.1+cu126 · NCCL 2.29.3

Overlap completed training on two and four GPUs at hidden dimension 64,
where every timed overlap attempt in the initial series ran out of memory.
Compared with blocking on the same cards, fixed overlap used 15.3% less peak
GPU memory on two GPUs and 7.9% less on four. Throughput fell by 10.7% and 8.7%,
respectively. [Initial graph-parallel benchmark](../report.md)

## Correctness

The original overlap backward pass could trigger a full autoregressive
checkpoint replay from each nested edge gradient call. The fix unpacks saved
inputs in the outer backward pass and rebuilds the local and remote edge MLP
graphs. The regression test requires one replay per checkpointed
autoregressive step. The runs used the fix in the working tree at `9f058a3`,
which explains the dirty flag in the raw JSON. The fix was subsequently
committed as `a69138e`.

All three CPU/Gloo checks and both CUDA/NCCL equivalence cases passed.
[CPU log](cpu-overlap-replay.log) ·
[Two-GPU test](validate-g2-blocking-h64-r4-4469.log) ·
[Four-GPU test](validate-g4-blocking-h64-r4-4470.log)

The 16 MiB pairwise NCCL probe on `studgpu-node06` measured 21.78 GB/s duplex
on two GPUs and 19.62–22.10 GB/s across the four-GPU pairs.
[Two-GPU probe](validate-g2-blocking-h64-r4-4469.transport.json) ·
[Four-GPU probe](validate-g4-blocking-h64-r4-4470.transport.json)

## Production smoke runs

All smoke jobs completed three training steps. Peak allocation includes the
first training step and is the maximum over ranks. The forward and backward
halo byte counters matched between modes at each GPU count.

| GPUs | Blocking peak/GPU | Overlap peak/GPU | Raw results |
| ---: | ---: | ---: | --- |
| 2 | 2.80 GiB | 2.38 GiB | [Blocking](smoke-g2-blocking-h64-r4-4474.json), [Overlap](smoke-g2-overlap-h64-r4-4471.json) |
| 4 | 2.43 GiB | 2.24 GiB | [Blocking](smoke-g4-blocking-h64-r4-4475.json), [Overlap](smoke-g4-overlap-h64-r4-4473.json) |

## Matched timing

Each run used FP32, hidden dimension 64, two processor layers, four
autoregressive steps, a logical batch size of two, activation checkpointing,
and zero DataLoader workers per rank. Timings cover 200 measured steps after
20 warmup steps in aggregate mode with telemetry off. Each pair ran once on
the same node and physical GPUs.

| GPUs | Node | Blocking examples/s | Overlap examples/s | Blocking peak/GPU | Overlap peak/GPU | Raw results |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | `studgpu-node06` | 0.558 | 0.498 | 2.80 GiB | 2.38 GiB | [Blocking](timed-g2-blocking-h64-r4-4476.json), [Overlap](timed-g2-overlap-h64-r4-4480.json) |
| 4 | `studgpu-node02` | 0.489 | 0.446 | 2.43 GiB | 2.24 GiB | [Blocking](timed-g4-blocking-h64-r5-4488.json), [Overlap](timed-g4-overlap-h64-r4-4486.json) |

The first four-GPU blocking run reached 0.297 examples/s on
`studgpu-node06`. Slurm allocated the overlap job on `studgpu-node02` with a
different PCIe layout, so a fresh blocking run on `studgpu-node02` supplied
the comparison in the table. [First four-GPU blocking result](timed-g4-blocking-h64-r4-4481.json)

Future timed graph-parallel jobs default to 10 warmup and 40 measured steps
to reduce GPU usage. The launcher allows both counts to be overridden.
[Launch instructions](../../../benchmarks/graph_parallel.md#scaling-measurements)
