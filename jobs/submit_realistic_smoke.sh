#!/bin/bash
# Cheap fit/correctness check for the realistic unrolled Graph-EFM workload.

set -euo pipefail

mkdir -p benchmark_results

sbatch \
    --time=00:20:00 \
    --export="ALL,RUN_KIND=realistic-smoke,HIDDEN_DIM=64,PROCESSOR_LAYERS=2,BATCH_SIZE=2,AR_STEPS_TRAIN=4,AR_STEPS_EVAL=4,PRECISION=32,GRAD_CHECKPOINTING=0,TIMING_MODE=phase,ENABLE_GPU_TELEMETRY=0,WARMUP_STEPS=5,BENCHMARK_STEPS=20" \
    jobs/benchmark_1080ti.sbatch
