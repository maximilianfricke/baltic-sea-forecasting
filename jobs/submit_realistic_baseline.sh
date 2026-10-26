#!/bin/bash
# Minimally instrumented throughput run for the realistic Graph-EFM workload.

set -euo pipefail

REPEAT="${1:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-200}"

mkdir -p benchmark_results

sbatch \
    --time=00:45:00 \
    --export="ALL,RUN_KIND=realistic-baseline-r${REPEAT},HIDDEN_DIM=64,PROCESSOR_LAYERS=2,BATCH_SIZE=2,AR_STEPS_TRAIN=4,AR_STEPS_EVAL=4,PRECISION=32,GRAD_CHECKPOINTING=0,TIMING_MODE=aggregate,ENABLE_GPU_TELEMETRY=0,WARMUP_STEPS=${WARMUP_STEPS},BENCHMARK_STEPS=${BENCHMARK_STEPS}" \
    jobs/benchmark_1080ti.sbatch
