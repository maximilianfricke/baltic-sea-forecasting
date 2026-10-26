#!/bin/bash
# Compare node-local staged data with direct reads from shared scratch.

set -euo pipefail

REPEAT="${1:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-500}"

mkdir -p benchmark_results

sbatch \
    --export="ALL,RUN_KIND=io-shared-r${REPEAT},STAGE_DATA=0,TIMING_MODE=aggregate,ENABLE_GPU_TELEMETRY=0,WARMUP_STEPS=${WARMUP_STEPS},BENCHMARK_STEPS=${BENCHMARK_STEPS}" \
    jobs/benchmark_1080ti.sbatch
