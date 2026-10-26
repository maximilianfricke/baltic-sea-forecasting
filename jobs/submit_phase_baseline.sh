#!/bin/bash
# Detailed phase timings are separate from headline aggregate throughput.

set -euo pipefail

REPEAT="${1:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-500}"

mkdir -p benchmark_results

sbatch \
    --export="ALL,RUN_KIND=phase-baseline-r${REPEAT},TIMING_MODE=phase,ENABLE_GPU_TELEMETRY=0,WARMUP_STEPS=${WARMUP_STEPS},BENCHMARK_STEPS=${BENCHMARK_STEPS}" \
    jobs/benchmark_1080ti.sbatch
