#!/bin/bash
# Repeat the final baseline with low-frequency telemetry to quantify overhead.

set -euo pipefail

REPEAT="${1:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-500}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-1000}"

mkdir -p benchmark_results

sbatch \
    --export="ALL,RUN_KIND=telemetry-baseline-r${REPEAT},TIMING_MODE=aggregate,ENABLE_GPU_TELEMETRY=1,TELEMETRY_INTERVAL_MS=${TELEMETRY_INTERVAL_MS},WARMUP_STEPS=${WARMUP_STEPS},BENCHMARK_STEPS=${BENCHMARK_STEPS}" \
    jobs/benchmark_1080ti.sbatch
