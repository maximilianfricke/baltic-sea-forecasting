#!/bin/bash
# Compare node-local staged data with direct reads from shared scratch.

set -euo pipefail

REPEATS="${REPEATS:-3}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-500}"

mkdir -p benchmark_results

for REPEAT in $(seq 1 "${REPEATS}"); do
    sbatch \
        --export="ALL,RUN_KIND=io-shared-r${REPEAT},STAGE_DATA=0,WARMUP_STEPS=${WARMUP_STEPS},BENCHMARK_STEPS=${BENCHMARK_STEPS}" \
        jobs/benchmark_1080ti.sbatch
done
