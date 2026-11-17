#!/bin/bash
# Submit one controlled smoke test or full 1/2/4-GPU benchmark.

set -euo pipefail

GPU_COUNT="${1:-}"
RUN_PHASE="${2:-}"
NUM_WORKERS="${NJORD_NUM_WORKERS:-0}"
PREFETCH_FACTOR="${NJORD_PREFETCH_FACTOR:-2}"
case "${GPU_COUNT}" in
    1)
        ACCOUNT="dphpc"
        ;;
    2|4)
        ACCOUNT="dphpc_4gpu"
        ;;
    *)
        echo "Usage: $0 1|2|4 smoke|full" >&2
        exit 2
        ;;
esac

case "${RUN_PHASE}" in
    smoke)
        JOB_TIME="${NJORD_JOB_TIME:-01:00:00}"
        WARMUP_STEPS=0
        MEASUREMENT_STEPS=1
        ;;
    full)
        JOB_TIME="${NJORD_JOB_TIME:-12:00:00}"
        WARMUP_STEPS="${NJORD_WARMUP_STEPS:-5}"
        MEASUREMENT_STEPS="${NJORD_MEASUREMENT_STEPS:-20}"
        ;;
    *)
        echo "Usage: $0 1|2|4 smoke|full" >&2
        exit 2
        ;;
esac

mkdir -p benchmark_results/model-parallel-real
sbatch \
    --account="${ACCOUNT}" \
    --gpus="1080ti:${GPU_COUNT}" \
    --time="${JOB_TIME}" \
    --job-name="njord-${RUN_PHASE}-${GPU_COUNT}g" \
    --output="benchmark_results/model-parallel-real/slurm-${RUN_PHASE}-${GPU_COUNT}g-%j.out" \
    --export="ALL,NJORD_GPU_COUNT=${GPU_COUNT},NJORD_DATA_MODE=rank-local,NJORD_NUM_WORKERS=${NUM_WORKERS},NJORD_PREFETCH_FACTOR=${PREFETCH_FACTOR},NJORD_WARMUP_STEPS=${WARMUP_STEPS},NJORD_MEASUREMENT_STEPS=${MEASUREMENT_STEPS},NJORD_PROGRESS_INTERVAL=0,NJORD_RUN_LABEL=${RUN_PHASE}" \
    jobs/model_parallel_real_benchmark.sbatch
