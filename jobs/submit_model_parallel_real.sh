#!/bin/bash
# Submit one controlled 1- or 2-GPU real-data model-parallel benchmark.

set -euo pipefail

GPU_COUNT="${1:-}"
case "${GPU_COUNT}" in
    1)
        ACCOUNT="dphpc"
        ;;
    2)
        ACCOUNT="dphpc_4gpu"
        ;;
    *)
        echo "Usage: $0 1|2" >&2
        exit 2
        ;;
esac

mkdir -p benchmark_results/model-parallel-real
sbatch \
    --account="${ACCOUNT}" \
    --gpus="1080ti:${GPU_COUNT}" \
    --job-name="njord-real-mp-${GPU_COUNT}g" \
    --export="ALL,NJORD_GPU_COUNT=${GPU_COUNT}" \
    jobs/model_parallel_real_benchmark.sbatch
