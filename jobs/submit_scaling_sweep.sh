#!/bin/bash
# Submit one bundled strong/weak training/inference job for one GPU count.

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 <course-account-tag> <gpu-count: 1|2|4>" >&2
    exit 2
fi

ACCOUNT="$1"
GPU_COUNT="$2"
if [[ "${GPU_COUNT}" != "1" && "${GPU_COUNT}" != "2" \
    && "${GPU_COUNT}" != "4" ]]; then
    echo "GPU count must be 1, 2, or 4" >&2
    exit 2
fi

mkdir -p benchmark_results/scaling
sbatch \
    --account="${ACCOUNT}" \
    --ntasks="${GPU_COUNT}" \
    --gpus="1080ti:${GPU_COUNT}" \
    jobs/benchmark_scaling_bundle.sbatch
