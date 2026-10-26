#!/bin/bash
# Submit a two-minute read-only resource probe for 1, 2, or 4 GTX 1080 Ti GPUs.

set -euo pipefail

GPU_COUNT="${1:-1}"
if [[ "${GPU_COUNT}" != "1" && "${GPU_COUNT}" != "2" \
    && "${GPU_COUNT}" != "4" ]]; then
    echo "GPU count must be 1, 2, or 4" >&2
    exit 2
fi

mkdir -p benchmark_results
sbatch --gpus="1080ti:${GPU_COUNT}" jobs/probe_gpu_allocation.sbatch
