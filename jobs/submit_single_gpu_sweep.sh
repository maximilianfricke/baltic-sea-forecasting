#!/bin/bash
# Submit one job containing three repeats of the single-GPU matrix.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 <course-account-tag>" >&2
    exit 2
fi

ACCOUNT="$1"
mkdir -p benchmark_results

sbatch \
    --account="${ACCOUNT}" \
    jobs/benchmark_single_gpu_sweep.sbatch
