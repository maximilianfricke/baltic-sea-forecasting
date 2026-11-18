#!/bin/bash
# Submit from the repository root. Extra arguments are passed to sbatch.
set -euo pipefail

RUN_KIND="${1:-}"
GPU_COUNT="${2:-}"
MODE="${3:-blocking}"
HIDDEN_DIM="${4:-64}"
REPETITION="${5:-1}"
if [[ ! "${RUN_KIND}" =~ ^(validate|smoke|timed|capacity)$ ||
      ! "${GPU_COUNT}" =~ ^(1|2|4)$ ||
      ! "${MODE}" =~ ^(blocking|overlap)$ ||
      ! "${HIDDEN_DIM}" =~ ^[1-9][0-9]*$ ||
      ! "${REPETITION}" =~ ^[1-9][0-9]*$ ||
      ( "${GPU_COUNT}" == 1 && ( "${RUN_KIND}" == validate || "${MODE}" == overlap ) ) ]]; then
    echo "Usage: $0 {validate|smoke|timed|capacity} {1|2|4} [blocking|overlap] [hidden_dim] [repetition] [sbatch options...]" >&2
    exit 2
fi
if (( $# > 5 )); then
    shift 5
else
    shift "$#"
fi

ACCOUNT=dphpc
if [[ "${GPU_COUNT}" != 1 ]]; then
    ACCOUNT=dphpc_4gpu
fi
mkdir -p benchmark_results/graph-parallel
sbatch --parsable --account="${ACCOUNT}" --gpus="1080ti:${GPU_COUNT}" \
    --ntasks=1 \
    --mem="$((24 * GPU_COUNT))G" "$@" \
    jobs/benchmark_graph_parallel_1080ti.sbatch \
    "${RUN_KIND}" "${GPU_COUNT}" "${MODE}" "${HIDDEN_DIM}" "${REPETITION}"
