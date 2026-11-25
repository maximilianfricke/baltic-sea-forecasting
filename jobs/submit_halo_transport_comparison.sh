#!/bin/bash
# Submit one GPU count only. CPU allocation remains under the course wrapper.
set -euo pipefail
GPU_COUNT="${1:-}"
PHASE="${2:-}"
case "$GPU_COUNT" in
    1) ACCOUNT=dphpc ;;
    2|4) ACCOUNT=dphpc_4gpu ;;
    *) echo "Usage: $0 1|2|4 smoke|full" >&2; exit 2 ;;
esac
case "$PHASE" in
    smoke) REPEATS=1; WARMUP=0; STEPS=1; JOB_TIME=00:15:00 ;;
    full) REPEATS=3; WARMUP=10; STEPS=50; JOB_TIME=01:00:00 ;;
    *) echo "Usage: $0 1|2|4 smoke|full" >&2; exit 2 ;;
esac
[[ -f benchmarks/compare_halo_transports.py ]] || { echo 'Run from the repository root' >&2; exit 2; }
mkdir -p benchmark_results/halo-transport
sbatch --account="$ACCOUNT" --gpus="1080ti:$GPU_COUNT" --time="$JOB_TIME" \
    --job-name="halo-ab-$PHASE-${GPU_COUNT}g" \
    --output="benchmark_results/halo-transport/slurm-$PHASE-${GPU_COUNT}g-%j.out" \
    --export="ALL,NJORD_AB_GPUS=$GPU_COUNT,NJORD_AB_PHASE=$PHASE,NJORD_AB_REPEATS=$REPEATS,NJORD_AB_WARMUP=$WARMUP,NJORD_AB_STEPS=$STEPS" \
    jobs/halo_transport_comparison.sbatch
