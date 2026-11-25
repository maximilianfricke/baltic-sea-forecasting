#!/bin/bash
# Submit one GPU count. The course wrapper grants the corresponding CPUs.
set -euo pipefail
GPUS="${1:-}"
PHASE="${2:-}"
case "$GPUS" in
    1) ACCOUNT=dphpc ;;
    2|4) ACCOUNT=dphpc_4gpu ;;
    *) echo "Usage: $0 1|2|4 smoke|full" >&2; exit 2 ;;
esac
case "$PHASE" in
    smoke) REPEATS=1; WARMUP=0; STEPS=1; LIMIT=00:15:00 ;;
    full) REPEATS=4; WARMUP=10; STEPS=50; LIMIT=01:00:00 ;;
    *) echo "Usage: $0 1|2|4 smoke|full" >&2; exit 2 ;;
esac
[[ -f message_passing/run.py ]] || { echo "Run from the repository root" >&2; exit 2; }
CPU_POLICY="${MP_CPU_POLICY:-disjoint}"
[[ "$CPU_POLICY" == shared || "$CPU_POLICY" == disjoint ]] || { echo "Invalid CPU policy" >&2; exit 2; }
mkdir -p benchmark_results/message-passing
sbatch --account="$ACCOUNT" --gpus="1080ti:$GPUS" --time="$LIMIT" \
    --job-name="mp-$PHASE-${GPUS}g" \
    --output="benchmark_results/message-passing/job-$PHASE-${GPUS}g-%j.out" \
    --export="ALL,MP_GPUS=$GPUS,MP_PHASE=$PHASE,MP_REPEATS=$REPEATS,MP_WARMUP=$WARMUP,MP_STEPS=$STEPS,MP_CPU_POLICY=$CPU_POLICY" \
    jobs/message_passing.sbatch
