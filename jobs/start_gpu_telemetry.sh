#!/bin/bash
# Source this file after RESULT_DIR is set. It starts one node-level sampler.

if [[ -z "${RESULT_DIR:-}" ]]; then
    echo "RESULT_DIR must be set before sourcing GPU telemetry" >&2
    return 2
fi

TELEMETRY_DIR="${RESULT_DIR}/telemetry"
mkdir -p "${TELEMETRY_DIR}"
TELEMETRY_PREFIX="${TELEMETRY_LABEL:+${TELEMETRY_LABEL}-}"
TELEMETRY_FILE="${TELEMETRY_DIR}/${TELEMETRY_PREFIX}gpu-"
TELEMETRY_FILE="${TELEMETRY_FILE}${SLURM_JOB_ID:-manual}.csv"

nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem \
    --format=csv \
    --loop-ms=200 > "${TELEMETRY_FILE}" &
TELEMETRY_PID=$!

stop_gpu_telemetry() {
    kill "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
}
trap stop_gpu_telemetry EXIT
