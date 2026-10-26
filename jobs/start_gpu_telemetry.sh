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
TELEMETRY_WORK_DIR="${TMPDIR:-${TELEMETRY_DIR}}/njord-telemetry"
mkdir -p "${TELEMETRY_WORK_DIR}"
TELEMETRY_LOCAL_FILE="${TELEMETRY_WORK_DIR}/${TELEMETRY_PREFIX}gpu-"
TELEMETRY_LOCAL_FILE="${TELEMETRY_LOCAL_FILE}${SLURM_JOB_ID:-manual}.csv"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-1000}"

echo "GPU telemetry interval: ${TELEMETRY_INTERVAL_MS} ms"

nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem \
    --format=csv \
    --loop-ms="${TELEMETRY_INTERVAL_MS}" > "${TELEMETRY_LOCAL_FILE}" &
TELEMETRY_PID=$!

stop_gpu_telemetry() {
    kill "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
    if [[ -f "${TELEMETRY_LOCAL_FILE}" ]]; then
        cp "${TELEMETRY_LOCAL_FILE}" "${TELEMETRY_FILE}"
    fi
}
trap stop_gpu_telemetry EXIT
