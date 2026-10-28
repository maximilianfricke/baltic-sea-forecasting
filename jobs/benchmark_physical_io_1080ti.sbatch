#!/bin/bash
# Long baseline using independent physical Zarr chunks for input-I/O timing.

#SBATCH --job-name=njord-physical-io
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_results/slurm-physical-io-%j.out

set -euo pipefail

export NJORD_JOB_STARTED_EPOCH_SECONDS="$(date +%s.%N)"
source "${SLURM_SUBMIT_DIR}/jobs/common.sh"
njord_init_cluster_job
njord_configure_training_job
PHYSICAL_DATA_ROOT="${PHYSICAL_DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
PHYSICAL_IO_MODE="${PHYSICAL_IO_MODE:-scratch}"
PHYSICAL_TIMING_MODE="${PHYSICAL_TIMING_MODE:-aggregate}"
PHYSICAL_TELEMETRY="${PHYSICAL_TELEMETRY:-off}"
RESULT_DIR="${REPO_ROOT}/benchmark_results/physical-io"
njord_require_dataset "${PHYSICAL_DATA_ROOT}"
if [[ "${PHYSICAL_TIMING_MODE}" != "aggregate" \
    && "${PHYSICAL_TIMING_MODE}" != "phase" ]]; then
    echo "PHYSICAL_TIMING_MODE must be aggregate or phase" >&2
    exit 2
fi
if [[ "${PHYSICAL_TELEMETRY}" != "off" \
    && "${PHYSICAL_TELEMETRY}" != "on" ]]; then
    echo "PHYSICAL_TELEMETRY must be off or on" >&2
    exit 2
fi

STAGING_START_NS="$(date +%s%N)"
case "${PHYSICAL_IO_MODE}" in
    scratch)
        JOB_DATA_DIR="${PHYSICAL_DATA_ROOT}"
        export NJORD_DATA_LOCATION="physical-repeat-shared-scratch"
        ;;
    tmpdir)
        JOB_DATA_DIR="${TMPDIR}/njord-physical-repeat-16"
        mkdir -p "${JOB_DATA_DIR}"
        cp -rL --preserve=timestamps \
            "${PHYSICAL_DATA_ROOT}/." "${JOB_DATA_DIR}/"
        export NJORD_DATA_LOCATION="physical-repeat-node-local-tmpdir"
        ;;
    *)
        echo "PHYSICAL_IO_MODE must be scratch or tmpdir" >&2
        exit 2
        ;;
esac
STAGING_END_NS="$(date +%s%N)"
STAGING_ELAPSED_NS="$((${STAGING_END_NS} - ${STAGING_START_NS}))"
export NJORD_DATA_STAGING_SECONDS="${STAGING_ELAPSED_NS}e-9"
export NJORD_DATA_STAGING_BYTES="$(du -sb "${JOB_DATA_DIR}" | cut -f1)"

mkdir -p "${RESULT_DIR}"
export RESULT_DIR
export ENABLE_GPU_TELEMETRY=0
if [[ "${PHYSICAL_TELEMETRY}" == "on" ]]; then
    export ENABLE_GPU_TELEMETRY=1
    export TELEMETRY_LABEL="physical-${PHYSICAL_IO_MODE}"
    source jobs/start_gpu_telemetry.sh
fi

echo "I/O mode: ${PHYSICAL_IO_MODE}"
echo "Timing mode: ${PHYSICAL_TIMING_MODE}"
echo "GPU telemetry: ${PHYSICAL_TELEMETRY}"
echo "Data source: ${JOB_DATA_DIR}"
echo "Data size: $(du -sh "${JOB_DATA_DIR}" | cut -f1)"
echo "Staging seconds: ${NJORD_DATA_STAGING_SECONDS}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit \
    --format=csv

RUN_KIND="${PHYSICAL_IO_MODE}-${PHYSICAL_TIMING_MODE}"
RUN_KIND="${RUN_KIND}-telemetry-${PHYSICAL_TELEMETRY}"
RUN_LABEL="physical-io-${RUN_KIND}-1080ti-h64-p2-b2-ar4-fp32-ckpt-on"
OUTPUT="${RESULT_DIR}/physical-${RUN_KIND}-${SLURM_JOB_ID}.json"
RUN_LOG="${RESULT_DIR}/physical-${RUN_KIND}-${SLURM_JOB_ID}.log"
set +e
njord_run_graph_efm_benchmark "${JOB_DATA_DIR}" 64 "${NUM_WORKERS}" \
    --benchmark_scaling_mode io \
    --benchmark_warmup_steps 20 \
    --benchmark_steps 200 \
    --benchmark_timing_mode "${PHYSICAL_TIMING_MODE}" \
    --benchmark_label "${RUN_LABEL}" \
    --benchmark_output "${OUTPUT}" 2>&1 | tee "${RUN_LOG}"
RUN_EXIT_CODE="${PIPESTATUS[0]}"
set -e

if [[ ! -f "${OUTPUT}" ]]; then
    if [[ "${RUN_EXIT_CODE}" == "0" ]]; then
        RUN_EXIT_CODE=1
    fi
    python benchmarks/write_failure_result.py \
        --output "${OUTPUT}" \
        --label "${RUN_LABEL}" \
        --log "${RUN_LOG}" \
        --hidden-dim 64 \
        --checkpointing on \
        --scaling-mode io \
        --exit-code "${RUN_EXIT_CODE}"
elif ! njord_result_completed "${OUTPUT}"
then
    RUN_EXIT_CODE=1
fi

if [[ "${PHYSICAL_TELEMETRY}" == "on" ]]; then
    stop_gpu_telemetry
    trap - EXIT
fi
if ! bash jobs/update_benchmark_archive.sh \
    "${OUTPUT}"; then
    echo "WARNING: result is safe in ${OUTPUT}; archive update failed" >&2
    if [[ "${RUN_EXIT_CODE}" == "0" ]]; then
        RUN_EXIT_CODE=1
    fi
fi
exit "${RUN_EXIT_CODE}"
