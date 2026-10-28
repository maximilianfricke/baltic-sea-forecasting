#!/bin/bash
# Short, controlled measurements needed for the project-direction decision.

#SBATCH --job-name=njord-decision
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=benchmark_results/slurm-decision-%j.out

set -euo pipefail

export NJORD_JOB_STARTED_EPOCH_SECONDS="$(date +%s.%N)"
source "${SLURM_SUBMIT_DIR}/jobs/common.sh"
njord_init_cluster_job
njord_configure_training_job
RESULT_DIR="${REPO_ROOT}/benchmark_results/decision-suite"
LOG_DIR="${RESULT_DIR}/logs"
PHYSICAL_DATA_ROOT="${PHYSICAL_DATA_ROOT:-${DEFAULT_DATA_ROOT}}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
njord_require_dataset "${PHYSICAL_DATA_ROOT}"
export NJORD_DATA_LOCATION="physical-repeat-shared-scratch"
export NJORD_DATA_STAGING_SECONDS=0
export NJORD_DATA_STAGING_BYTES="$(du -sb "${PHYSICAL_DATA_ROOT}" | cut -f1)"
export NJORD_GPU_NAME="$(nvidia-smi --query-gpu=name \
    --format=csv,noheader | head -n 1)"
GPU_QUERY="name,uuid,driver_version,memory.total,power.limit"
GPU_QUERY="${GPU_QUERY},clocks.max.sm"
nvidia-smi --query-gpu="${GPU_QUERY}" \
    --format=csv

run_case() {
    local label="$1"
    local hidden_dim="$2"
    local checkpointing="$3"
    local warmup="$4"
    local steps="$5"
    local output="${RESULT_DIR}/${label}-${SLURM_JOB_ID}.json"
    local log="${LOG_DIR}/${label}-${SLURM_JOB_ID}.log"
    local checkpoint_flag=()
    if [[ "${checkpointing}" == "off" ]]; then
        checkpoint_flag=(--benchmark_disable_grad_checkpointing)
    fi

    echo "Running ${label}: h=${hidden_dim}, checkpointing=${checkpointing}"
    set +e
    njord_run_graph_efm_benchmark \
        "${PHYSICAL_DATA_ROOT}" "${hidden_dim}" "${NUM_WORKERS}" \
        --benchmark_scaling_mode capacity \
        --benchmark_warmup_steps "${warmup}" \
        --benchmark_steps "${steps}" \
        --benchmark_timing_mode phase \
        --benchmark_label "${label}" \
        --benchmark_output "${output}" \
        "${checkpoint_flag[@]}" > "${log}" 2>&1
    local exit_code=$?
    set -e
    if [[ ! -f "${output}" ]]; then
        if [[ "${exit_code}" == "0" ]]; then
            exit_code=1
        fi
        python benchmarks/write_failure_result.py \
            --output "${output}" \
            --label "${label}" \
            --log "${log}" \
            --hidden-dim "${hidden_dim}" \
            --checkpointing "${checkpointing}" \
            --exit-code "${exit_code}"
    elif ! njord_result_completed "${output}"; then
        exit_code=1
    fi
    echo "${label} exit code: ${exit_code}"
}

# A controlled pair isolates recomputation cost and activation-memory savings.
run_case hidden-h64-ckpt-on 64 on 10 50
run_case hidden-h64-ckpt-off 64 off 10 50

# Short capacity probes locate the useful range without spending a full sweep.
run_case hidden-h128-ckpt-on 128 on 5 15
run_case hidden-h192-ckpt-on 192 on 3 8
run_case hidden-h256-ckpt-on 256 on 2 5

# Archive only this controlled suite. Plot generation happens on the allocated
# compute node; the user reviews, commits, and pushes the files manually.
if ! bash jobs/update_benchmark_archive.sh \
    "${RESULT_DIR}"; then
    echo "WARNING: results are safe in ${RESULT_DIR}; archive update failed" \
        >&2
    exit 1
fi
