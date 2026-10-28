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
. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
VENV_PATH="${NJORD_VENV:-${REPO_ROOT}/.venv}"
RESULT_DIR="${REPO_ROOT}/benchmark_results/decision-suite"
LOG_DIR="${RESULT_DIR}/logs"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-2}}"
NUM_WORKERS="$((ALLOCATED_CPUS > 1 ? ALLOCATED_CPUS - 1 : 0))"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
source jobs/stage_benchmark_data.sh
source "${VENV_PATH}/bin/activate"
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
    python -m neural_lam.train_model \
        --config_path "${JOB_DATA_DIR}/global_ocean_1.yaml" \
        --model graph_efm \
        --graph global_cluster_graph \
        --hidden_dim "${hidden_dim}" \
        --hidden_dim_grid "${hidden_dim}" \
        --hidden_dim_mesh_nodes "${hidden_dim}" \
        --hidden_dim_edge "${hidden_dim}" \
        --processor_layers 2 \
        --batch_size 2 \
        --precision 32 \
        --epochs 100 \
        --ar_steps_train 4 \
        --ar_steps_eval 4 \
        --num_workers "${NUM_WORKERS}" \
        --num_nodes 1 \
        --devices auto \
        --n_example_pred 0 \
        --benchmark_scaling_mode capacity \
        --benchmark_warmup_steps "${warmup}" \
        --benchmark_steps "${steps}" \
        --benchmark_timing_mode phase \
        --benchmark_label "${label}" \
        --benchmark_output "${output}" \
        "${checkpoint_flag[@]}" > "${log}" 2>&1
    local exit_code=$?
    set -e
    if [[ "${exit_code}" -ne 0 && ! -f "${output}" ]]; then
        python benchmarks/write_failure_result.py \
            --output "${output}" \
            --label "${label}" \
            --log "${log}" \
            --hidden-dim "${hidden_dim}" \
            --checkpointing "${checkpointing}" \
            --exit-code "${exit_code}"
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

bash jobs/update_benchmark_summary.sh "${REPO_ROOT}/benchmark_results"
# Import the existing cluster history as well as this suite. Plot generation
# and Git publication therefore happen on the allocated compute node, not on
# the shared login node.
if ! bash jobs/publish_benchmark_results.sh --push \
    "${REPO_ROOT}/benchmark_results"; then
    echo "WARNING: results are safe in ${RESULT_DIR}; Git publication failed" \
        >&2
fi
