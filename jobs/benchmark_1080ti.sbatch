#!/bin/bash
#SBATCH --job-name=njord-baseline
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_results/slurm-%j.out

set -euo pipefail

export NJORD_JOB_STARTED_EPOCH_SECONDS="$(date +%s.%N)"

. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
VENV_PATH="${NJORD_VENV:-${REPO_ROOT}/.venv}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmark_results}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
PROCESSOR_LAYERS="${PROCESSOR_LAYERS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
AR_STEPS_TRAIN="${AR_STEPS_TRAIN:-1}"
AR_STEPS_EVAL="${AR_STEPS_EVAL:-${AR_STEPS_TRAIN}}"
PRECISION="${PRECISION:-32}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-50}"
RUN_KIND="${RUN_KIND:-baseline}"
TIMING_MODE="${TIMING_MODE:-phase}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-2}}"
DEFAULT_WORKERS="$((ALLOCATED_CPUS > 1 ? ALLOCATED_CPUS - 1 : 0))"
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_WORKERS}}"
ENABLE_GPU_TELEMETRY="${ENABLE_GPU_TELEMETRY:-0}"
PUBLISH_RESULTS="${PUBLISH_RESULTS:-1}"
CHECKPOINT_LABEL="off"
CHECKPOINT_FLAG=()
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ "${GRAD_CHECKPOINTING:-0}" == "1" ]]; then
    CHECKPOINT_LABEL="explicit-on"
    CHECKPOINT_FLAG=(--grad_checkpointing)
elif [[ "${AR_STEPS_TRAIN}" -gt 1 ]]; then
    # ARProbModel enables checkpointing automatically for unrolled training.
    CHECKPOINT_LABEL="auto-on"
fi

mkdir -p "${RESULT_DIR}"
OUTPUT="${RESULT_DIR}/${RUN_KIND}-${SLURM_JOB_ID}.json"
source jobs/stage_benchmark_data.sh
source "${VENV_PATH}/bin/activate"
if [[ "${ENABLE_GPU_TELEMETRY}" == "1" ]]; then
    source jobs/start_gpu_telemetry.sh
fi

GPU_QUERY="name,uuid,driver_version,memory.total,power.limit"
GPU_QUERY="${GPU_QUERY},clocks.max.sm"
nvidia-smi --query-gpu="${GPU_QUERY}" \
    --format=csv

python -m neural_lam.train_model \
    --config_path "${JOB_DATA_DIR}/global_ocean_1.yaml" \
    --model graph_efm \
    --graph global_cluster_graph \
    --hidden_dim "${HIDDEN_DIM}" \
    --hidden_dim_grid "${HIDDEN_DIM}" \
    --hidden_dim_mesh_nodes "${HIDDEN_DIM}" \
    --hidden_dim_edge "${HIDDEN_DIM}" \
    --processor_layers "${PROCESSOR_LAYERS}" \
    --batch_size "${BATCH_SIZE}" \
    --precision "${PRECISION}" \
    --epochs 100 \
    --ar_steps_train "${AR_STEPS_TRAIN}" \
    --ar_steps_eval "${AR_STEPS_EVAL}" \
    --num_workers "${NUM_WORKERS}" \
    --num_nodes 1 \
    --devices auto \
    --n_example_pred 0 \
    --benchmark_warmup_steps "${WARMUP_STEPS}" \
    --benchmark_steps "${BENCHMARK_STEPS}" \
    --benchmark_timing_mode "${TIMING_MODE}" \
    --benchmark_label \
    "${RUN_KIND}-1080ti-h${HIDDEN_DIM}-p${PROCESSOR_LAYERS}-b${BATCH_SIZE}-ar${AR_STEPS_TRAIN}-fp${PRECISION}-ckpt-${CHECKPOINT_LABEL}-${TIMING_MODE}" \
    --benchmark_output "${OUTPUT}" \
    "${CHECKPOINT_FLAG[@]}"

if [[ "${ENABLE_GPU_TELEMETRY}" == "1" ]]; then
    stop_gpu_telemetry
    trap - EXIT
fi
bash jobs/update_benchmark_summary.sh "${REPO_ROOT}/benchmark_results"

if [[ "${PUBLISH_RESULTS}" == "1" ]]; then
    if ! bash jobs/publish_benchmark_results.sh --push "${RESULT_DIR}"; then
        echo "WARNING: result is safe in ${OUTPUT}, but Git publication failed" \
            >&2
    fi
fi
