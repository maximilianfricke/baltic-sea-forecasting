#!/bin/bash
#SBATCH --job-name=njord-baseline
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
#SBATCH --cpus-per-task=4
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
WARMUP_STEPS="${WARMUP_STEPS:-10}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-50}"
RUN_KIND="${RUN_KIND:-baseline}"
CHECKPOINT_LABEL="off"
CHECKPOINT_FLAG=()

if [[ "${GRAD_CHECKPOINTING:-0}" == "1" ]]; then
    CHECKPOINT_LABEL="on"
    CHECKPOINT_FLAG=(--grad_checkpointing)
fi

mkdir -p "${RESULT_DIR}"
source jobs/stage_benchmark_data.sh
source "${VENV_PATH}/bin/activate"
source jobs/start_gpu_telemetry.sh

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
    --batch_size 1 \
    --precision 32 \
    --epochs 100 \
    --ar_steps_train 1 \
    --ar_steps_eval 1 \
    --num_workers "${SLURM_CPUS_PER_TASK}" \
    --num_nodes 1 \
    --devices auto \
    --n_example_pred 0 \
    --benchmark_warmup_steps "${WARMUP_STEPS}" \
    --benchmark_steps "${BENCHMARK_STEPS}" \
    --benchmark_label \
    "${RUN_KIND}-1080ti-h${HIDDEN_DIM}-p${PROCESSOR_LAYERS}-ckpt-${CHECKPOINT_LABEL}" \
    --benchmark_output "${RESULT_DIR}/${RUN_KIND}-${SLURM_JOB_ID}.json" \
    "${CHECKPOINT_FLAG[@]}"

stop_gpu_telemetry
trap - EXIT
bash jobs/update_benchmark_summary.sh "${REPO_ROOT}/benchmark_results"
if ! bash jobs/generate_benchmark_plots.sh \
    "${REPO_ROOT}/benchmark_results"; then
    echo "Warning: benchmark completed, but plot generation failed" >&2
fi
