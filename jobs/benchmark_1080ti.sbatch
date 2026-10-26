#!/bin/bash
#SBATCH --job-name=njord-baseline
#SBATCH --gpus=1080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_results/slurm-%j.out

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

REPO_ROOT="${SLURM_SUBMIT_DIR}"
VENV_PATH="${NJORD_VENV:-${REPO_ROOT}/.venv}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmark_results}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
PROCESSOR_LAYERS="${PROCESSOR_LAYERS:-3}"
CHECKPOINT_LABEL="off"
CHECKPOINT_FLAG=()

if [[ "${GRAD_CHECKPOINTING:-0}" == "1" ]]; then
    CHECKPOINT_LABEL="on"
    CHECKPOINT_FLAG=(--grad_checkpointing)
fi

mkdir -p "${RESULT_DIR}"
source "${VENV_PATH}/bin/activate"
source jobs/start_gpu_telemetry.sh

GPU_QUERY="name,uuid,driver_version,memory.total,power.limit"
GPU_QUERY="${GPU_QUERY},clocks.max.sm"
nvidia-smi --query-gpu="${GPU_QUERY}" \
    --format=csv

python -m neural_lam.train_model \
    --config_path data/global_ocean_1.yaml \
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
    --benchmark_warmup_steps 5 \
    --benchmark_steps 30 \
    --benchmark_label \
    "1080ti-h${HIDDEN_DIM}-p${PROCESSOR_LAYERS}-ckpt-${CHECKPOINT_LABEL}" \
    --benchmark_output "${RESULT_DIR}/baseline-${SLURM_JOB_ID}.json" \
    "${CHECKPOINT_FLAG[@]}"
