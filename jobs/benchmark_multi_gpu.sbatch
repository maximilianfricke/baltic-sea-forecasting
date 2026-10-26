#!/bin/bash
#SBATCH --job-name=njord-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --overcommit
#SBATCH --gpus=1080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=benchmark_results/slurm-%j.out

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

REPO_ROOT="${SLURM_SUBMIT_DIR}"
VENV_PATH="${NJORD_VENV:-${REPO_ROOT}/.venv}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmark_results/scaling}"
GPU_COUNT="${SLURM_NTASKS:-1}"
SCALING_MODE="${SCALING_MODE:-weak}"
LOCAL_BATCH="${LOCAL_BATCH:-1}"
REPEAT="${REPEAT:-1}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
PROCESSOR_LAYERS="${PROCESSOR_LAYERS:-3}"
WORKLOAD="${WORKLOAD:-training}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ "${SCALING_MODE}" == "strong" ]]; then
    GLOBAL_BATCH="${GLOBAL_BATCH:-4}"
else
    GLOBAL_BATCH="$((LOCAL_BATCH * GPU_COUNT))"
fi

mkdir -p "${RESULT_DIR}"
source "${VENV_PATH}/bin/activate"
source jobs/start_gpu_telemetry.sh

GPU_QUERY="name,uuid,driver_version,memory.total,power.limit"
GPU_QUERY="${GPU_QUERY},clocks.max.sm"
nvidia-smi --query-gpu="${GPU_QUERY}" --format=csv

LABEL="${WORKLOAD}-${SCALING_MODE}-${GPU_COUNT}gpu-repeat${REPEAT}"
OUTPUT="${RESULT_DIR}/${LABEL}-${SLURM_JOB_ID}.json"

srun --overcommit --ntasks="${GPU_COUNT}" --gpu-bind=single:1 \
    python -m neural_lam.train_model \
    --config_path data/global_ocean_1.yaml \
    --model graph_efm \
    --graph global_cluster_graph \
    --hidden_dim "${HIDDEN_DIM}" \
    --hidden_dim_grid "${HIDDEN_DIM}" \
    --hidden_dim_mesh_nodes "${HIDDEN_DIM}" \
    --hidden_dim_edge "${HIDDEN_DIM}" \
    --processor_layers "${PROCESSOR_LAYERS}" \
    --batch_size "${LOCAL_BATCH}" \
    --precision 32 \
    --epochs 100 \
    --ar_steps_train 1 \
    --ar_steps_eval 1 \
    --num_workers 0 \
    --num_nodes 1 \
    --devices auto \
    --n_example_pred 0 \
    --benchmark_strategy ddp \
    --benchmark_scaling_mode "${SCALING_MODE}" \
    --benchmark_workload "${WORKLOAD}" \
    --benchmark_global_batch_size "${GLOBAL_BATCH}" \
    --benchmark_warmup_steps 10 \
    --benchmark_steps 50 \
    --benchmark_label "${LABEL}" \
    --benchmark_output "${OUTPUT}"
