#!/bin/bash
#SBATCH --job-name=njord-ddp-profile
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --overcommit
#SBATCH --gpus=1080ti:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_results/slurm-ddp-profile-%j.out

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

REPO_ROOT="${SLURM_SUBMIT_DIR}"
VENV_PATH="${NJORD_VENV:-${REPO_ROOT}/.venv}"
PROFILE_DIR="${PROFILE_DIR:-${REPO_ROOT}/benchmark_results/profiles/ddp}"
GPU_COUNT="${SLURM_NTASKS:-4}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "${PROFILE_DIR}"
source "${VENV_PATH}/bin/activate"
RESULT_DIR="${PROFILE_DIR}"
source jobs/start_gpu_telemetry.sh

srun --overcommit --ntasks="${GPU_COUNT}" --gpu-bind=single:1 \
    python -m neural_lam.train_model \
    --config_path data/global_ocean_1.yaml \
    --model graph_efm \
    --graph global_cluster_graph \
    --hidden_dim 64 \
    --hidden_dim_grid 64 \
    --hidden_dim_mesh_nodes 64 \
    --hidden_dim_edge 64 \
    --processor_layers 3 \
    --batch_size 1 \
    --precision 32 \
    --epochs 10 \
    --ar_steps_train 1 \
    --ar_steps_eval 1 \
    --num_workers 0 \
    --devices auto \
    --n_example_pred 0 \
    --benchmark_strategy ddp \
    --benchmark_scaling_mode weak \
    --benchmark_warmup_steps 3 \
    --benchmark_steps 5 \
    --benchmark_label "profile-ddp-${GPU_COUNT}gpu" \
    --benchmark_profiler_dir "${PROFILE_DIR}" \
    --benchmark_output "${PROFILE_DIR}/profile-${SLURM_JOB_ID}.json"

bash jobs/update_benchmark_summary.sh "${REPO_ROOT}/benchmark_results"
