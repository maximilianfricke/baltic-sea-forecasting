#!/bin/bash
#SBATCH --job-name=njord-profile
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_results/slurm-profile-%j.out

set -euo pipefail

export NJORD_JOB_STARTED_EPOCH_SECONDS="$(date +%s.%N)"

. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
VENV_PATH="${NJORD_VENV:-${REPO_ROOT}/.venv}"
PROFILE_DIR="${PROFILE_DIR:-${REPO_ROOT}/benchmark_results/profiles}"
mkdir -p "${PROFILE_DIR}"
source jobs/stage_benchmark_data.sh
source "${VENV_PATH}/bin/activate"
RESULT_DIR="${PROFILE_DIR}"
source jobs/start_gpu_telemetry.sh

python -m neural_lam.train_model \
    --config_path "${JOB_DATA_DIR}/global_ocean_1.yaml" \
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
    --num_workers "${SLURM_CPUS_PER_TASK}" \
    --devices auto \
    --n_example_pred 0 \
    --benchmark_warmup_steps 3 \
    --benchmark_steps 5 \
    --benchmark_label "profile-1080ti" \
    --benchmark_profiler_dir "${PROFILE_DIR}" \
    --benchmark_output "${PROFILE_DIR}/profile-${SLURM_JOB_ID}.json"

stop_gpu_telemetry
trap - EXIT
bash jobs/update_benchmark_summary.sh "${REPO_ROOT}/benchmark_results"
if ! bash jobs/generate_benchmark_plots.sh \
    "${REPO_ROOT}/benchmark_results"; then
    echo "Warning: profile completed, but plot generation failed" >&2
fi
