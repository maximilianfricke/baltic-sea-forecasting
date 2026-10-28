#!/bin/bash
#SBATCH --job-name=njord-profile
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
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
PROFILE_TRACE_DIR="${PROFILE_TRACE_DIR:-/scratch/users/${USER}/baltic-sea-forecasting-results/profiles}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-2}}"
NUM_WORKERS="$((ALLOCATED_CPUS > 1 ? ALLOCATED_CPUS - 1 : 0))"

# Defaults match the realistic baseline (submit_realistic_phase.sh), so the
# trace explains the same workload that the timing runs measure. Four-step
# unrolling enables gradient checkpointing automatically.
HIDDEN_DIM="${HIDDEN_DIM:-64}"
PROCESSOR_LAYERS="${PROCESSOR_LAYERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
AR_STEPS_TRAIN="${AR_STEPS_TRAIN:-4}"
AR_STEPS_EVAL="${AR_STEPS_EVAL:-${AR_STEPS_TRAIN}}"
PRECISION="${PRECISION:-32}"
CRPS_WEIGHT="${CRPS_WEIGHT:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-5}"
LABEL="profile-1080ti-h${HIDDEN_DIM}-p${PROCESSOR_LAYERS}-b${BATCH_SIZE}-ar${AR_STEPS_TRAIN}-fp${PRECISION}-crps${CRPS_WEIGHT}"

mkdir -p "${PROFILE_DIR}"
mkdir -p "${PROFILE_TRACE_DIR}"
source jobs/stage_benchmark_data.sh
source "${VENV_PATH}/bin/activate"

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
    --epochs 10 \
    --ar_steps_train "${AR_STEPS_TRAIN}" \
    --ar_steps_eval "${AR_STEPS_EVAL}" \
    --crps_weight "${CRPS_WEIGHT}" \
    --num_workers "${NUM_WORKERS}" \
    --devices auto \
    --n_example_pred 0 \
    --benchmark_warmup_steps "${WARMUP_STEPS}" \
    --benchmark_steps "${BENCHMARK_STEPS}" \
    --benchmark_label "${LABEL}" \
    --benchmark_profiler_dir "${PROFILE_TRACE_DIR}" \
    --benchmark_output "${PROFILE_DIR}/profile-${SLURM_JOB_ID}.json"

# Attribute forward, recompute and backward GPU time to model components.
# This runs here on the compute node because traces can be several GB.
TRACE_FILE="$(ls -t "${PROFILE_TRACE_DIR}"/*"${LABEL}"*.pt.trace.json 2>/dev/null | head -n 1 || true)"
if [[ -n "${TRACE_FILE}" ]]; then
    python benchmarks/attribute_profile.py "${TRACE_FILE}" \
        --output "${PROFILE_DIR}/attribution-${SLURM_JOB_ID}.json" \
        || echo "WARNING: profile attribution failed for ${TRACE_FILE}" >&2
else
    echo "WARNING: no trace found for label ${LABEL}" >&2
fi

bash jobs/update_benchmark_summary.sh "${REPO_ROOT}/benchmark_results"
