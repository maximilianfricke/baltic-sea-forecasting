#!/bin/bash
#SBATCH --job-name=njord-scaling-bundle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --overcommit
#SBATCH --gpus=1080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=benchmark_results/slurm-scaling-%j.out

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

REPO_ROOT="${SLURM_SUBMIT_DIR}"
VENV_PATH="${NJORD_VENV:-${REPO_ROOT}/.venv}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmark_results/scaling}"
GPU_COUNT="${SLURM_NTASKS:-1}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
PROCESSOR_LAYERS="${PROCESSOR_LAYERS:-3}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${RESULT_DIR}"
source "${VENV_PATH}/bin/activate"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit \
    --format=csv

FAILURES=0
for WORKLOAD in training inference; do
    for SCALING_MODE in strong weak; do
        if [[ "${SCALING_MODE}" == "strong" ]]; then
            LOCAL_BATCH="$((4 / GPU_COUNT))"
            GLOBAL_BATCH=4
        else
            LOCAL_BATCH=1
            GLOBAL_BATCH="${GPU_COUNT}"
        fi
        for REPEAT in 1 2 3; do
            LABEL="${WORKLOAD}-${SCALING_MODE}-${GPU_COUNT}gpu"
            LABEL="${LABEL}-repeat${REPEAT}"
            OUTPUT="${RESULT_DIR}/${LABEL}-${SLURM_JOB_ID}.json"
            TELEMETRY_LABEL="${LABEL}"
            source jobs/start_gpu_telemetry.sh

            if ! srun --overcommit --ntasks="${GPU_COUNT}" \
                --gpu-bind=single:1 \
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
                --benchmark_output "${OUTPUT}"; then
                FAILURES="$((FAILURES + 1))"
            fi
            stop_gpu_telemetry
            trap - EXIT
        done
    done
done

bash jobs/update_benchmark_summary.sh "${REPO_ROOT}/benchmark_results"

if [[ "${FAILURES}" -gt 0 ]]; then
    echo "${FAILURES} scaling configuration(s) failed" >&2
    exit 1
fi
