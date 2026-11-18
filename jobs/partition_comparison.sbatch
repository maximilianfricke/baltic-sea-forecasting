#!/bin/bash
# Compare partition strategies with the real-data model-parallel benchmark.
#
# One 4-GPU allocation runs every configuration on the same node, so all
# strategies see the same hardware. 2-GPU runs use GPUs 0 and 1. The order is
# repetition -> hidden size -> GPU count -> strategy, so the strategies of one
# comparison run back to back. Every run uses rank-zero-broadcast data, so no
# strategy gets rank-local shards. Predictions: local hypotheses file, H3.
#
# Submit from the repository root:
#   NJORD_DATA_ROOT=/scratch/users/$USER/njord-data \
#   NJORD_PARTITION_DIR=/scratch/users/$USER/partitions \
#   NJORD_VENV=/scratch/users/$USER/njord-venv \
#   sbatch --account=dphpc_4gpu --gpus=1080ti:4 --time=02:00:00 \
#       jobs/partition_comparison.sbatch

#SBATCH --job-name=njord-partitions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --output=benchmark_results/partition-comparison/slurm-%j.out

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

PYTHON="${NJORD_VENV:?Set NJORD_VENV}/bin/python"
CONFIG_PATH="${NJORD_DATA_ROOT:?Set NJORD_DATA_ROOT}/global_ocean_1.yaml"
PARTITION_DIR="${NJORD_PARTITION_DIR:?Set NJORD_PARTITION_DIR}"
STRATEGIES="${NJORD_STRATEGIES:-geographic lon_bands mtkahypar_km1 metis}"
HIDDEN_DIMS="${NJORD_HIDDEN_DIMS:-64 256}"
GPU_COUNTS="${NJORD_GPU_COUNTS:-2 4}"
REPETITIONS="${NJORD_REPETITIONS:-2}"
WARMUP_STEPS="${NJORD_WARMUP_STEPS:-5}"
MEASUREMENT_STEPS="${NJORD_MEASUREMENT_STEPS:-25}"
NUM_WORKERS="${NJORD_NUM_WORKERS:-2}"
RESULT_DIR="benchmark_results/partition-comparison/${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"

for path in "${PYTHON}" "${CONFIG_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Not found: ${path}" >&2
        exit 2
    fi
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader
echo "Strategies: ${STRATEGIES}; hidden: ${HIDDEN_DIMS}; GPUs: ${GPU_COUNTS}"
echo "Repetitions: ${REPETITIONS}; warm-up ${WARMUP_STEPS}; steps ${MEASUREMENT_STEPS}"

run() {
    # run LABEL GPU_COUNT HIDDEN STRATEGY_ARGS...
    local label="$1" gpus="$2" hidden="$3"
    shift 3
    local devices
    devices="$(seq -s, 0 $((gpus - 1)))"
    echo "=== $(date -u +%H:%M:%S) ${label}"
    CUDA_VISIBLE_DEVICES="${devices}" "${PYTHON}" -m torch.distributed.run \
        --standalone --nproc-per-node="${gpus}" \
        benchmarks/model_parallel_real.py \
        --config-path "${CONFIG_PATH}" \
        --graph global_cluster_graph \
        --hidden-dim "${hidden}" \
        --processor-layers 2 \
        --batch-size 2 \
        --ar-steps 1 \
        --num-workers "${NUM_WORKERS}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --steps "${MEASUREMENT_STEPS}" \
        --data-mode rank-zero-broadcast \
        --output "${RESULT_DIR}/${label}.json" \
        "$@"
    local status=$?
    echo "=== $(date -u +%H:%M:%S) ${label}: exit ${status}"
}

# Prediction check: one GPU cannot hold hidden 256 (expected to fail with
# an out-of-memory error; the job continues either way)
run "h256-1gpu-single" 1 256 --partition-strategy auto

for rep in $(seq 1 "${REPETITIONS}"); do
    for hidden in ${HIDDEN_DIMS}; do
        for gpus in ${GPU_COUNTS}; do
            for strategy in ${STRATEGIES}; do
                label="h${hidden}-${gpus}gpu-${strategy}-rep${rep}"
                if [[ "${strategy}" == "geographic" ]]; then
                    run "${label}" "${gpus}" "${hidden}" \
                        --partition-strategy geographic
                else
                    file="${PARTITION_DIR}/${strategy}_k${gpus}.npz"
                    if [[ ! -f "${file}" ]]; then
                        echo "=== ${label}: missing ${file}, skipped" >&2
                        continue
                    fi
                    run "${label}" "${gpus}" "${hidden}" \
                        --partition-strategy file --partition-file "${file}"
                fi
            done
        done
    done
done
echo "Results: ${RESULT_DIR}"
