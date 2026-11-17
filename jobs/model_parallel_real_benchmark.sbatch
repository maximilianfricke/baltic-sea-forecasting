#!/bin/bash
# Real-data GraphEFM strong-scaling benchmark. Resources come from submitter.

#SBATCH --job-name=njord-real-mp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=benchmark_results/slurm-real-mp-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

VENV_PATH="${NJORD_VENV:-${SLURM_SUBMIT_DIR}/.venv}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    echo "Python environment not found: ${VENV_PATH}" >&2
    exit 2
fi

GPU_COUNT="${NJORD_GPU_COUNT:?NJORD_GPU_COUNT must be set by submitter}"
DATA_PARENT="/scratch/users/${USER}"
DEFAULT_DATA_ROOT="${DATA_PARENT}/baltic-sea-forecasting-data"
DEFAULT_DATA_ROOT="${DEFAULT_DATA_ROOT}/physical_repeat_16"
DATA_ROOT="${NJORD_DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
NUM_WORKERS="${NJORD_NUM_WORKERS:-0}"
DATA_MODE="${NJORD_DATA_MODE:-rank-local}"
WARMUP_STEPS="${NJORD_WARMUP_STEPS:-1}"
MEASUREMENT_STEPS="${NJORD_MEASUREMENT_STEPS:-1}"
PROGRESS_INTERVAL="${NJORD_PROGRESS_INTERVAL:-1}"
CONFIG_PATH="${DATA_ROOT}/global_ocean_1.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Real benchmark dataset not found: ${CONFIG_PATH}" >&2
    exit 2
fi

DATA_ARGUMENTS=(--data-mode "${DATA_MODE}")
if [[ "${DATA_MODE}" == "rank-local" ]]; then
    DEFAULT_PARTITION_ROOT="${DATA_PARENT}/baltic-sea-forecasting-data"
    DEFAULT_PARTITION_ROOT="${DEFAULT_PARTITION_ROOT}/partitioned/physical_repeat_16"
    PARTITION_ROOT="${NJORD_PARTITIONED_DATA_ROOT:-${DEFAULT_PARTITION_ROOT}}"
    PARTITION_NAME="geographic_rcb_ws${GPU_COUNT}"
    PARTITION_SOURCE="${PARTITION_ROOT}/${PARTITION_NAME}"
    if [[ ! -f "${PARTITION_SOURCE}/manifest.json" ]]; then
        echo "Partitioned benchmark data not found: ${PARTITION_SOURCE}" >&2
        exit 2
    fi
    STAGED_ROOT="${TMPDIR:?TMPDIR is not set}/njord-partitioned"
    mkdir -p "${STAGED_ROOT}/${PARTITION_NAME}"
    STAGE_START_NS="$(date +%s%N)"
    rsync -a \
        "${PARTITION_SOURCE}/" \
        "${STAGED_ROOT}/${PARTITION_NAME}/"
    STAGE_END_NS="$(date +%s%N)"
    export NJORD_STAGING_MS="$(((STAGE_END_NS - STAGE_START_NS) / 1000000))"
    DATA_ARGUMENTS+=(--partitioned-data-root "${STAGED_ROOT}")
else
    export NJORD_STAGING_MS=0
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

RESULT_DIR="${NJORD_RESULT_DIR:-benchmark_results/model-parallel-real}"
RUN_LABEL="${NJORD_RUN_LABEL:-run}"
OUTPUT="${RESULT_DIR}/real-mp-${RUN_LABEL}-${DATA_MODE}-${GPU_COUNT}gpu-${SLURM_JOB_ID}.json"
mkdir -p "${RESULT_DIR}"

echo "World size: ${GPU_COUNT}"
echo "Dataset: ${DATA_ROOT}"
echo "Output: ${OUTPUT}"
echo "Data workers: ${NUM_WORKERS}"
echo "Data mode: ${DATA_MODE}"
echo "Data staging: ${NJORD_STAGING_MS} ms"
echo "Warm-up steps: ${WARMUP_STEPS}"
echo "Measurement steps: ${MEASUREMENT_STEPS}"
echo "Progress interval: ${PROGRESS_INTERVAL}"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader

"${VENV_PATH}/bin/python" -m torch.distributed.run \
    --standalone \
    --nproc-per-node="${GPU_COUNT}" \
    benchmarks/model_parallel_real.py \
    --config-path "${CONFIG_PATH}" \
    --graph global_cluster_graph \
    --hidden-dim 64 \
    --processor-layers 2 \
    --batch-size 2 \
    --ar-steps 1 \
    --num-workers "${NUM_WORKERS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --steps "${MEASUREMENT_STEPS}" \
    --progress-interval "${PROGRESS_INTERVAL}" \
    --partition-strategy auto \
    "${DATA_ARGUMENTS[@]}" \
    --output "${OUTPUT}"

echo "Result: ${OUTPUT}"
