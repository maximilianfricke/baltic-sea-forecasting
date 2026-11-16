#!/bin/bash
# Real-data GraphEFM strong-scaling benchmark. Resources come from submitter.

#SBATCH --job-name=njord-real-mp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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
DATA_ROOT="${NJORD_DATA_ROOT:-/scratch/users/${USER}/baltic-sea-forecasting-data/physical_repeat_16}"
CONFIG_PATH="${DATA_ROOT}/global_ocean_1.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Real benchmark dataset not found: ${CONFIG_PATH}" >&2
    exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

RESULT_DIR="benchmark_results/model-parallel-real"
OUTPUT="${RESULT_DIR}/real-mp-${GPU_COUNT}gpu-${SLURM_JOB_ID}.json"
mkdir -p "${RESULT_DIR}"

echo "World size: ${GPU_COUNT}"
echo "Dataset: ${DATA_ROOT}"
echo "Output: ${OUTPUT}"
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
    --num-workers 2 \
    --warmup-steps 5 \
    --steps 20 \
    --partition-strategy auto \
    --output "${OUTPUT}"

echo "Result: ${OUTPUT}"
