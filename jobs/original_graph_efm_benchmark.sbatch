#!/bin/bash
# Original GraphEFM reference, matched to the AR4 1/2/4-GPU experiment.
# Submit from the repository root; no training is executed on the login node.
#SBATCH --job-name=njord-original-ar4
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=benchmark_results/model-parallel-real/slurm-original-ar4-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

PYTHON="${NJORD_VENV:-${SLURM_SUBMIT_DIR}/.venv}/bin/python"
SOURCE_ROOT="${NJORD_DATA_ROOT:-/scratch/users/${USER}/baltic-sea-forecasting-data/physical_repeat_16}"
STAGED_ROOT="${TMPDIR:?TMPDIR missing}/njord-original-${SLURM_JOB_ID}/physical_repeat_16"
REFERENCE_INPUT="${NJORD_REFERENCE_INPUT:-full-grid}"
DATA_ARGUMENTS=(--data-mode rank-zero-broadcast)
OUTPUT_STEM="original-graph-efm-ar4-1gpu"
case "${REFERENCE_INPUT}" in
    full-grid)
        REFERENCE_RELATIVE="full/ar4-workers8-prefetch2-4944-4949-4952/real-mp-max-ar4-ckpt-full-rank-local-1gpu-4944.json"
        ;;
    ocean-shard)
        SHARD_PARENT="${NJORD_PARTITIONED_DATA_ROOT:-/scratch/users/${USER}/baltic-sea-forecasting-data/partitioned/physical_repeat_16}"
        SHARD_SOURCE="${SHARD_PARENT}/geographic_rcb_ws1"
        SHARD_STAGE="${TMPDIR}/njord-original-${SLURM_JOB_ID}/partitioned"
        test -f "${SHARD_SOURCE}/manifest.json"
        test -d "${SHARD_SOURCE}/rank_0/ocean.zarr"
        test -d "${SHARD_SOURCE}/rank_0/atmosphere.zarr"
        DATA_ARGUMENTS=(--data-mode rank-local --partitioned-data-root "${SHARD_STAGE}")
        OUTPUT_STEM="original-graph-efm-ar4-ocean-shard-1gpu"
        REFERENCE_RELATIVE="baseline/original-ar4-workers8-prefetch2-4981/original-graph-efm-ar4-1gpu-4981.json"
        ;;
    *)
        echo "Unknown reference input: ${REFERENCE_INPUT}" >&2
        exit 2
        ;;
esac
export NJORD_REFERENCE_RESULT="${SLURM_SUBMIT_DIR}/benchmark_results/model-parallel-real/${REFERENCE_RELATIVE}"
test -x "${PYTHON}"
test -f "${SOURCE_ROOT}/global_ocean_1.yaml"
test -f "${SOURCE_ROOT}/graphs/global_cluster_graph/mesh_features.pt"
test -f "${NJORD_REFERENCE_RESULT}"
mkdir -p "${STAGED_ROOT}" benchmark_results/model-parallel-real
STAGE_START_NS="$(date +%s%N)"
rsync -a "${SOURCE_ROOT}/" "${STAGED_ROOT}/"
SOURCE_END_NS="$(date +%s%N)"
export NJORD_SOURCE_STAGING_MS="$(((SOURCE_END_NS - STAGE_START_NS) / 1000000))"
export NJORD_SHARD_STAGING_MS=0
unset NJORD_SOURCE_PARTITIONED_DATA_ROOT
if [[ "${REFERENCE_INPUT}" == "ocean-shard" ]]; then
    mkdir -p "${SHARD_STAGE}/geographic_rcb_ws1"
    SHARD_START_NS="$(date +%s%N)"
    rsync -a "${SHARD_SOURCE}/" "${SHARD_STAGE}/geographic_rcb_ws1/"
    SHARD_END_NS="$(date +%s%N)"
    export NJORD_SHARD_STAGING_MS="$(((SHARD_END_NS - SHARD_START_NS) / 1000000))"
    export NJORD_SOURCE_PARTITIONED_DATA_ROOT="${SHARD_PARENT}"
fi
STAGE_END_NS="$(date +%s%N)"
export NJORD_STAGING_MS="$(((STAGE_END_NS - STAGE_START_NS) / 1000000))"
export NJORD_SOURCE_DATA_ROOT="${SOURCE_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
OUTPUT="benchmark_results/model-parallel-real/${OUTPUT_STEM}-${SLURM_JOB_ID}.json"
echo "Original GraphEFM: H64/L2/B2/AR4/FP32, checkpointing, 8 workers, prefetch 2"
echo "Input mode: ${REFERENCE_INPUT}; checkpoint early stop: original default (on)"
echo "Original full-grid source: ${SOURCE_ROOT}; staged: ${STAGED_ROOT}"
echo "Staging: ${NJORD_STAGING_MS} ms; output: ${OUTPUT}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

"${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node=1 \
    -m benchmarks.original_graph_efm \
    --config-path "${STAGED_ROOT}/global_ocean_1.yaml" \
    --graph global_cluster_graph --hidden-dim 64 --processor-layers 2 \
    --batch-size 2 --ar-steps 4 --num-workers 8 --prefetch-factor 2 \
    --warmup-steps "${NJORD_WARMUP_STEPS:-10}" \
    --steps "${NJORD_MEASUREMENT_STEPS:-50}" \
    --progress-interval 0 "${DATA_ARGUMENTS[@]}" --output "${OUTPUT}"
