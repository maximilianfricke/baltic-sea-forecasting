#!/bin/bash
# Edge-chunked single-GPU inference vs split inference at 0.25°.
# Predictions: local hypotheses file, H7.
#
#   NJORD_DATA_025=/scratch/users/$USER/njord-data-025/configs \
#   NJORD_VENV=/scratch/users/$USER/njord-venv \
#   sbatch --account=dphpc_4gpu --gpus=1080ti:4 --time=01:15:00 \
#       jobs/inference_chunking_vs_split.sbatch

#SBATCH --job-name=njord-infer-split
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
CONFIG="${NJORD_DATA_025:?Set NJORD_DATA_025}/analysis_ocean_30d.yaml"
GRAPH=global_cluster_analysis
PARTITIONS=benchmark_results/partition-comparison/partitions_025
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
RESULT_DIR="benchmark_results/partition-comparison/${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
COMMON=(--model graph_efm --processor_layers 2 --ar_steps_train 4
        --ar_steps_eval 4 --num_workers 0)

# (A) edge-chunked inference on one GPU
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m benchmarks.inference_memory \
    --device cuda --chunk-rows 65536 --check --hidden 64 \
    --output "${RESULT_DIR}/chunked-check-h64.json" -- \
    --config_path "${CONFIG}" --graph "${GRAPH}" "${COMMON[@]}"
echo "=== chunked check: exit $?"
for hidden in 96 112 128 192 256 288 320; do
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m benchmarks.inference_memory \
        --device cuda --chunk-rows 65536 --hidden "${hidden}" \
        --output "${RESULT_DIR}/chunked-h${hidden}.json" -- \
        --config_path "${CONFIG}" --graph "${GRAPH}" "${COMMON[@]}"
    echo "=== chunked h${hidden}: exit $?"
done

# (B) split inference, one launch per configuration
for gpus in 4 2; do
    devices="$(seq -s, 0 $((gpus - 1)))"
    for hidden in 64 112 256; do
        for strategy in geographic lon_bands; do
            if [[ "${strategy}" == geographic ]]; then
                partition=(--partition-strategy geographic)
            else
                partition=(--partition-strategy file --partition-file
                           "${PARTITIONS}/lon_bands_025_k${gpus}.npz")
            fi
            label="split-${gpus}gpu-${strategy}-h${hidden}"
            CUDA_VISIBLE_DEVICES="${devices}" "${PYTHON}" \
                -m torch.distributed.run --standalone \
                --nproc-per-node="${gpus}" -m benchmarks.inference_split \
                --config-path "${CONFIG}" --graph "${GRAPH}" \
                --hidden "${hidden}" "${partition[@]}" --steps 3 \
                --output "${RESULT_DIR}/${label}.json"
            echo "=== ${label}: exit $?"
        done
    done
done
echo "Results: ${RESULT_DIR}"
