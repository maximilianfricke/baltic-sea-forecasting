#!/bin/bash
# Peak memory of one GraphEFM inference step at 1° and 0.25° on one GPU.
# Predictions: local hypotheses file, H6.
#
#   NJORD_DATA_ROOT=/scratch/users/$USER/njord-data \
#   NJORD_DATA_025=/scratch/users/$USER/njord-data-025/configs \
#   NJORD_VENV=/scratch/users/$USER/njord-venv \
#   sbatch --account=dphpc --gpus=1080ti:1 --time=00:40:00 \
#       jobs/inference_memory.sbatch

#SBATCH --job-name=njord-inference-mem
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
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RESULT_DIR="benchmark_results/partition-comparison/${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

COMMON=(--model graph_efm --processor_layers 2 --ar_steps_train 4
        --ar_steps_eval 4 --num_workers 0)
"${PYTHON}" -m benchmarks.inference_memory --device cuda --hidden 64 256 \
    --output "${RESULT_DIR}/inference-1deg.json" -- \
    --config_path "${NJORD_DATA_ROOT:?}/global_ocean_1.yaml" \
    --graph global_cluster_graph "${COMMON[@]}"
echo "1-degree status: $?"
"${PYTHON}" -m benchmarks.inference_memory --device cuda \
    --hidden 64 96 112 128 192 256 \
    --output "${RESULT_DIR}/inference-025deg.json" -- \
    --config_path "${NJORD_DATA_025:?}/analysis_ocean_30d.yaml" \
    --graph global_cluster_analysis "${COMMON[@]}"
echo "0.25-degree status: $?"
