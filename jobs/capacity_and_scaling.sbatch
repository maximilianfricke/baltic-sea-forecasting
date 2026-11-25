#!/bin/bash
# Largest hidden size that fits, and a hidden-size sweep on 1/2/4 GPUs.
#
# Same harness settings as jobs/partition_comparison.sbatch (batch 2, AR1,
# rank-zero broadcast, geographic split) so the results extend job 4467.
# Capacity points run 1 warm-up + 2 measured steps; out-of-memory is an
# expected outcome and the job continues. Predictions: local hypotheses
# file, H4 and H5.
#
# Submit from the repository root:
#   NJORD_DATA_ROOT=/scratch/users/$USER/njord-data \
#   NJORD_VENV=/scratch/users/$USER/njord-venv \
#   sbatch --account=dphpc_4gpu --gpus=1080ti:4 --time=02:00:00 \
#       jobs/capacity_and_scaling.sbatch

#SBATCH --job-name=njord-capacity
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
CAPACITY_K2="${NJORD_CAPACITY_K2:-320 336 352}"
CAPACITY_K4="${NJORD_CAPACITY_K4:-496 528 560}"
SWEEP_HIDDEN="${NJORD_SWEEP_HIDDEN:-64 128 192 256}"
SINGLE_GPU_HIDDEN="${NJORD_SINGLE_GPU_HIDDEN:-64 128}"
REPETITIONS="${NJORD_REPETITIONS:-2}"
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
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" -c "import neural_lam, neural_lam.distributed_message_passing; \
print('neural_lam from', neural_lam.__file__)" || exit 2

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader

run() {
    # run LABEL GPU_COUNT HIDDEN WARMUP STEPS
    local label="$1" gpus="$2" hidden="$3" warmup="$4" steps="$5"
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
        --warmup-steps "${warmup}" \
        --steps "${steps}" \
        --data-mode rank-zero-broadcast \
        --partition-strategy "$([[ ${gpus} == 1 ]] && echo auto || echo geographic)" \
        --output "${RESULT_DIR}/${label}.json"
    echo "=== $(date -u +%H:%M:%S) ${label}: exit $?"
}

# H4: capacity
for hidden in ${CAPACITY_K2}; do
    run "capacity-h${hidden}-2gpu" 2 "${hidden}" 1 2
done
for hidden in ${CAPACITY_K4}; do
    run "capacity-h${hidden}-4gpu" 4 "${hidden}" 1 2
done

# H5: sweep, repetition outermost so drift affects all sizes alike
for rep in $(seq 1 "${REPETITIONS}"); do
    for hidden in ${SWEEP_HIDDEN}; do
        for gpus in 4 2; do
            run "h${hidden}-${gpus}gpu-geographic-rep${rep}" "${gpus}" "${hidden}" 5 25
        done
    done
    for hidden in ${SINGLE_GPU_HIDDEN}; do
        run "h${hidden}-1gpu-single-rep${rep}" 1 "${hidden}" 5 25
    done
done
echo "Results: ${RESULT_DIR}"
