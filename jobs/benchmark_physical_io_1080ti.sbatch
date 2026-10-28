#!/bin/bash
# Long baseline using independent physical Zarr chunks for input-I/O timing.

#SBATCH --job-name=njord-physical-io
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_results/slurm-physical-io-%j.out

set -euo pipefail

export NJORD_JOB_STARTED_EPOCH_SECONDS="$(date +%s.%N)"
. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
VENV_PATH="${NJORD_VENV:-${REPO_ROOT}/.venv}"
DEFAULT_DATA_ROOT="/scratch/users/${USER}/baltic-sea-forecasting-data"
DEFAULT_DATA_ROOT="${DEFAULT_DATA_ROOT}/physical_repeat_16"
PHYSICAL_DATA_ROOT="${PHYSICAL_DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
PHYSICAL_IO_MODE="${PHYSICAL_IO_MODE:-scratch}"
RESULT_DIR="${REPO_ROOT}/benchmark_results/physical-io"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-2}}"
NUM_WORKERS="$((ALLOCATED_CPUS > 1 ? ALLOCATED_CPUS - 1 : 0))"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ ! -f "${PHYSICAL_DATA_ROOT}/global_ocean_1.yaml" ]]; then
    echo "Physical dataset not found: ${PHYSICAL_DATA_ROOT}" >&2
    exit 2
fi

STAGING_START_NS="$(date +%s%N)"
case "${PHYSICAL_IO_MODE}" in
    scratch)
        JOB_DATA_DIR="${PHYSICAL_DATA_ROOT}"
        export NJORD_DATA_LOCATION="physical-repeat-shared-scratch"
        ;;
    tmpdir)
        JOB_DATA_DIR="${TMPDIR}/njord-physical-repeat-16"
        mkdir -p "${JOB_DATA_DIR}"
        cp -rL --preserve=timestamps \
            "${PHYSICAL_DATA_ROOT}/." "${JOB_DATA_DIR}/"
        export NJORD_DATA_LOCATION="physical-repeat-node-local-tmpdir"
        ;;
    *)
        echo "PHYSICAL_IO_MODE must be scratch or tmpdir" >&2
        exit 2
        ;;
esac
STAGING_END_NS="$(date +%s%N)"
STAGING_ELAPSED_NS="$((${STAGING_END_NS} - ${STAGING_START_NS}))"
export NJORD_DATA_STAGING_SECONDS="${STAGING_ELAPSED_NS}e-9"
export NJORD_DATA_STAGING_BYTES="$(du -sb "${JOB_DATA_DIR}" | cut -f1)"

mkdir -p "${RESULT_DIR}"
source "${VENV_PATH}/bin/activate"
export RESULT_DIR
export ENABLE_GPU_TELEMETRY=1
export TELEMETRY_LABEL="physical-${PHYSICAL_IO_MODE}"
source jobs/start_gpu_telemetry.sh

echo "I/O mode: ${PHYSICAL_IO_MODE}"
echo "Data source: ${JOB_DATA_DIR}"
echo "Data size: $(du -sh "${JOB_DATA_DIR}" | cut -f1)"
echo "Staging seconds: ${NJORD_DATA_STAGING_SECONDS}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit \
    --format=csv

OUTPUT="${RESULT_DIR}/physical-${PHYSICAL_IO_MODE}-${SLURM_JOB_ID}.json"
RUN_LOG="${RESULT_DIR}/physical-${PHYSICAL_IO_MODE}-${SLURM_JOB_ID}.log"
set +e
python -m neural_lam.train_model \
    --config_path "${JOB_DATA_DIR}/global_ocean_1.yaml" \
    --model graph_efm \
    --graph global_cluster_graph \
    --hidden_dim 64 \
    --hidden_dim_grid 64 \
    --hidden_dim_mesh_nodes 64 \
    --hidden_dim_edge 64 \
    --processor_layers 2 \
    --batch_size 2 \
    --precision 32 \
    --epochs 100 \
    --ar_steps_train 4 \
    --ar_steps_eval 4 \
    --num_workers "${NUM_WORKERS}" \
    --num_nodes 1 \
    --devices auto \
    --n_example_pred 0 \
    --benchmark_scaling_mode io \
    --benchmark_warmup_steps 20 \
    --benchmark_steps 200 \
    --benchmark_timing_mode phase \
    --benchmark_label \
    "physical-io-${PHYSICAL_IO_MODE}-1080ti-h64-p2-b2-ar4-fp32-ckpt-on" \
    --benchmark_output "${OUTPUT}" 2>&1 | tee "${RUN_LOG}"
RUN_EXIT_CODE="${PIPESTATUS[0]}"
set -e

if [[ ! -f "${OUTPUT}" ]]; then
    if [[ "${RUN_EXIT_CODE}" == "0" ]]; then
        RUN_EXIT_CODE=1
    fi
    python benchmarks/write_failure_result.py \
        --output "${OUTPUT}" \
        --label \
        "physical-io-${PHYSICAL_IO_MODE}-1080ti-h64-p2-b2-ar4-fp32-ckpt-on" \
        --log "${RUN_LOG}" \
        --hidden-dim 64 \
        --checkpointing on \
        --scaling-mode io \
        --exit-code "${RUN_EXIT_CODE}"
elif ! python -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
sys.exit(0 if result.get("status") == "completed" else 1)
' "${OUTPUT}"
then
    RUN_EXIT_CODE=1
fi

stop_gpu_telemetry
trap - EXIT
if ! bash jobs/publish_benchmark_results.sh --push \
    "${OUTPUT}"; then
    echo "WARNING: result is safe in ${OUTPUT}; Git publication failed" >&2
fi
exit "${RUN_EXIT_CODE}"
