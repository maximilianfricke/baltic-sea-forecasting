#!/bin/bash
#SBATCH --job-name=njord-single-sweep
#SBATCH --gpus=1080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=benchmark_results/slurm-single-sweep-%j.out

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

REPO_ROOT="${SLURM_SUBMIT_DIR}"
VENV_PATH="${NJORD_VENV:-${REPO_ROOT}/.venv}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmark_results/single-gpu}"

mkdir -p "${RESULT_DIR}"
source "${VENV_PATH}/bin/activate"
source jobs/start_gpu_telemetry.sh

nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit,clocks.max.sm \
    --format=csv

FAILURES=0
for REPEAT in 1 2 3; do
    RUN_DIR="${RESULT_DIR}/repeat-${REPEAT}"
    if ! python benchmarks/run_local_sweep.py \
        --group all \
        --output-dir "${RUN_DIR}" \
        --max-workers "${SLURM_CPUS_PER_TASK}" \
        --warmup 10 \
        --steps 50; then
        FAILURES="$((FAILURES + 1))"
    fi
done

if [[ "${FAILURES}" -gt 0 ]]; then
    echo "${FAILURES} repeat(s) contained failed configurations" >&2
    exit 1
fi
