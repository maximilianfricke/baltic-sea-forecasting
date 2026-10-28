#!/bin/bash
# Rebuild the tracked archive after a completed run whose archive step failed.

#SBATCH --job-name=njord-archive
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=benchmark_results/slurm-archive-%j.out

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
VENV_PATH="${NJORD_VENV:-${REPO_ROOT}/.venv}"
source "${VENV_PATH}/bin/activate"

BENCHMARK_INPUT="${BENCHMARK_INPUT:-benchmark_results/decision-suite}"
bash jobs/update_benchmark_archive.sh "${BENCHMARK_INPUT}"
