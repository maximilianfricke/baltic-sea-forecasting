#!/bin/bash
# Rebuild the tracked archive after a completed run whose archive step failed.

#SBATCH --job-name=njord-archive
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=benchmark_results/slurm-archive-%j.out

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/jobs/common.sh"
njord_init_cluster_job

BENCHMARK_INPUT="${BENCHMARK_INPUT:-benchmark_results/decision-suite}"
bash jobs/update_benchmark_archive.sh "${BENCHMARK_INPUT}"
