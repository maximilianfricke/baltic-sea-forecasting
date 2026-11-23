#!/bin/bash
# Control B: original model and checkpointing, existing one-GPU ocean input.
#SBATCH --job-name=njord-orig-shard
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=benchmark_results/model-parallel-real/slurm-original-ocean-shard-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export NJORD_REFERENCE_INPUT=ocean-shard
exec bash jobs/original_graph_efm_benchmark.sbatch
