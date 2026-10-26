#!/bin/bash
# Read-only probe: report the resources Slurm actually grants to this job.
# Override the default GPU count at submission time; do not request CPUs here.

#SBATCH --job-name=njord-resource-probe
#SBATCH --account=dphpc
#SBATCH --gpus=1080ti:1
#SBATCH --time=00:02:00
#SBATCH --output=benchmark_results/resource-probe-%j.out

set -euo pipefail

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "job_gpus=${SLURM_JOB_GPUS:-}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "cpus_on_node=${SLURM_CPUS_ON_NODE:-}"
echo "mem_per_node_mb=${SLURM_MEM_PER_NODE:-}"
echo "tmpdir=${TMPDIR:-}"
echo "cpu_affinity=$(taskset -pc $$ 2>&1)"
du -sh "${TMPDIR}"
nvidia-smi --query-gpu=index,name,uuid,memory.total,power.limit \
    --format=csv
