#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3
PYTHON="$PWD/.venv/bin/python"
DATA="/scratch/users/${USER}/baltic-sea-forecasting-data/physical_repeat_16"
RESULTS="$PWD/benchmark_results/message-passing/${MP_PHASE}-${MP_GPUS}gpu-${SLURM_JOB_ID}"
mkdir -p "$RESULTS"
[[ -x "$PYTHON" && -f "$DATA/global_ocean_1.yaml" ]]
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv > "$RESULTS/gpus.csv"
nvidia-smi topo -m > "$RESULTS/topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/allocation.txt"
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$MP_GPUS" \
    -m message_passing.run --config-path "$DATA/global_ocean_1.yaml" \
    --workers 8 --warmup "$MP_WARMUP" --steps "$MP_STEPS" \
    --output "$RESULTS/result.json"
