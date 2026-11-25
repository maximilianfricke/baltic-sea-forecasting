#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3
PYTHON="$PWD/.venv/bin/python"
GPUS="${MP_GPUS:?}"
CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:?Missing CPU allocation}}"
DATA="/scratch/users/${USER}/baltic-sea-forecasting-data/physical_repeat_16"
SHARDS="/scratch/users/${USER}/baltic-sea-forecasting-data/partitioned/physical_repeat_16"
STAGED="${TMPDIR:?TMPDIR is required}/message-passing-${SLURM_JOB_ID}"
RESULTS="$PWD/benchmark_results/message-passing/${MP_PHASE}-${GPUS}gpu-${SLURM_JOB_ID}"
[[ -x "$PYTHON" && -f "$DATA/global_ocean_1.yaml" && -f "$SHARDS/geographic_rcb_ws${GPUS}/manifest.json" ]]
mkdir -p "$STAGED/original" "$STAGED/shards/geographic_rcb_ws${GPUS}" "$RESULTS"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export DASK_SCHEDULER=synchronous TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MLFLOW_DISABLE_AGENT_HINT=1 MPLCONFIGDIR="$STAGED/matplotlib-cache"
START_NS="$(date +%s%N)"
# The original store supplies static/global statistics and the numerical gate.
# Each rank uses only its shard for cached/live measured input.
rsync -a "$DATA/" "$STAGED/original/"
rsync -a "$SHARDS/geographic_rcb_ws${GPUS}/" "$STAGED/shards/geographic_rcb_ws${GPUS}/"
END_NS="$(date +%s%N)"
"$PYTHON" - "$RESULTS/staging.json" "$START_NS" "$END_NS" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"elapsed_ms":(int(sys.argv[3])-int(sys.argv[2]))/1e6})+'\n')
PY
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv > "$RESULTS/gpus.csv"
nvidia-smi topo -m > "$RESULTS/topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/allocation.txt"
lscpu --extended=CPU,NODE,SOCKET,CORE > "$RESULTS/cpu-topology.txt"
"$PYTHON" -m pip freeze > "$RESULTS/packages.txt"
for ((REPEAT=0; REPEAT<MP_REPEATS; REPEAT++)); do
    srun --exact --ntasks=1 --cpus-per-task="$CPUS" --gpus="1080ti:$GPUS" \
        --cpu-bind=verbose,threads \
        "$PYTHON" -m message_passing.launch launch --ranks "$GPUS" --expected-cpus "$CPUS" \
        --cpu-policy "$MP_CPU_POLICY" -- \
        --config-path "$STAGED/original/global_ocean_1.yaml" \
        --shard-root "$STAGED/shards" --backend both --input-mode cached \
        --workers 0 --prefetch 2 --repeat "$REPEAT" \
        --warmup "$MP_WARMUP" --steps "$MP_STEPS" \
        --output "$RESULTS/repeat-${REPEAT}.json"
done
"$PYTHON" -m message_passing.report --directory "$RESULTS" --repeats "$MP_REPEATS"
echo "Final result: $RESULTS/comparison.json"
