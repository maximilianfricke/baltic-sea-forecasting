#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3
PYTHON="${SLURM_SUBMIT_DIR}/.venv/bin/python"
GPU_COUNT="${NJORD_AB_GPUS:?Missing GPU count}"
RESULTS="benchmark_results/halo-transport/${NJORD_AB_PHASE}-${GPU_COUNT}gpu-${SLURM_JOB_ID}"
DATA_ROOT="/scratch/users/${USER}/baltic-sea-forecasting-data/physical_repeat_16"
PARTITIONS="/scratch/users/${USER}/baltic-sea-forecasting-data/partitioned/physical_repeat_16"
LAYOUT="geographic_rcb_ws${GPU_COUNT}"
STAGED="${TMPDIR:?TMPDIR is required}/njord-halo-ab-${SLURM_JOB_ID}"
[[ -x "$PYTHON" ]]
[[ -f "$DATA_ROOT/global_ocean_1.yaml" ]]
[[ -f "$DATA_ROOT/graphs/global_cluster_graph/mesh_features.pt" ]]
[[ -f "$PARTITIONS/$LAYOUT/manifest.json" ]]
mkdir -p "$RESULTS" "$STAGED/$LAYOUT"
START_NS="$(date +%s%N)"
rsync -a "$PARTITIONS/$LAYOUT/" "$STAGED/$LAYOUT/"
END_NS="$(date +%s%N)"
"$PYTHON" - "$RESULTS/staging.json" "$START_NS" "$END_NS" "$PARTITIONS/$LAYOUT" "$STAGED/$LAYOUT" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(elapsed_ms=(int(sys.argv[3])-int(sys.argv[2]))/1e6,source=sys.argv[4],target=sys.argv[5]),indent=2)+'\n')
PY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export DASK_SCHEDULER=synchronous TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MPLCONFIGDIR="${STAGED}/matplotlib-cache" MLFLOW_DISABLE_AGENT_HINT=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv > "$RESULTS/gpus.csv"
nvidia-smi topo -m > "$RESULTS/topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/slurm.txt"
"$PYTHON" -m pip freeze > "$RESULTS/packages.txt"
for ((REPEAT=0; REPEAT<NJORD_AB_REPEATS; REPEAT++)); do
    "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$GPU_COUNT" \
        -m benchmarks.compare_halo_transports run \
        --config-path "$DATA_ROOT/global_ocean_1.yaml" \
        --partitioned-data-root "$STAGED" \
        --output "$RESULTS/repeat-${REPEAT}.json" --repeat "$REPEAT" \
        --warmup-steps "$NJORD_AB_WARMUP" --steps "$NJORD_AB_STEPS" \
        --cached-batches 4 --num-workers 8 --prefetch-factor 2 --deterministic on
done
"$PYTHON" -m benchmarks.compare_halo_transports summarize --directory "$RESULTS" --repeats "$NJORD_AB_REPEATS"
echo "Final result: $RESULTS/comparison.json"
