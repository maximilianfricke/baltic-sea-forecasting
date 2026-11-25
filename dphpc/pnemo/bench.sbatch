#!/bin/bash
#SBATCH --job-name=njord-pnemo-bench
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:08:00
#SBATCH --output=dphpc/pnemo/results/bench-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the PhysicsNeMo worktree}"
. /etc/profile.d/modules.sh
module add cuda/12.9
PYTHON="/scratch/users/$USER/venvs/njord-pascal/bin/python"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 DASK_SCHEDULER=synchronous MLFLOW_DISABLE_AGENT_HINT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
K="${1:?Pass 1, 2 or 4}"
[[ "$K" == 1 || "$K" == 2 || "$K" == 4 ]]
STAGE="${2:-full}"
[[ "$STAGE" == full || "$STAGE" == balance ]]
# Additional arguments (including --reference-dir) go directly to the driver.
shift "$(( $# >= 2 ? 2 : 1 ))"
RESULTS="$PWD/dphpc/pnemo/results/bench-${STAGE}-k${K}-job${SLURM_JOB_ID}"
mkdir -p "$RESULTS"
exec >"$RESULTS/job.log" 2>&1
"$PYTHON" - "$K" <<'PY'
import sys, torch
K = int(sys.argv[1])
assert torch.cuda.is_available() and torch.cuda.device_count() == K
assert all(torch.cuda.get_device_capability(i) == (6, 1) for i in range(K))
PY
NJORD_BENCH_CONFIG="$("$PYTHON" -m dphpc.pnemo.stage_data --output "$RESULTS")"
"$PYTHON" -m dphpc.pnemo.bench --K "$K" --stage "$STAGE" --output "$RESULTS" --config-path "$NJORD_BENCH_CONFIG" "$@"
