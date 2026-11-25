#!/bin/bash
#SBATCH --job-name=njord-pnemo-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=dphpc/pnemo/results/equivalence-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the PhysicsNeMo worktree}"
. /etc/profile.d/modules.sh
module add cuda/12.9
PYTHON="/scratch/users/$USER/venvs/njord-pascal/bin/python"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DASK_SCHEDULER=synchronous MLFLOW_DISABLE_AGENT_HINT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
K="${1:?Pass 2 or 4}"
[[ "$K" == 2 || "$K" == 4 ]]
"$PYTHON" - "$K" <<'PY'
import sys,torch
K=int(sys.argv[1])
assert torch.cuda.is_available() and torch.cuda.device_count()==K
assert all(torch.cuda.get_device_capability(i)==(6,1) for i in range(K)), 'Use GTX 1080 Ti'
PY
# This must pass before a benchmark command can be added to a future job.
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$K" \
    -m dphpc.pnemo.checks --backend nccl --hidden-dim 64 --processor-layers 2 \
    --batch-size 2 --output "dphpc/pnemo/results/equivalence-k${K}-job${SLURM_JOB_ID}.json"
