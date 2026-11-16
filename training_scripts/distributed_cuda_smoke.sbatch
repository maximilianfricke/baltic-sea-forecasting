#!/bin/bash
# Two-GPU NCCL correctness smoke for receiver-owned halo message passing.

#SBATCH --job-name=njord-mp-smoke
#SBATCH --account=dphpc_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-njord-mp-smoke-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p benchmark_results

. /etc/profile.d/modules.sh
module purge
module add cuda/12.6.3

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing ${SLURM_SUBMIT_DIR}/.venv/bin/python" >&2
    exit 2
fi

export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NJORD_GIT_COMMIT="$(git rev-parse HEAD)"
if [[ -z "$(git status --porcelain)" ]]; then
    export NJORD_GIT_DIRTY=false
else
    export NJORD_GIT_DIRTY=true
fi

MESSAGE_RESULT="benchmark_results/distributed-message-smoke-${SLURM_JOB_ID}.json"
CORE_RESULT="benchmark_results/distributed-core-smoke-${SLURM_JOB_ID}.json"
export NJORD_CORE_SMOKE_RESULT="${CORE_RESULT}"

nvidia-smi --query-gpu=index,name,uuid,memory.total \
    --format=csv,noheader
.venv/bin/python -c \
    'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda}")'

.venv/bin/python -m torch.distributed.run \
    --standalone \
    --nproc-per-node=2 \
    tests/distributed_cuda_smoke.py \
    --output "${MESSAGE_RESULT}"

.venv/bin/python -m torch.distributed.run \
    --standalone \
    --nproc-per-node=2 \
    tests/test_distributed_graph_efm.py

echo "Result files:"
echo "  ${MESSAGE_RESULT}"
echo "  ${CORE_RESULT}"
