#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --time=01:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=njord
#SBATCH --output=/scratch/users/%u/logs/%j_%x.out
#SBATCH --error=/scratch/users/%u/logs/%j_%x.err

set -euo pipefail

# Environment
. /etc/profile.d/modules.sh
module add cuda/13.0

source /scratch/users/$USER/venvs/njord/bin/activate

cd ~/baltic-sea-forecasting

export WANDB_MODE=online
export PYTHONUNBUFFERED=1

# Runtime diagnostics
echo "========================================"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Job name:        ${SLURM_JOB_NAME}"
echo "Start time:      $(date -Is)"
echo "Hostname:        $(hostname)"
echo "Working dir:     $(pwd)"
echo "Submit dir:      ${SLURM_SUBMIT_DIR}"
echo "Allocated nodes: ${SLURM_JOB_NUM_NODES:-unknown}"
echo "Node list:       ${SLURM_JOB_NODELIST:-unknown}"
echo "CPUs per task:   ${SLURM_CPUS_PER_TASK:-unknown}"
echo "GPU allocation:  ${SLURM_JOB_GPUS:-unknown}"
echo "Arguments:       $*"
echo "========================================"

echo "Loaded modules:"
module list

echo "Python environment:"
which python
python --version

echo "GPU information:"
nvidia-smi

echo "Training configuration:"
ls -lh sample_data/global_ocean_1.yaml

# Capture exit status and completion time
trap 'status=$?; echo "End time: $(date -Is)"; echo "Exit status: $status"' EXIT

# Training
python -m neural_lam.train_model \
  --config_path sample_data/global_ocean_1.yaml \
  --num_workers 2 \
  --batch_size 1 \
  --num_nodes 1 \
  --num_sanity_val_steps 0 \
  --ar_steps_eval 2 \
  --val_steps_to_log 1 2 \
  --lr 0.001 \
  --epochs 1 \
  "$@"