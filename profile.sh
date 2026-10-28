#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --time=00:30:00
#SBATCH --gpus=5060ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=njord-profile
#SBATCH --output=profile-%j.log

set -euo pipefail
. /etc/profile.d/modules.sh
module add cuda/13.0
source /scratch/users/"$USER"/venvs/njord/bin/activate
cd /home/nrottstegge/baltic-sea-forecasting
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS=1

out="output/profile-${SLURM_JOB_ID:-local}"
mkdir -p "$out"
nvidia-smi

# A fresh process per width releases all CUDA allocations after each run.
# All hidden dimensions inherit hidden_dim unless overridden in "$@".
read -r -a widths <<< "${WIDTHS:-16 32 64 128 256 512 1024}"
for width in "${widths[@]}"; do
    status=0
    python -m neural_lam.train_model \
        --config_path sample_data/global_ocean_1.yaml \
        --model graph_efm --graph global_hierarchical_graph \
        --hidden_dim "$width" --processor_layers 3 \
        --batch_size 1 --ar_steps_train 1 \
        --kl_beta 0.1 --crps_weight 0 --optimizer adamw \
        --precision 32-true --num_workers 2 \
        --ar_steps_eval 1 --val_steps_to_log 1 \
        --profile_warmup 3 --profile_steps 10 \
        --profile_output "$out/results.csv" \
        "$@" > "$out/width-${width}.log" 2>&1 || status=$?
    if [ "$status" -eq 86 ]; then
        echo "width=$width: CUDA OOM; see $out/results.csv"
        break
    elif [ "$status" -ne 0 ]; then
        tail -n 40 "$out/width-${width}.log"
        exit "$status"
    fi
    tail -n 1 "$out/width-${width}.log"
done
echo "Results: $out/results.csv"
