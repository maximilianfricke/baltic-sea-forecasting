#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --time=00:10:00
#SBATCH --gpus=5060ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=njord-stages
#SBATCH --output=stages-%j.log

set -euo pipefail
. /etc/profile.d/modules.sh
module add cuda/13.0
source /scratch/users/"$USER"/venvs/njord/bin/activate
cd /home/nrottstegge/baltic-sea-forecasting
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"

out="output/stages-${SLURM_JOB_ID:-local}"
mkdir -p "$out"
command=(python -m neural_lam.train_model
    --config_path sample_data/global_ocean_1.yaml
    --model graph_efm --graph global_hierarchical_graph
    --hidden_dim 128 --processor_layers 3 --batch_size 1
    --optimizer adamw --kl_beta 0.1 --crps_weight 0
    --ar_steps_train 1 --precision 32-true --num_workers 2
    --ar_steps_eval 1 --val_steps_to_log 1
    --profile_warmup 3 --profile_steps 10 --profile_stages
    --profile_output "$out/results.csv" "$@")

# Optional Nsight capture. CUDA modules may select an older nsys on PATH.
if [ "${NSYS:-0}" = 1 ]; then
    profiler="${NSYS_BIN:-/usr/local/bin/nsys}"
    "$profiler" --version
    command=("$profiler" profile --trace=cuda,nvtx,osrt --sample=none
        -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0
        --capture-range=nvtx --nvtx-capture=profile --capture-range-end=stop
        -o "$out/trace" "${command[@]}")
fi
"${command[@]}" 2>&1 | tee "$out/train.log"
python scripts/plot_stage_profile.py "$out/results.csv"
echo "Results and figures: $out"
