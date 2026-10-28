#!/bin/bash
# Shared setup and validation for NJORD Slurm jobs.

njord_init_cluster_job() {
    . /etc/profile.d/modules.sh
    module purge
    module add cuda/12.6.3

    REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit the job from the repository root}"
    cd "${REPO_ROOT}"
    VENV_PATH="${NJORD_VENV:-${REPO_ROOT}/.venv}"
    if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
        echo "Python environment not found: ${VENV_PATH}" >&2
        return 2
    fi
    source "${VENV_PATH}/bin/activate"
}

njord_configure_training_job() {
    DEFAULT_DATA_ROOT="/scratch/users/${USER}/baltic-sea-forecasting-data"
    DEFAULT_DATA_ROOT="${DEFAULT_DATA_ROOT}/physical_repeat_16"
    ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-2}}"
    NUM_WORKERS="$((ALLOCATED_CPUS > 1 ? ALLOCATED_CPUS - 1 : 0))"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
}

njord_require_dataset() {
    local data_root="$1"
    if [[ ! -f "${data_root}/global_ocean_1.yaml" ]]; then
        echo "Physical dataset not found: ${data_root}" >&2
        return 2
    fi
}

njord_result_completed() {
    local output="$1"
    python - "${output}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
sys.exit(0 if result.get("status") == "completed" else 1)
PY
}

njord_run_graph_efm_benchmark() {
    local data_root="$1"
    local hidden_dim="$2"
    local num_workers="$3"
    shift 3
    python -m neural_lam.train_model \
        --config_path "${data_root}/global_ocean_1.yaml" \
        --model graph_efm \
        --graph global_cluster_graph \
        --hidden_dim "${hidden_dim}" \
        --hidden_dim_grid "${hidden_dim}" \
        --hidden_dim_mesh_nodes "${hidden_dim}" \
        --hidden_dim_edge "${hidden_dim}" \
        --processor_layers 2 \
        --batch_size 2 \
        --precision 32 \
        --epochs 100 \
        --ar_steps_train 4 \
        --ar_steps_eval 4 \
        --num_workers "${num_workers}" \
        --num_nodes 1 \
        --devices auto \
        --n_example_pred 0 \
        "$@"
}
