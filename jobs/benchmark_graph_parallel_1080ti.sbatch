#!/bin/bash
# Use submit_graph_parallel.sh to request the matching GPU count.

#SBATCH --job-name=njord-graph
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_results/graph-parallel/slurm-%j.out

set -euo pipefail

RUN_KIND="$1"
GPU_COUNT="$2"
MODE="$3"
HIDDEN_DIM="$4"
REPETITION="$5"
export NJORD_JOB_STARTED_EPOCH_SECONDS="$(date +%s.%N)"
source "${SLURM_SUBMIT_DIR}/jobs/common.sh"
njord_init_cluster_job
njord_configure_training_job
RESULT_DIR="${REPO_ROOT}/benchmark_results/graph-parallel"
export ENABLE_GPU_TELEMETRY=0
mkdir -p "${RESULT_DIR}"

LABEL="${RUN_KIND}-g${GPU_COUNT}-${MODE}-h${HIDDEN_DIM}-r${REPETITION}"
STEM="${RESULT_DIR}/${LABEL}-${SLURM_JOB_ID}"
{
    scontrol show job "${SLURM_JOB_ID}" -o
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit \
        --format=csv
    nvidia-smi topo -m
} > "${STEM}.topology.log" 2>&1

check_cuda() {
    python -c '
import sys
import torch

count = int(sys.argv[1])
assert torch.cuda.device_count() == count
assert count == 1 or torch.distributed.is_nccl_available()
for index in range(count):
    assert "1080 Ti" in torch.cuda.get_device_name(index)
    assert torch.ones(1, device=f"cuda:{index}").item() == 1
' "${GPU_COUNT}"
}

if [[ "${RUN_KIND}" == "validate" ]]; then
    check_cuda
    python -m torch.distributed.run --standalone --nproc-per-node="${GPU_COUNT}" \
        -m benchmarks.graph_parallel_probe --output "${STEM}.transport.json"
    python -m pytest --noconftest -q -rA --tb=short \
        "tests/test_graph_parallel.py::test_cuda_distributed_equivalence[${GPU_COUNT}]" \
        2>&1 | tee "${STEM}.log"
    exit
fi

DATA_ROOT="${GRAPH_PARALLEL_DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
njord_require_dataset "${DATA_ROOT}"
test -d "${DATA_ROOT}/graphs/global_cluster_graph"
export NJORD_DATA_LOCATION="physical-repeat-shared-scratch"
export NJORD_DATA_STAGING_SECONDS=0
export NJORD_DATA_STAGING_BYTES="$(du -sb "${DATA_ROOT}" | cut -f1)"

case "${RUN_KIND}" in
    smoke) WARMUP=1; STEPS=2 ;;
    timed) WARMUP=20; STEPS=200 ;;
    capacity) WARMUP=1; STEPS=4 ;;
esac
OVERLAP_ARGS=()
if [[ "${MODE}" == "overlap" ]]; then
    OVERLAP_ARGS=(--graph_parallel_overlap)
fi
OUTPUT="${STEM}.json"
RUN_LOG="${STEM}.log"
set +e
{
    check_cuda && njord_run_graph_efm_benchmark --torchrun "${GPU_COUNT}" \
        "${DATA_ROOT}" "${HIDDEN_DIM}" 0 \
        --seed 42 \
        --graph_parallel_size "${GPU_COUNT}" "${OVERLAP_ARGS[@]}" \
        --benchmark_scaling_mode strong \
        --benchmark_warmup_steps "${WARMUP}" \
        --benchmark_steps "${STEPS}" \
        --benchmark_timing_mode aggregate \
        --benchmark_label "${LABEL}" \
        --benchmark_output "${OUTPUT}"
} 2>&1 | tee "${RUN_LOG}"
RUN_EXIT_CODE="${PIPESTATUS[0]}"
set -e

if [[ ! -f "${OUTPUT}" ]]; then
    if [[ "${RUN_EXIT_CODE}" == 0 ]]; then
        RUN_EXIT_CODE=1
    fi
    python -m benchmarks.write_failure_result \
        --output "${OUTPUT}" --label "${LABEL}" --log "${RUN_LOG}" \
        --hidden-dim "${HIDDEN_DIM}" --checkpointing on \
        --world-size "${GPU_COUNT}" --scaling-mode graph --exit-code "${RUN_EXIT_CODE}"
fi
if [[ "${RUN_EXIT_CODE}" != 0 ]]; then
    exit "${RUN_EXIT_CODE}"
fi
python - "${OUTPUT}" "${STEPS}" "${GPU_COUNT}" <<'PY'
import json
import sys

path, steps, ranks = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    result = json.load(handle)
if (
    result["status"] != "completed"
    or result["protocol"]["completed_measurement_steps"] != int(steps)
    or result["distributed"]["world_size"] != int(ranks)
    or result["distributed"]["global_batch_size"] != 2
):
    raise SystemExit(f"Incomplete or mismatched benchmark result: {path}")
PY
