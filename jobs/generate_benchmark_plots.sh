#!/bin/bash
# Regenerate comparison and measurement-window telemetry plots.

set -euo pipefail

RESULT_ROOT="${1:-benchmark_results}"
PLOT_DIR="${RESULT_ROOT}/plots"

mkdir -p "${PLOT_DIR}"
python benchmarks/plot_results.py \
    "${RESULT_ROOT}" \
    --output-dir "${PLOT_DIR}"

if find -L "${RESULT_ROOT}" -type f -name "*gpu-*.csv" -print -quit \
    | grep -q .; then
    python benchmarks/plot_telemetry.py \
        "${RESULT_ROOT}" \
        --benchmark-results "${RESULT_ROOT}" \
        --output-dir "${PLOT_DIR}"
fi
