#!/bin/bash
# Rebuild one comparison table from all raw benchmark JSON files.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${PWD}}"
SUMMARY_ROOT="${1:-${REPO_ROOT}/benchmark_results}"
SUMMARY_OUTPUT="${SUMMARY_ROOT}/all-benchmarks.csv"
SUMMARY_TEMP="${SUMMARY_OUTPUT}.tmp-${SLURM_JOB_ID:-$$}"
REPORT_OUTPUT="${SUMMARY_ROOT}/benchmark-report.md"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_EXECUTABLE="${REPO_ROOT}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXECUTABLE="$(command -v python)"
else
    PYTHON_EXECUTABLE="$(command -v python3)"
fi

mkdir -p "${SUMMARY_ROOT}"
"${PYTHON_EXECUTABLE}" benchmarks/summarize_results.py \
    "${SUMMARY_ROOT}" > "${SUMMARY_TEMP}"
mv "${SUMMARY_TEMP}" "${SUMMARY_OUTPUT}"
echo "Benchmark summary written to ${SUMMARY_OUTPUT}"

"${PYTHON_EXECUTABLE}" benchmarks/analyze_results.py \
    "${SUMMARY_ROOT}" --output "${REPORT_OUTPUT}"
echo "Benchmark report written to ${REPORT_OUTPUT}"
