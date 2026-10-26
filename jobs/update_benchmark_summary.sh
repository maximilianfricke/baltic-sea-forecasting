#!/bin/bash
# Rebuild one comparison table from all raw benchmark JSON files.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${PWD}}"
SUMMARY_ROOT="${1:-${REPO_ROOT}/benchmark_results}"
SUMMARY_OUTPUT="${SUMMARY_ROOT}/all-benchmarks.csv"
SUMMARY_TEMP="${SUMMARY_OUTPUT}.tmp-${SLURM_JOB_ID:-$$}"

mkdir -p "${SUMMARY_ROOT}"
python benchmarks/summarize_results.py "${SUMMARY_ROOT}" > "${SUMMARY_TEMP}"
mv "${SUMMARY_TEMP}" "${SUMMARY_OUTPUT}"
echo "Benchmark summary written to ${SUMMARY_OUTPUT}"
