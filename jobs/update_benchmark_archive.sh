#!/bin/bash
# Copy benchmark evidence into the tracked archive and rebuild derived files.
# This script deliberately performs no git add, commit, pull, or push.

set -euo pipefail

INPUTS=("$@")
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
ARCHIVE_ROOT="${BENCHMARK_ARCHIVE_ROOT:-benchmark_archive}"
if [[ "${#INPUTS[@]}" -eq 0 ]]; then
    INPUTS=(benchmark_results)
fi

if [[ -n "${NJORD_VENV:-}" && -x "${NJORD_VENV}/bin/python" ]]; then
    PYTHON_EXECUTABLE="${NJORD_VENV}/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_EXECUTABLE="${REPO_ROOT}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXECUTABLE="$(command -v python)"
else
    PYTHON_EXECUTABLE="$(command -v python3)"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/njord-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

mkdir -p "${ARCHIVE_ROOT}/runs" "${ARCHIVE_ROOT}/plots"
"${PYTHON_EXECUTABLE}" benchmarks/archive_results.py \
    "${INPUTS[@]}" --output-root "${ARCHIVE_ROOT}"
"${PYTHON_EXECUTABLE}" benchmarks/summarize_results.py \
    "${ARCHIVE_ROOT}/runs" > "${ARCHIVE_ROOT}/all-benchmarks.csv"
"${PYTHON_EXECUTABLE}" benchmarks/analyze_results.py \
    "${ARCHIVE_ROOT}/runs" \
    --output "${ARCHIVE_ROOT}/benchmark-report.md"
"${PYTHON_EXECUTABLE}" benchmarks/plot_results.py \
    "${ARCHIVE_ROOT}/runs" --output-dir "${ARCHIVE_ROOT}/plots"

if find "${ARCHIVE_ROOT}/telemetry" -type f -name '*.csv' \
    -print -quit 2>/dev/null | grep -q .; then
    "${PYTHON_EXECUTABLE}" benchmarks/plot_telemetry.py \
        "${ARCHIVE_ROOT}/telemetry" \
        --benchmark-results "${ARCHIVE_ROOT}/runs" \
        --output-dir "${ARCHIVE_ROOT}/plots"
fi

echo "Benchmark archive updated under ${ARCHIVE_ROOT}/"
echo "Review and publish it manually with git."
