#!/bin/bash
# Archive compact benchmark evidence, regenerate reports/plots, and optionally
# commit and push it. Raw logs and large profiler traces remain untracked.

set -euo pipefail

PUSH_RESULTS=0
INPUTS=()
for argument in "$@"; do
    if [[ "${argument}" == "--push" ]]; then
        PUSH_RESULTS=1
    else
        INPUTS+=("${argument}")
    fi
done

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

git add "${ARCHIVE_ROOT}"
if git diff --cached --quiet -- "${ARCHIVE_ROOT}"; then
    echo "Benchmark archive is already up to date"
    exit 0
fi

RUN_IDENTIFIERS="$(git diff --cached --name-only -- "${ARCHIVE_ROOT}/runs" \
    | sed -E 's#^.*/##; s#\.json$##' | paste -sd, -)"
COMMIT_MESSAGE="Archive benchmark results"
if [[ -n "${RUN_IDENTIFIERS}" ]]; then
    COMMIT_MESSAGE="Archive benchmark results: ${RUN_IDENTIFIERS}"
fi

GIT_NAME="$(git config user.name || true)"
GIT_EMAIL="$(git config user.email || true)"
GIT_NAME="${GIT_NAME:-NJORD benchmark automation}"
GIT_EMAIL="${GIT_EMAIL:-benchmark@localhost}"
git -c user.name="${GIT_NAME}" -c user.email="${GIT_EMAIL}" \
    commit -m "${COMMIT_MESSAGE}"

if [[ "${PUSH_RESULTS}" != "1" ]]; then
    echo "Archive committed locally; rerun with --push to publish it"
    exit 0
fi

BRANCH="${BENCHMARK_PUBLISH_BRANCH:-$(git branch --show-current)}"
if [[ -z "${BRANCH}" ]]; then
    echo "Cannot publish from a detached Git checkout" >&2
    exit 2
fi

for attempt in 1 2 3; do
    git fetch origin "${BRANCH}"
    git rebase "origin/${BRANCH}"
    if git push origin "HEAD:${BRANCH}"; then
        echo "Benchmark archive published to origin/${BRANCH}"
        exit 0
    fi
    echo "Push attempt ${attempt} failed; retrying after refresh" >&2
done

echo "Archive is committed locally, but could not be pushed" >&2
exit 1
