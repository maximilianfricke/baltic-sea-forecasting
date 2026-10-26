#!/bin/bash
# Source this file from a compute job after REPO_ROOT has been set.

if [[ -z "${REPO_ROOT:-}" ]]; then
    echo "REPO_ROOT must be set before staging benchmark data" >&2
    return 2
fi

if [[ "${STAGE_DATA:-1}" == "1" ]]; then
    JOB_DATA_DIR="${TMPDIR}/njord-data"
    mkdir -p "${JOB_DATA_DIR}"
    STAGING_START_NS="$(date +%s%N)"
    echo "Staging benchmark data to ${JOB_DATA_DIR}"
    cp -rL "${REPO_ROOT}/data/." "${JOB_DATA_DIR}/"
    STAGING_END_NS="$(date +%s%N)"
    export NJORD_DATA_LOCATION="node-local-tmpdir"
else
    JOB_DATA_DIR="${REPO_ROOT}/data"
    STAGING_START_NS="$(date +%s%N)"
    STAGING_END_NS="${STAGING_START_NS}"
    export NJORD_DATA_LOCATION="shared-scratch-symlink"
fi

export JOB_DATA_DIR
export NJORD_DATA_STAGING_SECONDS="$((
    STAGING_END_NS - STAGING_START_NS
))e-9"
export NJORD_DATA_STAGING_BYTES="$(du -sb "${JOB_DATA_DIR}/" | cut -f1)"

echo "Data source: ${NJORD_DATA_LOCATION}"
echo "Data size: $(du -sh "${JOB_DATA_DIR}/" | cut -f1)"
echo "Staging seconds: ${NJORD_DATA_STAGING_SECONDS}"
