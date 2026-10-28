#!/bin/bash
# Source this file from a compute job after REPO_ROOT has been set.

if [[ -z "${REPO_ROOT:-}" ]]; then
    echo "REPO_ROOT must be set before staging benchmark data" >&2
    return 2
fi

# Directory holding the neural-lam config, datastores and graphs/. Defaults
# to the repository's data/ (the public 1 degree sample).
DATA_SOURCE_DIR="${DATA_SOURCE_DIR:-${REPO_ROOT}/data}"

if [[ "${STAGE_DATA:-1}" == "1" ]]; then
    JOB_DATA_DIR="${TMPDIR}/njord-data"
    mkdir -p "${JOB_DATA_DIR}"
    STAGING_START_NS="$(date +%s%N)"
    echo "Staging benchmark data from ${DATA_SOURCE_DIR} to ${JOB_DATA_DIR}"
    # Preserve only timestamps (not ownership/xattrs) so MDP's YAML-vs-Zarr
    # freshness check sees the same ordering as in the source dataset.
    cp -rL --preserve=timestamps \
        "${DATA_SOURCE_DIR}/." "${JOB_DATA_DIR}/"
    STAGING_END_NS="$(date +%s%N)"
    export NJORD_DATA_LOCATION="node-local-tmpdir"
else
    JOB_DATA_DIR="${DATA_SOURCE_DIR}"
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
