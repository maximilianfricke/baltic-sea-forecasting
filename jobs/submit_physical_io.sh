#!/bin/bash
# Submit the physical-repeat I/O baseline from shared scratch or local TMPDIR.

set -euo pipefail

MODE="${1:-scratch}"
if [[ "${MODE}" != "scratch" && "${MODE}" != "tmpdir" ]]; then
    echo "Usage: $0 [scratch|tmpdir]" >&2
    exit 2
fi

mkdir -p benchmark_results
sbatch --export="ALL,PHYSICAL_IO_MODE=${MODE}" \
    jobs/benchmark_physical_io_1080ti.sbatch
