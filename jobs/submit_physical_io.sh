#!/bin/bash
# Submit a physical-repeat run with explicit I/O and instrumentation modes.

set -euo pipefail

MODE="${1:-scratch}"
TIMING_MODE="${2:-aggregate}"
TELEMETRY="${3:-off}"
if [[ "${MODE}" != "scratch" && "${MODE}" != "tmpdir" ]]; then
    echo "Usage: $0 [scratch|tmpdir] [aggregate|phase] [off|on]" >&2
    exit 2
fi
if [[ "${TIMING_MODE}" != "aggregate" && "${TIMING_MODE}" != "phase" ]]; then
    echo "Usage: $0 [scratch|tmpdir] [aggregate|phase] [off|on]" >&2
    exit 2
fi
if [[ "${TELEMETRY}" != "off" && "${TELEMETRY}" != "on" ]]; then
    echo "Usage: $0 [scratch|tmpdir] [aggregate|phase] [off|on]" >&2
    exit 2
fi

mkdir -p benchmark_results
EXPORTS="ALL,PHYSICAL_IO_MODE=${MODE}"
EXPORTS="${EXPORTS},PHYSICAL_TIMING_MODE=${TIMING_MODE}"
EXPORTS="${EXPORTS},PHYSICAL_TELEMETRY=${TELEMETRY}"
sbatch --export="${EXPORTS}" \
    jobs/benchmark_physical_io_1080ti.sbatch
