#!/bin/bash
# Submit archive/report/plot regeneration to a compute node.

set -euo pipefail

INPUT="${1:-benchmark_results/decision-suite}"
mkdir -p benchmark_results
sbatch --export="ALL,BENCHMARK_INPUT=${INPUT}" \
    jobs/archive_existing_results.sbatch
