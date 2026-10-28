#!/bin/bash
# Submit the small checkpointing/capacity suite used for project planning.

set -euo pipefail

mkdir -p benchmark_results
sbatch jobs/decision_suite_1080ti.sbatch
