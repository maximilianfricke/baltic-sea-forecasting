# Package E reproduction

Branch `nils-pnemo`, based on `global-model` at `76b30d7`. Original model files
are unchanged. Raw measurements and generated figures belong outside git in
`package-e-results` (extract `package-e-results.tar.gz` there).

Environment: Python 3.12, torch 2.14.1+cu126, CUDA 12.6, NCCL 2.29.3,
PyG 2.3.1, PyMetis 2025.2.2, Mt-KaHyPar 1.7.1; driver 580.178.04.
The cluster environment is `/scratch/users/nrottstegge/venvs/njord-pascal`.
Install the base project dependencies, then `pip install -r dphpc/pnemo/requirements.txt`.
PhysicsNeMo 2.3.0 is vendored: only imports changed; see its `SOURCE.json`
and `LICENSE.txt`. Installing the full PhysicsNeMo package is unnecessary.

## Order

From a fresh checkout, with the original 1-degree dataset and graph at
`/scratch/users/nrottstegge/njord-data/sample_data`:

```bash
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DASK_SCHEDULER=synchronous MLFLOW_DISABLE_AGENT_HINT=1
PY=/scratch/users/$USER/venvs/njord-pascal/bin/python
# CPU: regenerate maps, compare every node-array hash, audit, write rank data.
"$PY" -m dphpc.pnemo.reweight --write-ranks
mkdir -p dphpc/pnemo/results
# GPU: submit yourself; wait for both equivalence jobs before benchmarks.
sbatch --account=dphpc_4gpu --gpus=1080ti:2 dphpc/pnemo/equivalence.sbatch 2
sbatch --account=dphpc_4gpu --gpus=1080ti:4 dphpc/pnemo/equivalence.sbatch 4
```

Then submit `dphpc/pnemo/bench.sbatch 1 full`, `2 full`, `4 full` in that
order (accounts `dphpc`, `dphpc_4gpu`, `dphpc_4gpu`; respectively 1, 2, 4
`1080ti` GPUs; limits 30, 60, 60 minutes). Pass the completed K=1 directory
as `--reference-dir` to K=2/4 and K=2 as `--k2-reference-dir` to K=4.
Submit `bench.sbatch 2 balance` and `4 balance`, each with
`--partitions-dir partitions/1deg-unweighted --reference-dir <K1-full-directory>`.
Use `--nodelist=studgpu-node02` for the original placement. Outputs are
`dphpc/pnemo/results/bench-<stage>-k<K>-job<JOBID>`.
The driver requires committed `dphpc/`; full-run loss references must come
from the same commit. Commit before submission and leave files untouched.

CPU analysis of the archived final measurements:

```bash
E=/absolute/path/to/package-e-results
"$PY" -m dphpc.pnemo.report "$E/measurements/5490" "$E/measurements/5491" \
  "$E/measurements/5492" --balance-results "$E/measurements/5497" \
  "$E/measurements/5498" --accounting "$E/report/job-accounting.txt" \
  --balance-accounting "$E/report/job-accounting-balance.txt" \
  --output dphpc/pnemo/results/report-full
"$PY" -m dphpc.pnemo.plot_partitions
```

The report validates the archived source commits (`1f09517`, `4a20e0d`).
Jobs 5437/5438: K=2/4 equivalence (outputs, forecast, loss, gradients, SGD,
resolved-gradient AdamW; inferred source commit `fd180da`). Jobs 5490–5492:
Tables 1–3 and Figures 1–3. Jobs 5497/5498: Table 4 and four additional
Figure 2 points. Ownership figures use the saved rank assignments.
This local worktree reuses existing rank data through an ignored `partitions`
symlink; fresh checkouts regenerate it. Owner maps are small committed NPZs; the generator verifies them before
writing data. Weighted/unweighted Mt-KaHyPar runs use separate processes,
as in the original preparation. Unweighted grid-imbalance warnings are expected.

## Limits

1-degree GraphEFM training, one node, GTX 1080 Ti, FP32, batch 2, AR4,
2 processor layers. H64 all four partitioners; H128/H256 only Mt-KaHyPar.
Four CPU-pinned batches cycle through 5 warmup + 20 measured steps.
Cached timing excludes loading/H2D and begins after sync/barrier; live timing
includes them. Rank maxima, median, 4-step-block/launch bootstrap (4,000,
seed 42), ratio-of-medians speedup. Single-launch CI: within_launch_only.
K=1 checkpoint early stop is on; K>1 fully recomputes. Retain outliers.
One separate profiler step supplies the compute bound; NCCL includes waiting.
One loader worker/rank, 2 CPUs/GPU. No overlap optimization, no prediction
for 0.25-degree graphs, other GPUs, multi-node runs or full training.
