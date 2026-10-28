# NJORD benchmark workflow

The benchmark code is intentionally separate from the model implementation.
Only two official GTX-1080-Ti jobs are kept:

1. `decision_suite_1080ti.sbatch` compares gradient checkpointing and probes
   hidden-dimension capacity.
2. `benchmark_physical_io_1080ti.sbatch` measures a long baseline while
   reading independent physical Zarr chunks from shared scratch or `$TMPDIR`.

Both jobs use the same GraphEFM workload: hidden dimension 64, two processor
layers, batch size 2, four autoregressive steps, and FP32. The decision suite
also probes hidden dimensions 128, 192, and 256. FP32 is used because the GTX
1080 Ti has no native BF16 support.

## Physical I/O dataset

The original January dataset is too short for a sustained AR=4 measurement.
Build a self-contained physical-repeat proxy locally:

```bash
python -m benchmarks.build_physical_repeat_dataset \
  --source-root data \
  --destination-root data/physical_repeat_16 \
  --repeats 16
```

This writes independent Zarr chunks with monotonically increasing timestamps.
It expands the storage workload and epoch length but does not create new
scientific observations. The 16-repeat dataset contains 496 time steps, of
which 479 belong to the training split, and occupies roughly 2.2 GB.

Transfer it to personal scratch, never to Git or the cluster home directory:

```bash
rsync -av --progress --partial \
  data/physical_repeat_16/ \
  mfricke@student-cluster1.inf.ethz.ch:\
/scratch/users/mfricke/baltic-sea-forecasting-data/physical_repeat_16/
```

## Cluster runs

Use the CUDA-12.6 environment and run jobs sequentially. Commit and push the
generated archive manually after every job before submitting the next one.

```bash
source /scratch/users/mfricke/venvs/njord-py312-cu126/bin/activate
bash jobs/submit_decision_suite.sh
# after reviewing and committing that result: clean baseline repetition 1
bash jobs/submit_physical_io.sh scratch aggregate off
```

Run the clean aggregate baseline three times, always waiting, reviewing, and
committing before the next submission. Then run one instrumented diagnostic:

```bash
bash jobs/submit_physical_io.sh scratch phase on
```

The optional node-local comparison is only needed if the scratch results show
a material input bottleneck:

```bash
bash jobs/submit_physical_io.sh tmpdir aggregate off
```

`aggregate off` avoids per-batch synchronization and background telemetry, so
it is the headline throughput measurement. `phase on` synchronizes each batch
and samples the GPU once per second; it is used for diagnosis, not for the
headline throughput number.

Each job requests the `dphpc` account, one GTX 1080 Ti, 24 GiB host memory,
and CUDA 12.6.3. The scripts do not request a CPU count because the course
scheduler determines it. The job reads the actual allocation and assigns at
most all but one CPU to DataLoader workers.

## Measurements

The JSON result is the source of truth. It contains:

- every command-line argument, Git revision, Slurm job ID, software version,
  GPU model, CUDA runtime, driver, power limit, and clock limit;
- parameter, graph-node, and graph-edge counts;
- warm-up and measurement protocol;
- forward, backward, optimizer, data-wait, and end-to-end times for every
  measured batch, plus mean, median, standard deviation, p95, p99, and max;
- allocated/reserved GPU memory, peak process RSS, throughput, staging time,
  data location, and dataset length;
- optional one-second GPU utilization, memory, power, temperature, and clock
  telemetry.

Phase timings synchronize every batch and are intended for bottleneck
analysis. Their sum must not be presented as an instrumentation-free headline
throughput number. Capacity failures are retained as evidence.

## Result archiving and manual publication

At the end of every job, `update_benchmark_archive.sh`:

1. copies immutable raw JSON and compact telemetry to `benchmark_archive/`;
2. rebuilds `all-benchmarks.csv` and `benchmark-report.md`;
3. regenerates the PNG plots.

It deliberately does not stage, commit, pull, or push anything. After the job
has completed, inspect and publish the generated files manually:

```bash
git status --short benchmark_archive
git add benchmark_archive
git diff --cached --stat
git commit -m "Archive benchmark job JOB_ID"
git push origin max
```

Replace `JOB_ID` with the Slurm job ID. Then run `git pull --ff-only origin max`
on the local computer to retrieve the result and plots.

Datasets, Slurm logs, checkpoints, and large traces remain outside Git. If the
archive update fails, the original result remains under `benchmark_results/`.
Regenerate it in a short Slurm allocation rather than running the plotting and
analysis programs directly on the login node:

```bash
bash jobs/submit_archive_recovery.sh benchmark_results/decision-suite
```

This recovery job only reads existing results and rebuilds the tracked archive;
it does not repeat training and does not execute Git commands.

## Interpretation boundary

- The physical-repeat dataset is a controlled I/O proxy, not additional
  scientific data.
- The decision suite identifies checkpointing cost and the capacity boundary;
  the long physical run provides the stable end-to-end baseline.
- Multi-GPU speedup, halo-exchange cost, and communication overlap are not yet
  measured. New distributed jobs should be added only after their resource
  geometry and correctness checks are implemented and reviewed.
- Official comparisons must use the same GPU type, data location, model
  settings, precision, software environment, and protocol.
