# Git-tracked benchmark archive

This directory contains compact, reviewable benchmark evidence. Each completed
run keeps its immutable JSON source of truth under `runs/`. Derived files are
rebuilt from those JSONs:

- `manifest.csv`: content hash and provenance for every archived result;
- `all-benchmarks.csv`: one comparison row per run;
- `benchmark-report.md`: human-readable comparison and measurement caveats;
- `plots/`: regenerated PNG/PDF figures and the compact benchmark table;
- `telemetry/`: optional small `nvidia-smi` CSVs keyed by Slurm job ID.

Raw Slurm logs, datasets, checkpoints, and PyTorch profiler traces are not
committed. They remain in `benchmark_results/` or scratch because they are
large, noisy, or reproducible from the archived evidence.

Archive every result currently present on a machine and publish it:

```bash
bash jobs/publish_benchmark_results.sh --push benchmark_results
```

The standard GTX-1080-Ti batch job performs this automatically by default.
Set `PUBLISH_RESULTS=0` for an intentionally private or experimental run.
