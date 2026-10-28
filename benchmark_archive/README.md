# Git-tracked benchmark archive

This directory contains compact, reviewable benchmark evidence. Each completed
run keeps its immutable JSON source of truth under `runs/`. Derived files are
rebuilt from those JSONs:

- `manifest.csv`: content hash and provenance for every archived result;
- `all-benchmarks.csv`: one comparison row per run;
- `benchmark-report.md`: human-readable comparison and measurement caveats;
- `plots/`: regenerated PNG figures and telemetry summary;
- `telemetry/`: optional small `nvidia-smi` CSVs keyed by Slurm job ID.
- `NJORD_Baseline_and_Parallelization_Report_2026-10-28.docx`: reviewed
  English project report containing the current interpretation and plan.

Raw Slurm logs, datasets, checkpoints, and PyTorch profiler traces are not
committed. They remain in `benchmark_results/` or scratch because they are
large, noisy, or reproducible from the archived evidence.

Rebuild the archive from results inside an allocated compute job:

```bash
bash jobs/update_benchmark_archive.sh benchmark_results
```

The rebuild replaces all derived plot/table files. Raw JSON runs, telemetry,
and the reviewed Word report are preserved.

The decision-suite and physical-I/O jobs update the archive automatically but
never run Git commands. After each job, review `git status` and manually add,
commit, and push `benchmark_archive/`. An archive failure never deletes the
original result under `benchmark_results/`.

If a job completed its measurements but failed while building this archive,
submit the lightweight recovery job instead of rerunning the measurements:

```bash
bash jobs/submit_archive_recovery.sh benchmark_results/decision-suite
```
