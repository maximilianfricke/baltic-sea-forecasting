"""Stage only the fixed 1-degree loader inputs and graph to job-local storage."""

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .partitions import CONFIG, GRAPH


def stage(output, tmpdir, config=CONFIG):
    start = time.perf_counter()
    source = config.parent
    paths = [
        config.name,
        "global_ocean_1_mdp_config.yaml",
        "global_era5_1_mdp_config.yaml",
        "global_ocean_1_mdp_config.zarr",
        "global_era5_1_mdp_config.zarr",
        f"graphs/{GRAPH}",
    ]
    source_bytes = 0
    for name in paths:
        path = source / name
        if not path.exists():
            raise FileNotFoundError(path)
        files = path.rglob("*") if path.is_dir() else [path]
        source_bytes += sum(p.stat().st_size + 4096 for p in files)
    needed = int(source_bytes * 1.1) + 64 * 2**20
    selected = config
    report = dict(
        status="skipped",
        source=str(source),
        source_bytes=source_bytes,
        required_bytes_with_headroom=needed,
        config_path=str(config),
    )
    if not tmpdir or not Path(tmpdir).is_dir():
        report["reason"] = "TMPDIR is unset or is not a directory"
    else:
        available = shutil.disk_usage(tmpdir).free
        report.update(tmpdir=str(tmpdir), available_bytes=available)
        if available < needed:
            report["reason"] = (
                "Dataset does not fit in TMPDIR with 10% + 64 MiB headroom"
            )
        else:
            target = (
                Path(tmpdir)
                / f"njord-1deg-{os.environ.get('SLURM_JOB_ID', 'local')}"
            )
            target.mkdir()  # Refuse to reuse an incomplete or stale copy.
            try:
                for name in paths:
                    src, dst = source / name, target / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    (
                        shutil.copytree(src, dst)
                        if src.is_dir()
                        else shutil.copy2(src, dst)
                    )
            except OSError as error:
                if error.errno != errno.ENOSPC:
                    raise
                shutil.rmtree(target)
                report["reason"] = (
                    "TMPDIR ran out of space during copying; using shared data"
                )
            else:
                selected = target / config.name
                report.update(
                    status="staged", config_path=str(selected), paths=paths
                )
    report["elapsed_s"] = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    (output / "staging.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Input staging: " + json.dumps(report), file=sys.stderr, flush=True)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(stage(args.output, os.environ.get("TMPDIR")))
