"""Record an expected capacity-probe failure as benchmark evidence."""

# Standard library
import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--hidden-dim", required=True, type=int)
    parser.add_argument("--checkpointing", required=True, choices=("on", "off"))
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()

    error = ""
    if args.log.exists():
        error = args.log.read_text(encoding="utf-8", errors="replace")[-8000:]
    result = {
        "schema_version": 4,
        "label": args.label,
        "started_at_utc": None,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "error": error,
        "exit_code": args.exit_code,
        "git": {"commit": _git_commit(), "dirty": None},
        "hardware": {
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": {"name": os.environ.get("NJORD_GPU_NAME")},
        },
        "distributed": {"world_size": 1, "scaling_mode": "capacity"},
        "model": {
            "class": "GraphEFM",
            "gradient_checkpointing": args.checkpointing == "on",
        },
        "arguments": {
            "hidden_dim": args.hidden_dim,
            "processor_layers": 2,
            "batch_size": 2,
            "ar_steps_train": 4,
            "precision": "32",
        },
        "protocol": {"workload": "training"},
        "summary": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
