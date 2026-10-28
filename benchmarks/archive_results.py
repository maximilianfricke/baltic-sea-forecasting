"""Copy compact benchmark evidence into the Git-tracked archive."""

# Standard library
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
MANIFEST_FIELDS = (
    "archive_file",
    "sha256",
    "source_file",
    "slurm_job_id",
    "label",
    "status",
    "started_at_utc",
    "git_commit",
    "archived_at_utc",
)


def _result_paths(inputs):
    paths = set()
    for input_path in inputs:
        if input_path.is_file() and input_path.suffix == ".json":
            paths.add(input_path.resolve())
        elif input_path.is_dir():
            for root, _, names in os.walk(input_path, followlinks=True):
                for name in names:
                    if name.endswith(".json") and not name.endswith(
                        ".pt.trace.json"
                    ):
                        paths.add((Path(root) / name).resolve())
    return sorted(paths)


def _safe(value):
    return SAFE_NAME.sub("-", str(value)).strip("-") or "unknown"


def _load_manifest(path):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["sha256"]: row
            for row in csv.DictReader(handle)
            if row.get("sha256")
        }


def _telemetry_candidates(inputs, job_id):
    if not job_id:
        return []
    candidates = set()
    pattern = f"*gpu-{job_id}.csv"
    for input_path in inputs:
        roots = [input_path] if input_path.is_dir() else [input_path.parent]
        for root in roots:
            candidates.update(root.rglob(pattern))
    return sorted(path for path in candidates if path.is_file())


def archive(inputs, output_root):
    runs_dir = output_root / "runs"
    telemetry_dir = output_root / "telemetry"
    manifest_path = output_root / "manifest.csv"
    runs_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    archived_at = datetime.now(timezone.utc).isoformat()
    added = 0

    for source in _result_paths(inputs):
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in manifest:
            continue
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict) or "schema_version" not in result:
            continue

        hardware = result.get("hardware") or {}
        git = result.get("git") or {}
        job_id = hardware.get("slurm_job_id")
        identity = job_id or "local"
        label = result.get("label") or source.stem
        destination_name = (
            f"{_safe(identity)}-{_safe(label)}-{digest[:10]}.json"
        )
        destination = runs_dir / destination_name
        destination.write_bytes(raw)
        manifest[digest] = {
            "archive_file": str(destination.relative_to(output_root)),
            "sha256": digest,
            "source_file": str(source),
            "slurm_job_id": job_id or "",
            "label": label,
            "status": result.get("status", ""),
            "started_at_utc": result.get("started_at_utc", ""),
            "git_commit": git.get("commit", ""),
            "archived_at_utc": archived_at,
        }
        for telemetry in _telemetry_candidates(inputs, job_id):
            telemetry_destination = telemetry_dir / telemetry.name
            if not telemetry_destination.exists():
                shutil.copy2(telemetry, telemetry_destination)
        added += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(
            sorted(
                manifest.values(),
                key=lambda row: (
                    row.get("started_at_utc", ""),
                    row.get("archive_file", ""),
                ),
            )
        )
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    added = archive(args.inputs, args.output_root)
    print(f"Archived {added} new benchmark result(s)")


if __name__ == "__main__":
    main()
