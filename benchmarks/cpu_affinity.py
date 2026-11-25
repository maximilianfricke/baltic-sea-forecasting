"""Benchmark-only CPU placement, restricted to a verified Slurm step mask.

The launcher runs before torch is imported. Each rank then inherits a distinct
part of that mask. Repairs happen outside timing; a changed mask during timing
rejects the result. No machine-wide CPU IDs are guessed or requested here.
"""

import argparse
import errno
import json
import os
from pathlib import Path
import sys


POLICY = "slurm-disjoint-ranks-v1"
MASK_ENV = "NJORD_VERIFIED_JOB_CPUS"
REPORT_ENV = "NJORD_CPU_LAUNCH_REPORT"


def split_cpus(cpus, ranks):
    cpus = sorted(set(cpus))
    if ranks < 1 or len(cpus) < ranks:
        raise RuntimeError("Each GPU rank needs at least one allocated logical CPU")
    return [cpus[i * len(cpus) // ranks:(i + 1) * len(cpus) // ranks]
            for i in range(ranks)]


def write_report(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def launch(expected_cpus, metadata, command):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Launch this benchmark inside the provided srun job step")
    cpus = sorted(os.sched_getaffinity(0))
    # An unrestricted whole-node mask is not proof of a job allocation either.
    if expected_cpus < 1 or len(cpus) != expected_cpus:
        raise RuntimeError(
            f"Slurm step exposes CPUs {cpus}, expected {expected_cpus}. "
            "Refusing to guess additional CPU IDs; check the srun binding."
        )
    if not command:
        raise ValueError("Missing Python command after --")
    report = {"policy": POLICY, "job_cpu_ids": cpus,
              "expected_logical_cpus": expected_cpus,
              "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "slurm_step_id": os.environ["SLURM_STEP_ID"],
              "inherited_affinity_environment": {k: os.environ.get(k) for k in (
                  "SLURM_CPU_BIND", "OMP_PROC_BIND", "OMP_PLACES",
                  "GOMP_CPU_AFFINITY", "KMP_AFFINITY", "NCCL_IGNORE_CPU_AFFINITY")}}
    write_report(metadata, report)
    os.environ[MASK_ENV] = json.dumps(cpus)
    os.environ[REPORT_ENV] = str(Path(metadata).resolve())
    # Disable inherited OpenMP pinning, while keeping one compute thread/rank.
    os.environ["OMP_PROC_BIND"] = "false"
    os.environ["KMP_AFFINITY"] = "disabled"
    os.environ.pop("GOMP_CPU_AFFINITY", None)
    os.environ.pop("OMP_PLACES", None)
    print(f"CPU allocation verified: {cpus}; policy={POLICY}", flush=True)
    os.execv(sys.executable, [sys.executable, *command])


class AffinityGuard:
    def __init__(self, job_cpus=None, rank=0, ranks=1, output=None):
        self.job_cpus = sorted(set(job_cpus)) if job_cpus is not None else None
        self.rank, self.ranks, self.output = rank, ranks, output
        self.history = []
        if self.job_cpus is None:
            self.target = None
        else:
            if not 0 <= rank < ranks:
                raise ValueError("Invalid local rank")
            self.target = split_cpus(self.job_cpus, ranks)[rank]

    @classmethod
    def from_environment(cls):
        encoded = os.environ.get(MASK_ENV)
        if encoded is None:
            return cls()
        rank = int(os.environ["LOCAL_RANK"])
        ranks = int(os.environ["LOCAL_WORLD_SIZE"])
        launch_path = Path(os.environ[REPORT_ENV])
        report = json.loads(launch_path.read_text())
        cpus = json.loads(encoded)
        if cpus != report["job_cpu_ids"] or report["policy"] != POLICY:
            raise RuntimeError("CPU mask differs from the Slurm launch report")
        return cls(cpus, rank, ranks,
                   launch_path.with_name(f"{launch_path.stem}-rank-{rank}.json"))

    @staticmethod
    def thread_masks():
        masks = {}
        for entry in Path("/proc/self/task").iterdir():
            try:
                masks[int(entry.name)] = sorted(os.sched_getaffinity(int(entry.name)))
            except ProcessLookupError:
                pass  # A CUDA/helper thread may have exited while enumerating.
        return masks

    def checkpoint(self, stage, repair=False):
        if self.target is None:
            return
        before = sorted(os.sched_getaffinity(0))
        masks = self.thread_masks()
        changed = []
        if repair:
            # Set the main thread first, so newly created threads inherit it.
            os.sched_setaffinity(0, self.target)
            for tid, mask in masks.items():
                if mask != self.target:
                    try:
                        os.sched_setaffinity(tid, self.target)
                        changed.append(tid)
                    except OSError as error:
                        if error.errno != errno.ESRCH:
                            raise
        after = sorted(os.sched_getaffinity(0))
        current = self.thread_masks()
        bad = {tid: mask for tid, mask in current.items() if mask != self.target}
        passed = after == self.target and not bad
        self.history.append({"stage": stage, "main_before": before,
                             "main_after": after, "repaired_thread_ids": changed,
                             "thread_masks": current, "passed": passed})
        if self.output is not None:
            write_report(self.output, self.report())
        if not passed:
            raise RuntimeError(
                f"CPU affinity changed at {stage}: expected {self.target}, "
                f"main={after}, threads={bad}. Reject this timing run."
            )
        if changed or len(self.history) == 1:
            print(f"rank {self.rank}: CPU affinity {stage}: {before} -> "
                  f"{after}; repaired {len(changed)} thread(s)", flush=True)

    def report(self):
        return {"policy": POLICY if self.target is not None else "unmanaged",
                "job_cpu_ids": self.job_cpus, "rank_cpu_ids": self.target,
                "rank": self.rank, "checkpoints": self.history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-cpus", type=int, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    launch(args.expected_cpus, args.metadata, command)


if __name__ == "__main__":
    main()
