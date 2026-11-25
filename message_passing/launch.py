"""Stdlib-only CPU placement before torch imports. No import-time mutations."""

# Standard library
import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def cpu_shares(cpus, ranks, policy):
    cpus = sorted(set(cpus))
    if ranks < 1 or len(cpus) < ranks:
        raise ValueError("Each rank needs at least one allocated logical CPU")
    if policy == "shared":
        return [cpus[:] for _ in range(ranks)]
    if policy != "disjoint":
        raise ValueError(f"Unknown CPU policy: {policy}")
    return [
        cpus[i * len(cpus) // ranks : (i + 1) * len(cpus) // ranks]
        for i in range(ranks)
    ]


def affinity_snapshot(stage):
    snapshot = {
        "stage": stage,
        "main": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
    }
    snapshot["threads"] = {}
    tasks = Path("/proc/self/task")
    if tasks.is_dir():
        for entry in tasks.iterdir():
            try:
                snapshot["threads"][entry.name] = sorted(
                    os.sched_getaffinity(int(entry.name))
                )
            except ProcessLookupError:
                pass
    return snapshot


def assert_main_affinity():
    encoded = os.environ.get("MP_RANK_CPUS")
    if encoded and sorted(os.sched_getaffinity(0)) != json.loads(encoded):
        raise RuntimeError("Rank main-thread CPU mask changed; reject this run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--ranks", type=int, choices=(1, 2, 4), required=True)
    launch.add_argument("--expected-cpus", type=int, required=True)
    launch.add_argument(
        "--cpu-policy", choices=("shared", "disjoint"), default="disjoint"
    )
    launch.add_argument("arguments", nargs=argparse.REMAINDER)
    rank = sub.add_parser("rank")
    rank.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = (
        args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    )
    if args.command == "launch":
        if not os.environ.get("SLURM_JOB_ID") or not os.environ.get(
            "SLURM_STEP_ID"
        ):
            raise RuntimeError(
                "The GPU launcher must run in the supplied Slurm step"
            )
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) != args.expected_cpus:
            raise RuntimeError(
                f"Step mask {cpus} does not match "
                f"{args.expected_cpus} allocated CPUs"
            )
        cpu_shares(cpus, args.ranks, args.cpu_policy)
        os.environ["MP_JOB_CPUS"] = json.dumps(cpus)
        os.environ["MP_CPU_POLICY"] = args.cpu_policy
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = "1"
        os.environ["OMP_PROC_BIND"] = "false"
        os.environ["KMP_AFFINITY"] = "disabled"
        for name in ("GOMP_CPU_AFFINITY", "OMP_PLACES"):
            os.environ.pop(name, None)
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={args.ranks}",
                "-m",
                "message_passing.launch",
                "rank",
                "--",
                *forwarded,
            ],
        )
    cpus = json.loads(os.environ["MP_JOB_CPUS"])
    target = cpu_shares(
        cpus, int(os.environ["LOCAL_WORLD_SIZE"]), os.environ["MP_CPU_POLICY"]
    )[int(os.environ["LOCAL_RANK"])]
    before = affinity_snapshot("before_rank_placement")
    os.sched_setaffinity(0, target)
    os.environ["MP_RANK_CPUS"] = json.dumps(target)
    os.environ["MP_INITIAL_CPU_TRACE"] = json.dumps(
        [before, affinity_snapshot("before_torch_import")]
    )
    sys.argv = ["message_passing.run", *forwarded]
    runpy.run_module("message_passing.run", run_name="__main__")


if __name__ == "__main__":
    main()
