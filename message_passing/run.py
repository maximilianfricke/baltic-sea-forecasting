"""First P2P run: original AR4 model and live original data, no CPU binding."""

# Standard library
import argparse
import hashlib
import math
import os
import socket
import subprocess
import time
from datetime import timedelta
from pathlib import Path

# Third-party
import torch
import torch.distributed as dist
from torch.utils.data import default_collate

# Local
from .data import data_loader, local_batch, ownership
from .model import DistributedGraphEFM
from .partition import graph_fingerprint
from .results import digest_tensor, distribution, write_json
from .validation import optimizer, validate_original
from .workload import load_workload

ROOT = Path(__file__).resolve().parents[1]
BASE = "76b30d778382939963219142053fcf6f2d54c54e"


def cpu_snapshot():
    """Observe CPU state without changing or repairing it."""
    return {
        "affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "torch_threads": torch.get_num_threads(),
    }


def source_identity():
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE], cwd=ROOT, text=True
    ).splitlines()
    if subprocess.check_output(
        ["git", "diff", BASE, "--", *files], cwd=ROOT, text=True
    ):
        raise RuntimeError("An original global-model file changed")
    paths = [
        p
        for p in (ROOT / "message_passing").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    paths += [
        ROOT / "jobs/message_passing.sbatch",
        ROOT / "jobs/submit_message_passing.sh",
    ]
    return {
        "base": BASE,
        "original_files_verified": len(files),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device, function):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000


def model_step(model, opt, batch, seed, device):
    opt.zero_grad(set_to_none=True)
    start = time.perf_counter()
    output, forward = timed(device, lambda: model(batch, seed))
    _, backward = timed(device, output["loss"].backward)
    _, grad_sync = timed(device, model.synchronize_gradients)
    _, opt_time = timed(device, opt.step)
    loss = output["loss"].detach().clone()
    _, reduce = timed(device, lambda: dist.all_reduce(loss))
    wall = (time.perf_counter() - start) * 1000
    if not math.isfinite(float(loss)):
        raise RuntimeError("Non-finite loss")
    return {
        "model_wall_ms": wall,
        "forward_ms": forward,
        "backward_ms": backward,
        "gradient_sync_ms": grad_sync,
        "optimizer_ms": opt_time,
        "metric_reduce_ms": reduce,
        "loss": float(loss),
    }


def run(args):
    if args.steps < 1 or args.warmup < 0 or args.workers < 0:
        raise ValueError("Invalid run length or worker count")
    snapshots = {"after_import": cpu_snapshot()}
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = (
        torch.device("cpu")
        if args.cpu
        else torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo" if args.cpu else "nccl", timeout=timedelta(minutes=5)
    )
    dist.barrier()
    snapshots["after_first_collective"] = cpu_snapshot()
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        if size not in (1, 2, 4):
            raise ValueError("Expected 1, 2 or 4 ranks")
        setup = time.perf_counter()
        source = source_identity()
        torch.manual_seed(args.seed)
        dataset, original, config = load_workload(args)
        owners = ownership(original, size, args.owners)
        graph_sha = graph_fingerprint(original)
        model = DistributedGraphEFM(
            original,
            owners,
            device,
            config=config,
            datastore=original._datastore,
        )
        initial = {
            name: t.detach().cpu().clone()
            for name, t in model.state_dict().items()
        }
        # One real batch for correctness, not a cache used for timing.
        full = tuple(
            t.to(device)
            for t in default_collate(
                [dataset[i] for i in range(args.batch_size)]
            )
        )
        local = model.local_batch(full)
        validation = validate_original(
            original.to(device), model, full, local, args.seed + 100, args.lr
        )
        original.zero_grad(set_to_none=True)
        original.to("cpu")
        del original, full, local
        model.load_state_dict(initial)
        model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        loader = data_loader(dataset, args.batch_size, args.workers, args.seed)
        iterator = iter(loader)
        opt = optimizer(model, args.lr)
        setup_ms = (time.perf_counter() - setup) * 1000
        records = []
        if rank == 0:
            print(
                "Original-model correctness check passed; starting P2P run",
                flush=True,
            )
        for index in range(args.warmup + args.steps):
            start = time.perf_counter()
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                cpu_batch = next(iterator)
            read_ms = (time.perf_counter() - start) * 1000
            selected = time.perf_counter()
            cpu_batch = local_batch(cpu_batch, model.nodes["grid"])
            select_ms = (time.perf_counter() - selected) * 1000
            gpu_batch, h2d_ms = timed(
                device, lambda: tuple(t.to(device) for t in cpu_batch)
            )
            _, wait_ms = timed(device, dist.barrier)
            if index == args.warmup:
                snapshots["after_warmup"] = cpu_snapshot()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            record = model_step(
                model, opt, gpu_batch, args.seed + 1000 + index, device
            )
            record.update(
                data_read_ms=read_ms,
                partition_select_ms=select_ms,
                host_to_device_ms=h2d_ms,
                rank_wait_ms=wait_ms,
                end_to_end_ms=(time.perf_counter() - start) * 1000,
                batch_samples=gpu_batch[0].shape[0],
                measurement_step=index - args.warmup,
            )
            if index >= args.warmup:
                records.append(record)
            del gpu_batch
            if rank == 0 and (
                (index + 1) % 10 == 0 or index + 1 == args.warmup + args.steps
            ):
                print(
                    f"P2P step {index + 1}/{args.warmup + args.steps}",
                    flush=True,
                )
        snapshots["after_measurement"] = cpu_snapshot()
        payload = {
            "rank": rank,
            "hostname": socket.gethostname(),
            "cpu_snapshots": snapshots,
            "validation": validation,
            "records": records,
            "setup_ms": setup_ms,
            "nodes": {n: len(ids) for n, ids in model.nodes.items()},
            "owner_sha256": {n: digest_tensor(t) for n, t in owners.items()},
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
        }
        ranks = [None] * size
        dist.all_gather_object(ranks, payload)
        if rank == 0:
            combined = []
            for observations in zip(
                *(r["records"] for r in ranks), strict=True
            ):
                critical = max(
                    range(size), key=lambda r: observations[r]["model_wall_ms"]
                )
                row = dict(observations[critical])
                row["model_critical_rank"] = critical
                row["application_wall_ms"] = max(
                    r["end_to_end_ms"] for r in observations
                )
                combined.append(row)
            summary = {
                key: distribution([r[key] for r in combined])
                for key in combined[0]
                if key.endswith("_ms") or key == "loss"
            }
            count = sum(r["batch_samples"] for r in combined)
            summary["model_samples_per_second"] = (
                count * 1000 / sum(r["model_wall_ms"] for r in combined)
            )
            summary["application_samples_per_second"] = (
                count * 1000 / sum(r["application_wall_ms"] for r in combined)
            )
            write_json(
                args.output,
                {
                    "status": "completed",
                    "world_size": size,
                    "transport": "p2p",
                    "performance_valid": not args.cpu,
                    "source": source,
                    "graph_sha256": graph_sha,
                    "model_args": vars(model.model.args),
                    "checkpoint_early_stop": "original default",
                    "input": {
                        "path": str(args.config_path),
                        "mode": "live original Zarr; rank selects owned nodes",
                        "workers_per_rank": args.workers,
                        "prefetch_factor": loader.prefetch_factor,
                        "shuffle_seed": args.seed,
                    },
                    "workload": {
                        "warmup_steps": args.warmup,
                        "measurement_steps": args.steps,
                        "global_batch_size": args.batch_size,
                    },
                    "runtime": {
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "nccl": (
                            torch.cuda.nccl.version()
                            if device.type == "cuda"
                            else None
                        ),
                    },
                    "cpu_policy": "inherited; observed only, never modified",
                    "all_original_checks_passed": all(
                        r["validation"]["passed"] for r in ranks
                    ),
                    "ranks": ranks,
                    "records": combined,
                    "summary": summary,
                },
            )
            print(f"Result: {args.output}", flush=True)
        opt.zero_grad(set_to_none=True)
        del iterator, loader, opt
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--graph", default="global_cluster_graph")
    parser.add_argument("--owners", type=Path)
    parser.add_argument("--cpu", action="store_true")
    for name, value in (
        ("ar-steps", 4),
        ("hidden-dim", 64),
        ("processor-layers", 2),
        ("batch-size", 2),
        ("seed", 42),
        ("workers", 8),
        ("warmup", 10),
        ("steps", 50),
    ):
        parser.add_argument("--" + name, type=int, default=value)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if int(os.environ.get("RANK", 0)) == 0:
            write_json(
                args.output,
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "cpu": cpu_snapshot(),
                },
            )
        raise


if __name__ == "__main__":
    main()
