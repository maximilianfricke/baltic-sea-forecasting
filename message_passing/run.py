"""One original-model validation and benchmark entry point for 1/2/4 GPUs."""

# Standard library
import argparse
import copy
import hashlib
import json
import math
import os
import platform
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
from .data import data_loader, local_cpu_batch, ownership, rank_dataset
from .launch import affinity_snapshot, assert_main_affinity
from .model import DistributedGraphEFM
from .partition import graph_fingerprint
from .report import digest_tensor, distribution, write_json
from .validation import optimizer, validate_original
from .workload import load_workload

ROOT = Path(__file__).resolve().parents[1]
BASE = "76b30d778382939963219142053fcf6f2d54c54e"


def source_identity():
    """All original tracked files, not only thirteen selected model files."""
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE], cwd=ROOT, text=True
    ).splitlines()
    changed = subprocess.check_output(
        ["git", "diff", BASE, "--", *files], cwd=ROOT, text=True
    )
    if changed:
        raise RuntimeError(
            "An original global-model file changed; "
            "restore the reference before benchmarking"
        )
    source_paths = [
        p
        for p in sorted((ROOT / "message_passing").rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ] + [
        ROOT / "jobs/message_passing.sbatch",
        ROOT / "jobs/submit_message_passing.sh",
    ]
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_paths
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--", "message_passing"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    return {
        "base": BASE,
        "original_files_verified": len(files),
        "commit": commit,
        "dirty": dirty,
        "sources": sources,
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


def step(model, opt, batch, seed, device):
    opt.zero_grad(set_to_none=True)
    begin = time.perf_counter()
    output, forward_ms = timed(device, lambda: model(batch, seed))
    _, backward_ms = timed(device, output["loss"].backward)
    _, gradient_sync_ms = timed(device, model.synchronize_gradients)
    _, optimizer_ms = timed(device, opt.step)
    total = output["loss"].detach().clone()
    _, metric_reduce_ms = timed(device, lambda: dist.all_reduce(total))
    model_wall_ms = (time.perf_counter() - begin) * 1000
    loss = float(total)
    if not math.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    return dict(
        model_wall_ms=model_wall_ms,
        forward_ms=forward_ms,
        backward_ms=backward_ms,
        gradient_sync_ms=gradient_sync_ms,
        optimizer_ms=optimizer_ms,
        metric_reduce_ms=metric_reduce_ms,
        loss=loss,
    )


def benchmark(model, cache, dataset, initial, args, device, backend):
    model.transport.backend = backend
    model.load_state_dict(initial)
    model.train()
    opt = optimizer(model, args.lr)
    loader = data_loader(dataset, args) if args.input_mode == "live" else None
    iterator = iter(loader) if loader is not None else None
    records = []
    for index in range(args.warmup + args.steps):
        begin = time.perf_counter()
        if iterator is None:
            cpu_batch = cache[index % len(cache)]
        else:
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                cpu_batch = next(iterator)
            cpu_batch = local_cpu_batch(
                cpu_batch, model.nodes["grid"], args.data_source == "shards"
            )
        input_ms = (time.perf_counter() - begin) * 1000
        batch, h2d_ms = timed(
            device, lambda: tuple(t.to(device) for t in cpu_batch)
        )
        _, readiness_ms = timed(device, dist.barrier)
        if index == args.warmup:
            assert_main_affinity()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        record = step(model, opt, batch, args.seed + 1000 + index, device)
        record.update(
            input_wait_ms=input_ms,
            h2d_ms=h2d_ms,
            readiness_ms=readiness_ms,
            end_to_end_ms=(time.perf_counter() - begin) * 1000,
            measurement_step=index - args.warmup,
            batch_slot=index % len(cache) if iterator is None else None,
            batch_samples=batch[0].shape[0],
        )
        if index >= args.warmup:
            records.append(record)
        del batch
        if dist.get_rank() == 0 and (
            (index + 1) % 10 == 0 or index + 1 == args.warmup + args.steps
        ):
            print(
                f"{backend}: {index + 1}/{args.warmup + args.steps} steps",
                flush=True,
            )
    assert_main_affinity()
    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    opt.zero_grad(set_to_none=True)
    # Dispose the iterator/loader to stop workers before the next case.
    del iterator, loader, opt
    return {"records": records, "peak_allocated_bytes": peak}


def run(args):
    if (
        args.steps < 1
        or args.warmup < 0
        or args.cached_batches < 1
        or args.workers < 0
        or args.prefetch < 1
    ):
        raise ValueError("Invalid step/cache/loader settings")
    if os.environ.get("SLURM_JOB_ID") and not os.environ.get("MP_JOB_CPUS"):
        raise RuntimeError(
            "Use the supplied Slurm launcher to verify CPU placement"
        )
    trace = json.loads(os.environ.get("MP_INITIAL_CPU_TRACE", "[]"))
    trace.append(affinity_snapshot("after_torch_import"))
    torch.set_num_threads(1)
    trace.append(affinity_snapshot("after_thread_setup"))
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cpu")
    if not args.cpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; --cpu is correctness-only")
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    trace.append(affinity_snapshot("after_cuda"))
    dist.init_process_group(
        "gloo" if args.cpu else "nccl", timeout=timedelta(minutes=10)
    )
    dist.barrier()
    trace.append(affinity_snapshot("after_first_collective"))
    # A single explicit main-thread restoration after library initialization.
    # Helper threads are observed, never blindly repinned.
    if os.environ.get("MP_RANK_CPUS"):
        os.sched_setaffinity(0, json.loads(os.environ["MP_RANK_CPUS"]))
    trace.append(affinity_snapshot("runtime_ready"))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        if size not in (1, 2, 4):
            raise ValueError("Expected 1, 2 or 4 ranks")
        started = time.perf_counter()
        identity = source_identity()
        torch.manual_seed(args.seed)
        dataset, original, config = load_workload(args)
        original.grad_checkpointing = args.checkpointing
        owners = ownership(original, size, args.owners)
        graph_sha = graph_fingerprint(original)
        candidate = DistributedGraphEFM(
            original,
            owners,
            device,
            config=config,
            datastore=original._datastore,
        )
        initial = {
            n: t.detach().cpu().clone()
            for n, t in candidate.state_dict().items()
        }
        original_initial = copy.deepcopy(original.state_dict())
        selected = (
            rank_dataset(args, config, original, owners, rank, size)
            if args.data_source == "shards"
            else dataset
        )
        cache_start = time.perf_counter()
        cache = [
            local_cpu_batch(
                batch, candidate.nodes["grid"], args.data_source == "shards"
            )
            for batch in data_loader(selected, args, finite_cache=True)
        ]
        cache_ms = (time.perf_counter() - cache_start) * 1000
        full_cpu = default_collate([dataset[i] for i in range(args.batch_size)])
        full = tuple(t.to(device) for t in full_cpu)
        local = tuple(t.to(device) for t in cache[0])
        backends = (
            ["p2p", "physicsnemo"] if args.backend == "both" else [args.backend]
        )
        if args.repeat % 2:
            backends.reverse()
        if args.cpu and "physicsnemo" in backends and size > 1:
            raise RuntimeError(
                "Native PhysicsNeMo unequal all-to-all requires CUDA/NCCL; "
                "use the explicit CPU test emulation"
            )
        validations = {}
        for backend in backends:
            original.to(device).load_state_dict(original_initial)
            candidate.load_state_dict(initial)
            candidate.transport.backend = backend
            checked = validate_original(
                original, candidate, full, local, args.seed + 100, args.lr
            )
            checks = [None] * size
            dist.all_gather_object(checks, checked)
            validations[backend] = checks
            if rank == 0:
                print(
                    f"{backend}: original AR4 predictions/loss/gradients/"
                    "AdamW/inference PASS",
                    flush=True,
                )
            original.zero_grad(set_to_none=True)
            original.to("cpu")
        del full, local, full_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()
        assert_main_affinity()
        metadata = {
            "rank": rank,
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": (
                torch.cuda.nccl.version() if device.type == "cuda" else None
            ),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "cpu_trace": trace,
            "cpu_ready": affinity_snapshot("before_measurement"),
            "cache_prepare_ms": cache_ms,
            "setup_ms": (time.perf_counter() - started) * 1000,
            "cache_sha256": [
                [digest_tensor(t) for t in batch] for batch in cache
            ],
            "owned_nodes": {n: len(ids) for n, ids in candidate.nodes.items()},
            "owner_sha256": {n: digest_tensor(t) for n, t in owners.items()},
        }
        controls = {
            "world_size": size,
            "graph_sha256": graph_sha,
            "model_args": vars(original.args),
            "checkpointing": args.checkpointing,
            "checkpoint_early_stop": False,
            "deterministic_algorithms": True,
            "distribution_validate_args": (
                torch.distributions.Distribution._validate_args
            ),
            "input_mode": args.input_mode,
            "data_source": args.data_source,
            "workers_per_rank": args.workers,
            "prefetch_factor": args.prefetch if args.workers else None,
            "cache_batches": len(cache),
            "pin_memory": False,
            "warmup": args.warmup,
            "steps": args.steps,
            "cpu_policy": os.environ.get("MP_CPU_POLICY", "unmanaged"),
            "job_cpus": json.loads(os.environ.get("MP_JOB_CPUS", "null")),
            "source": identity,
            "initial_state_sha256": {
                n: digest_tensor(t) for n, t in initial.items()
            },
        }
        all_metadata = [None] * size
        dist.all_gather_object(all_metadata, metadata)
        controls["rank_controls"] = [
            {
                key: r[key]
                for key in (
                    "cache_sha256",
                    "owned_nodes",
                    "owner_sha256",
                    "cpu_ready",
                    "hostname",
                    "torch",
                    "cuda",
                    "nccl",
                    "gpu",
                )
            }
            for r in all_metadata
        ]
        # Thread IDs differ between launches. The comparable placement control
        # is the main-thread mask; full thread traces stay in raw metadata.
        for row in controls["rank_controls"]:
            row["cpu_ready"] = row["cpu_ready"]["main"]
        controls_sha = hashlib.sha256(
            json.dumps(controls, sort_keys=True).encode()
        ).hexdigest()
        cases = {}
        for backend in backends:
            observation = benchmark(
                candidate, cache, selected, initial, args, device, backend
            )
            ranks = [None] * size
            dist.all_gather_object(ranks, observation)
            if rank == 0:
                records = []
                for values in zip(*(r["records"] for r in ranks), strict=True):
                    critical = max(
                        range(size), key=lambda r: values[r]["model_wall_ms"]
                    )
                    row = dict(values[critical])
                    row["critical_rank"] = critical
                    application_rank = max(
                        range(size), key=lambda r: values[r]["end_to_end_ms"]
                    )
                    row["application_critical_rank"] = application_rank
                    row["application_wall_ms"] = values[application_rank][
                        "end_to_end_ms"
                    ]
                    records.append(row)
                summary = {
                    key: distribution([r[key] for r in records])
                    for key in records[0]
                    if key.endswith("_ms") or key == "loss"
                }
                summary["global_batch_size"] = args.batch_size
                measured_samples = sum(r["batch_samples"] for r in records)
                summary["measured_samples"] = measured_samples
                summary["model_samples_per_second"] = (
                    measured_samples
                    * 1000
                    / sum(r["model_wall_ms"] for r in records)
                )
                summary["application_samples_per_second"] = (
                    measured_samples
                    * 1000
                    / sum(r["application_wall_ms"] for r in records)
                )
                cases[backend] = {
                    "records": records,
                    "rank_records": ranks,
                    "summary": summary,
                }
        metadata["cpu_final"] = affinity_snapshot("after_measurement")
        dist.all_gather_object(all_metadata, metadata)
        if rank == 0:
            write_json(
                args.output,
                {
                    "status": "completed",
                    "controls": controls,
                    "controls_sha256": controls_sha,
                    "validation": validations,
                    "ranks": all_metadata,
                    "cases": cases,
                    "backend_order": backends,
                    "timer": (
                        "Model wall step after H2D and readiness; "
                        "cached preparation/setup excluded"
                    ),
                    "performance_valid": not args.cpu,
                },
            )
            print(f"Result: {args.output}", flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path)
    parser.add_argument("--graph", default="global_cluster_graph")
    parser.add_argument("--owners", type=Path)
    parser.add_argument(
        "--data-source", choices=("original", "shards"), default="shards"
    )
    parser.add_argument(
        "--input-mode", choices=("cached", "live"), default="cached"
    )
    parser.add_argument(
        "--backend", choices=("p2p", "physicsnemo", "both"), default="both"
    )
    parser.add_argument(
        "--checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cpu", action="store_true")
    for name, default in (
        ("ar-steps", 4),
        ("hidden-dim", 64),
        ("processor-layers", 2),
        ("batch-size", 2),
        ("seed", 42),
        ("workers", 0),
        ("prefetch", 2),
        ("cached-batches", 4),
        ("warmup", 10),
        ("steps", 50),
        ("repeat", 0),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data_source == "shards" and args.shard_root is None:
        parser.error("--shard-root is required for shard input")
    try:
        run(args)
    except Exception as error:
        if int(os.environ.get("RANK", 0)) == 0:
            write_json(
                args.output,
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise


if __name__ == "__main__":
    main()
