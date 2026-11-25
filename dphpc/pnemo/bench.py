"""Fixed GraphEFM benchmark; job driver and isolated torchrun case worker."""

import argparse
import csv
from datetime import timedelta
import hashlib
import itertools
import math
import socket
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile
from torch.utils.checkpoint import set_checkpoint_early_stop
from torch.utils.data import DataLoader, default_collate

from .distributed_graph_efm import DistributedGraphEFM
from .partitions import CONFIG, DEFAULT, GRAPH, METHODS, graph_data, load_owners
from .static_metrics import partition_metrics
from .workload import load_workload
from .rank_dataset import RankDataset

BASE_FIELDS = [
    "job_id",
    "stage",
    "K",
    "partitioner",
    "hidden_dim",
    "checkpointing",
    "live_loader",
    "repeat",
    "repeats",
    "mode",
    "rank",
    "hostname",
    "commit",
    "seed",
    "check_seed",
    "parameters",
    "data_location",
    "partitions_dir",
    "reference_commit",
    "case_id",
    "log",
]
METRICS = [
    "e2e_ms",
    "gpu_ms",
    "load_wait_ms",
    "load_ms",
    "h2d_ms",
    "loading_ms",
    "halo_forward_ms",
    "halo_recompute_ms",
    "halo_forward_bytes_per_step",
]
PHASES = [
    "init_s",
    "setup_s",
    "preload_s",
    "check_s",
    "warmup_s",
    "timed_s",
    "profile_s",
]
FIELDS = (
    BASE_FIELDS
    + ["status", "check", "check_loss", "reference_loss"]
    + PHASES
    + METRICS
    + [
        "warmup_steps",
        "measured_steps",
        "cached_batches",
        "workers",
        "prefetch_factor",
        "batch_size",
        "peak_memory_min_mib",
        "peak_memory_max_mib",
        "compute_only_ms_min",
        "compute_only_ms_max",
        "samples_per_s",
    ]
)
RAW_FIELDS = BASE_FIELDS + ["step", "batch_slot"] + METRICS


def collate_owned(samples):
    return tuple(default_collate(samples))


def make_loader(source, seed, sampler):
    return DataLoader(
        source,
        batch_size=2,
        sampler=sampler,
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        multiprocessing_context="spawn",
        collate_fn=collate_owned,
        generator=torch.Generator().manual_seed(seed + 1),
    )


class HaloEvents:
    """Event pairs around the existing PN primitive; no transport changes."""

    def __init__(self, relations, hidden_dim=64):
        self.enabled = True
        self.pool = []
        self.reset()
        self.bytes = {}
        for name, relation in relations.items():
            graph = relation.graph
            self.bytes[name] = (
                partition_metrics(graph.graph_partition)[
                    "bytes_received_per_exchange"
                ]
                * hidden_dim
                // 64
            )
            original = graph.get_src_node_features_in_local_graph

            def timed(features, original=original, name=name):
                if not self.enabled:
                    return original(features)
                index = len(self.calls)
                if index == len(self.pool):
                    self.pool.append(
                        (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                    )
                start, end = self.pool[index]
                self.calls.append((name, self.phase, start, end))
                start.record()
                result = original(features)
                end.record()
                return result

            graph.get_src_node_features_in_local_graph = timed

    def reset(self):
        self.calls = []
        self.phase = "forward"

    def values(self):
        forward = sum(
            a.elapsed_time(b) for _, p, a, b in self.calls if p == "forward"
        )
        recompute = sum(
            a.elapsed_time(b) for _, p, a, b in self.calls if p == "backward"
        )
        received = sum(
            self.bytes[n] for n, p, _, _ in self.calls if p == "forward"
        )
        return forward, recompute, received


def kernel_summary(trace):
    events = json.loads(Path(trace).read_text())["traceEvents"]
    kernels = [
        e
        for e in events
        if "kernel" in e.get("cat", "").lower() and e.get("ph") == "X"
    ]
    nccl = sum(e["dur"] for e in kernels if "nccl" in e.get("name", "").lower())
    compute = sum(
        e["dur"] for e in kernels if "nccl" not in e.get("name", "").lower()
    )
    if not kernels:
        raise RuntimeError("Profiler produced no CUDA kernel activities")
    return dict(
        nccl_kernel_us=nccl,
        compute_kernel_us=compute,
        nccl_kernel_share=nccl / (nccl + compute) if nccl + compute else 0,
        kernel_count=len(kernels),
    )


def cases(K, stage="full"):
    if stage == "balance":
        specs = (
            [(m, 64, "on", False, 3) for m in ("metis", "mtkahypar_km1")]
            if K in (2, 4)
            else []
        )
    elif K == 1:
        specs = [
            ("original", 64, "on", False, 3),
            ("original", 128, "on", False, 1),
            ("original", 64, "on", True, 1),
        ]
    else:
        specs = [(m, 64, "on", False, 3) for m in METHODS]
        specs += [
            ("mtkahypar_km1", 64, "off", False, 1),
            ("mtkahypar_km1", 128, "on", False, 1),
            ("mtkahypar_km1", 256, "on", False, 1),
            ("mtkahypar_km1", 64, "on", True, 1),
        ]
    return [
        dict(
            partitioner=m,
            hidden_dim=h,
            checkpointing=c,
            live_loader=live,
            repeat=r,
            repeats=n,
        )
        for r in range(max((s[4] for s in specs), default=0))
        for m, h, c, live, n in specs
        if r < n
    ]


def median_summary(launch_steps, seed=42):
    """Percentile bootstrap: independent launches, then four-step cycle blocks."""
    values = np.asarray(launch_steps, dtype=float)
    if (
        values.ndim != 2
        or 0 in values.shape
        or values.shape[1] % 4
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Expected finite launch-by-step data, steps divisible by four"
        )
    launches, steps = values.shape
    rng = np.random.default_rng(seed)
    launch_ids = rng.integers(launches, size=(4000, launches, 1))
    block_ids = rng.integers(steps // 4, size=(4000, launches, steps // 4))
    samples = values.reshape(launches, steps // 4, 4)[launch_ids, block_ids]
    medians = np.median(samples.reshape(4000, -1), axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return dict(
        median=float(np.median(values)),
        ci95_low=float(low),
        ci95_high=float(high),
        min=float(values.min()),
        max=float(values.max()),
        ci_scope=(
            "launch_and_cycle_blocks" if launches > 1 else "within_launch_only"
        ),
    )


def case_name(args):
    return (
        f"{getattr(args, 'partitions_dir', DEFAULT).name}-{args.partitioner}-h{args.hidden_dim}-{args.checkpointing}-"
        f"{'live' if args.live_loader else 'cached'}-r{args.repeat}"
    )


def identity(args):
    return dict(
        job_id=os.environ.get("SLURM_JOB_ID", "local"),
        stage=args.stage,
        K=args.K,
        partitioner=args.partitioner,
        hidden_dim=args.hidden_dim,
        checkpointing=args.checkpointing,
        live_loader=int(args.live_loader),
        repeat=args.repeat,
        repeats=args.repeats,
        mode="train",
        rank=int(os.environ.get("RANK", "0")),
        hostname=socket.gethostname(),
        commit=args.commit,
        seed=args.seed,
        check_seed=1042,
        parameters=getattr(args, "parameters", ""),
        partitions_dir=str(args.partitions_dir),
        reference_commit=getattr(args, "reference_commit", args.commit),
        data_location=str(
            args.config_path
            if args.K == 1
            else args.partitions_dir
            / args.partitioner
            / f"k{args.K}"
            / f"rank{os.environ.get('RANK', '0')}.pt"
        ),
        case_id=case_name(args),
        log=f"cases/{case_name(args)}/worker.log",
    )


def append_row(path, row, fields=FIELDS):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def worker(args):
    torch.distributions.Distribution.set_default_validate_args(False)
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    assert not torch.are_deterministic_algorithms_enabled()
    if args.stage == "balance":
        args.reference_commit = json.loads(
            (args.reference_dir / "reference-loss-h64.json").read_text()
        )["commit"]
    rank = int(os.environ["RANK"])
    initial_affinity = sorted(os.sched_getaffinity(0))
    phase_start = time.perf_counter()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    assert torch.cuda.get_device_capability(device) == (6, 1), "Use GTX 1080 Ti"
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    assert dist.get_world_size() == args.K
    dist.barrier()
    torch.cuda.synchronize(
        device
    )  # Include lazy NCCL initialization in init_s.
    phases = {name: "" for name in PHASES}
    case_dir = args.output / "cases" / case_name(args)
    case_dir.mkdir(parents=True, exist_ok=True)

    check_state = {}

    def phase(name, start):
        phases[name] = time.perf_counter() - start
        if rank == 0:
            progress = dict(identity(args), **phases, **check_state)
            (case_dir / "phases.json").write_text(
                json.dumps(progress, indent=2) + "\n"
            )
            print(f"{name}={phases[name]:.3f}s", flush=True)

    phase("init_s", phase_start)
    phase_start = time.perf_counter()
    torch.manual_seed(args.seed)
    source, template, config = load_workload(
        args, include_global_dataset=args.K == 1
    )
    args.parameters = sum(p.numel() for p in template.parameters())
    distributed = None
    if args.K == 1:
        template.log_dict = lambda *a, **kw: None
        net, relations = template.to(device), {}
    else:
        info = graph_data(args.config_path, args.graph)
        owners, shards = load_owners(
            args.partitions_dir, args.partitioner, args.K, info
        )
        source = RankDataset(
            args, config, shards[rank], len(info[3].get_xy("state", True))
        )
        del shards
        distributed = DistributedGraphEFM(
            template, owners, info[2], config, info[3], device
        )
        net, relations = distributed.model, distributed.relations
    del template
    assert sum(p.numel() for p in net.parameters()) == args.parameters
    net.train()
    net.grad_checkpointing = args.checkpointing == "on"
    phase("setup_s", phase_start)
    phase_start = time.perf_counter()
    if len(source) < 8:
        raise ValueError(
            "Four distinct batch2 batches require at least eight samples"
        )
    batch_count = 1 if args.reference_only else (26 if args.live_loader else 4)
    loader = make_loader(
        source,
        args.seed,
        sampler=[i % 8 for i in range(2 * batch_count)],
    )
    loader_iterator = iter(loader)
    if args.live_loader and not args.reference_only:
        first = next(loader_iterator)
        iterator = itertools.chain((first,), loader_iterator)
        cached_batches = None
        cached_load_ms = None
    else:
        cached_batches, cached_load_ms = [], []
        for _ in range(batch_count):
            load_start = time.perf_counter()
            cached_batches.append(next(loader_iterator))
            cached_load_ms.append((time.perf_counter() - load_start) * 1000)
        first = cached_batches[0]
        del loader_iterator, loader
    phase("preload_s", phase_start)
    rank_metadata = dict(
        identity(args),
        initial_affinity=initial_affinity,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        nccl_version=torch.cuda.nccl.version(),
        gpu=str(torch.cuda.get_device_properties(device)),
        gpu_uuid=str(
            getattr(torch.cuda.get_device_properties(device), "uuid", "")
        ),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        batch_indices=[[0, 1], [2, 3], [4, 5], [6, 7]],
    )
    (case_dir / f"metadata-rank{rank}.json").write_text(
        json.dumps(rank_metadata, indent=2) + "\n"
    )
    halo = HaloEvents(relations, hidden_dim=args.hidden_dim)
    finite = torch.ones((), dtype=torch.bool, device=device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    def train_step(batch):
        optimizer.zero_grad(set_to_none=True)
        halo.phase = "forward"
        with set_checkpoint_early_stop(False if distributed else True):
            loss = net.training_step(batch)
        if distributed:
            total = loss.detach().clone()
            dist.all_reduce(total)
            loss = total + (loss - loss.detach())
        halo.phase = "backward"
        loss.backward()
        if distributed:
            distributed.synchronize_gradients()
        value = loss.detach()
        optimizer.step()
        finite.logical_and_(torch.isfinite(value))
        return value

    # Fixed seed and batch0, with pre-update loss. Roll the check step back completely.
    phase_start = time.perf_counter()
    initial = {
        name: value.detach().cpu().clone()
        for name, value in net.state_dict().items()
    }
    batch = tuple(t.to(device, non_blocking=True) for t in first)
    torch.cuda.synchronize(device)
    torch.manual_seed(1042)
    halo.enabled = False
    if args.reference_only:
        assert args.K == 1
        with torch.no_grad():
            loss = net.training_step(batch)
    else:
        loss = train_step(batch)
    torch.cuda.synchronize(device)
    check_loss = float(loss)
    reference_file = (
        args.reference_dir / f"reference-loss-h{args.hidden_dim}.json"
    )
    signature = dict(
        hidden_dim=args.hidden_dim,
        parameters=args.parameters,
        seed=args.seed,
        check_seed=1042,
        batch_indices=[0, 1],
        commit=args.commit,
        config_sha256=hashlib.sha256(args.config_path.read_bytes()).hexdigest(),
    )
    if rank == 0 and args.K == 1 and not reference_file.exists():
        reference_file.parent.mkdir(parents=True, exist_ok=True)
        reference_file.write_text(
            json.dumps(
                dict(
                    signature,
                    check_loss=check_loss,
                    reference_only=args.reference_only,
                ),
                indent=2,
            )
            + "\n"
        )
    dist.barrier()
    reference = json.loads(reference_file.read_text())
    assert all(
        reference[key] == value
        for key, value in signature.items()
        if key != "commit" or args.stage != "balance"
    ), "Mismatched K=1 loss reference"
    args.reference_commit = reference["commit"]
    expected_loss = reference["check_loss"]
    passed = math.isfinite(check_loss) and math.isclose(
        check_loss, expected_loss, rel_tol=2e-5, abs_tol=0
    )
    net.load_state_dict(initial)
    del initial, batch, loss, first
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    optimizer.zero_grad(set_to_none=True)
    finite.fill_(True)
    torch.manual_seed(args.seed + 1000 + args.repeat)
    halo.enabled = True
    torch.cuda.synchronize(device)
    check_state.update(
        check="pass" if passed else "fail",
        check_loss=check_loss,
        reference_loss=expected_loss,
    )
    phase("check_s", phase_start)
    common = dict(
        identity(args),
        **phases,
        check="pass" if passed else "fail",
        check_loss=check_loss,
        reference_loss=expected_loss,
        warmup_steps=5,
        measured_steps=0,
        cached_batches=0 if args.live_loader else 4,
        workers=1,
        prefetch_factor=2,
        batch_size=2,
    )
    if rank == 0:
        print(
            f"check_loss={check_loss:.9g} reference={expected_loss:.9g} check={common['check']}",
            flush=True,
        )
    if args.reference_only or not passed:
        if rank == 0 and not args.reference_only:
            append_row(
                args.output / "bench.csv", dict(common, status="check_fail")
            )
        dist.destroy_process_group()
        return

    def cpu_batch(index):
        return next(iterator) if args.live_loader else cached_batches[index % 4]

    phase_start = time.perf_counter()
    for index in range(5):
        batch = tuple(t.to(device, non_blocking=True) for t in cpu_batch(index))
        halo.reset()
        train_step(batch)
        del batch
        torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.synchronize(device)
    phase("warmup_s", phase_start)
    phase_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    copy_start, copy_end, gpu_start, gpu_end = [
        torch.cuda.Event(enable_timing=True) for _ in range(4)
    ]
    measurements = []
    with (case_dir / f"raw-rank{rank}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=RAW_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for index in range(20):
            halo.reset()
            if args.live_loader:
                begin = time.perf_counter()
                cpu = cpu_batch(5 + index)
                wait_ms = (time.perf_counter() - begin) * 1000
            else:
                cpu = cpu_batch(5 + index)
                wait_ms = 0.0
            copy_start.record()
            batch = tuple(t.to(device, non_blocking=True) for t in cpu)
            copy_end.record()
            if not args.live_loader:
                torch.cuda.synchronize(device)
                dist.barrier()
                begin = time.perf_counter()
            gpu_start.record()
            train_step(batch)
            gpu_end.record()
            gpu_end.synchronize()
            wall_ms = (time.perf_counter() - begin) * 1000
            h2d_ms = copy_start.elapsed_time(copy_end)
            forward, recompute, received = halo.values()
            load_ms = (
                wait_ms if args.live_loader else cached_load_ms[(5 + index) % 4]
            )
            values = [
                wall_ms,
                gpu_start.elapsed_time(gpu_end),
                wait_ms,
                load_ms,
                h2d_ms,
                load_ms + h2d_ms,
                forward,
                recompute,
                received,
            ]
            measurements.append(values)
            writer.writerow(
                dict(
                    identity(args),
                    step=index,
                    batch_slot=(5 + index) % 4,
                    **dict(zip(METRICS, values)),
                )
            )
            handle.flush()
            del batch, cpu
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    phase("timed_s", phase_start)
    values = torch.tensor(measurements, dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    medians = torch.quantile(values, 0.5, dim=0).cpu().tolist()
    low, high = [torch.tensor(peak, device=device) for _ in range(2)]
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)

    # One separate profiled step, after timing. No shape/stack retention.
    phase_start = time.perf_counter()
    halo.enabled = False
    batch = tuple(t.to(device, non_blocking=True) for t in cpu_batch(25))
    torch.cuda.synchronize(device)
    dist.barrier()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as prof:
        train_step(batch)
        torch.cuda.synchronize(device)
    trace = case_dir / f"profile-rank{rank}.json"
    prof.export_chrome_trace(str(trace))
    kernels = kernel_summary(trace)
    (case_dir / f"kernels-rank{rank}.json").write_text(
        json.dumps(dict(identity(args), **kernels), indent=2) + "\n"
    )
    compute = torch.tensor(kernels["compute_kernel_us"] / 1000, device=device)
    compute_low, compute_high = compute.clone(), compute.clone()
    dist.all_reduce(compute_low, op=dist.ReduceOp.MIN)
    dist.all_reduce(compute_high, op=dist.ReduceOp.MAX)
    phase("profile_s", phase_start)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    assert bool(finite), "Nonfinite training loss; results invalid"
    if rank == 0:
        row = dict(
            common,
            **phases,
            status="ok",
            measured_steps=20,
            **dict(zip(METRICS, medians)),
            peak_memory_min_mib=float(low),
            peak_memory_max_mib=float(high),
            compute_only_ms_min=float(compute_low),
            compute_only_ms_max=float(compute_high),
            samples_per_s=2000 / medians[0],
        )
        append_row(args.output / "bench.csv", row)
        print(json.dumps(row), flush=True)
    dist.destroy_process_group()


def launch(args, extra, log):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--monitor-interval=1",
        f"--nproc_per_node={args.K}",
        *extra,
    ]
    with log.open("w") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT
        )
    if result.returncode:
        message = log.read_text()
        if any(
            token in message
            for token in (
                "CUDA_OOM",
                "CUDA out of memory",
                "torch.OutOfMemoryError",
            )
        ):
            return "oom"
        raise RuntimeError(f"Case failed (not OOM); see {log}")
    return "ok"


def summarize_job(output, reference_dir=None, k2_reference_dir=None):
    def groups(root):
        launches = list(csv.DictReader((root / "bench.csv").open()))
        raw = list(csv.DictReader((root / "raw_steps.csv").open()))
        result = {}
        for row in launches:
            key = (
                int(row["K"]),
                row["partitioner"],
                int(row["hidden_dim"]),
                row["checkpointing"],
                int(row["live_loader"]),
            )
            result.setdefault(key, []).append(row)
        stats = {}
        for key, rows in result.items():
            valid = [
                r for r in rows if r["status"] == "ok" and r["check"] == "pass"
            ]
            metrics = {}
            for metric in METRICS:
                arrays = []
                for row in valid:
                    data = [r for r in raw if r["case_id"] == row["case_id"]]
                    assert (
                        len(data) == 20 * key[0]
                    ), f"Incomplete raw case {row['case_id']}"
                    arrays.append(
                        [
                            max(
                                float(r[metric])
                                for r in data
                                if int(r["step"]) == s
                            )
                            for s in range(20)
                        ]
                    )
                if arrays:
                    metrics[metric] = median_summary(arrays)
            stats[key] = (rows, valid, metrics)
        return stats

    current = groups(output)
    baseline = (
        groups(reference_dir)
        if reference_dir and reference_dir != output
        else current
    )
    k2 = (
        groups(k2_reference_dir)
        if k2_reference_dir and k2_reference_dir != output
        else current
    )
    result = []
    for key, (rows, valid, metrics) in current.items():
        K, method, hidden, checkpointing, live = key
        row = {
            name: rows[0][name]
            for name in (
                "stage",
                "K",
                "partitioner",
                "hidden_dim",
                "checkpointing",
                "live_loader",
                "hostname",
                "commit",
                "repeats",
            )
        }
        row["partitions_dir"] = rows[0].get("partitions_dir", str(DEFAULT))
        row["reference_commit"] = rows[0].get(
            "reference_commit", rows[0]["commit"]
        )
        row.update(
            successful_repeats=len(valid),
            oom_repeats=sum(r["status"] == "oom" for r in rows),
            check=(
                "pass"
                if valid
                and all(
                    r["check"] == "pass" for r in rows if r["status"] != "oom"
                )
                else "fail"
            ),
        )
        for metric in METRICS:
            for statistic in ("median", "ci95_low", "ci95_high", "min", "max"):
                row[f"{metric}_{statistic}"] = metrics.get(metric, {}).get(
                    statistic, ""
                )
        row["ci_scope"] = metrics.get("e2e_ms", {}).get(
            "ci_scope", "unavailable"
        )
        row["baseline_K"] = 2 if hidden == 256 else 1
        row["baseline_checkpointing"] = "on"
        basekey = (
            (2, "mtkahypar_km1", 256, "on", live)
            if hidden == 256
            else (1, "original", hidden, "on", live)
        )
        base = (k2 if hidden == 256 else baseline).get(basekey)
        for metric in ("e2e_ms", "gpu_ms"):
            row[f"{metric}_speedup"] = (
                base[2][metric]["median"] / metrics[metric]["median"]
                if base and metric in base[2] and metric in metrics
                else ""
            )
        result.append(row)
    path = output / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(result[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(result)
    return result


def driver(args):
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "dphpc/"], text=True
    )
    if dirty.strip():
        raise RuntimeError(
            "Commit all dphpc/ changes before benchmarking:\n" + dirty
        )
    plan = cases(args.K, args.stage)
    if not plan:
        raise ValueError(f"No {args.stage} cases for K={args.K}")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "bench.csv").exists():
        raise FileExistsError("Use a fresh job directory")
    if args.K != 1 and args.reference_dir is None:
        raise ValueError(
            "Pass the completed K=1 job directory as --reference-dir"
        )
    args.reference_dir = (args.reference_dir or args.output).resolve()
    if args.stage == "balance":
        if args.partitions_dir.name != "1deg-unweighted":
            raise ValueError(
                "Balance stage requires --partitions-dir partitions/1deg-unweighted"
            )
        reference = json.loads(
            (args.reference_dir / "reference-loss-h64.json").read_text()
        )
        args.reference_commit = reference["commit"]
        # This experiment reuses the old K=1 reference intentionally. Its numerical
        # model, adapter, input normalization and library implementation must match.
        subprocess.run(
            [
                "git",
                "diff",
                "--exit-code",
                args.reference_commit,
                "HEAD",
                "--",
                "neural_lam/",
                "dphpc/pnemo/distributed_graph_efm.py",
                "dphpc/pnemo/workload.py",
                "dphpc/pnemo/rank_dataset.py",
                "dphpc/third_party/",
            ],
            check=True,
        )
    metadata = dict(
        stage=args.stage,
        K=args.K,
        hostname=socket.gethostname(),
        git_commit=args.commit,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nccl=torch.cuda.nccl.version(),
        deterministic=False,
        seed=args.seed,
        check_seed=1042,
        warmup_steps=5,
        measured_steps=20,
        cycled_batches=4,
        workers_per_rank=1,
        prefetch_factor=2,
        pin_memory=True,
        batch_size=2,
        ar_steps=4,
        config=str(args.config_path),
        partitions=str(args.partitions_dir),
        reference_commit=getattr(args, "reference_commit", args.commit),
        case_order=plan,
        reference_dir=str(args.reference_dir),
        k2_reference_dir=(
            str(args.k2_reference_dir) if args.k2_reference_dir else None
        ),
        staging=json.loads((args.output / "staging.json").read_text()),
        ci_method="percentile bootstrap: launches and four-step blocks, 4000 resamples",
    )
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    shutil.copyfile(args.config_path, args.output / "config.yaml")
    for command, name in [
        (["hostname"], "hostname.txt"),
        (["nvidia-smi", "-q"], "nvidia-smi.txt"),
        (["nvidia-smi", "topo", "-m"], "topology.txt"),
        (
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pci.bus_id,driver_version",
                "--format=csv",
            ],
            "gpus.csv",
        ),
        (
            ["scontrol", "show", "job", str(os.environ.get("SLURM_JOB_ID"))],
            "slurm.txt",
        ),
        ([sys.executable, "-m", "pip", "freeze"], "packages.txt"),
    ]:
        with (args.output / name).open("w") as handle:
            subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
    from .static_metrics import write_metrics

    static_path = args.output / "partition_metrics.csv"
    write_metrics(
        static_path,
        K=args.K if args.K > 1 else None,
        root=args.partitions_dir,
        methods=tuple(dict.fromkeys(c["partitioner"] for c in plan))
        if args.K > 1 else METHODS,
    )
    statics = list(csv.DictReader(static_path.open()))
    wide = (
        [
            dict(
                row,
                hidden_dim=h,
                **{
                    key: str(int(row[key]) * h // 64)
                    for key in (
                        "bytes_received_per_exchange",
                        "bytes_sent_per_exchange",
                    )
                },
            )
            for row in statics
            if row["partitioner"] == "mtkahypar_km1"
            for h in (128, 256)
        ]
        if args.stage == "full"
        else []
    )
    statics = [dict(row, hidden_dim=64) for row in statics] + wide
    for row in statics:
        row.update(
            hostname=socket.gethostname(),
            commit=args.commit,
            partitions_dir=str(args.partitions_dir),
        )
    with (args.output / "partition_metrics.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(statics[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(statics)

    def worker_command(case, reference_only=False):
        extra = [
            "-m",
            "dphpc.pnemo.bench",
            "--worker",
            "--K",
            str(args.K),
            "--stage",
            args.stage,
            "--output",
            str(args.output),
            "--config-path",
            str(args.config_path),
            "--reference-dir",
            str(args.reference_dir),
            "--partitions-dir",
            str(args.partitions_dir),
            "--seed",
            str(args.seed),
        ]
        for name in (
            "partitioner",
            "hidden_dim",
            "checkpointing",
            "repeat",
            "repeats",
        ):
            extra += ["--" + name.replace("_", "-"), str(case[name])]
        if case["live_loader"]:
            extra += ["--live-loader"]
        if reference_only:
            extra += ["--reference-only"]
        return extra

    # H256 has no K=1 performance case, but still needs the same-H K=1 loss.
    if args.K == 1 and args.stage == "full":
        ref = dict(
            partitioner="original",
            hidden_dim=256,
            checkpointing="on",
            live_loader=False,
            repeat=0,
            repeats=1,
        )
        status = launch(
            args, worker_command(ref, True), args.output / "reference-h256.log"
        )
        if status != "ok":
            raise RuntimeError(
                "K=1 H256 forward-only loss reference OOM; no unverified full sweep"
            )
    for case in plan:
        for name, value in case.items():
            setattr(args, name, value)
        folder = args.output / "cases" / case_name(args)
        folder.mkdir(parents=True, exist_ok=True)
        print(f"CASE {case_name(args)} repeats={args.repeats}", flush=True)
        status = launch(args, worker_command(case), folder / "worker.log")
        if status == "oom":
            progress = (
                json.loads((folder / "phases.json").read_text())
                if (folder / "phases.json").exists()
                else {}
            )
            row = dict(
                identity(args),
                **{
                    k: progress[k]
                    for k in (*PHASES, "parameters", "reference_commit")
                    if k in progress
                },
                status="oom",
                check=progress.get("check", "not_run"),
                check_loss=progress.get("check_loss", ""),
                reference_loss=progress.get("reference_loss", ""),
                measured_steps=0,
                warmup_steps=5,
                cached_batches=0 if args.live_loader else 4,
                workers=1,
                prefetch_factor=2,
                batch_size=2,
            )
            append_row(args.output / "bench.csv", row)
        raw = []
        for path in sorted(folder.glob("raw-rank*.csv")):
            raw += list(csv.DictReader(path.open()))
        if raw:
            target = args.output / "raw_steps.csv"
            exists = target.exists()
            with target.open("a", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=RAW_FIELDS, lineterminator="\n"
                )
                if not exists:
                    writer.writeheader()
                writer.writerows(raw)
        rows = list(csv.DictReader((args.output / "bench.csv").open()))
        if rows[-1]["check"] == "fail":
            raise RuntimeError(
                "Initial loss check failed; see bench.csv; benchmark aborted"
            )
    rows = list(csv.DictReader((args.output / "bench.csv").open()))
    assert len(rows) == len(plan) and {r["case_id"] for r in rows} == {
        case_name(argparse.Namespace(**c, partitions_dir=args.partitions_dir))
        for c in plan
    }
    if not (args.output / "raw_steps.csv").exists():
        with (args.output / "raw_steps.csv").open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=RAW_FIELDS).writeheader()
    summarize_job(args.output, args.reference_dir, args.k2_reference_dir)
    print(
        f"Complete: {len(rows)} launch rows, {args.output/'summary.csv'}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, choices=(1, 2, 4))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stage", choices=("full", "balance"), default="full"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--k2-reference-dir", type=Path)
    parser.add_argument("--partitioner", choices=("original", *METHODS))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--checkpointing", choices=("on", "off"), default="on")
    parser.add_argument("--live-loader", action="store_true")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-path", type=Path, default=CONFIG)
    parser.add_argument("--partitions-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    args.partitions_dir = args.partitions_dir.resolve()
    if args.dry_run:
        for stage in (
            (args.stage,)
            if "--stage" in sys.argv
            else ("full", "balance")
        ):
            for K in (1, 2, 4):
                if args.K is not None and args.K != K:
                    continue
                plan = cases(K, stage)
                print(
                    f"{stage} K={K}: expected launch rows={len(plan)}, raw rows={20*K*len(plan)}, summary rows={len({(c['partitioner'],c['hidden_dim'],c['checkpointing'],c['live_loader']) for c in plan})}"
                )
                for c in plan:
                    print(
                        "  "
                        + json.dumps(
                            dict(
                                c,
                                partitions_dir=str(args.partitions_dir),
                                case_id=case_name(
                                    argparse.Namespace(
                                        **c, partitions_dir=args.partitions_dir
                                    )
                                ),
                            )
                        )
                    )
        return
    if args.output is None:
        parser.error("--output is required")
    args.output = args.output.resolve()
    if args.reference_dir:
        args.reference_dir = args.reference_dir.resolve()
    if args.k2_reference_dir:
        args.k2_reference_dir = args.k2_reference_dir.resolve()
    if args.K is None:
        parser.error("--K is required")
    (
        args.processor_layers,
        args.batch_size,
        args.ar_steps,
        args.graph,
        args.lr,
    ) = (2, 2, 4, GRAPH, 0.001)
    args.commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if args.worker:
        args.reference_dir = (args.reference_dir or args.output).resolve()
    try:
        worker(args) if args.worker else driver(args)
    except torch.OutOfMemoryError:
        print("CUDA_OOM", flush=True)
        raise


if __name__ == "__main__":
    main()
