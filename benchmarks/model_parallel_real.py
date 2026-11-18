"""Real-data strong-scaling benchmark for model-parallel GraphEFM.

The default workload matches the max-branch H64/L2/B2/AR4/FP32 baseline,
including per-step gradient checkpointing. The loader uses the original
global-model AR4 scripts' eight persistent workers and prefetch factor two.
Forecast feedback remains rank-local throughout the differentiable rollout.
"""

from __future__ import annotations

# Standard library
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Third-party
import numpy as np
import torch
import torch.distributed as dist
import xarray as xr

# First-party
from neural_lam.config import load_config_and_datastores
from neural_lam.datastore import MDPDatastore
from neural_lam.models.distributed_graph_efm import (
    build_partition_from_graph_efm,
    distributed_core_from_graph_efm,
)
from neural_lam.models.distributed_graph_efm_training import (
    DistributedGraphEFMTraining,
)
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.weather_dataset import WeatherDataset


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(rank, message):
    if rank == 0:
        print(f"[{_utc_now()}] {message}", flush=True)


def _summary(values):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction):
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": ordered[0],
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _run_git(*arguments):
    return subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _git_metadata():
    revision = _run_git("rev-parse", "HEAD")
    tracked_worktree = _run_git("diff", "--quiet")
    tracked_index = _run_git("diff", "--cached", "--quiet")
    untracked = _run_git("ls-files", "--others", "--exclude-standard")
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "tracked_dirty": (
            tracked_worktree.returncode != 0 or tracked_index.returncode != 0
        ),
        "untracked_file_count": (
            len(untracked.stdout.splitlines())
            if untracked.returncode == 0
            else None
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark model-parallel GraphEFM on real Njord data"
    )
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--graph", default="global_cluster_graph")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--processor-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="batches prefetched by each DataLoader worker",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=0,
        help="print every Nth step; 0 disables step progress output",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latitude-cut", type=float, default=-5.0)
    parser.add_argument("--longitude-seam", type=float, default=120.0)
    parser.add_argument(
        "--data-mode",
        choices=("rank-zero-broadcast", "rank-local"),
        default="rank-zero-broadcast",
    )
    parser.add_argument(
        "--partitioned-data-root",
        type=Path,
        help=(
            "Directory containing geographic_rcb_wsN/rank_R stores; "
            "required for --data-mode=rank-local"
        ),
    )
    parser.add_argument(
        "--partition-strategy",
        choices=("auto", "geographic", "latitude", "random", "file"),
        default="auto",
    )
    parser.add_argument(
        "--partition-file",
        type=Path,
        help=(
            "Owners per grid node and mesh level (.npz, see "
            "load_partition_file); required for --partition-strategy=file"
        ),
    )
    parser.add_argument("--ar-steps", type=int, default=4)
    args = parser.parse_args()
    if args.warmup_steps < 0 or args.steps < 1:
        parser.error("warmup-steps must be >=0 and steps must be >=1")
    if args.progress_interval < 0:
        parser.error("progress-interval must be >=0")
    if args.num_workers < 0:
        parser.error("num-workers must be >=0")
    if args.prefetch_factor < 1:
        parser.error("prefetch-factor must be >=1")
    if args.ar_steps < 1:
        parser.error("ar-steps must be positive")
    if (args.partition_strategy == "file") != (args.partition_file is not None):
        parser.error(
            "--partition-file is required for, and only used by, "
            "--partition-strategy=file"
        )
    if args.partition_file is not None and not args.partition_file.is_file():
        parser.error(f"partition file not found: {args.partition_file}")
    if args.partition_strategy == "file" and args.data_mode == "rank-local":
        parser.error(
            "rank-local data shards are built for the geographic partition; "
            "use --data-mode=rank-zero-broadcast with a partition file"
        )
    return args


def _model_args(args):
    """Build the normal GraphEFM arguments for the controlled workload."""

    return SimpleNamespace(
        ar_steps_train=args.ar_steps,
        batch_size=args.batch_size,
        crps_alpha=0.95,
        crps_weight=0.0,
        encoder_processor_layers=1,
        ensemble_size=5,
        epochs=100,
        grad_checkpointing=True,
        graph=args.graph,
        hidden_dim=args.hidden_dim,
        hidden_dim_edge=args.hidden_dim,
        hidden_dim_grid=args.hidden_dim,
        hidden_dim_mesh_nodes=args.hidden_dim,
        hidden_layers=1,
        input_steps=2,
        kl_beta=1.0,
        latent_dim=None,
        learn_prior=1,
        loss="wmse",
        lr=args.lr,
        metrics_watch=[],
        min_lr=args.lr,
        muon_exclude_patterns=[],
        muon_momentum=0.95,
        muon_weight_decay=0.0,
        n_example_pred=0,
        num_future_atmosphere_steps=0,
        num_future_boundary_steps=0,
        num_future_forcing_steps=0,
        num_latents_plot=4,
        num_past_atmosphere_steps=2,
        num_past_boundary_steps=2,
        num_past_forcing_steps=2,
        output_std=False,
        optimizer="adamw",
        prior_dist="isotropic",
        prior_processor_layers=1,
        processor_layers=args.processor_layers,
        restore_opt=False,
        sample_obs_noise=0,
        seed=args.seed,
        soap_betas=[0.95, 0.95],
        soap_muon_betas=[0.95, 0.98],
        soap_muon_disable_sqrt=False,
        soap_precondition_frequency=10,
        soap_weight_decay=0.0,
        train_distr_crps=False,
        val_steps_to_log=[1, 2, 3],
        var_leads_metrics_watch={},
    )


def _make_dataset(
    args,
    model_args,
    config,
    datastore,
    datastore_boundary,
    datastore_atmosphere,
    statistics_datastore=None,
    statistics_datastore_boundary=None,
    statistics_datastore_atmosphere=None,
):
    return WeatherDataset(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        split="train",
        ar_steps=args.ar_steps,
        input_steps=model_args.input_steps,
        standardize=True,
        num_past_forcing_steps=model_args.num_past_forcing_steps,
        num_future_forcing_steps=model_args.num_future_forcing_steps,
        current_forcing_step=True,
        current_boundary_step=True,
        current_atmosphere_step=True,
        num_past_boundary_steps=model_args.num_past_boundary_steps,
        num_future_boundary_steps=model_args.num_future_boundary_steps,
        num_past_atmosphere_steps=model_args.num_past_atmosphere_steps,
        num_future_atmosphere_steps=model_args.num_future_atmosphere_steps,
        use_atmosphere_g2m=False,
        density_channel=getattr(config.training, "density_channel", None),
        statistics_datastore=statistics_datastore,
        statistics_datastore_boundary=statistics_datastore_boundary,
        statistics_datastore_atmosphere=statistics_datastore_atmosphere,
    )


def _load_workload(args, include_global_dataset=True):
    loaded = load_config_and_datastores(config_path=str(args.config_path))
    (
        config,
        datastore,
        datastore_boundary,
        datastore_atmosphere,
        statistics_datastore,
        statistics_datastore_boundary,
        statistics_datastore_atmosphere,
    ) = loaded
    model_args = _model_args(args)
    dataset = None
    if include_global_dataset:
        dataset = _make_dataset(
            args,
            model_args,
            config,
            datastore,
            datastore_boundary,
            datastore_atmosphere,
            statistics_datastore,
            statistics_datastore_boundary,
            statistics_datastore_atmosphere,
        )
    model = GraphEFM(
        model_args,
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        statistics_datastore=statistics_datastore,
        statistics_datastore_boundary=statistics_datastore_boundary,
        statistics_datastore_atmosphere=statistics_datastore_atmosphere,
    )
    return dataset, model, config


def _load_rank_local_dataset(args, config, rank, world_size):
    if args.partitioned_data_root is None:
        raise ValueError(
            "--partitioned-data-root is required for rank-local loading"
        )
    rank_root = (
        args.partitioned_data_root
        / f"geographic_rcb_ws{world_size}"
        / f"rank_{rank}"
    )
    ocean_store = rank_root / "ocean.zarr"
    atmosphere_store = rank_root / "atmosphere.zarr"
    config_root = args.config_path.parent
    ocean_config = config_root / config.datastore.config_path
    ocean = _open_rank_local_datastore(ocean_config, ocean_store)

    if config.datastore_boundary is not None:
        raise NotImplementedError(
            "rank-local benchmark does not yet support boundary datastores"
        )
    atmosphere = None
    if config.datastore_atmosphere is not None:
        atmosphere_config = (
            config_root / config.datastore_atmosphere.config_path
        )
        atmosphere = _open_rank_local_datastore(
            atmosphere_config,
            atmosphere_store,
        )

    dataset = _make_dataset(
        args,
        _model_args(args),
        config,
        ocean,
        None,
        atmosphere,
        statistics_datastore=ocean,
        statistics_datastore_atmosphere=atmosphere,
    )
    return dataset, rank_root


def _open_rank_local_datastore(config_path, store_path):
    """Reuse the MDP schema while swapping in one benchmark-only shard."""

    if not store_path.is_dir():
        raise FileNotFoundError(f"rank-local Zarr store not found: {store_path}")
    datastore = MDPDatastore(config_path)
    datastore._ds = xr.open_zarr(store_path, consolidated=True)
    datastore._available_splits = list(
        datastore._ds.splits.split_name.values
    )
    return datastore


def _data_loader_options(num_workers, prefetch_factor, device):
    options = {
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        options.update(
            {
                "multiprocessing_context": "spawn",
                "prefetch_factor": prefetch_factor,
            }
        )
    return options


class RankZeroBatchStream:
    """Load each real batch once on rank zero and broadcast it to all ranks."""

    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
        prefetch_factor,
        rank,
        device,
        seed=42,
    ):
        self.rank = rank
        self.device = device
        self.metadata = None
        self.iterator = None
        self.loader = None
        self.last_timings = {}
        if rank == 0:
            self.loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                generator=torch.Generator().manual_seed(seed),
                **_data_loader_options(
                    num_workers,
                    prefetch_factor,
                    device,
                ),
            )
            self.iterator = iter(self.loader)

    def _next_cpu_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

    @staticmethod
    def _dtype_from_name(name):
        return getattr(torch, name.removeprefix("torch."))

    def next(self):
        read_start = time.perf_counter()
        if self.rank == 0:
            cpu_batch = tuple(self._next_cpu_batch())
            read_end = time.perf_counter()
            transfer_start = read_end
            device_batch = tuple(
                tensor.to(self.device, non_blocking=True)
                for tensor in cpu_batch
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            transfer_end = time.perf_counter()
            metadata = [
                (tuple(tensor.shape), str(tensor.dtype))
                for tensor in device_batch
            ]
        else:
            read_end = read_start
            transfer_start = read_start
            transfer_end = read_start
            device_batch = None
            metadata = None

        broadcast_start = time.perf_counter()
        container = [metadata]
        dist.broadcast_object_list(container, src=0, device=self.device)
        self.metadata = container[0]

        if self.rank != 0:
            device_batch = tuple(
                torch.empty(
                    shape,
                    dtype=self._dtype_from_name(dtype_name),
                    device=self.device,
                )
                for shape, dtype_name in self.metadata
            )

        for tensor in device_batch:
            if tensor.numel():
                dist.broadcast(tensor, src=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        broadcast_end = time.perf_counter()
        self.last_timings = {
            "data_read_ms": (read_end - read_start) * 1000,
            "host_to_device_ms": (transfer_end - transfer_start) * 1000,
            "batch_broadcast_ms": (
                broadcast_end - broadcast_start
            )
            * 1000,
        }
        return device_batch


class RankLocalBatchStream:
    """Load the same temporal samples from each rank's spatial Zarr shard."""

    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
        prefetch_factor,
        device,
        seed=42,
    ):
        self.device = device
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed),
            **_data_loader_options(
                num_workers,
                prefetch_factor,
                device,
            ),
        )
        self.iterator = iter(self.loader)
        self.last_timings = {}

    def _next_cpu_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def next(self):
        read_start = time.perf_counter()
        cpu_batch = tuple(self._next_cpu_batch())
        read_end = time.perf_counter()
        transfer_start = read_end
        device_batch = tuple(
            tensor.to(self.device, non_blocking=True) for tensor in cpu_batch
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        transfer_end = time.perf_counter()
        self.last_timings = {
            "data_read_ms": (read_end - read_start) * 1000,
            "host_to_device_ms": (transfer_end - transfer_start) * 1000,
            "batch_broadcast_ms": 0.0,
        }
        return device_batch


def _synchronize_data_phase(device, world_size):
    """Align ranks before forward and return synchronization wait time."""

    wait_start = time.perf_counter()
    if world_size > 1:
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - wait_start) * 1000


def _partition_file_metadata(path):
    """Path, name and checksum of the partition file, so results name it."""

    if path is None:
        return None
    with np.load(path) as data:
        name = str(data["name"]) if "name" in data.files else None
        graph = (
            str(data["graph_sha256"]) if "graph_sha256" in data.files else None
        )
    return {
        "path": str(path),
        "name": name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        # Checked against the model's graph when present; files without it
        # are only checked for sizes and ranks
        "graph_sha256": graph,
        "graph_verified": graph is not None,
    }


def _partition_strategy(requested, world_size):
    if requested != "auto":
        return requested
    return "geographic" if world_size in (1, 2, 4) else "random"


def _build_sharded_training_modules(model, args, rank, world_size, device):
    strategy = _partition_strategy(args.partition_strategy, world_size)
    if strategy == "latitude" and world_size != 2:
        raise ValueError("latitude partitioning currently requires two GPUs")
    cpu_partition = build_partition_from_graph_efm(
        model,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        strategy=strategy,
        latitude_cut=args.latitude_cut,
        longitude_seam=args.longitude_seam,
        partition_file=args.partition_file,
    )
    partition = cpu_partition.to(device)
    core = distributed_core_from_graph_efm(
        model,
        partition,
        device=device,
    )

    # Keep the real embedding, clamping, loss and static-feature frontend, but
    # remove the unpartitioned core before moving that frontend to the GPU.
    # This avoids holding both complete and sharded GNN cores on every rank.
    model.prior_model = None
    model.encoder = None
    model.decoder = None
    model.to(device)
    model.train()
    core.train()
    return model, core, partition, strategy


def _forward_local_loss(
    training_model,
    batch,
    generator,
    batch_is_local=False,
):
    output = training_model(
        batch, generator=generator, batch_is_local=batch_is_local
    )
    return (
        output["loss"],
        output["negative_likelihood"].detach(),
        output["kl"].detach(),
    )


def _gradient_buckets(parameters, bucket_bytes=25 * 1024 * 1024):
    buckets = []
    current = []
    current_bytes = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_bytes = parameter.grad.numel() * parameter.grad.element_size()
        if current and current_bytes + grad_bytes > bucket_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(parameter)
        current_bytes += grad_bytes
    if current:
        buckets.append(current)
    return buckets


def _synchronize_gradients(parameters, world_size):
    if world_size == 1:
        return 0, 0
    total_bytes = 0
    buckets = _gradient_buckets(parameters)
    for bucket in buckets:
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for parameter in bucket:
            count = parameter.grad.numel()
            parameter.grad.copy_(
                flat[offset : offset + count].view_as(parameter.grad)
            )
            offset += count
        total_bytes += flat.numel() * flat.element_size()
    return total_bytes, len(buckets)


def _global_scalar(local_value):
    value = local_value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return float(value.item())


def _global_metrics(*local_values):
    """Sum scalar diagnostics with one collective instead of one per value."""

    values = torch.stack([value.detach() for value in local_values])
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return tuple(float(value.item()) for value in values)


def _batch_fingerprint(batch):
    return {
        "shapes": [list(tensor.shape) for tensor in batch],
        "dtypes": [str(tensor.dtype) for tensor in batch],
        "sums": [
            float(tensor.double().sum().item()) if tensor.numel() else 0.0
            for tensor in batch
        ],
    }


def _distributed_batch_fingerprint(batch, batch_is_local):
    """Describe one global sample independent of the number of partitions."""

    fingerprint = _batch_fingerprint(batch)
    if not batch_is_local:
        return fingerprint

    spatial_tensors = {0, 1, 2, 4}
    global_shapes = []
    global_sums = []
    for index, tensor in enumerate(batch):
        shape = list(tensor.shape)
        value = torch.tensor(
            float(tensor.double().sum().item()) if tensor.numel() else 0.0,
            dtype=torch.float64,
            device=tensor.device,
        )
        if index in spatial_tensors:
            node_count = torch.tensor(
                shape[-2],
                dtype=torch.long,
                device=tensor.device,
            )
            dist.all_reduce(node_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            shape[-2] = int(node_count.item())
        elif tensor.numel() == 0 and len(shape) >= 2:
            shape[-2] = 0
        global_shapes.append(shape)
        global_sums.append(float(value.item()))
    return {
        "shapes": global_shapes,
        "dtypes": fingerprint["dtypes"],
        "sums": global_sums,
    }


def _dataset_metadata(dataset):
    datastore = dataset.datastore
    source = getattr(datastore, "_ds", None)
    benchmark_attributes = {}
    if source is not None:
        for key, value in source.attrs.items():
            if not key.startswith("benchmark_"):
                continue
            if hasattr(value, "item"):
                value = value.item()
            if not isinstance(value, (str, int, float, bool, type(None))):
                value = str(value)
            benchmark_attributes[key] = value

    time_values = dataset.da_state.time.values
    return {
        "root": str(datastore.root_path),
        "train_samples": len(dataset),
        "train_time_steps": int(dataset.da_state.sizes["time"]),
        "train_time_start": str(time_values[0]),
        "train_time_end": str(time_values[-1]),
        "state_variables": datastore.get_vars_names("state"),
        "forcing_variables": datastore.get_vars_names("forcing"),
        "source_benchmark_attributes": benchmark_attributes,
    }


def _partition_metadata(partition):
    relations = {
        "g2m": (partition.g2m,),
        "m2g": (partition.m2g,),
        "m2m": partition.m2m,
        "mesh_up": partition.mesh_up,
        "mesh_down": partition.mesh_down,
    }
    return {
        "rank": partition.rank,
        "owned_full_grid_nodes": int(partition.full_grid_nodes.numel()),
        "owned_interior_grid_nodes": int(
            partition.interior_grid_nodes.numel()
        ),
        "owned_mesh_nodes_by_level": [
            int(nodes.numel()) for nodes in partition.mesh_nodes
        ],
        "relations": {
            name: [
                {
                    "owned_edges": int(part.edge_ids.numel()),
                    "halo_nodes": int(part.halo_nodes.numel()),
                }
                for part in parts
            ]
            for name, parts in relations.items()
        },
    }


def _hardware_metadata(device, rank):
    props = torch.cuda.get_device_properties(device)
    return {
        "rank": rank,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": {
            "index": device.index,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
        },
    }


def _step_record(
    training_model,
    optimizer,
    parameters,
    batch,
    generator,
    world_size,
    batch_is_local=False,
    progress_callback=None,
):
    optimizer.zero_grad(set_to_none=True)
    events = {
        name: torch.cuda.Event(enable_timing=True)
        for name in (
            "start",
            "forward_end",
            "backward_end",
            "gradient_sync_end",
            "optimizer_end",
            "metric_reduce_end",
        )
    }
    if progress_callback is not None:
        progress_callback("forward started")
    events["start"].record()
    loss, local_nll, local_kl = _forward_local_loss(
        training_model,
        batch,
        generator,
        batch_is_local=batch_is_local,
    )
    events["forward_end"].record()
    if progress_callback is not None:
        torch.cuda.synchronize()
        duration_ms = events["start"].elapsed_time(events["forward_end"])
        progress_callback(
            f"forward completed in {duration_ms:.1f}ms"
        )
        progress_callback("backward started")
    loss.backward()
    events["backward_end"].record()
    if progress_callback is not None:
        torch.cuda.synchronize()
        duration_ms = events["forward_end"].elapsed_time(
            events["backward_end"]
        )
        progress_callback(
            f"backward completed in {duration_ms:.1f}ms"
        )
        progress_callback("gradient synchronization started")
    synchronized_bytes, all_reduce_calls = _synchronize_gradients(
        parameters,
        world_size,
    )
    events["gradient_sync_end"].record()
    if progress_callback is not None:
        torch.cuda.synchronize()
        duration_ms = events["backward_end"].elapsed_time(
            events["gradient_sync_end"]
        )
        progress_callback(
            f"gradient synchronization completed in {duration_ms:.1f}ms"
        )
        progress_callback("optimizer started")
    optimizer.step()
    events["optimizer_end"].record()
    if progress_callback is not None:
        torch.cuda.synchronize()
        duration_ms = events["gradient_sync_end"].elapsed_time(
            events["optimizer_end"]
        )
        progress_callback(
            f"optimizer completed in {duration_ms:.1f}ms"
        )
        progress_callback("metric reduction started")
    global_loss, global_nll, global_kl = _global_metrics(
        loss,
        local_nll,
        local_kl,
    )
    events["metric_reduce_end"].record()
    torch.cuda.synchronize()
    if progress_callback is not None:
        duration_ms = events["optimizer_end"].elapsed_time(
            events["metric_reduce_end"]
        )
        progress_callback(
            f"metric reduction completed in {duration_ms:.1f}ms"
        )

    return {
        "batch_samples": int(batch[0].shape[0]),
        "forward_ms": events["start"].elapsed_time(events["forward_end"]),
        "backward_ms": events["forward_end"].elapsed_time(
            events["backward_end"]
        ),
        "gradient_sync_ms": events["backward_end"].elapsed_time(
            events["gradient_sync_end"]
        ),
        "optimizer_ms": events["gradient_sync_end"].elapsed_time(
            events["optimizer_end"]
        ),
        "compute_ms": events["start"].elapsed_time(events["optimizer_end"]),
        "metric_reduce_ms": events["optimizer_end"].elapsed_time(
            events["metric_reduce_end"]
        ),
        "step_ms": events["start"].elapsed_time(events["metric_reduce_end"]),
        "loss": global_loss,
        "negative_likelihood": global_nll,
        "kl": global_kl,
        "gradient_sync_bytes": synchronized_bytes,
        "gradient_all_reduce_calls": all_reduce_calls,
        "memory_allocated_bytes": torch.cuda.memory_allocated(),
        "memory_reserved_bytes": torch.cuda.memory_reserved(),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _aggregate_records(rank_payloads):
    record_count = min(len(payload["records"]) for payload in rank_payloads)
    time_keys = (
        "data_distribution_ms",
        "data_read_ms",
        "host_to_device_ms",
        "batch_broadcast_ms",
        "data_rank_wait_ms",
        "forward_ms",
        "backward_ms",
        "gradient_sync_ms",
        "optimizer_ms",
        "compute_ms",
        "metric_reduce_ms",
        "step_ms",
        "end_to_end_ms",
    )
    memory_keys = (
        "memory_allocated_bytes",
        "memory_reserved_bytes",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
    )
    aggregate = []
    for index in range(record_count):
        records = [payload["records"][index] for payload in rank_payloads]
        item = {"measurement_step": index}
        for key in time_keys + memory_keys:
            item[key] = max(record[key] for record in records)
        for key in (
            "batch_samples",
            "loss",
            "negative_likelihood",
            "kl",
            "gradient_sync_bytes",
            "gradient_all_reduce_calls",
        ):
            item[key] = records[0][key]
        aggregate.append(item)
    return aggregate


def _result_summary(records, batch_size, measurement_window_ms):
    summary = {
        key: _summary([record[key] for record in records])
        for key in (
            "data_distribution_ms",
            "data_read_ms",
            "host_to_device_ms",
            "batch_broadcast_ms",
            "data_rank_wait_ms",
            "forward_ms",
            "backward_ms",
            "gradient_sync_ms",
            "optimizer_ms",
            "compute_ms",
            "metric_reduce_ms",
            "step_ms",
            "end_to_end_ms",
            "loss",
        )
    }
    compute_mean = summary["compute_ms"]["mean"]
    end_to_end_mean = summary["end_to_end_ms"]["mean"]
    sample_count = sum(record["batch_samples"] for record in records)
    mean_batch_size = sample_count / len(records)
    summary.update(
        {
            "global_batch_size": batch_size,
            "measured_samples": sample_count,
            "compute_samples_per_second": mean_batch_size * 1000 / compute_mean,
            "end_to_end_samples_per_second": (
                mean_batch_size * 1000 / end_to_end_mean
            ),
            "observed_window_ms": measurement_window_ms,
            "observed_samples_per_second": (
                sample_count * 1000 / measurement_window_ms
            ),
            "peak_memory_allocated_bytes_per_gpu": max(
                record["peak_memory_allocated_bytes"] for record in records
            ),
            "peak_memory_reserved_bytes_per_gpu": max(
                record["peak_memory_reserved_bytes"] for record in records
            ),
        }
    )
    return summary


def main():
    args = _parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("the official real-data benchmark requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    batch_is_local = args.data_mode == "rank-local"
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    started_at = _utc_now()
    benchmark_start = time.perf_counter()
    initialization_timings = {
        "data_staging_ms": float(os.environ.get("NJORD_STAGING_MS", 0.0))
    }
    _progress(
        rank,
        f"distributed runtime ready: rank={rank}, world_size={world_size}, "
        f"device={device}",
    )

    try:
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        phase_start = time.perf_counter()
        _progress(rank, f"loading real workload from {args.config_path}")
        dataset, reference_model, config = _load_workload(
            args,
            include_global_dataset=not batch_is_local,
        )
        initialization_timings["workload_load_ms"] = (
            time.perf_counter() - phase_start
        ) * 1000
        _progress(
            rank,
            "real workload loaded in "
            f"{initialization_timings['workload_load_ms'] / 1000:.2f}s; "
            f"global_dataset_loaded={dataset is not None}",
        )

        phase_start = time.perf_counter()
        _progress(rank, "building graph partition and sharded GNN core")
        model, core, partition, strategy = _build_sharded_training_modules(
            reference_model,
            args,
            rank,
            world_size,
            device,
        )
        del reference_model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initialization_timings["sharded_model_build_ms"] = (
            time.perf_counter() - phase_start
        ) * 1000
        _progress(
            rank,
            "sharded GNN core ready in "
            f"{initialization_timings['sharded_model_build_ms'] / 1000:.2f}s; "
            f"partition_strategy={strategy}",
        )

        rank_data_root = None
        if batch_is_local:
            phase_start = time.perf_counter()
            dataset, rank_data_root = _load_rank_local_dataset(
                args,
                config,
                rank,
                world_size,
            )
            shard_nodes = torch.as_tensor(
                dataset.datastore._ds.global_surface_index.values,
                dtype=torch.long,
            )
            if not torch.equal(
                shard_nodes,
                partition.interior_grid_nodes.detach().cpu(),
            ):
                raise ValueError(
                    "rank-local Zarr node order does not match graph partition"
                )
            initialization_timings["rank_local_dataset_load_ms"] = (
                time.perf_counter() - phase_start
            ) * 1000
            rank_dataset_seconds = (
                initialization_timings["rank_local_dataset_load_ms"] / 1000
            )
            _progress(
                rank,
                "rank-local dataset ready in "
                f"{rank_dataset_seconds:.2f}s; "
                f"train_samples={len(dataset)}",
            )

        phase_start = time.perf_counter()
        training_model = DistributedGraphEFMTraining(model, core, partition)
        parameters = list(training_model.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        if batch_is_local:
            stream = RankLocalBatchStream(
                dataset,
                args.batch_size,
                args.num_workers,
                args.prefetch_factor,
                device,
                seed=args.seed,
            )
        else:
            stream = RankZeroBatchStream(
                dataset,
                args.batch_size,
                args.num_workers,
                args.prefetch_factor,
                rank,
                device,
                seed=args.seed,
            )
        initialization_timings["optimizer_and_stream_ms"] = (
            time.perf_counter() - phase_start
        ) * 1000
        _progress(
            rank,
            "optimizer and batch stream ready in "
            f"{initialization_timings['optimizer_and_stream_ms'] / 1000:.2f}s",
        )

        # Partition-independent preflight loss on the first physical batch.
        phase_start = time.perf_counter()
        _progress(rank, "preflight: distributing first physical batch")
        dist.barrier()
        preflight_batch = stream.next()
        _synchronize_data_phase(device, world_size)
        initialization_timings["preflight_batch_ms"] = (
            time.perf_counter() - phase_start
        ) * 1000
        _progress(
            rank,
            "preflight batch ready in "
            f"{initialization_timings['preflight_batch_ms'] / 1000:.2f}s",
        )
        fingerprint = _distributed_batch_fingerprint(
            preflight_batch,
            batch_is_local,
        )
        if rank != 0:
            fingerprint = None
        preflight_generator = torch.Generator(device=device).manual_seed(
            args.seed + 100_000
        )
        phase_start = time.perf_counter()
        _progress(rank, "preflight: forward pass started")
        with torch.no_grad():
            preflight_local, _, _ = _forward_local_loss(
                training_model,
                preflight_batch,
                preflight_generator,
                batch_is_local=batch_is_local,
            )
        preflight_loss = _global_scalar(preflight_local)
        torch.cuda.synchronize()
        initialization_timings["preflight_forward_ms"] = (
            time.perf_counter() - phase_start
        ) * 1000
        initialization_timings["total_before_training_ms"] = (
            time.perf_counter() - benchmark_start
        ) * 1000
        _progress(
            rank,
            "preflight forward completed in "
            f"{initialization_timings['preflight_forward_ms'] / 1000:.2f}s; "
            f"global_loss={preflight_loss:.6g}",
        )

        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        rank_records = []
        measurement_window_start = None
        total_steps = args.warmup_steps + args.steps
        for step in range(total_steps):
            measurement_step = step - args.warmup_steps
            is_measurement = measurement_step >= 0
            phase_name = "measurement" if is_measurement else "warmup"
            phase_step = measurement_step if is_measurement else step
            phase_total = args.steps if is_measurement else args.warmup_steps
            show_progress = (
                args.progress_interval > 0
                and (
                    phase_step % args.progress_interval == 0
                    or phase_step + 1 == phase_total
                )
            )
            if show_progress:
                _progress(
                    rank,
                    f"{phase_name} step {phase_step + 1}/{phase_total} started",
                )
            if step == args.warmup_steps:
                dist.barrier()
                torch.cuda.synchronize()
                measurement_window_start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
            data_start = time.perf_counter()
            batch = stream.next()
            data_rank_wait_ms = _synchronize_data_phase(
                device,
                world_size,
            )
            data_distribution_ms = (
                time.perf_counter() - data_start
            ) * 1000

            progress_callback = None
            if show_progress and rank == 0:
                step_prefix = (
                    f"{phase_name} step {phase_step + 1}/{phase_total}: "
                )

                def progress_callback(message):
                    _progress(rank, step_prefix + message)

            record = _step_record(
                training_model,
                optimizer,
                parameters,
                batch,
                generator,
                world_size,
                batch_is_local=batch_is_local,
                progress_callback=progress_callback,
            )
            record["data_distribution_ms"] = data_distribution_ms
            record["data_rank_wait_ms"] = data_rank_wait_ms
            record.update(stream.last_timings)
            record["end_to_end_ms"] = (
                data_distribution_ms + record["step_ms"]
            )
            if step >= args.warmup_steps:
                record["measurement_step"] = step - args.warmup_steps
                rank_records.append(record)
            if show_progress:
                _progress(
                    rank,
                    f"{phase_name} step {phase_step + 1}/{phase_total} "
                    f"completed: data={record['data_distribution_ms']:.1f}ms, "
                    f"data_wait={record['data_rank_wait_ms']:.1f}ms, "
                    f"forward={record['forward_ms']:.1f}ms, "
                    f"backward={record['backward_ms']:.1f}ms, "
                    f"gradient_sync={record['gradient_sync_ms']:.1f}ms, "
                    f"optimizer={record['optimizer_ms']:.1f}ms, "
                    f"loss={record['loss']:.6g}",
                )

        torch.cuda.synchronize()
        measurement_window_ms = (
            time.perf_counter() - measurement_window_start
        ) * 1000

        payload = {
            "rank": rank,
            "records": rank_records,
            "hardware": _hardware_metadata(device, rank),
            "partition": _partition_metadata(partition),
            "initialization_timings": initialization_timings,
            "measurement_window_ms": measurement_window_ms,
        }
        _progress(rank, "collecting per-rank records")
        payloads = [None] * world_size
        dist.all_gather_object(payloads, payload)

        if rank == 0:
            records = _aggregate_records(payloads)
            measurement_window_ms = max(
                payload["measurement_window_ms"] for payload in payloads
            )
            result = {
                "schema_version": 4,
                "status": "completed",
                "benchmark": "real_data_model_parallel_graph_efm",
                "scaling_mode": "strong",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "git": _git_metadata(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "backend": "nccl",
                "world_size": world_size,
                "partition_strategy": strategy,
                "partition_file": _partition_file_metadata(
                    args.partition_file
                ),
                "workload": {
                    "config_path": str(args.config_path),
                    "dataset_root": str(args.config_path.parent),
                    "rank_data_root": (
                        str(rank_data_root) if rank_data_root else None
                    ),
                    "dataset_samples": len(dataset),
                    "data_kind": "physical Njord Zarr samples",
                    "dataset": _dataset_metadata(dataset),
                    "graph": args.graph,
                    "hidden_dim": args.hidden_dim,
                    "hidden_dim_grid": model.args.hidden_dim_grid,
                    "hidden_dim_mesh_nodes": model.args.hidden_dim_mesh_nodes,
                    "hidden_dim_edge": model.args.hidden_dim_edge,
                    "latent_dim": model.args.latent_dim or args.hidden_dim,
                    "learn_prior": model.args.learn_prior,
                    "prior_dist": model.args.prior_dist,
                    "output_std": model.output_std,
                    "sample_obs_noise": model.sample_obs_noise,
                    "processor_layers": args.processor_layers,
                    "batch_size_global": args.batch_size,
                    "global_surface_grid_nodes": int(
                        model.num_interior_nodes
                    ),
                    "ar_steps": args.ar_steps,
                    "gradient_checkpointing": model.grad_checkpointing,
                    "checkpoint_scope": "complete autoregressive step",
                    "checkpoint_early_stop": False,
                    "hidden_layers": model.args.hidden_layers,
                    "encoder_processor_layers": (
                        model.args.encoder_processor_layers
                    ),
                    "prior_processor_layers": (
                        model.args.prior_processor_layers
                    ),
                    "input_steps": model.input_steps,
                    "loss": model.args.loss,
                    "loss_time_reduction": "sum",
                    "kl_beta": model.kl_beta,
                    "crps_weight": model.crps_weight,
                    "optimizer": "adamw",
                    "optimizer_betas": [0.9, 0.95],
                    "optimizer_weight_decay": 0.1,
                    "shuffle": True,
                    "sampler_seed": args.seed,
                    "deterministic_algorithms": (
                        torch.are_deterministic_algorithms_enabled()
                    ),
                    "drop_last": False,
                    "pin_memory": False,
                    "baseline_reference": "global-model-ar4-elbo-fp32",
                    "precision": "float32",
                    "num_workers": args.num_workers,
                    "prefetch_factor": (
                        args.prefetch_factor if args.num_workers > 0 else None
                    ),
                    "seed": args.seed,
                    "learning_rate": args.lr,
                    "warmup_steps": args.warmup_steps,
                    "measurement_steps": args.steps,
                    "progress_interval": args.progress_interval,
                    "model_parallel_same_batch_on_all_ranks": True,
                    "data_mode": args.data_mode,
                    "rank_zero_loads_and_broadcasts_batch": (
                        not batch_is_local
                    ),
                    "replicated_frontend_parameters": True,
                    "rank_local_frontend_compute": True,
                    "data_phase_rank_barrier": True,
                    "partitioned_components": [
                        "prior",
                        "posterior",
                        "grid_to_mesh",
                        "mesh_up",
                        "mesh_intra_level",
                        "mesh_down",
                        "mesh_to_grid",
                    ],
                },
                "model": {
                    "frontend_parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "core_parameters_per_rank": sum(
                        parameter.numel() for parameter in core.parameters()
                    ),
                    "trainable_parameters_per_rank": sum(
                        parameter.numel() for parameter in parameters
                    ),
                    "gradient_sync_rule": "sum model-parallel contributions",
                },
                "preflight": {
                    "loss": preflight_loss,
                    "batch_fingerprint": fingerprint,
                },
                "rank_metadata": payloads,
                "records": records,
                "summary": _result_summary(
                    records,
                    args.batch_size,
                    measurement_window_ms,
                ),
                "interpretation": {
                    "comparable_across_world_sizes": (
                        args.progress_interval == 0
                    ),
                    "global_batch_size_is_constant": True,
                    "last_batch_may_be_smaller": True,
                    "compare_speed_with_same_gpu_model_only": True,
                    "limitations": [
                        "full autoregressive feedback stays rank-local; "
                        "halo exchange runs in forward and checkpoint replay",
                        "embedding parameters are replicated; rank-local mode "
                        "computes embeddings only for owned nodes and edges",
                        "phase timing uses CUDA events; headline observed "
                        "throughput uses wall time",
                        "a rank barrier after batch loading attributes loader "
                        "imbalance to data_rank_wait_ms instead of forward_ms",
                        "progress output adds diagnostic synchronization "
                        "when progress_interval is nonzero",
                    ],
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(args.output)
            _progress(rank, f"result JSON safely written to {args.output}")
            print(
                "Real-data model-parallel benchmark written to "
                f"{args.output}"
            )
            print(json.dumps(result["summary"], indent=2, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
