"""Low-overhead training benchmark support for neural-lam."""

# Standard library
import json
import os
import platform
import resource
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Third-party
import pytorch_lightning as pl
import torch


def _mps_available():
    """Return whether this process can use the Apple Metal backend."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _synchronize():
    """Wait for queued accelerator work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif _mps_available():
        torch.mps.synchronize()


def _summary(values):
    """Return stable summary statistics for a sequence of measurements."""
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
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _git_metadata():
    """Read the revision without making benchmark execution depend on Git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _edge_count(edge_index):
    """Count edges in a tensor or BufferList of edge-index tensors."""
    if edge_index is None:
        return None
    if isinstance(edge_index, torch.Tensor):
        return int(edge_index.shape[-1])
    try:
        return sum(int(item.shape[-1]) for item in edge_index)
    except (TypeError, AttributeError, IndexError):
        return None


def _node_count(node_features):
    """Count nodes in a tensor or BufferList of feature tensors."""
    if node_features is None:
        return None
    if isinstance(node_features, torch.Tensor):
        return int(node_features.shape[0])
    try:
        return sum(int(item.shape[0]) for item in node_features)
    except (TypeError, AttributeError, IndexError):
        return None


def _model_metadata(model):
    """Collect architecture and graph sizes from a neural-lam model."""
    model = getattr(model, "forecast_model", model)
    graph = {}
    for name in (
        "g2m_edge_index",
        "m2m_edge_index",
        "mesh_up_edge_index",
        "mesh_down_edge_index",
        "m2g_edge_index",
    ):
        count = _edge_count(getattr(model, name, None))
        if count is not None:
            graph[f"{name.removesuffix('_index')}_count"] = count

    grid_features = getattr(model, "grid_static_features", None)
    mesh_features = getattr(model, "mesh_static_features", None)
    parameters = list(model.parameters())
    trainable_parameters = [p for p in parameters if p.requires_grad]
    return {
        "class": type(model).__name__,
        "parameters_total": sum(p.numel() for p in parameters),
        "parameters_trainable": sum(p.numel() for p in trainable_parameters),
        "parameter_bytes": sum(
            p.numel() * p.element_size() for p in parameters
        ),
        "trainable_parameter_bytes": sum(
            p.numel() * p.element_size() for p in trainable_parameters
        ),
        "gradient_checkpointing": bool(
            getattr(model, "grad_checkpointing", False)
        ),
        "graph": {
            "grid_node_count": _node_count(grid_features),
            "mesh_node_count": _node_count(mesh_features),
            **graph,
        },
    }


def _hardware_metadata():
    """Collect enough environment information to reproduce a result."""
    result = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cpu_count_visible": os.cpu_count(),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
        "slurm_mem_per_node_mb": os.environ.get("SLURM_MEM_PER_NODE"),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "index": device,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
            "device_count": torch.cuda.device_count(),
        }
        try:
            nvidia_smi = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device}",
                    "--query-gpu=uuid,driver_version,power.limit,clocks.max.sm",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            result["gpu"]["nvidia_smi"] = nvidia_smi
        except (OSError, subprocess.CalledProcessError):
            result["gpu"]["nvidia_smi"] = None
    elif _mps_available():
        result["gpu"] = {
            "index": 0,
            "name": "Apple MPS",
            "compute_capability": None,
            "total_memory_bytes": torch.mps.recommended_max_memory(),
            "device_count": 1,
        }
    else:
        result["gpu"] = None
    return result


class InferenceBenchmarkModule(pl.LightningModule):
    """Run forecasting only, without metric aggregation or plotting."""

    def __init__(self, forecast_model, ensemble_size=1):
        super().__init__()
        self.forecast_model = forecast_model
        self.ensemble_size = ensemble_size

    def test_step(self, batch, batch_idx):
        del batch_idx
        (
            init_states,
            _,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            _,
        ) = batch

        def expand(tensor):
            return tensor.repeat_interleave(self.ensemble_size, dim=0)

        init_states = expand(init_states)
        forcing = expand(forcing)
        boundary_forcing = expand(boundary_forcing)
        atmosphere_forcing = expand(atmosphere_forcing)
        if self.forecast_model.input_steps == 1:
            previous = init_states[:, 0]
            previous_previous = None
        else:
            previous_previous = init_states[:, 0]
            previous = init_states[:, 1]

        for step in range(forcing.shape[1]):
            prediction, _ = self.forecast_model.predict_step(
                previous,
                previous_previous,
                forcing[:, step],
                boundary_forcing[:, step],
                atmosphere_forcing[:, step],
            )
            if self.forecast_model.input_steps > 1:
                previous_previous = previous
            previous = prediction
        return None


class BaselineBenchmark(pl.Callback):
    """Measure training steps after warm-up and write a self-contained JSON."""

    def __init__(
        self,
        output_path,
        warmup_steps,
        measurement_steps,
        run_label,
        args,
    ):
        super().__init__()
        if warmup_steps < 0 or measurement_steps < 1:
            raise ValueError(
                "Benchmark needs >=0 warm-up and >=1 measured step"
            )
        self.output_path = Path(output_path)
        self.warmup_steps = warmup_steps
        self.measurement_steps = measurement_steps
        self.run_label = run_label
        self.args = vars(args).copy()
        self.workload = self.args.get("benchmark_workload", "training")
        self.timing_mode = self.args.get(
            "benchmark_timing_mode", "phase"
        )
        self.records = []
        self.seen_steps = 0
        self.measured_steps = 0
        self.last_batch_end = None
        self._events = None
        self._times = None
        self.started_at = None
        self.measurement_started_at = None
        self.measurement_finished_at = None
        self.startup = {}
        self.aggregate_start = None
        self.error = None
        self._mps_peak_allocated = 0
        self._mps_peak_reserved = 0
        self._mps_batch_memory = []

    def _sample_mps_memory(self):
        if _mps_available():
            self._mps_batch_memory.append(
                (
                    torch.mps.current_allocated_memory(),
                    torch.mps.driver_allocated_memory(),
                )
            )

    def on_fit_start(self, trainer, pl_module):
        del trainer, pl_module
        self._run_start()

    def on_test_start(self, trainer, pl_module):
        del trainer, pl_module
        self._run_start()

    def _run_start(self):
        now = datetime.now(timezone.utc)
        self.started_at = now.isoformat()
        job_started_epoch = os.environ.get(
            "NJORD_JOB_STARTED_EPOCH_SECONDS"
        )
        if job_started_epoch is not None:
            try:
                self.startup["job_to_fit_start_seconds"] = (
                    now.timestamp() - float(job_started_epoch)
                )
            except ValueError:
                pass
        for environment_name, output_name in (
            ("NJORD_DATA_STAGING_SECONDS", "data_staging_seconds"),
            ("NJORD_DATA_STAGING_BYTES", "data_staging_bytes"),
        ):
            value = os.environ.get(environment_name)
            if value is not None:
                try:
                    self.startup[output_name] = float(value)
                except ValueError:
                    pass
        data_location = os.environ.get("NJORD_DATA_LOCATION")
        if data_location is not None:
            self.startup["data_location"] = data_location
        _synchronize()
        self.last_batch_end = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del trainer, pl_module, batch, batch_idx
        if self.timing_mode == "aggregate":
            if self.seen_steps == self.warmup_steps:
                _synchronize()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                self.measurement_started_at = datetime.now(
                    timezone.utc
                ).isoformat()
                self.aggregate_start = time.perf_counter()
            return
        _synchronize()
        now = time.perf_counter()
        data_wait = (
            0.0 if self.last_batch_end is None else now - self.last_batch_end
        )
        self._times = {"start": now, "data_wait": data_wait}
        self._mps_batch_memory = []
        self._sample_mps_memory()
        if (
            self.seen_steps == self.warmup_steps
            and self.measurement_started_at is None
        ):
            self.measurement_started_at = datetime.now(
                timezone.utc
            ).isoformat()
        if torch.cuda.is_available():
            if self.seen_steps == self.warmup_steps:
                torch.cuda.reset_peak_memory_stats()
            self._events = {
                name: torch.cuda.Event(enable_timing=True)
                for name in ("start", "forward_end", "backward_end", "end")
            }
            self._events["start"].record()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_before_backward(self, trainer, pl_module, loss):
        del trainer, pl_module, loss
        self._mark("forward_end")

    def on_after_backward(self, trainer, pl_module):
        del trainer, pl_module
        self._mark("backward_end")

    def _mark(self, name):
        if self._events is not None:
            self._events[name].record()
        elif _mps_available():
            # MPS is asynchronous but does not expose CUDA-style timing events.
            torch.mps.synchronize()
            self._sample_mps_memory()
        if self._times is not None:
            self._times[name] = time.perf_counter()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        del trainer, pl_module, outputs, batch, batch_idx
        if self.timing_mode == "aggregate":
            if self.seen_steps >= self.warmup_steps:
                self.measured_steps += 1
                if self.measured_steps == self.measurement_steps:
                    _synchronize()
                    end = time.perf_counter()
                    total_ms = (end - self.aggregate_start) * 1e3
                    mean_ms = total_ms / self.measured_steps
                    record = {
                        "measurement_step": None,
                        "measurement_steps": self.measured_steps,
                        "aggregate_window_ms": total_ms,
                        "step_ms": mean_ms,
                        "step_wall_ms": mean_ms,
                        "end_to_end_wall_ms": mean_ms,
                    }
                    if torch.cuda.is_available():
                        record.update(
                            {
                                "gpu_memory_allocated_bytes": (
                                    torch.cuda.memory_allocated()
                                ),
                                "gpu_memory_reserved_bytes": (
                                    torch.cuda.memory_reserved()
                                ),
                                "gpu_peak_memory_allocated_bytes": (
                                    torch.cuda.max_memory_allocated()
                                ),
                                "gpu_peak_memory_reserved_bytes": (
                                    torch.cuda.max_memory_reserved()
                                ),
                            }
                        )
                    self.records.append(record)
                    self.measurement_finished_at = datetime.now(
                        timezone.utc
                    ).isoformat()
            self.seen_steps += 1
            return
        self._mark("end")
        _synchronize()
        end = time.perf_counter()
        self.last_batch_end = end

        if self.seen_steps >= self.warmup_steps:
            if self._events is not None:
                step_ms = self._events["start"].elapsed_time(
                    self._events["end"]
                )
                forward_ms = self._events["start"].elapsed_time(
                    self._events["forward_end"]
                )
                backward_ms = self._events["forward_end"].elapsed_time(
                    self._events["backward_end"]
                )
                optimizer_ms = self._events["backward_end"].elapsed_time(
                    self._events["end"]
                )
            else:
                step_ms = (self._times["end"] - self._times["start"]) * 1e3
                forward_ms = (
                    self._times["forward_end"] - self._times["start"]
                ) * 1e3
                backward_ms = (
                    self._times["backward_end"]
                    - self._times["forward_end"]
                ) * 1e3
                optimizer_ms = (
                    self._times["end"] - self._times["backward_end"]
                ) * 1e3

            record = {
                "measurement_step": len(self.records),
                "step_ms": step_ms,
                "step_wall_ms": (end - self._times["start"]) * 1e3,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "optimizer_and_hooks_ms": optimizer_ms,
                "data_wait_ms": self._times["data_wait"] * 1e3,
            }
            record["end_to_end_wall_ms"] = (
                record["step_wall_ms"] + record["data_wait_ms"]
            )
            if torch.cuda.is_available():
                record.update(
                    {
                        "gpu_memory_allocated_bytes": (
                            torch.cuda.memory_allocated()
                        ),
                        "gpu_memory_reserved_bytes": (
                            torch.cuda.memory_reserved()
                        ),
                        "gpu_peak_memory_allocated_bytes": (
                            torch.cuda.max_memory_allocated()
                        ),
                        "gpu_peak_memory_reserved_bytes": (
                            torch.cuda.max_memory_reserved()
                        ),
                    }
                )
            elif _mps_available():
                allocated = torch.mps.current_allocated_memory()
                reserved = torch.mps.driver_allocated_memory()
                sampled_peak_allocated = max(
                    sample[0] for sample in self._mps_batch_memory
                )
                sampled_peak_reserved = max(
                    sample[1] for sample in self._mps_batch_memory
                )
                self._mps_peak_allocated = max(
                    self._mps_peak_allocated, sampled_peak_allocated
                )
                self._mps_peak_reserved = max(
                    self._mps_peak_reserved, sampled_peak_reserved
                )
                record.update(
                    {
                        "gpu_memory_allocated_bytes": allocated,
                        "gpu_memory_reserved_bytes": reserved,
                        # MPS exposes no resettable peak counter. These are the
                        # largest observations at synchronized phase borders.
                        "gpu_peak_memory_allocated_bytes": (
                            self._mps_peak_allocated
                        ),
                        "gpu_peak_memory_reserved_bytes": (
                            self._mps_peak_reserved
                        ),
                    }
                )
            self.records.append(record)
            self.measured_steps += 1
            self.measurement_finished_at = datetime.now(
                timezone.utc
            ).isoformat()
        self.seen_steps += 1

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        self._mark("forward_end")
        self._mark("backward_end")
        self.on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )

    def on_exception(self, trainer, pl_module, exception):
        del trainer, pl_module
        self.error = f"{type(exception).__name__}: {exception}"

    def on_fit_end(self, trainer, pl_module):
        self._finish(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        self._finish(trainer, pl_module)

    def _finish(self, trainer, pl_module):
        rank_payload = {
            "rank": trainer.global_rank,
            "hostname": socket.gethostname(),
            "records": self.records,
            "process_peak_rss_bytes": self._process_peak_rss_bytes(),
        }
        distributed_ready = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if distributed_ready:
            rank_payloads = [None] * trainer.world_size
            torch.distributed.all_gather_object(rank_payloads, rank_payload)
        else:
            rank_payloads = [rank_payload]
        if trainer.is_global_zero:
            self._write_result(trainer, pl_module, rank_payloads)

    @staticmethod
    def _process_peak_rss_bytes():
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, while macOS reports bytes.
        return max_rss if sys.platform == "darwin" else max_rss * 1024

    @staticmethod
    def _aggregate_rank_records(rank_payloads):
        """Use the slowest rank per step for distributed wall-clock metrics."""
        if len(rank_payloads) == 1:
            return rank_payloads[0]["records"]
        record_count = min(
            len(payload["records"]) for payload in rank_payloads
        )
        time_keys = (
            "aggregate_window_ms",
            "step_ms",
            "step_wall_ms",
            "end_to_end_wall_ms",
            "forward_ms",
            "backward_ms",
            "optimizer_and_hooks_ms",
            "data_wait_ms",
        )
        memory_keys = (
            "gpu_memory_allocated_bytes",
            "gpu_memory_reserved_bytes",
            "gpu_peak_memory_allocated_bytes",
            "gpu_peak_memory_reserved_bytes",
        )
        aggregate = []
        for step in range(record_count):
            rank_records = [
                payload["records"][step] for payload in rank_payloads
            ]
            record = {"measurement_step": step}
            for key in time_keys + memory_keys:
                values = [
                    rank_record[key]
                    for rank_record in rank_records
                    if key in rank_record
                ]
                if values:
                    record[key] = max(values)
            aggregate.append(record)
        return aggregate

    def _write_result(self, trainer, model, rank_payloads):
        records = self._aggregate_rank_records(rank_payloads)
        metric_names = (
            "aggregate_window_ms",
            "step_ms",
            "step_wall_ms",
            "end_to_end_wall_ms",
            "forward_ms",
            "backward_ms",
            "optimizer_and_hooks_ms",
            "data_wait_ms",
        )
        summaries = {
            name: _summary(
                [record[name] for record in records if name in record]
            )
            for name in metric_names
        }
        if self.timing_mode == "aggregate":
            for name in ("step_ms", "step_wall_ms", "end_to_end_wall_ms"):
                if summaries[name] is not None:
                    summaries[name]["p95"] = None
                    summaries[name]["p99"] = None
        step_mean = (
            summaries["step_wall_ms"]["mean"] if records else None
        )
        end_to_end_mean = (
            summaries["end_to_end_wall_ms"]["mean"]
            if records
            else None
        )
        batch_size = self.args.get("batch_size")
        global_batch_size = self.args.get("benchmark_global_batch_size")
        if global_batch_size is None and batch_size is not None:
            global_batch_size = batch_size * trainer.world_size
            if self.workload == "inference":
                global_batch_size *= self.args.get(
                    "benchmark_inference_ensemble_size", 1
                )
        compute_samples_per_second = (
            global_batch_size * 1000.0 / step_mean
            if step_mean and global_batch_size is not None
            else None
        )
        samples_per_second = (
            global_batch_size * 1000.0 / end_to_end_mean
            if end_to_end_mean and global_batch_size is not None
            else None
        )
        memory_keys = (
            "gpu_memory_allocated_bytes",
            "gpu_memory_reserved_bytes",
            "gpu_peak_memory_allocated_bytes",
            "gpu_peak_memory_reserved_bytes",
        )
        memory = {
            key: max(
                (record.get(key, 0) for record in records), default=None
            )
            for key in memory_keys
        }
        model_metadata = _model_metadata(model)
        trainable_bytes = model_metadata["trainable_parameter_bytes"]
        world_size = trainer.world_size
        # DDP synchronizes one gradient value per trainable parameter. The
        # ring-all-reduce wire-volume formula is an estimate; communication
        # time is measured separately by the PyTorch/NCCL profiler trace.
        estimated_ring_bytes = (
            2 * (world_size - 1) / world_size * trainable_bytes
            if self.workload == "training" and world_size > 1
            else 0
        )
        data_module = trainer.datamodule
        split = "test" if self.workload == "inference" else "train"
        dataset_metadata = getattr(
            data_module, "benchmark_dataset_metadata", {}
        ).get(split, {})
        result = {
            "schema_version": 4,
            "label": self.run_label,
            "started_at_utc": self.started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if self.error else "completed",
            "error": self.error,
            "git": _git_metadata(),
            "hardware": _hardware_metadata(),
            "distributed": {
                "world_size": world_size,
                "global_rank": trainer.global_rank,
                "strategy": type(trainer.strategy).__name__,
                "scaling_mode": self.args.get(
                    "benchmark_scaling_mode", "single"
                ),
                "global_batch_size": global_batch_size,
                "estimated_gradient_payload_bytes_per_step": (
                    trainable_bytes if self.workload == "training" else 0
                ),
                "estimated_ring_allreduce_bytes_per_rank_per_step": (
                    estimated_ring_bytes
                ),
                "rank_results": rank_payloads,
            },
            "model": model_metadata,
            "arguments": self.args,
            "dataset": {
                "split": split,
                **dataset_metadata,
            },
            "startup": self.startup,
            "instrumentation": {
                "timing_mode": self.timing_mode,
                "per_batch_accelerator_synchronization": (
                    self.timing_mode == "phase"
                ),
                "gpu_telemetry_enabled": (
                    os.environ.get("ENABLE_GPU_TELEMETRY", "0") == "1"
                ),
                "gpu_telemetry_interval_ms": (
                    os.environ.get("TELEMETRY_INTERVAL_MS")
                ),
                "pytorch_profiler_enabled": bool(
                    self.args.get("benchmark_profiler_dir")
                ),
            },
            "protocol": {
                "warmup_steps": self.warmup_steps,
                "workload": self.workload,
                "requested_measurement_steps": self.measurement_steps,
                "completed_measurement_steps": self.measured_steps,
                "measurement_started_at_utc": (
                    self.measurement_started_at
                ),
                "measurement_finished_at_utc": (
                    self.measurement_finished_at
                ),
                "validation_enabled": False,
                "external_logger_enabled": False,
                "phase_timing_source": (
                    "cuda_events"
                    if torch.cuda.is_available()
                    else "synchronized_wall_clock"
                    if _mps_available()
                    else "wall_clock"
                ),
                "memory_measurement": (
                    "allocator_peak"
                    if torch.cuda.is_available()
                    else "maximum_synchronized_phase_boundary"
                    if _mps_available()
                    else None
                ),
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
            },
            "summary": {
                **summaries,
                "samples_per_second": samples_per_second,
                "compute_samples_per_second": compute_samples_per_second,
                "process_peak_rss_bytes": max(
                    payload["process_peak_rss_bytes"]
                    for payload in rank_payloads
                ),
                "process_peak_rss_bytes_sum": sum(
                    payload["process_peak_rss_bytes"]
                    for payload in rank_payloads
                ),
                **memory,
            },
            "steps": records,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(
            self.output_path.suffix + ".tmp"
        )
        temporary_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.output_path)
        print(f"Benchmark written to {self.output_path.resolve()}")
