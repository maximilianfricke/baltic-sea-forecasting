"""One-GPU reference using the unchanged original GraphEFM.training_step.

Same H64/L2/B2/AR4/FP32, optimizer, shuffle and RNG schedule as the archived
distributed runs. The original full-grid Zarr/WeatherDataset path is used;
there is no graph partitioning, halo exchange or distributed model wrapper.
Only metric logging is captured by the benchmark instead of Lightning.
"""

# Standard library
import json
import math
import os
import subprocess
import time
from pathlib import Path

# Third-party
import torch
import torch.distributed as dist
from torch import nn

from benchmarks.compare_model_parallel_real import _fingerprints_match
from benchmarks.model_parallel_real import (
    RankLocalBatchStream,
    _batch_fingerprint,
    _dataset_metadata,
    _git_metadata,
    _global_scalar,
    _hardware_metadata,
    _load_workload,
    _parse_args,
    _progress,
    _result_summary,
    _step_record,
    _utc_now,
)

SOURCE_COMMIT = "76b30d778382939963219142053fcf6f2d54c54e"
SOURCE_FILES = (
    "neural_lam/models/graph_efm.py",
    "neural_lam/models/ar_model.py",
    "neural_lam/models/ar_prob_model.py",
    "neural_lam/models/base_latent_encoder.py",
    "neural_lam/models/base_graph_latent_decoder.py",
    "neural_lam/models/hi_graph_latent_encoder.py",
    "neural_lam/models/hi_graph_latent_decoder.py",
    "neural_lam/interaction_net.py",
    "neural_lam/utils.py",
    "neural_lam/metrics.py",
    "neural_lam/loss_weighting.py",
    "neural_lam/weather_dataset.py",
    "neural_lam/datastore/mdp.py",
)


def original_source_identity():
    """Reject accidental model/loader changes relative to our frozen base."""
    blobs = {}
    for filename in SOURCE_FILES:
        expected = subprocess.check_output(
            ["git", "rev-parse", f"{SOURCE_COMMIT}:{filename}"], text=True
        ).strip()
        actual = subprocess.check_output(
            ["git", "hash-object", filename], text=True
        ).strip()
        if actual != expected:
            raise RuntimeError(f"original reference source changed: {filename}")
        blobs[filename] = actual
    return {"commit": SOURCE_COMMIT, "matching_git_blobs": blobs}


def comparison_fingerprint(batch):
    """Use the distributed convention for unused, zero-element inputs.

    Original WeatherDataset retains a rectangular grid extent in its empty
    boundary placeholder. Rank-local comparisons use extent zero instead.
    Normalize only this metadata, leaving all inputs and actual values intact.
    """
    fingerprint = _batch_fingerprint(batch)
    for tensor, shape in zip(batch, fingerprint["shapes"]):
        if tensor.numel() == 0 and len(shape) >= 2:
            shape[-2] = 0
    return fingerprint


class OriginalGraphEFMTraining(nn.Module):
    """Capture original ELBO logs without changing rollout or loss code."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.metrics = {}
        self.model.log_dict = self._capture_metrics

    def _capture_metrics(self, values, **kwargs):
        self.metrics = values

    def forward(self, batch, generator=None, *, batch_is_local=False):
        # Original Normal.rsample uses the default generator. The runner seeds
        # that generator to the same stream as the distributed explicit RNG.
        self.metrics = {}
        loss = self.model.training_step(batch)
        return {
            "loss": loss,
            "negative_likelihood": -self.metrics["elbo_likelihood"],
            "kl": self.metrics["elbo_kl"],
        }


def main():
    args = _parse_args()
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("the original reference requires exactly one GPU")
    source = original_source_identity()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("the official reference benchmark requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    dist.init_process_group("nccl", device_id=device)
    started = _utc_now()
    start = time.perf_counter()
    try:
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        _progress(0, "loading original full-grid GraphEFM and WeatherDataset")
        dataset, model, _ = _load_workload(args)
        model.to(device).train()
        training = OriginalGraphEFMTraining(model)
        parameters = list(training.parameters())
        if sum(p.numel() for p in parameters) != 645020:
            raise ValueError(
                "reference configuration must have 645,020 parameters"
            )
        if not model.grad_checkpointing:
            raise ValueError("reference requires gradient checkpointing")
        optimizer = torch.optim.AdamW(
            parameters, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        # This stream is an ordinary DataLoader; no rank slicing occurs here.
        stream = RankLocalBatchStream(
            dataset,
            args.batch_size,
            args.num_workers,
            args.prefetch_factor,
            device,
            seed=args.seed,
        )
        batch = stream.next()
        fingerprint = comparison_fingerprint(batch)
        original_batch_shapes = [list(tensor.shape) for tensor in batch]
        torch.cuda.manual_seed(args.seed + 100_000)
        with torch.no_grad():
            preflight = _global_scalar(training(batch)["loss"])
        reference_check = None
        reference_path = os.environ.get("NJORD_REFERENCE_RESULT")
        if reference_path:
            reference = json.loads(Path(reference_path).read_text())
            same_batch = _fingerprints_match(
                fingerprint, reference["preflight"]["batch_fingerprint"]
            )
            close_loss = math.isclose(
                preflight,
                reference["preflight"]["loss"],
                rel_tol=1e-5,
                abs_tol=1.0,
            )
            if not same_batch or not close_loss:
                raise RuntimeError(
                    "original/distributed preflight mismatch: "
                    f"same_batch={same_batch}, loss={preflight:.9g}, "
                    f"reference_loss={reference['preflight']['loss']:.9g}"
                )
            reference_check = {
                "path": reference_path,
                "same_batch": same_batch,
                "preflight_loss_close": close_loss,
                "distributed_preflight_loss": reference["preflight"]["loss"],
            }
        torch.cuda.synchronize(device)
        initialization_ms = (time.perf_counter() - start) * 1000
        _progress(0, f"original preflight passed; loss={preflight:.9g}")
        # Match the model-parallel generator seeded with seed+1. The original
        # checkpoint restores this default CUDA RNG during backward replay.
        torch.cuda.manual_seed(args.seed + 1)
        generator = torch.cuda.default_generators[0]
        records = []
        window_start = None
        for step in range(args.warmup_steps + args.steps):
            if step == args.warmup_steps:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                window_start = time.perf_counter()
            data_start = time.perf_counter()
            batch = stream.next()
            data_ms = (time.perf_counter() - data_start) * 1000
            record = _step_record(
                training, optimizer, parameters, batch, generator, 1
            )
            if not math.isfinite(record["loss"]):
                raise RuntimeError(f"non-finite loss at step {step}")
            record.update(stream.last_timings)
            record.update(
                {
                    "data_distribution_ms": data_ms,
                    "data_rank_wait_ms": 0.0,
                    "end_to_end_ms": data_ms + record["step_ms"],
                }
            )
            if step >= args.warmup_steps:
                record["measurement_step"] = step - args.warmup_steps
                records.append(record)
        torch.cuda.synchronize(device)
        window_ms = (time.perf_counter() - window_start) * 1000
        result = {
            "schema_version": 1,
            "status": "completed",
            "benchmark": "original_graph_efm_ar4_reference",
            "execution_mode": "original_unpartitioned",
            "world_size": 1,
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "git": _git_metadata(),
            "original_source": source,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "backend": "nccl",
            "workload": {
                "config_path": str(args.config_path),
                "source_data_root": os.environ.get("NJORD_SOURCE_DATA_ROOT"),
                "dataset": _dataset_metadata(dataset),
                "dataset_samples": len(dataset),
                "data_kind": "physical Njord Zarr samples",
                "global_surface_grid_nodes": int(model.num_interior_nodes),
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
                "hidden_layers": model.args.hidden_layers,
                "encoder_processor_layers": (
                    model.args.encoder_processor_layers
                ),
                "prior_processor_layers": model.args.prior_processor_layers,
                "batch_size_global": args.batch_size,
                "ar_steps": args.ar_steps,
                "precision": "float32",
                "gradient_checkpointing": model.grad_checkpointing,
                "checkpoint_scope": "complete autoregressive step",
                "checkpoint_early_stop": True,
                "input_steps": model.input_steps,
                "num_workers": args.num_workers,
                "prefetch_factor": (
                    args.prefetch_factor if args.num_workers else None
                ),
                "pin_memory": False,
                "shuffle": True,
                "drop_last": False,
                "seed": args.seed,
                "sampler_seed": args.seed,
                "learning_rate": args.lr,
                "optimizer": "adamw",
                "optimizer_betas": [0.9, 0.95],
                "optimizer_weight_decay": 0.1,
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "baseline_reference": "global-model-ar4-elbo-fp32",
                "loss": model.args.loss,
                "loss_time_reduction": "sum",
                "kl_beta": model.kl_beta,
                "crps_weight": model.crps_weight,
                "warmup_steps": args.warmup_steps,
                "measurement_steps": args.steps,
                "data_mode": "original-full-grid-zarr",
                "halo_exchange": False,
                "data_phase_rank_barrier": False,
                "rank_zero_loads_and_broadcasts_batch": False,
                "progress_interval": 0,
            },
            "model": {
                "trainable_parameters": sum(p.numel() for p in parameters)
            },
            "preflight": {
                "loss": preflight,
                "batch_fingerprint": fingerprint,
                "original_batch_shapes": original_batch_shapes,
                "empty_spatial_dimensions_canonicalized": True,
                "distributed_reference_check": reference_check,
            },
            "rank_metadata": [
                {
                    "rank": 0,
                    "hardware": _hardware_metadata(device, 0),
                    "initialization_timings": {
                        "total_before_training_ms": initialization_ms,
                        "data_staging_ms": float(
                            os.environ.get("NJORD_STAGING_MS", 0)
                        ),
                    },
                    "records": records,
                }
            ],
            "records": records,
            "summary": _result_summary(records, args.batch_size, window_ms),
            "interpretation": {
                "same_model_source_as_distributed_base": True,
                "loader_reads_original_full_grid_before_ocean_mask": True,
                "end_to_end_comparison_includes_input_layout_difference": True,
                "original_checkpoint_early_stop_preserved": True,
                "timing_protocol_shared_with_distributed_runs": True,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(args.output)
        print(json.dumps(result["summary"], indent=2), flush=True)
        _progress(0, f"Result: {args.output}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
