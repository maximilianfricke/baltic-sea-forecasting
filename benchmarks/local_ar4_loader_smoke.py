"""Functional CPU/Gloo smoke with the real AR4 graph and eight workers/rank.

This deliberately reports no GPU scaling claim. Runs 1/2/4 ranks sequentially,
using the production rank-local loader, complete H64 model, checkpointed
backward and AdamW. Only two optimizer steps per run are needed here.
"""

# Standard library
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

# Avoid BLAS thread pools in each of the 8/16/32 spawned loader processes.
for variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Third-party
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# First-party
from benchmarks.model_parallel_real import (
    RankLocalBatchStream,
    _build_sharded_training_modules,
    _distributed_batch_fingerprint,
    _global_scalar,
    _load_rank_local_dataset,
    _load_workload,
    _synchronize_gradients,
)
from neural_lam.models.distributed_graph_efm_training import (
    DistributedGraphEFMTraining,
)


def _worker(rank, world_size, init_method, config_path, shard_root, result_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=init_method,
        timeout=timedelta(seconds=300),
    )
    stream = None
    start = time.perf_counter()
    try:
        args = SimpleNamespace(
            config_path=Path(config_path), partitioned_data_root=Path(shard_root),
            graph="global_cluster_graph", ar_steps=4, hidden_dim=64,
            processor_layers=2, batch_size=2, lr=.001, seed=42,
            partition_strategy="geographic", latitude_cut=-5., longitude_seam=120.,
        )
        _, reference, config = _load_workload(args, include_global_dataset=False)
        frontend, core, partition, _ = _build_sharded_training_modules(
            reference, args, rank, world_size, torch.device("cpu")
        )
        training = DistributedGraphEFMTraining(frontend, core, partition)
        dataset, _ = _load_rank_local_dataset(args, config, rank, world_size)
        torch.testing.assert_close(
            torch.as_tensor(dataset.datastore._ds.global_surface_index.values),
            partition.interior_grid_nodes,
        )
        assert frontend.grad_checkpointing
        assert sum(p.numel() for p in training.parameters()) == 645020
        parameters = list(training.parameters())
        optimizer = torch.optim.AdamW(
            parameters, lr=.001, betas=(.9, .95), weight_decay=.1
        )
        print(f"rank={rank}/{world_size}: starting 8 persistent spawn workers, P2", flush=True)
        stream = RankLocalBatchStream(
            dataset, 2, 8, 2, torch.device("cpu"), seed=42
        )
        assert stream.loader.num_workers == 8
        assert stream.loader.prefetch_factor == 2
        records = []
        fingerprint = None
        generator = torch.Generator().manual_seed(43)
        for step in range(2):
            read_start = time.perf_counter()
            batch = stream.next()
            data_ms = (time.perf_counter() - read_start) * 1000
            if step == 0:
                fingerprint = _distributed_batch_fingerprint(batch, True)
            optimizer.zero_grad(set_to_none=True)
            forward_start = time.perf_counter()
            output = training(batch, generator)
            forward_ms = (time.perf_counter() - forward_start) * 1000
            assert torch.isfinite(output["loss"])
            backward_start = time.perf_counter()
            output["loss"].backward()
            backward_ms = (time.perf_counter() - backward_start) * 1000
            _synchronize_gradients(parameters, world_size)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            loss = _global_scalar(output["loss"])
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in parameters)
            records.append({
                "step": step, "global_loss": loss,
                "data_wait_ms": data_ms, "forward_cpu_ms": forward_ms,
                "backward_cpu_ms": backward_ms,
            })
            if rank == 0:
                print(f"{world_size} ranks: AR4 step {step + 1}/2 passed; loss={loss:.9g}", flush=True)
        local = {
            "rank": rank, "owned_grid_nodes": partition.interior_grid_nodes.numel(),
            "workers": stream.loader.num_workers,
            "prefetch_factor": stream.loader.prefetch_factor,
            "elapsed_seconds": time.perf_counter() - start,
            "records": records,
        }
        reports = [None] * world_size
        dist.all_gather_object(reports, local)
        if rank == 0:
            Path(result_path).write_text(json.dumps({
                "status": "completed", "backend": "gloo", "device": "cpu",
                "performance_comparable_to_cuda": False,
                "world_size": world_size, "ar_steps": 4,
                "gradient_checkpointing": True, "hidden_dim": 64,
                "processor_layers": 2, "global_batch_size": 2,
                "parameters_per_rank": 645020,
                "num_workers_per_rank": 8, "prefetch_factor": 2,
                "batch_fingerprint": fingerprint, "ranks": reports,
            }, indent=2) + "\n")
    finally:
        if stream is not None:
            # Explicit cleanup matters when the next run spawns 16/32 workers.
            stream.iterator._shutdown_workers()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--config-path", type=Path, default=Path("data/physical_repeat_16/global_ocean_1.yaml"))
    parser.add_argument("--shard-root", type=Path, default=Path("data/partitioned/physical_repeat_16"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--check-only", action="store_true",
        help="compare existing results without repeating the training runs",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for world_size in args.world_sizes:
        if world_size not in (1, 2, 4):
            parser.error("world sizes must be 1, 2 or 4")
        result_path = args.output_dir / f"ar4-workers8-prefetch2-{world_size}rank.json"
        if not args.check_only:
            with tempfile.TemporaryDirectory(prefix="njord-ar4-gloo-") as tmp:
                mp.spawn(
                    _worker,
                    args=(world_size, f"file://{tmp}/init", str(args.config_path),
                          str(args.shard_root), str(result_path)),
                    nprocs=world_size, join=True,
                )
        results.append(json.loads(result_path.read_text()))
    reference = results[0]
    for result in results[1:]:
        left, right = reference["batch_fingerprint"], result["batch_fingerprint"]
        assert left["shapes"] == right["shapes"]
        assert left["dtypes"] == right["dtypes"]
        assert all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)
                   for a, b in zip(left["sums"], right["sums"]))
        for expected, actual in zip(reference["ranks"][0]["records"], result["ranks"][0]["records"]):
            assert math.isclose(expected["global_loss"], actual["global_loss"], rel_tol=1e-4)
    summary = {
        "status": "completed", "test": "physical_ar4_workers8_prefetch2",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "world_sizes": args.world_sizes,
        "same_batch_and_close_losses": True if len(results) > 1 else None,
        "performance_comparable_to_cuda": False,
        "results": [f"ar4-workers8-prefetch2-{w}rank.json" for w in args.world_sizes],
    }
    (args.output_dir / "validation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
