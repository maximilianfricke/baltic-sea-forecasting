"""Peak memory and time of one GraphEFM inference step split over GPUs.

Launch with torchrun (one process per GPU). Each rank embeds only its own
grid nodes and graph edges (no replicated full-graph embedding), then runs
the sharded prior, latent sample and decoder of the model-parallel core
(``DistributedGraphEFMCore.inference_forward``) with ``no_grad``. The
latent noise is drawn for the global top mesh and selected per rank, so the
forecast does not depend on the number of ranks.

``--check`` (small runs): before sharding, compute the unsplit forecast with
the same noise and compare each rank's owned rows with it.

Usage:
    torchrun --standalone --nproc-per-node=4 -m benchmarks.inference_split \
        --config-path CONFIG --graph GRAPH --hidden 64 256 \
        --partition-strategy geographic --output out.json
"""

# Standard library
import argparse
import datetime
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

# Third-party
import torch
import torch.distributed as dist

# First-party
from benchmarks.model_parallel_real import (
    _build_sharded_training_modules,
    _embed_local_graph,
    _load_workload,
    _num_latent_mesh_nodes,
)
from neural_lam.distributed_message_passing import select_owned


def _local_frontend(model, partition, previous, previous_previous, forcing,
                    atmosphere_forcing):
    """Interior grid embedding of the rank's own nodes only (the previous-
    state part of the rank-local frontend of model_parallel_real.py)."""
    if model.boundary_forced or model.use_atmosphere_g2m:
        raise NotImplementedError(
            "local frontend supports the global-ocean configuration without "
            "boundary or atmosphere graph nodes"
        )
    if not torch.equal(partition.full_grid_nodes,
                       partition.interior_grid_nodes):
        raise ValueError("full and interior grid ownership differ")
    nodes = partition.interior_grid_nodes
    batch_size = previous.shape[0]
    static = model.expand_to_batch(
        model.grid_static_features.index_select(0, nodes), batch_size
    )
    parts = [select_owned(previous, nodes)]
    if previous_previous is not None:
        parts.append(select_owned(previous_previous, nodes))
    parts.extend((select_owned(forcing, nodes), static))
    if model.concat_atmosphere:
        parts.append(select_owned(atmosphere_forcing, nodes))
    return model.interior_embedder(torch.cat(parts, dim=-1))


def _noise(model, batch_size, device, seed):
    latent_dim = model.args.latent_dim or model.args.hidden_dim
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch_size, _num_latent_mesh_nodes(model), latent_dim,
                       generator=generator).to(device)


def _reference(model, batch, noise):
    """Unsplit forecast with the given latent noise (before sharding)."""
    init, _, forcing, boundary, atmosphere, _ = batch
    previous_previous, previous = init[:, 0], init[:, 1]
    grid_prev, grid_prev_interior, graph_emb = model.embedd_all(
        previous, previous_previous, forcing[:, 0], boundary[:, 0],
        atmosphere[:, 0],
    )
    prior = model.prior_model(grid_prev, graph_emb=graph_emb)
    latent = prior.mean + prior.stddev * noise
    mean, _ = model.decoder(grid_prev, grid_prev_interior, latent, previous,
                            graph_emb)
    return model.get_clamped_new_state(mean - previous, previous)


def _split_step(model, core, partition, batch, noise):
    init, _, forcing, boundary, atmosphere, _ = batch
    previous_previous, previous = init[:, 0], init[:, 1]
    interior = _local_frontend(model, partition, previous, previous_previous,
                               forcing[:, 0], atmosphere[:, 0])
    graph = _embed_local_graph(model, partition, previous.shape[0])
    local_previous = select_owned(previous, partition.interior_grid_nodes)
    output = core.inference_forward(
        interior, interior, local_previous, graph,
        select_owned(noise, partition.mesh_nodes[-1]),
    )
    return model.get_clamped_new_state(
        output["pred_mean"] - local_previous, local_previous
    )


def run(args, hidden, rank, world_size, device):
    workload = SimpleNamespace(
        config_path=Path(args.config_path), graph=args.graph,
        hidden_dim=hidden, processor_layers=args.processor_layers,
        batch_size=1, ar_steps=1, lr=1e-3, seed=42, num_workers=0,
        prefetch_factor=2, partition_strategy=args.partition_strategy,
        partition_file=args.partition_file, latitude_cut=-5.0,
        longitude_seam=120.0,
    )
    # Same seed on every rank: identical initial weights (as in
    # model_parallel_real.py), otherwise ranks would run different models
    torch.manual_seed(workload.seed)
    dataset, model, _ = _load_workload(workload, include_global_dataset=True)
    batch = torch.utils.data.default_collate([dataset[0]])
    noise = _noise(model, 1, torch.device("cpu"), seed=0)
    entry = {"hidden": hidden}
    reference = None
    if args.check:
        model.eval()
        with torch.no_grad():
            reference = _reference(model, batch, noise)
    model, core, partition, strategy = _build_sharded_training_modules(
        model, workload, rank, world_size, device
    )
    model.eval()
    core.eval()
    batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
    noise = noise.to(device)
    entry["partition_strategy"] = strategy
    with torch.no_grad():
        prediction = _split_step(model, core, partition, batch, noise)
        if reference is not None:
            owned = select_owned(reference, partition.interior_grid_nodes)
            entry["max_abs_diff_vs_unsplit"] = float(
                (prediction.cpu() - owned).abs().max()
            )
        del prediction
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.steps):
            dist.barrier()
            start = time.perf_counter()
            _split_step(model, core, partition, batch, noise)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    peak = (torch.cuda.max_memory_allocated() if device.type == "cuda"
            else 0)
    stats = torch.tensor([max(times), peak], dtype=torch.float64,
                         device=device)
    gathered = [torch.zeros_like(stats) for _ in range(world_size)]
    dist.all_gather(gathered, stats)
    entry["step_ms_per_rank"] = [float(g[0]) for g in gathered]
    entry["peak_allocated_bytes_per_rank"] = [int(g[1]) for g in gathered]
    entry["step_ms"] = max(entry["step_ms_per_rank"])
    entry["owned_grid_nodes"] = int(partition.interior_grid_nodes.numel())
    if "max_abs_diff_vs_unsplit" in entry:
        diff = torch.tensor([entry["max_abs_diff_vs_unsplit"]],
                            dtype=torch.float64, device=device)
        dist.all_reduce(diff, op=dist.ReduceOp.MAX)
        entry["max_abs_diff_vs_unsplit"] = float(diff)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--hidden", type=int, nargs="+", required=True)
    parser.add_argument("--processor-layers", type=int, default=2)
    parser.add_argument("--partition-strategy", default="geographic",
                        choices=("geographic", "random", "file"))
    parser.add_argument("--partition-file", type=Path)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-minutes", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if args.device == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        # Short timeout: if one rank runs out of memory, the others should
        # fail within minutes instead of waiting the default 10
        dist.init_process_group(
            "nccl", device_id=device,
            timeout=datetime.timedelta(minutes=args.timeout_minutes),
        )
    else:
        device = torch.device("cpu")
        # INIT_METHOD (e.g. file://...) allows launching without torchrun
        dist.init_process_group(
            "gloo", init_method=os.environ.get("INIT_METHOD"),
            rank=rank, world_size=world_size,
        )
    results = []
    for hidden in args.hidden:
        try:
            entry = run(args, hidden, rank, world_size, device)
            entry["status"] = "ok"
        except torch.OutOfMemoryError as e:
            entry = {"hidden": hidden, "status": "oom",
                     "error": str(e).splitlines()[0]}
        if rank == 0:
            print(json.dumps(entry), flush=True)
        results.append(entry)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if rank == 0:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"world_size": world_size, "graph": args.graph,
                       "results": results}, f, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
