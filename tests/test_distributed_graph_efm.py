"""End-to-end tests for model-parallel hierarchical Graph-EFM."""

# Standard library
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

# Third-party
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# First-party
from neural_lam.distributed_message_passing import (
    balanced_random_owners,
    build_graph_partition,
    latitude_cut_owners,
    select_graph_embeddings,
    select_owned,
)
from neural_lam.geographic_partitioning import (
    geographic_rcb_owners,
    partition_indices,
)
from neural_lam.models.distributed_graph_efm import (
    build_partition_from_graph_efm,
    distributed_core_from_graph_efm,
)
from neural_lam.models.hi_graph_latent_decoder import HiGraphLatentDecoder
from neural_lam.models.hi_graph_latent_encoder import HiGraphLatentEncoder


WORLD_SIZE = 2
BATCH_SIZE = 2
HIDDEN_DIM = 4
LATENT_DIM = 3
STATE_DIM = 2
CUDA_RTOL = 1e-4
CUDA_ATOL = 2e-5


def _assert_close(actual, expected, **kwargs):
    if actual.is_cuda:
        kwargs.setdefault("rtol", CUDA_RTOL)
        kwargs.setdefault("atol", CUDA_ATOL)
    torch.testing.assert_close(actual, expected, **kwargs)


def _toy_hierarchy():
    g2m = torch.tensor(
        [
            [0, 2, 1, 3, 4, 5, 0, 5],
            [0, 0, 1, 1, 2, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    m2g = torch.tensor(
        [
            [0, 1, 0, 2, 1, 2, 1, 3, 2, 3, 0, 3],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ],
        dtype=torch.long,
    )
    m2m = (
        torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 0],
                [1, 0, 2, 1, 3, 2, 0, 3],
            ],
            dtype=torch.long,
        ),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    mesh_up = (
        torch.tensor(
            [[0, 1, 2, 3], [0, 0, 1, 1]],
            dtype=torch.long,
        ),
    )
    mesh_down = (
        torch.tensor(
            [[0, 0, 1, 1], [0, 1, 2, 3]],
            dtype=torch.long,
        ),
    )
    grid_owners = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    mesh_owners = (
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    return {
        "g2m": g2m,
        "m2g": m2g,
        "m2m": m2m,
        "mesh_up": mesh_up,
        "mesh_down": mesh_down,
        "grid_owners": grid_owners,
        "mesh_owners": mesh_owners,
    }


def _global_inputs(requires_grad, device=torch.device("cpu")):
    generator = torch.Generator().manual_seed(4200)

    def make(*shape):
        tensor = torch.randn(*shape, generator=generator).to(device)
        return tensor.requires_grad_(requires_grad)

    graph_emb = {
        "mesh": [
            make(BATCH_SIZE, 4, HIDDEN_DIM),
            make(BATCH_SIZE, 2, HIDDEN_DIM),
        ],
        "g2m": make(BATCH_SIZE, 8, HIDDEN_DIM),
        "m2g": make(BATCH_SIZE, 12, HIDDEN_DIM),
        "m2m": [
            make(BATCH_SIZE, 8, HIDDEN_DIM),
            make(BATCH_SIZE, 2, HIDDEN_DIM),
        ],
        "mesh_up": [make(BATCH_SIZE, 4, HIDDEN_DIM)],
        "mesh_down": [make(BATCH_SIZE, 4, HIDDEN_DIM)],
    }
    return {
        "grid_prev": make(BATCH_SIZE, 6, HIDDEN_DIM),
        "grid_current": make(BATCH_SIZE, 6, HIDDEN_DIM),
        "grid_interior": make(BATCH_SIZE, 6, HIDDEN_DIM),
        "last_state": make(BATCH_SIZE, 6, STATE_DIM),
        "posterior_noise": make(BATCH_SIZE, 2, LATENT_DIM).detach(),
        "prior_noise": make(BATCH_SIZE, 2, LATENT_DIM).detach(),
        "graph_emb": graph_emb,
    }


def _reference_models(graph):
    torch.manual_seed(101)
    prior = HiGraphLatentEncoder(
        LATENT_DIM,
        graph["g2m"],
        graph["m2m"],
        graph["mesh_up"],
        HIDDEN_DIM,
        HIDDEN_DIM,
        HIDDEN_DIM,
        HIDDEN_DIM,
        intra_level_layers=1,
        hidden_layers=1,
        output_dist="isotropic",
        num_grid_con_mesh_nodes=4,
    )
    encoder = HiGraphLatentEncoder(
        LATENT_DIM,
        graph["g2m"],
        graph["m2m"],
        graph["mesh_up"],
        HIDDEN_DIM,
        HIDDEN_DIM,
        HIDDEN_DIM,
        HIDDEN_DIM,
        intra_level_layers=1,
        hidden_layers=1,
        output_dist="diagonal",
        num_grid_con_mesh_nodes=4,
    )
    decoder = HiGraphLatentDecoder(
        graph["g2m"],
        graph["m2m"],
        graph["m2g"],
        graph["mesh_up"],
        graph["mesh_down"],
        HIDDEN_DIM,
        HIDDEN_DIM,
        HIDDEN_DIM,
        HIDDEN_DIM,
        LATENT_DIM,
        STATE_DIM,
        intra_level_layers=1,
        hidden_layers=1,
        output_std=True,
        num_grid_con_mesh_nodes=4,
        num_interior_nodes=6,
    )
    return prior, encoder, decoder


def _graph_efm_model(graph, reference_models):
    args = SimpleNamespace(
        hidden_dim=HIDDEN_DIM,
        hidden_dim_grid=HIDDEN_DIM,
        hidden_dim_mesh_nodes=HIDDEN_DIM,
        hidden_dim_edge=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        learn_prior=True,
        prior_processor_layers=1,
        encoder_processor_layers=1,
        processor_layers=1,
        hidden_layers=1,
        prior_dist="isotropic",
        output_std=True,
    )
    return SimpleNamespace(
        args=args,
        hierarchical_graph=True,
        g2m_edge_index=graph["g2m"],
        m2g_edge_index=graph["m2g"],
        m2m_edge_index=graph["m2m"],
        mesh_up_edge_index=graph["mesh_up"],
        mesh_down_edge_index=graph["mesh_down"],
        mesh_static_features=(torch.zeros(4, 1), torch.zeros(2, 1)),
        num_state_vars=STATE_DIM,
        prior_model=reference_models[0],
        encoder=reference_models[1],
        decoder=reference_models[2],
    )


def _reference_training_forward(models, inputs):
    prior_model, encoder, decoder = models
    graph_emb = inputs["graph_emb"]
    prior = prior_model(inputs["grid_prev"], graph_emb=graph_emb)
    posterior = encoder(inputs["grid_current"], graph_emb=graph_emb)
    latent = (
        posterior.mean
        + posterior.stddev * inputs["posterior_noise"]
    )
    pred_mean, pred_std = decoder(
        inputs["grid_prev"],
        inputs["grid_interior"],
        latent,
        inputs["last_state"],
        graph_emb,
    )
    kl = torch.distributions.kl_divergence(posterior, prior).sum(dim=(1, 2))
    return {
        "prior": prior,
        "posterior": posterior,
        "latent_sample": latent,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "local_kl": kl,
    }


def _reference_inference_forward(models, inputs):
    prior_model, _, decoder = models
    graph_emb = inputs["graph_emb"]
    prior = prior_model(inputs["grid_prev"], graph_emb=graph_emb)
    latent = prior.mean + prior.stddev * inputs["prior_noise"]
    pred_mean, pred_std = decoder(
        inputs["grid_prev"],
        inputs["grid_interior"],
        latent,
        inputs["last_state"],
        graph_emb,
    )
    return {
        "prior": prior,
        "latent_sample": latent,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
    }


def _local_inputs(global_inputs, partition):
    def local(tensor, indices, requires_grad=True):
        result = select_owned(tensor.detach(), indices).clone()
        return result.requires_grad_(requires_grad)

    global_graph = global_inputs["graph_emb"]
    local_graph = select_graph_embeddings(global_graph, partition)

    for key in ("g2m", "m2g"):
        local_graph[key] = local_graph[key].detach().clone().requires_grad_()
    for key in ("mesh", "m2m", "mesh_up", "mesh_down"):
        local_graph[key] = [
            tensor.detach().clone().requires_grad_()
            for tensor in local_graph[key]
        ]

    return {
        "grid_prev": local(
            global_inputs["grid_prev"],
            partition.full_grid_nodes,
        ),
        "grid_current": local(
            global_inputs["grid_current"],
            partition.full_grid_nodes,
        ),
        "grid_interior": local(
            global_inputs["grid_interior"],
            partition.interior_grid_nodes,
        ),
        "last_state": local(
            global_inputs["last_state"],
            partition.interior_grid_nodes,
        ),
        "posterior_noise": local(
            global_inputs["posterior_noise"],
            partition.mesh_nodes[-1],
            requires_grad=False,
        ),
        "prior_noise": local(
            global_inputs["prior_noise"],
            partition.mesh_nodes[-1],
            requires_grad=False,
        ),
        "graph_emb": local_graph,
    }


def _assert_local_outputs(local_output, reference_output, partition):
    top_nodes = partition.mesh_nodes[-1]
    grid_nodes = partition.interior_grid_nodes
    for distribution_name in ("prior", "posterior"):
        if distribution_name not in local_output:
            continue
        local_dist = local_output[distribution_name]
        reference_dist = reference_output[distribution_name]
        _assert_close(
            local_dist.mean,
            select_owned(reference_dist.mean, top_nodes),
        )
        _assert_close(
            local_dist.stddev,
            select_owned(reference_dist.stddev, top_nodes),
        )
    _assert_close(
        local_output["latent_sample"],
        select_owned(reference_output["latent_sample"], top_nodes),
    )
    _assert_close(
        local_output["pred_mean"],
        select_owned(reference_output["pred_mean"], grid_nodes),
    )
    _assert_close(
        local_output["pred_std"],
        select_owned(reference_output["pred_std"], grid_nodes),
    )
    if "local_kl" in local_output:
        reference_kl = torch.distributions.kl_divergence(
            reference_output["posterior"],
            reference_output["prior"],
        ).index_select(1, top_nodes).sum(dim=(1, 2))
        _assert_close(local_output["local_kl"], reference_kl)


def _assert_local_grad(local_tensor, global_tensor, indices):
    _assert_close(
        local_tensor.grad,
        select_owned(global_tensor.grad, indices),
    )


def _assert_input_gradients(local_inputs, global_inputs, partition):
    _assert_local_grad(
        local_inputs["grid_prev"],
        global_inputs["grid_prev"],
        partition.full_grid_nodes,
    )
    _assert_local_grad(
        local_inputs["grid_current"],
        global_inputs["grid_current"],
        partition.full_grid_nodes,
    )
    _assert_local_grad(
        local_inputs["grid_interior"],
        global_inputs["grid_interior"],
        partition.interior_grid_nodes,
    )
    _assert_local_grad(
        local_inputs["last_state"],
        global_inputs["last_state"],
        partition.interior_grid_nodes,
    )

    local_graph = local_inputs["graph_emb"]
    global_graph = global_inputs["graph_emb"]
    _assert_local_grad(
        local_graph["g2m"],
        global_graph["g2m"],
        partition.g2m.edge_ids,
    )
    _assert_local_grad(
        local_graph["m2g"],
        global_graph["m2g"],
        partition.m2g.edge_ids,
    )
    for local_rep, global_rep, nodes in zip(
        local_graph["mesh"],
        global_graph["mesh"],
        partition.mesh_nodes,
    ):
        _assert_local_grad(local_rep, global_rep, nodes)
    for key, parts in (
        ("m2m", partition.m2m),
        ("mesh_up", partition.mesh_up),
        ("mesh_down", partition.mesh_down),
    ):
        for local_rep, global_rep, part in zip(
            local_graph[key],
            global_graph[key],
            parts,
        ):
            _assert_local_grad(local_rep, global_rep, part.edge_ids)


def _distributed_case(rank, device=torch.device("cpu")):
    graph = _toy_hierarchy()
    partition = build_graph_partition(
        graph["g2m"],
        graph["m2g"],
        graph["m2m"],
        graph["grid_owners"],
        graph["grid_owners"],
        graph["mesh_owners"],
        rank,
        WORLD_SIZE,
        graph["mesh_up"],
        graph["mesh_down"],
    ).to(device)
    reference_models = tuple(
        model.to(device) for model in _reference_models(graph)
    )
    model = _graph_efm_model(graph, reference_models)
    core = distributed_core_from_graph_efm(model, partition)
    core_devices = {
        tensor.device
        for tensor in (*core.parameters(), *core.buffers())
    }
    assert core_devices == {device}

    reference_inputs = _global_inputs(requires_grad=True, device=device)
    local_inputs = _local_inputs(reference_inputs, partition)

    with torch.no_grad():
        reference_inference = _reference_inference_forward(
            reference_models,
            reference_inputs,
        )
        local_inference = core.inference_forward(
            local_inputs["grid_prev"],
            local_inputs["grid_interior"],
            local_inputs["last_state"],
            local_inputs["graph_emb"],
            local_inputs["prior_noise"],
        )
        _assert_local_outputs(
            local_inference,
            reference_inference,
            partition,
        )

    reference_output = _reference_training_forward(
        reference_models,
        reference_inputs,
    )
    reference_loss = (
        reference_output["pred_mean"].square().sum()
        + reference_output["pred_std"].square().sum()
        + reference_output["local_kl"].sum()
    )
    reference_loss.backward()

    local_output = core.training_forward(
        local_inputs["grid_prev"],
        local_inputs["grid_current"],
        local_inputs["grid_interior"],
        local_inputs["last_state"],
        local_inputs["graph_emb"],
        local_inputs["posterior_noise"],
    )
    _assert_local_outputs(local_output, reference_output, partition)
    local_loss = (
        local_output["pred_mean"].square().sum()
        + local_output["pred_std"].square().sum()
        + local_output["local_kl"].sum()
    )
    local_loss.backward()

    _assert_input_gradients(local_inputs, reference_inputs, partition)
    for name, local_parameter, reference_parameter in core.parameter_pairs(
        *reference_models
    ):
        assert local_parameter.grad is not None, name
        assert reference_parameter.grad is not None, name
        dist.all_reduce(local_parameter.grad, op=dist.ReduceOp.SUM)
        _assert_close(
            local_parameter.grad,
            reference_parameter.grad,
            msg=lambda message, param_name=name: f"{param_name}: {message}",
        )


def _distributed_worker(rank, init_method):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        _distributed_case(rank)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _distributed_cuda_worker():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NCCL integration smoke")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"expected {WORLD_SIZE} ranks, received {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        started_at = datetime.now(timezone.utc).isoformat()
        torch.cuda.reset_peak_memory_stats(device)
        wall_start = time.perf_counter()
        _distributed_case(rank, device=device)
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - wall_start
        local_metrics = torch.tensor(
            [
                wall_seconds,
                torch.cuda.max_memory_allocated(device) / (1024**2),
            ],
            device=device,
            dtype=torch.float64,
        )
        gathered_metrics = [
            torch.zeros_like(local_metrics) for _ in range(world_size)
        ]
        dist.all_gather(gathered_metrics, local_metrics)
        dist.barrier()
        if rank == 0:
            print("Two-GPU hierarchical GraphEFM core smoke: PASS")
            output = os.environ.get("NJORD_CORE_SMOKE_RESULT")
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result = {
                    "schema_version": 1,
                    "status": "completed",
                    "test": "hierarchical_distributed_graph_efm_core_smoke",
                    "comparable_to_training_baseline": False,
                    "comparison_note": (
                        "Synthetic hierarchical toy graph correctness test; "
                        "wall time includes reference and assertion work."
                    ),
                    "started_at_utc": started_at,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "git_commit": os.environ.get("NJORD_GIT_COMMIT"),
                    "git_dirty": os.environ.get("NJORD_GIT_DIRTY"),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "backend": "nccl",
                    "world_size": world_size,
                    "torch_version": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "comparison_tolerances": {
                        "rtol": CUDA_RTOL,
                        "atol": CUDA_ATOL,
                    },
                    "validated_components": [
                        "prior",
                        "posterior",
                        "grid_to_mesh",
                        "mesh_up",
                        "mesh_intra_level",
                        "mesh_down",
                        "mesh_to_grid",
                        "forward",
                        "backward",
                        "input_gradients",
                        "summed_parameter_gradients",
                    ],
                    "rank_metrics": [
                        {
                            "rank": peer,
                            "wall_clock_smoke_seconds": values[0],
                            "peak_allocated_mib": values[1],
                        }
                        for peer, values in enumerate(
                            metrics.cpu().tolist()
                            for metrics in gathered_metrics
                        )
                    ],
                }
                output_path.write_text(
                    json.dumps(result, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"Wrote {output_path}")
        print(
            f"rank={rank} peak_memory_mib="
            f"{torch.cuda.max_memory_allocated(device) / (1024**2):.1f}"
        )
    finally:
        dist.destroy_process_group()


def test_balanced_random_owners_are_reproducible_and_balanced():
    owners = balanced_random_owners(11, 3, seed=7)
    torch.testing.assert_close(owners, balanced_random_owners(11, 3, seed=7))
    counts = torch.bincount(owners, minlength=3)
    assert int(counts.max() - counts.min()) <= 1


def test_latitude_cut_owners_split_at_equator():
    lon_lat = torch.tensor(
        [
            [-120.0, -60.0],
            [20.0, -0.1],
            [150.0, 0.0],
            [-30.0, 75.0],
        ]
    )
    torch.testing.assert_close(
        latitude_cut_owners(lon_lat),
        torch.tensor([0, 0, 1, 1]),
    )


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_geographic_rcb_is_balanced_complete_and_deterministic(world_size):
    longitude = torch.linspace(-179.0, 179.0, 13)
    latitude = torch.linspace(-80.0, 80.0, 7)
    lon_grid, lat_grid = torch.meshgrid(
        longitude,
        latitude,
        indexing="ij",
    )
    lon_lat = torch.stack((lon_grid.flatten(), lat_grid.flatten()), dim=1)

    owners = geographic_rcb_owners(
        lon_lat,
        world_size=world_size,
        longitude_seam=120.0,
    )
    torch.testing.assert_close(
        owners,
        geographic_rcb_owners(
            lon_lat,
            world_size=world_size,
            longitude_seam=120.0,
        ),
    )
    counts = torch.bincount(owners, minlength=world_size)
    assert int(counts.max() - counts.min()) <= 1

    indices = partition_indices(owners, world_size)
    combined = torch.cat(indices)
    assert combined.numel() == lon_lat.shape[0]
    assert torch.unique(combined).numel() == lon_lat.shape[0]
    torch.testing.assert_close(
        torch.sort(combined).values,
        torch.arange(lon_lat.shape[0]),
    )


def test_geographic_rcb_four_way_partitions_are_geographically_ordered():
    lon_lat = torch.tensor(
        [
            [130.0, -60.0],
            [-80.0, -60.0],
            [130.0, -20.0],
            [-80.0, -20.0],
            [130.0, 20.0],
            [-80.0, 20.0],
            [130.0, 60.0],
            [-80.0, 60.0],
        ]
    )
    owners = geographic_rcb_owners(
        lon_lat,
        world_size=4,
        longitude_seam=120.0,
    )
    assert set(owners[:4].tolist()) == {0, 1}
    assert set(owners[4:].tolist()) == {2, 3}
    assert owners[0] == owners[2]
    assert owners[1] == owners[3]
    assert owners[4] == owners[6]
    assert owners[5] == owners[7]


def test_existing_graph_efm_can_be_converted_to_distributed_core():
    graph = _toy_hierarchy()
    reference_models = _reference_models(graph)
    model = _graph_efm_model(graph, reference_models)

    partition = build_partition_from_graph_efm(
        model,
        rank=0,
        world_size=2,
        strategy="random",
    )
    core = distributed_core_from_graph_efm(model, partition)
    for _, local_parameter, reference_parameter in core.parameter_pairs(
        *reference_models
    ):
        assert torch.equal(local_parameter, reference_parameter)


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is required for the model-parallel test",
)
def test_full_hierarchical_graph_efm_forward_backward(tmp_path):
    init_method = f"file://{tmp_path / 'graph_efm_gloo_init'}"
    mp.spawn(
        _distributed_worker,
        args=(init_method,),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    _distributed_cuda_worker()
