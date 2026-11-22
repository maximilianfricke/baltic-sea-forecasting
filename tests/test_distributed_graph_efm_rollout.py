"""AR4 equivalence against the original GraphEFM training_step.

Uses the original encoders, decoder, embeddings, WMSE and feedback methods on
a small two-level graph. Gloo covers 1/2/4 processes locally. The same case can
be launched with torchrun on CUDA to validate NCCL before physical-data jobs.
"""

# Standard library
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

# Third-party
import pytorch_lightning as pl
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

# First-party
from benchmarks.model_parallel_real import (
    _model_args, _synchronize_gradients, _load_workload,
    _build_sharded_training_modules,
)
from neural_lam import metrics, utils
from neural_lam.distributed_message_passing import (
    build_graph_partition, select_owned,
)
from neural_lam.models.distributed_graph_efm import distributed_core_from_graph_efm
from neural_lam.models.distributed_graph_efm_training import DistributedGraphEFMTraining
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.models.hi_graph_latent_decoder import HiGraphLatentDecoder
from neural_lam.models.hi_graph_latent_encoder import HiGraphLatentEncoder


class ToyGraphEFM(GraphEFM):
    """Original training implementation with a locally constructed graph."""

    def __init__(self, checkpointing=True):
        pl.LightningModule.__init__(self)
        self.args = _model_args(SimpleNamespace(
            ar_steps=4, batch_size=2, graph="toy", hidden_dim=4,
            processor_layers=2, lr=0.001, seed=42,
        ))
        self.args.latent_dim = 3
        self.input_steps = 2
        self.grad_checkpointing = checkpointing
        self.boundary_forced = False
        self.atmosphere_forced = True
        self.use_atmosphere_g2m = False
        self.concat_atmosphere = True
        self.hierarchical_graph = True
        self.embedd_m2m = True
        self.output_std = False
        self.sample_obs_noise = False
        self.use_density = False
        self.kl_beta = 1.0
        self.crps_weight = 0.0
        self.num_state_vars = 2
        self.predictions = []
        self.register_buffer("grid_static_features", torch.randn(8, 1))
        self.register_buffer("loss_mask", torch.tensor(
            [[1., 0.], [.5, 1.], [1., 1.], [1., .5]] * 2
        ))
        self.register_buffer("per_var_std", torch.tensor([1., 1.5]))
        for name in ("lower_upper", "lower", "upper"):
            self.register_buffer("clamp_" + name + "_idx", torch.empty(0, dtype=torch.long))
        ring = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                             [1, 0, 2, 1, 3, 2, 0, 3]])
        self.m2m_edge_index = (ring, ring.clone())
        up = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        self.mesh_up_edge_index = (up,)
        self.mesh_down_edge_index = (up.flip(0),)
        self.g2m_edge_index = torch.tensor([
            list(range(8)) + list(range(8)),
            [i % 4 for i in range(8)] + [(i + 1) % 4 for i in range(8)],
        ])
        self.m2g_edge_index = self.g2m_edge_index.flip(0)
        self.mesh_static_features = [torch.randn(4, 1), torch.randn(4, 1)]
        self.m2m_features = [torch.randn(8, 1), torch.randn(8, 1)]
        self.mesh_up_features = [torch.randn(4, 1)]
        self.mesh_down_features = [torch.randn(4, 1)]
        self.register_buffer("g2m_features", torch.randn(16, 1))
        self.register_buffer("m2g_features", torch.randn(16, 1))
        # 2 states x 2 channels + forcing + static + atmosphere = 7.
        self.interior_embedder = utils.make_mlp([7, 4, 4])
        self.interior_current_embedder = utils.make_mlp([9, 4, 4])
        self.g2m_embedder = utils.make_mlp([1, 4, 4])
        self.m2g_embedder = utils.make_mlp([1, 4, 4])
        for name, count in (("mesh", 2), ("m2m", 2), ("mesh_up", 1), ("mesh_down", 1)):
            setattr(self, name + "_embedders", nn.ModuleList([
                utils.make_mlp([1, 4, 4]) for _ in range(count)
            ]))
        common = (3, self.g2m_edge_index, self.m2m_edge_index,
                  self.mesh_up_edge_index, 4, 4, 4, 4)
        self.prior_model = HiGraphLatentEncoder(
            *common, intra_level_layers=1, output_dist="isotropic",
            num_grid_con_mesh_nodes=4,
        )
        self.encoder = HiGraphLatentEncoder(
            *common, intra_level_layers=1, output_dist="diagonal",
            num_grid_con_mesh_nodes=4,
        )
        self.decoder = HiGraphLatentDecoder(
            self.g2m_edge_index, self.m2m_edge_index, self.m2g_edge_index,
            self.mesh_up_edge_index, self.mesh_down_edge_index,
            4, 4, 4, 4, 3, 2, intra_level_layers=2,
            output_std=False, num_grid_con_mesh_nodes=4, num_interior_nodes=8,
        )
        self.loss = metrics.wmse

    def log_dict(self, *args, **kwargs):
        pass

    def compute_step_loss(self, *args):
        output = super().compute_step_loss(*args)
        self.predictions.append(output[2].detach())
        return output

    def to(self, device):
        # Production uses BufferList; keep toy list tensors on the same device.
        super().to(device)
        for name in ("mesh_static_features", "m2m_features", "mesh_up_features", "mesh_down_features"):
            setattr(self, name, [tensor.to(device) for tensor in getattr(self, name)])
        return self


def _batch(device):
    gen = torch.Generator().manual_seed(781)
    shapes = ((2, 2, 8, 2), (2, 4, 8, 2), (2, 4, 8, 1),
              (2, 4, 0, 0), (2, 4, 8, 1), (2, 4))
    return tuple(torch.randn(*shape, generator=gen).to(device) for shape in shapes)


def rollout_case(rank, world_size, checkpointing=True, device=torch.device("cpu")):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    reference = ToyGraphEFM(checkpointing).to(device)
    frontend = copy.deepcopy(reference)
    grid_owners = torch.arange(8) % world_size
    mesh_owners = (torch.arange(4) % world_size, (torch.arange(4) + 1) % world_size)
    partition = build_graph_partition(
        reference.g2m_edge_index, reference.m2g_edge_index,
        reference.m2m_edge_index, grid_owners, grid_owners, mesh_owners,
        rank, world_size, reference.mesh_up_edge_index,
        reference.mesh_down_edge_index,
    ).to(device)
    core = distributed_core_from_graph_efm(frontend, partition)
    core_pairs = list(core.parameter_pairs(
        reference.prior_model, reference.encoder, reference.decoder
    ))
    frontend.prior_model = frontend.encoder = frontend.decoder = None
    training = DistributedGraphEFMTraining(frontend, core, partition)
    batch = _batch(device)
    local_batch = tuple(
        select_owned(tensor, partition.interior_grid_nodes)
        if index in (0, 1, 2, 4) else tensor
        for index, tensor in enumerate(batch)
    )
    reference_opt = torch.optim.AdamW(reference.parameters(), lr=.001, betas=(.9, .95), weight_decay=.1)
    local_opt = torch.optim.AdamW(training.parameters(), lr=.001, betas=(.9, .95), weight_decay=.1)
    # Two updates also validate checkpoint noise replay and optimizer state.
    max_forecast_error = max_gradient_error = 0.0
    for update in range(2):
        reference_opt.zero_grad(set_to_none=True)
        local_opt.zero_grad(set_to_none=True)
        reference.predictions = []
        seed = 1000 + update
        torch.manual_seed(seed)
        expected_loss = reference.training_step(batch)
        expected_predictions = torch.stack(reference.predictions[:4], dim=1)
        expected_loss.backward()
        generator = torch.Generator(device=device).manual_seed(seed)
        output = training(local_batch, generator, return_predictions=True)
        torch.testing.assert_close(
            output["predictions"],
            select_owned(expected_predictions, partition.interior_grid_nodes),
            rtol=3e-4, atol=3e-5,
        )
        max_forecast_error = max(max_forecast_error, float((
            output["predictions"].detach()
            - select_owned(expected_predictions, partition.interior_grid_nodes)
        ).abs().max()))
        summed_loss = output["loss"].detach().clone()
        dist.all_reduce(summed_loss)
        torch.testing.assert_close(summed_loss, expected_loss, rtol=1e-5, atol=1e-4)
        output["loss"].backward()
        _synchronize_gradients(list(training.parameters()), world_size)
        reference_params = dict(reference.named_parameters())
        pairs = core_pairs + [
            (name, parameter, reference_params[name])
            for name, parameter in frontend.named_parameters()
        ]
        for name, local, original in pairs:
            assert local.grad is not None, name
            torch.testing.assert_close(local.grad, original.grad, rtol=5e-4, atol=5e-4, msg=name)
            max_gradient_error = max(max_gradient_error, float(
                (local.grad - original.grad).abs().max()
            ))
        reference_opt.step()
        local_opt.step()
        for name, local, original in pairs:
            torch.testing.assert_close(local, original, rtol=2e-4, atol=2e-4, msg=name)
    return {
        "rank": rank,
        "autoregressive_steps": 4,
        "gradient_checkpointing": checkpointing,
        "validated_optimizer_updates": 2,
        "max_forecast_absolute_error": max_forecast_error,
        "max_gradient_absolute_error": max_gradient_error,
    }


def _worker(rank, world_size, init_method, checkpointing):
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            init_method=init_method, timeout=timedelta(seconds=90))
    try:
        rollout_case(rank, world_size, checkpointing)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_ar4_forecasts_gradients_and_optimizer_match_original(tmp_path, world_size, checkpointing):
    mp.spawn(_worker, args=(world_size, f"file://{tmp_path / 'init'}", checkpointing),
             nprocs=world_size, join=True)


@pytest.mark.skipif(
    not Path("data/physical_repeat_16/global_ocean_1.yaml").exists(),
    reason="local physical-repeat dataset is optional",
)
def test_physical_max_baseline_ar4_full_graph(tmp_path):
    """Exercise the actual H64/L2/B2 model, masks, clamping and density feedback."""
    torch.set_num_threads(2)
    torch.manual_seed(42)
    args = SimpleNamespace(
        config_path=Path("data/physical_repeat_16/global_ocean_1.yaml"),
        graph="global_cluster_graph", ar_steps=4, hidden_dim=64,
        processor_layers=2, batch_size=2, lr=.001, seed=42,
        partition_strategy="geographic", latitude_cut=-5., longitude_seam=120.,
    )
    dataset, reference, _ = _load_workload(args)
    reference.log_dict = lambda *a, **kw: None
    reference.train()
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    frontend, core, partition, _ = _build_sharded_training_modules(
        copy.deepcopy(reference), args, 0, 1, torch.device("cpu")
    )
    training = DistributedGraphEFMTraining(frontend, core, partition)
    assert len(dataset) == 473
    assert reference.grad_checkpointing and frontend.grad_checkpointing
    assert sum(p.numel() for p in training.parameters()) == 645020
    dist.init_process_group("gloo", rank=0, world_size=1,
                            init_method=f"file://{tmp_path / 'physical-init'}")
    try:
        torch.manual_seed(143)
        expected = reference.training_step(batch)
        expected.backward()
        actual = training(batch, torch.Generator().manual_seed(143))
        torch.testing.assert_close(actual["loss"], expected, rtol=1e-5, atol=1.)
        actual["loss"].backward()
        pairs = list(core.parameter_pairs(
            reference.prior_model, reference.encoder, reference.decoder
        ))
        reference_params = dict(reference.named_parameters())
        pairs += [(n, p, reference_params[n]) for n, p in frontend.named_parameters()]
        for name, local, original in pairs:
            torch.testing.assert_close(local.grad, original.grad,
                                        rtol=2e-3, atol=.1, msg=name)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        report = rollout_case(rank, dist.get_world_size(), device=device)
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, report)
        dist.barrier()
        if rank == 0:
            print("AR4 checkpointed GraphEFM forecasts, loss, gradients and AdamW: PASS", flush=True)
            if os.environ.get("NJORD_AR4_SMOKE_RESULT"):
                output = Path(os.environ["NJORD_AR4_SMOKE_RESULT"])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({
                    "status": "completed", "test": "ar4_checkpoint_equivalence",
                    "backend": "nccl", "world_size": dist.get_world_size(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "ranks": reports,
                }, indent=2) + "\n")
    finally:
        dist.destroy_process_group()
