"""Distributed numerical checks using real mesh modules and CPU/Gloo.

Run with ``python -m pytest tests/test_graph_parallel.py -q``. No dataset or
GPU is needed; the same worker can additionally run on CUDA/NCCL.
"""

# Standard library
import copy
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

# Third-party
import pytest
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# First-party
from neural_lam import metrics, utils
from neural_lam.graph_parallel import (
    configure_graph_parallel,
    parameter_reductions,
    synchronize_gradients,
)
from neural_lam.graph_parallel_strategy import GraphParallelStrategy
from neural_lam.graphs.partition import (
    EdgePartition,
    MeshPartition,
    graph_fingerprint,
    spatial_owners,
)
from neural_lam.interaction_net import InteractionNet, PropagationNet
from neural_lam.models.constant_latent_encoder import ConstantLatentEncoder
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.models.hi_graph_latent_decoder import HiGraphLatentDecoder
from neural_lam.models.hi_graph_latent_encoder import HiGraphLatentEncoder


class TinyEFM(GraphEFM):
    """Synthetic graph fixture using GraphEFM's actual training/forecast methods."""

    def __init__(self, layers=2, learned_prior=True, beta=0.1):
        pl.LightningModule.__init__(self)
        self.args = SimpleNamespace(crps_alpha=0.95, seed=42, optimizer="adamw")
        self.hierarchical_graph = True
        self.input_steps = 1
        self.boundary_forced = self.atmosphere_forced = (
            self.concat_atmosphere
        ) = False
        self.output_std = True
        self.sample_obs_noise = True
        self.use_density = False
        self.grad_checkpointing = False
        self.restore_opt = True
        self.train_distr_crps = False
        self.kl_beta, self.crps_weight = beta, 0.1
        self.num_interior_nodes = self.num_total_grid_nodes = 6
        self.mesh_static_features = utils.BufferList(
            [torch.randn(n, 2) for n in (5, 3, 1)], persistent=False
        )
        self.register_buffer(
            "grid_static_features", torch.randn(6, 2), persistent=False
        )
        self.register_buffer("loss_mask", torch.ones(6, 2), persistent=False)
        self.register_buffer("per_var_std", torch.ones(2), persistent=False)
        self.loss = metrics.get_metric("nll")
        same_edges = [
            torch.tensor([[0, 1, 1, 2, 3, 4, 0, 4], [0, 0, 1, 2, 3, 4, 4, 1]]),
            torch.tensor([[0, 1, 2, 2], [0, 1, 2, 0]]),
            torch.tensor([[0], [0]]),
        ]
        up_edges = [
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 2]]),
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
        ]
        down_edges = [e.flip(0) for e in up_edges]
        self.m2m_edge_index = utils.BufferList(same_edges, persistent=False)
        self.mesh_up_edge_index = utils.BufferList(up_edges, persistent=False)
        self.mesh_down_edge_index = utils.BufferList(
            down_edges, persistent=False
        )
        for kind, edges in (
            ("m2m", same_edges),
            ("mesh_up", up_edges),
            ("mesh_down", down_edges),
        ):
            setattr(
                self,
                f"{kind}_features",
                utils.BufferList(
                    [torch.randn(e.shape[1], 2) for e in edges],
                    persistent=False,
                ),
            )
        self.register_buffer(
            "g2m_edge_index",
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 0]]),
            persistent=False,
        )
        self.register_buffer(
            "m2g_edge_index", self.g2m_edge_index.flip(0), persistent=False
        )
        self.register_buffer(
            "g2m_features", torch.randn(6, 2), persistent=False
        )
        self.register_buffer(
            "m2g_features", torch.randn(6, 2), persistent=False
        )
        for name, count in (
            ("mesh_embedders", 3),
            ("m2m_embedders", 3),
            ("mesh_up_embedders", 2),
            ("mesh_down_embedders", 2),
        ):
            setattr(
                self,
                name,
                torch.nn.ModuleList(
                    [utils.make_mlp([2, 4, 4]) for _ in range(count)]
                ),
            )
        self.embedd_m2m = layers > 0
        self.interior_embedder = utils.make_mlp([5, 4, 4])
        self.interior_current_embedder = utils.make_mlp([7, 4, 4])
        self.g2m_embedder = utils.make_mlp([2, 4, 4])
        self.m2g_embedder = utils.make_mlp([2, 4, 4])
        encoder_args = (
            2,
            self.g2m_edge_index,
            self.m2m_edge_index,
            self.mesh_up_edge_index,
            4,
            4,
            4,
            4,
            layers,
        )
        self.encoder = HiGraphLatentEncoder(
            *encoder_args, output_dist="diagonal", num_grid_con_mesh_nodes=5
        )
        self.prior_model = (
            HiGraphLatentEncoder(
                *encoder_args, output_dist="diagonal", num_grid_con_mesh_nodes=5
            )
            if learned_prior
            else ConstantLatentEncoder(2, 1)
        )
        self.decoder = HiGraphLatentDecoder(
            self.g2m_edge_index,
            self.m2m_edge_index,
            self.m2g_edge_index,
            self.mesh_up_edge_index,
            self.mesh_down_edge_index,
            4,
            4,
            4,
            4,
            2,
            2,
            layers,
            num_grid_con_mesh_nodes=5,
            num_interior_nodes=6,
        )
        self.logged = {}

    def get_clamped_new_state(self, delta, previous):
        return previous + delta

    def log_dict(self, values, **kwargs):
        self.logged = {key: value.detach() for key, value in values.items()}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


def _check_layer(
    rank,
    size,
    net_type,
    mean,
    empty,
    overlap=False,
    device="cpu",
    no_edges=False,
):
    torch.manual_seed(15)
    dtype = torch.float64 if device == "cpu" else torch.float32
    rtol, atol = (1e-9, 1e-10) if device == "cpu" else (1e-4, 1e-5)
    edges = torch.tensor([[0, 0, 1, 2, 3, 4, 4], [0, 2, 1, 2, 3, 4, 1]])
    if no_edges:
        edges = torch.empty(2, 0, dtype=torch.long)
    owners = (
        torch.tensor([0, 1, 0, 1, 0])
        if not empty
        else torch.zeros(5, dtype=torch.long)
    )
    plan = EdgePartition(edges, owners, owners, rank, size)
    plan.overlap = overlap
    reference = net_type(
        edges, 4, num_rec=5, aggr="mean" if mean else "sum"
    ).to(device=device, dtype=dtype)
    local = copy.deepcopy(reference)
    local.graph_edge_partition, local.num_rec = plan, len(plan.owned_receivers)
    sender = torch.randn(
        2, 5, 4, dtype=dtype, device=device, requires_grad=True
    )
    receiver = torch.randn(
        2, 5, 4, dtype=dtype, device=device, requires_grad=True
    )
    edge = torch.randn(
        2, edges.shape[1], 4, dtype=dtype, device=device, requires_grad=True
    )
    expected_node, expected_edge = reference(sender, receiver, edge)
    (expected_node.square().sum() + expected_edge.square().sum()).backward()
    local_sender = sender.detach()[:, plan.owned_senders].requires_grad_()
    local_receiver = receiver.detach()[:, plan.owned_receivers].requires_grad_()
    local_edge = edge.detach()[:, plan.edge_ids].requires_grad_()
    node_out, edge_out = local(local_sender, local_receiver, local_edge)
    torch.testing.assert_close(
        node_out, expected_node[:, plan.owned_receivers], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        edge_out, expected_edge[:, plan.edge_ids], rtol=rtol, atol=atol
    )
    (node_out.square().sum() + edge_out.square().sum()).backward()
    for actual, expected in (
        (local_sender.grad, sender.grad[:, plan.owned_senders]),
        (local_receiver.grad, receiver.grad[:, plan.owned_receivers]),
        (local_edge.grad, edge.grad[:, plan.edge_ids]),
    ):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    synchronize_gradients(
        local, {name: "sum" for name, _ in local.named_parameters()}
    )
    for actual, expected in zip(local.parameters(), reference.parameters()):
        torch.testing.assert_close(
            actual.grad, expected.grad, rtol=rtol, atol=atol
        )


def _check_model(
    rank,
    size,
    layers,
    learned_prior,
    beta,
    checkpoint,
    overlap=False,
    device="cpu",
):
    torch.manual_seed(37)
    dtype = torch.float64 if device == "cpu" else torch.float32
    rtol, atol = (1e-7, 1e-8) if device == "cpu" else (1e-4, 1e-5)
    reference = TinyEFM(layers, learned_prior, beta).to(dtype=dtype)
    local = copy.deepcopy(reference)
    local.args.graph_parallel_overlap = overlap
    context = configure_graph_parallel(local, size, rank)
    reference.to(device)
    local.to(device)
    reductions = parameter_reductions(local)
    local.graph_parameter_reductions = reductions
    local.grad_checkpointing = reference.grad_checkpointing = checkpoint
    assert set(reference.state_dict()) == set(local.state_dict())
    assert sum(t.numel() for t in local.mesh_static_features) < sum(
        t.numel() for t in reference.mesh_static_features
    )
    batch = (
        torch.randn(2, 1, 6, 2, dtype=dtype, device=device),
        torch.randn(2, 2, 6, 2, dtype=dtype, device=device),
        torch.randn(2, 2, 6, 1, dtype=dtype, device=device),
        torch.empty(2, 2, 0, 0, dtype=dtype, device=device),
        torch.empty(2, 2, 0, 0, dtype=dtype, device=device),
        None,
    )
    # Lightning's sanity validation runs under inference_mode before training.
    step_args = (
        batch[0],
        batch[1][:, 0],
        batch[2][:, 0],
        batch[3][:, 0],
        batch[4][:, 0],
    )
    with torch.inference_mode():
        torch.manual_seed(101)
        expected_step = reference.compute_step_loss(*step_args)
        torch.manual_seed(101)
        actual_step = local.compute_step_loss(*step_args)
        for actual_tensor, expected_tensor in zip(actual_step, expected_step):
            if actual_tensor is not None:
                torch.testing.assert_close(
                    actual_tensor, expected_tensor, rtol=rtol, atol=atol
                )
    torch.manual_seed(123)
    expected = reference.training_step(batch)
    expected.backward()
    torch.manual_seed(123)
    actual = local.training_step(batch)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for key in reference.logged:
        torch.testing.assert_close(
            local.logged[key], reference.logged[key], rtol=rtol, atol=atol
        )
    actual.backward()
    synchronize_gradients(local, reductions)
    for (name, param), (_, expected_param) in zip(
        local.named_parameters(), reference.named_parameters()
    ):
        assert (param.grad is None) == (expected_param.grad is None), name
        if param.grad is not None:
            torch.testing.assert_close(
                param.grad,
                expected_param.grad,
                rtol=rtol,
                atol=atol,
                msg=lambda m: f"{name}: {m}",
            )
    actual_opt = local.configure_optimizers()
    expected_opt = reference.configure_optimizers()
    actual_opt.step()
    expected_opt.step()
    for param, expected_param in zip(
        local.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(param, expected_param, rtol=rtol, atol=atol)
    payload = {
        "state_dict": local.state_dict(),
        "optimizer_states": [actual_opt.state_dict()],
    }
    local.on_save_checkpoint(payload)
    # Reload parameter/optimizer/RNG state and reproduce the next random draw.
    noise_dist = torch.distributions.Normal(
        torch.zeros(3, device=device), torch.ones(3, device=device)
    )
    noise = local._rsample(noise_dist)
    local.on_load_checkpoint(payload)
    replay = local._rsample(noise_dist)
    torch.testing.assert_close(noise, replay, rtol=0, atol=0)
    assert context.partition.size == size
    dist.barrier()


def _distributed_worker(rank, size, rendezvous, checkpoint_dir, backend="gloo"):
    torch.set_num_threads(1)
    device = "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    dist.init_process_group(
        backend,
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=90),
    )
    try:
        for net_type, mean, empty in (
            (InteractionNet, False, False),
            (InteractionNet, True, False),
            (PropagationNet, True, False),
            (InteractionNet, False, True),
        ):
            _check_layer(rank, size, net_type, mean, empty, device=device)
            _check_layer(
                rank, size, net_type, mean, empty, overlap=True, device=device
            )
        for args in (
            (2, True, 0.1, False),
            (2, False, 0.1, False),
            (0, True, 0.0, False),
            (2, True, 0.1, True),
        ):
            _check_model(rank, size, *args, device=device)
        for overlap in (False, True):
            _check_layer(
                rank,
                size,
                InteractionNet,
                False,
                False,
                overlap=overlap,
                device=device,
                no_edges=True,
            )
        _check_model(
            rank, size, 2, True, 0.1, False, overlap=True, device=device
        )
        _check_model(
            rank, size, 2, True, 0.1, True, overlap=True, device=device
        )
        _check_strategy(rank, size, Path(checkpoint_dir), backend)
    finally:
        dist.destroy_process_group()


def _check_strategy(rank, size, checkpoint_dir, backend):
    os.environ["LOCAL_RANK"] = str(rank)
    torch.manual_seed(94)
    dtype = torch.float64 if backend == "gloo" else torch.float32
    model = TinyEFM(layers=1).to(dtype=dtype)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(4, 1, 6, 2, dtype=dtype),
        torch.randn(4, 2, 6, 2, dtype=dtype),
        torch.randn(4, 2, 6, 1, dtype=dtype),
        torch.empty(4, 2, 0, 0, dtype=dtype),
        torch.empty(4, 2, 0, 0, dtype=dtype),
        torch.zeros(4, 2),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    path = str(checkpoint_dir / "graph_parallel.ckpt")
    for max_steps, checkpoint in ((1, None), (2, path)):
        if checkpoint:
            torch.manual_seed(94)
            model = TinyEFM(layers=1).to(dtype=dtype)
        trainer = pl.Trainer(
            strategy=GraphParallelStrategy(size, process_group_backend=backend),
            accelerator="cpu" if backend == "gloo" else "gpu",
            devices=size,
            max_steps=max_steps,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            use_distributed_sampler=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            gradient_clip_val=0.1,
        )
        trainer.fit(model, train_dataloaders=loader, ckpt_path=checkpoint)
        assert trainer.global_step == max_steps
        assert not isinstance(
            trainer.strategy.model, torch.nn.parallel.DistributedDataParallel
        )
        packed = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        peer = packed.clone()
        dist.broadcast(peer, src=0)
        torch.testing.assert_close(packed, peer, rtol=0, atol=0)
        trainer.save_checkpoint(path)
        dist.barrier()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("size", [2, 4])
def test_distributed_equivalence(tmp_path, size):
    mp.spawn(
        _distributed_worker,
        args=(size, (tmp_path / "rendezvous").as_uri(), str(tmp_path)),
        nprocs=size,
        join=True,
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="Two CUDA GPUs and NCCL required",
)
def test_cuda_distributed_equivalence(tmp_path):
    mp.spawn(
        _distributed_worker,
        args=(2, (tmp_path / "rendezvous").as_uri(), str(tmp_path), "nccl"),
        nprocs=2,
        join=True,
    )


def test_partition_contract(tmp_path):
    positions = [
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        torch.zeros(1, 2),
    ]
    edges = [torch.tensor([[0, 1], [1, 2]])]
    fingerprint = graph_fingerprint(positions, edges)
    owners = spatial_owners(positions, 4)
    assert all(
        torch.equal(a, b) for a, b in zip(owners, spatial_owners(positions, 4))
    )
    partition = MeshPartition(owners, 4, fingerprint)
    partition.validate([3, 1], fingerprint)
    path = tmp_path / "partition.pt"
    partition.save(path)
    assert MeshPartition.load(path).identity == partition.identity
    with pytest.raises(ValueError, match="fingerprint"):
        partition.validate([3, 1], "wrong")
    with pytest.raises(ValueError, match="int64"):
        MeshPartition([owners[0].float(), owners[1]], 4, fingerprint).validate(
            [3, 1], fingerprint
        )
