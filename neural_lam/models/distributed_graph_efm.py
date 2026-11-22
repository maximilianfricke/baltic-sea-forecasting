"""Model-parallel Graph-EFM components.

The classes in this module mirror the original latent encoders and decoders,
but keep only locally owned grid nodes, mesh nodes and receiver-owned edges.
Cross-partition sender representations are fetched by the differentiable halo
exchange in :mod:`neural_lam.distributed_message_passing`.
"""

# Standard library
from pathlib import Path
from typing import Iterable, Tuple

# Third-party
import torch
from torch import nn

# First-party
from neural_lam.distributed_message_passing import (
    DistributedFlexibleNet,
    DistributedGNNSequence,
    DistributedInteractionNet,
    GraphPartition,
    balanced_random_owners,
    build_graph_partition,
    latitude_cut_owners,
)
from neural_lam.geographic_partitioning import geographic_rcb_owners
from neural_lam.models.base_graph_latent_decoder import BaseGraphLatentDecoder
from neural_lam.models.base_latent_encoder import BaseLatentEncoder


def _copy_module(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def _common_module_device(modules: Iterable[nn.Module]) -> torch.device:
    """Return the single device used by a collection of reference modules."""

    devices = set()
    for module in modules:
        devices.update(parameter.device for parameter in module.parameters())
        devices.update(buffer.device for buffer in module.buffers())
    if len(devices) > 1:
        raise ValueError(
            "reference GraphEFM modules must place all parameters and "
            "buffers on one device"
        )
    if not devices:
        return torch.device("cpu")
    return devices.pop()


def _reference_sequence_modules(sequence: nn.Module) -> Tuple[nn.Module, ...]:
    return tuple(module for _, module in sequence.named_children())


def _copy_gnn_sequence(
    target: DistributedGNNSequence,
    source: nn.Module,
) -> None:
    source_layers = _reference_sequence_modules(source)
    if len(target.layers) != len(source_layers):
        raise ValueError("reference GNN sequence has a different depth")
    for target_layer, source_layer in zip(target.layers, source_layers):
        _copy_module(target_layer.local_net, source_layer)


def _parameter_pairs(
    target: nn.Module,
    source: nn.Module,
    prefix: str,
) -> Iterable[Tuple[str, nn.Parameter, nn.Parameter]]:
    target_params = dict(target.named_parameters())
    source_params = dict(source.named_parameters())
    if target_params.keys() != source_params.keys():
        raise ValueError(f"parameter layout mismatch in {prefix}")
    for name in target_params:
        yield prefix + name, target_params[name], source_params[name]


def _sequence_parameter_pairs(
    target: DistributedGNNSequence,
    source: nn.Module,
    prefix: str,
) -> Iterable[Tuple[str, nn.Parameter, nn.Parameter]]:
    source_layers = _reference_sequence_modules(source)
    for layer_index, (target_layer, source_layer) in enumerate(
        zip(target.layers, source_layers)
    ):
        yield from _parameter_pairs(
            target_layer.local_net,
            source_layer,
            f"{prefix}{layer_index}.",
        )


class DistributedGraphLatentEncoder(BaseLatentEncoder):
    """Sharded single-level Graph-EFM prior or posterior encoder."""

    def __init__(
        self,
        partition: GraphPartition,
        latent_dim: int,
        hidden_dim: int,
        processor_layers: int,
        hidden_layers: int = 1,
        output_dist: str = "isotropic",
        hidden_dim_grid: int = None,
        hidden_dim_mesh_nodes: int = None,
        hidden_dim_edge: int = None,
    ):
        super().__init__(latent_dim, output_dist)
        if partition.num_levels != 1:
            raise ValueError("single-level encoder requires one mesh level")

        hidden_dim_grid = hidden_dim_grid or hidden_dim
        hidden_dim_mesh_nodes = hidden_dim_mesh_nodes or hidden_dim
        hidden_dim_edge = hidden_dim_edge or hidden_dim_grid

        self.g2m_gnn = DistributedFlexibleNet(
            partition.g2m,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=hidden_dim_mesh_nodes,
            edge_dim=hidden_dim_edge,
            hidden_layers=hidden_layers,
            propagation=True,
            aggr="mean",
        )
        self.pre_mesh_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim_grid, hidden_dim),
        )
        self.processor = DistributedGNNSequence(
            partition.m2m[0],
            processor_layers,
            hidden_layers,
            hidden_dim,
        )
        self.latent_param_map = _make_latent_param_map(
            hidden_dim,
            hidden_layers,
            self.output_dim,
        )

    def compute_dist_params(self, grid_rep, graph_emb, **kwargs):
        mesh_rep = self.g2m_gnn(
            grid_rep,
            graph_emb["mesh"],
            graph_emb["g2m"],
        )
        mesh_rep = self.pre_mesh_proj(mesh_rep)
        mesh_rep, _ = self.processor(mesh_rep, graph_emb["m2m"])
        return self.latent_param_map(mesh_rep)

    def load_from_reference(self, reference) -> None:
        _copy_module(self.g2m_gnn.local_net, reference.g2m_gnn)
        _copy_module(self.pre_mesh_proj, reference.pre_mesh_proj)
        _copy_gnn_sequence(self.processor, reference.processor)
        _copy_module(self.latent_param_map, reference.latent_param_map)

    def parameter_pairs(self, reference, prefix=""):
        yield from _parameter_pairs(
            self.g2m_gnn.local_net,
            reference.g2m_gnn,
            prefix + "g2m_gnn.",
        )
        yield from _parameter_pairs(
            self.pre_mesh_proj,
            reference.pre_mesh_proj,
            prefix + "pre_mesh_proj.",
        )
        yield from _sequence_parameter_pairs(
            self.processor,
            reference.processor,
            prefix + "processor.",
        )
        yield from _parameter_pairs(
            self.latent_param_map,
            reference.latent_param_map,
            prefix + "latent_param_map.",
        )


class DistributedHiGraphLatentEncoder(BaseLatentEncoder):
    """Sharded hierarchical Graph-EFM prior or posterior encoder."""

    def __init__(
        self,
        partition: GraphPartition,
        latent_dim: int,
        hidden_dim: int,
        hidden_dim_grid: int,
        hidden_dim_mesh_nodes: int,
        hidden_dim_edge: int,
        intra_level_layers: int,
        hidden_layers: int = 1,
        output_dist: str = "isotropic",
    ):
        super().__init__(latent_dim, output_dist)
        if partition.num_levels < 2:
            raise ValueError("hierarchical encoder requires multiple levels")

        self.g2m_gnn = DistributedFlexibleNet(
            partition.g2m,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=hidden_dim_mesh_nodes,
            edge_dim=hidden_dim_edge,
            hidden_layers=hidden_layers,
            propagation=True,
            aggr="mean",
        )
        self.mesh_up_gnns = nn.ModuleList(
            [
                DistributedInteractionNet(
                    part,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                    propagation=True,
                )
                for part in partition.mesh_up
            ]
        )
        self.intra_level_gnns = nn.ModuleList(
            [
                DistributedGNNSequence(
                    part,
                    intra_level_layers,
                    hidden_layers,
                    hidden_dim,
                )
                for part in partition.m2m
            ]
        )
        self.pre_mesh_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim_grid, hidden_dim),
        )
        self.latent_param_map = _make_latent_param_map(
            hidden_dim,
            hidden_layers,
            self.output_dim,
        )

    def compute_dist_params(self, grid_rep, graph_emb, **kwargs):
        current_mesh_rep = self.g2m_gnn(
            grid_rep,
            graph_emb["mesh"][0],
            graph_emb["g2m"],
        )
        current_mesh_rep = self.pre_mesh_proj(current_mesh_rep)
        current_mesh_rep, _ = self.intra_level_gnns[0](
            current_mesh_rep,
            graph_emb["m2m"][0],
        )

        for level, up_gnn in enumerate(self.mesh_up_gnns):
            current_mesh_rep = up_gnn(
                current_mesh_rep,
                graph_emb["mesh"][level + 1],
                graph_emb["mesh_up"][level],
            )
            current_mesh_rep, _ = self.intra_level_gnns[level + 1](
                current_mesh_rep,
                graph_emb["m2m"][level + 1],
            )
        return self.latent_param_map(current_mesh_rep)

    def load_from_reference(self, reference) -> None:
        _copy_module(self.g2m_gnn.local_net, reference.g2m_gnn)
        _copy_module(self.pre_mesh_proj, reference.pre_mesh_proj)
        _copy_module(self.latent_param_map, reference.latent_param_map)
        for target, source in zip(self.mesh_up_gnns, reference.mesh_up_gnns):
            _copy_module(target.local_net, source)
        for target, source in zip(
            self.intra_level_gnns,
            reference.intra_level_gnns,
        ):
            _copy_gnn_sequence(target, source)

    def parameter_pairs(self, reference, prefix=""):
        yield from _parameter_pairs(
            self.g2m_gnn.local_net,
            reference.g2m_gnn,
            prefix + "g2m_gnn.",
        )
        yield from _parameter_pairs(
            self.pre_mesh_proj,
            reference.pre_mesh_proj,
            prefix + "pre_mesh_proj.",
        )
        yield from _parameter_pairs(
            self.latent_param_map,
            reference.latent_param_map,
            prefix + "latent_param_map.",
        )
        for level, (target, source) in enumerate(
            zip(self.mesh_up_gnns, reference.mesh_up_gnns)
        ):
            yield from _parameter_pairs(
                target.local_net,
                source,
                f"{prefix}mesh_up_gnns.{level}.",
            )
        for level, (target, source) in enumerate(
            zip(self.intra_level_gnns, reference.intra_level_gnns)
        ):
            yield from _sequence_parameter_pairs(
                target,
                source,
                f"{prefix}intra_level_gnns.{level}.",
            )


def _make_latent_param_map(hidden_dim, hidden_layers, output_dim):
    # Local import avoids a circular dependency through neural_lam.utils.
    from neural_lam import utils

    return utils.make_mlp(
        [hidden_dim] * (hidden_layers + 1) + [output_dim],
        layer_norm=False,
    )


class DistributedConstantLatentEncoder(BaseLatentEncoder):
    """Local shard of Graph-EFM's non-learned constant prior."""

    def __init__(self, latent_dim, num_mesh_nodes, output_dist="isotropic"):
        super().__init__(latent_dim, output_dist)
        self.num_mesh_nodes = num_mesh_nodes

    def compute_dist_params(self, grid_rep, **kwargs):
        return torch.ones(
            grid_rep.shape[0],
            self.num_mesh_nodes,
            self.output_dim,
            device=grid_rep.device,
        )

    def load_from_reference(self, reference) -> None:
        if self.output_dist != reference.output_dist:
            raise ValueError("constant prior distribution does not match")

    def parameter_pairs(self, reference, prefix=""):
        del reference, prefix
        return iter(())


class DistributedGraphLatentDecoder(BaseGraphLatentDecoder):
    """Sharded single-level Graph-EFM decoder."""

    def __init__(
        self,
        partition: GraphPartition,
        hidden_dim: int,
        latent_dim: int,
        decode_dim: int,
        grid_state_dim: int,
        processor_layers: int,
        hidden_layers: int = 1,
        output_std: bool = True,
        hidden_dim_mesh_nodes: int = None,
        hidden_dim_edge: int = None,
    ):
        super().__init__(
            hidden_dim,
            decode_dim,
            latent_dim,
            grid_state_dim,
            hidden_layers,
            output_std,
        )
        if partition.num_levels != 1:
            raise ValueError("single-level decoder requires one mesh level")

        hidden_dim_mesh_nodes = hidden_dim_mesh_nodes or hidden_dim
        hidden_dim_edge = hidden_dim_edge or decode_dim
        self.g2m_gnn = DistributedFlexibleNet(
            partition.g2m,
            send_node_dim=decode_dim,
            rec_node_dim=hidden_dim_mesh_nodes,
            edge_dim=hidden_dim_edge,
            hidden_layers=hidden_layers,
            propagation=True,
            aggr="mean",
        )
        self.pre_processor_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(decode_dim, hidden_dim),
        )
        self.processor = DistributedGNNSequence(
            partition.m2m[0],
            processor_layers,
            hidden_layers,
            hidden_dim,
        )
        self.post_mesh_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, decode_dim),
        )
        self.m2g_gnn = DistributedFlexibleNet(
            partition.m2g,
            send_node_dim=decode_dim,
            rec_node_dim=decode_dim,
            edge_dim=hidden_dim_edge,
            hidden_layers=hidden_layers,
            propagation=False,
            aggr="sum",
        )

    def combine_with_latent(
        self,
        original_grid_rep,
        latent_rep,
        residual_grid_rep,
        graph_emb,
    ):
        mesh_rep = self.g2m_gnn(
            original_grid_rep,
            latent_rep,
            graph_emb["g2m"],
        )
        mesh_rep = self.pre_processor_proj(mesh_rep)
        mesh_rep, _ = self.processor(mesh_rep, graph_emb["m2m"])
        mesh_rep = self.post_mesh_proj(mesh_rep)
        return self.m2g_gnn(
            mesh_rep,
            residual_grid_rep,
            graph_emb["m2g"],
        )

    def load_from_reference(self, reference) -> None:
        _copy_module(self.grid_update_mlp, reference.grid_update_mlp)
        _copy_module(self.latent_embedder, reference.latent_embedder)
        _copy_module(self.param_map, reference.param_map)
        _copy_module(self.g2m_gnn.local_net, reference.g2m_gnn)
        _copy_module(self.pre_processor_proj, reference.pre_processor_proj)
        _copy_gnn_sequence(self.processor, reference.processor)
        _copy_module(self.post_mesh_proj, reference.post_mesh_proj)
        _copy_module(self.m2g_gnn.local_net, reference.m2g_gnn)

    def parameter_pairs(self, reference, prefix=""):
        module_pairs = (
            (
                self.grid_update_mlp,
                reference.grid_update_mlp,
                "grid_update_mlp",
            ),
            (
                self.latent_embedder,
                reference.latent_embedder,
                "latent_embedder",
            ),
            (self.param_map, reference.param_map, "param_map"),
            (self.g2m_gnn.local_net, reference.g2m_gnn, "g2m_gnn"),
            (
                self.pre_processor_proj,
                reference.pre_processor_proj,
                "pre_processor_proj",
            ),
            (
                self.post_mesh_proj,
                reference.post_mesh_proj,
                "post_mesh_proj",
            ),
            (self.m2g_gnn.local_net, reference.m2g_gnn, "m2g_gnn"),
        )
        for target, source, name in module_pairs:
            yield from _parameter_pairs(
                target,
                source,
                f"{prefix}{name}.",
            )
        yield from _sequence_parameter_pairs(
            self.processor,
            reference.processor,
            prefix + "processor.",
        )


class DistributedHiGraphLatentDecoder(BaseGraphLatentDecoder):
    """Sharded hierarchical Graph-EFM decoder."""

    def __init__(
        self,
        partition: GraphPartition,
        hidden_dim: int,
        hidden_dim_grid: int,
        hidden_dim_mesh_nodes: int,
        hidden_dim_edge: int,
        latent_dim: int,
        output_dim: int,
        intra_level_layers: int,
        hidden_layers: int = 1,
        output_std: bool = True,
    ):
        super().__init__(
            hidden_dim,
            hidden_dim_grid,
            latent_dim,
            output_dim,
            hidden_layers,
            output_std,
        )
        if partition.num_levels < 2:
            raise ValueError("hierarchical decoder requires multiple levels")

        self.g2m_gnn = DistributedFlexibleNet(
            partition.g2m,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=hidden_dim_mesh_nodes,
            edge_dim=hidden_dim_edge,
            hidden_layers=hidden_layers,
            propagation=True,
            aggr="mean",
        )
        self.m2g_gnn = DistributedFlexibleNet(
            partition.m2g,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=hidden_dim_grid,
            edge_dim=hidden_dim_edge,
            hidden_layers=hidden_layers,
            propagation=False,
            aggr="sum",
        )
        self.mesh_up_gnns = nn.ModuleList(
            [
                DistributedInteractionNet(
                    part,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for part in partition.mesh_up
            ]
        )
        self.mesh_down_gnns = nn.ModuleList(
            [
                DistributedInteractionNet(
                    part,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                    propagation=True,
                )
                for part in partition.mesh_down
            ]
        )
        self.intra_up_gnns = nn.ModuleList(
            [
                DistributedGNNSequence(
                    part,
                    intra_level_layers,
                    hidden_layers,
                    hidden_dim,
                )
                for part in partition.m2m
            ]
        )
        self.intra_down_gnns = nn.ModuleList(
            [
                DistributedGNNSequence(
                    part,
                    intra_level_layers,
                    hidden_layers,
                    hidden_dim,
                )
                for part in partition.m2m[:-1]
            ]
        )
        self.pre_mesh_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim_grid, hidden_dim),
        )
        self.post_mesh_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim_grid),
        )

    def combine_with_latent(
        self,
        original_grid_rep,
        latent_rep,
        residual_grid_rep,
        graph_emb,
    ):
        current_mesh_rep = self.g2m_gnn(
            original_grid_rep,
            graph_emb["mesh"][0],
            graph_emb["g2m"],
        )
        current_mesh_rep = self.pre_mesh_proj(current_mesh_rep)

        mesh_level_reps = []
        m2m_level_reps = []
        receiver_reps = list(graph_emb["mesh"][1:-1]) + [latent_rep]
        for level, up_gnn in enumerate(self.mesh_up_gnns):
            new_mesh_rep, new_m2m_rep = self.intra_up_gnns[level](
                current_mesh_rep,
                graph_emb["m2m"][level],
            )
            mesh_level_reps.append(new_mesh_rep)
            m2m_level_reps.append(new_m2m_rep)
            current_mesh_rep = up_gnn(
                new_mesh_rep,
                receiver_reps[level],
                graph_emb["mesh_up"][level],
            )

        current_mesh_rep, _ = self.intra_up_gnns[-1](
            current_mesh_rep,
            graph_emb["m2m"][-1],
        )

        num_transitions = len(self.mesh_down_gnns)
        for reverse_index, down_gnn in enumerate(
            reversed(self.mesh_down_gnns)
        ):
            level = num_transitions - reverse_index - 1
            current_mesh_rep = down_gnn(
                current_mesh_rep,
                mesh_level_reps[level],
                graph_emb["mesh_down"][level],
            )
            current_mesh_rep, _ = self.intra_down_gnns[level](
                current_mesh_rep,
                m2m_level_reps[level],
            )

        current_mesh_rep = self.post_mesh_proj(current_mesh_rep)
        return self.m2g_gnn(
            current_mesh_rep,
            residual_grid_rep,
            graph_emb["m2g"],
        )

    def load_from_reference(self, reference) -> None:
        simple_modules = (
            (self.grid_update_mlp, reference.grid_update_mlp),
            (self.latent_embedder, reference.latent_embedder),
            (self.param_map, reference.param_map),
            (self.g2m_gnn.local_net, reference.g2m_gnn),
            (self.m2g_gnn.local_net, reference.m2g_gnn),
            (self.pre_mesh_proj, reference.pre_mesh_proj),
            (self.post_mesh_proj, reference.post_mesh_proj),
        )
        for target, source in simple_modules:
            _copy_module(target, source)
        for target, source in zip(self.mesh_up_gnns, reference.mesh_up_gnns):
            _copy_module(target.local_net, source)
        for target, source in zip(
            self.mesh_down_gnns,
            reference.mesh_down_gnns,
        ):
            _copy_module(target.local_net, source)
        for target, source in zip(self.intra_up_gnns, reference.intra_up_gnns):
            _copy_gnn_sequence(target, source)
        for target, source in zip(
            self.intra_down_gnns,
            reference.intra_down_gnns,
        ):
            _copy_gnn_sequence(target, source)

    def parameter_pairs(self, reference, prefix=""):
        simple_modules = (
            (
                self.grid_update_mlp,
                reference.grid_update_mlp,
                "grid_update_mlp",
            ),
            (
                self.latent_embedder,
                reference.latent_embedder,
                "latent_embedder",
            ),
            (self.param_map, reference.param_map, "param_map"),
            (self.g2m_gnn.local_net, reference.g2m_gnn, "g2m_gnn"),
            (self.m2g_gnn.local_net, reference.m2g_gnn, "m2g_gnn"),
            (self.pre_mesh_proj, reference.pre_mesh_proj, "pre_mesh_proj"),
            (self.post_mesh_proj, reference.post_mesh_proj, "post_mesh_proj"),
        )
        for target, source, name in simple_modules:
            yield from _parameter_pairs(
                target,
                source,
                f"{prefix}{name}.",
            )
        for level, (target, source) in enumerate(
            zip(self.mesh_up_gnns, reference.mesh_up_gnns)
        ):
            yield from _parameter_pairs(
                target.local_net,
                source,
                f"{prefix}mesh_up_gnns.{level}.",
            )
        for level, (target, source) in enumerate(
            zip(self.mesh_down_gnns, reference.mesh_down_gnns)
        ):
            yield from _parameter_pairs(
                target.local_net,
                source,
                f"{prefix}mesh_down_gnns.{level}.",
            )
        for level, (target, source) in enumerate(
            zip(self.intra_up_gnns, reference.intra_up_gnns)
        ):
            yield from _sequence_parameter_pairs(
                target,
                source,
                f"{prefix}intra_up_gnns.{level}.",
            )
        for level, (target, source) in enumerate(
            zip(self.intra_down_gnns, reference.intra_down_gnns)
        ):
            yield from _sequence_parameter_pairs(
                target,
                source,
                f"{prefix}intra_down_gnns.{level}.",
            )


class DistributedGraphEFMCore(nn.Module):
    """End-to-end sharded prior, posterior and decoder computation."""

    def __init__(self, prior_model, encoder, decoder):
        super().__init__()
        self.prior_model = prior_model
        self.encoder = encoder
        self.decoder = decoder

    def training_forward(
        self,
        grid_prev_rep,
        grid_current_rep,
        grid_prev_interior_rep,
        last_state,
        graph_emb,
        latent_noise=None,
    ):
        """Run the full variational training path on locally owned nodes."""

        posterior_dist = self.encoder(grid_current_rep, graph_emb=graph_emb)
        if latent_noise is None:
            latent_noise = torch.randn_like(posterior_dist.mean)
        latent_sample = (
            posterior_dist.mean + posterior_dist.stddev * latent_noise
        )
        pred_mean, pred_std = self.decoder(
            grid_prev_rep,
            grid_prev_interior_rep,
            latent_sample,
            last_state,
            graph_emb,
        )
        prior_dist = self.prior_model(grid_prev_rep, graph_emb=graph_emb)
        local_kl = torch.distributions.kl_divergence(
            posterior_dist,
            prior_dist,
        ).sum(dim=(1, 2))
        return {
            "prior": prior_dist,
            "posterior": posterior_dist,
            "latent_sample": latent_sample,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "local_kl": local_kl,
        }

    def inference_forward(
        self,
        grid_prev_rep,
        grid_prev_interior_rep,
        last_state,
        graph_emb,
        latent_noise=None,
    ):
        """Run prior sampling and decoding on locally owned nodes."""

        prior_dist = self.prior_model(grid_prev_rep, graph_emb=graph_emb)
        if latent_noise is None:
            latent_noise = torch.randn_like(prior_dist.mean)
        latent_sample = prior_dist.mean + prior_dist.stddev * latent_noise
        pred_mean, pred_std = self.decoder(
            grid_prev_rep,
            grid_prev_interior_rep,
            latent_sample,
            last_state,
            graph_emb,
        )
        return {
            "prior": prior_dist,
            "latent_sample": latent_sample,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
        }

    def load_from_reference(self, prior_model, encoder, decoder) -> None:
        self.prior_model.load_from_reference(prior_model)
        self.encoder.load_from_reference(encoder)
        self.decoder.load_from_reference(decoder)

    def parameter_pairs(self, prior_model, encoder, decoder):
        yield from self.prior_model.parameter_pairs(
            prior_model,
            prefix="prior_model.",
        )
        yield from self.encoder.parameter_pairs(
            encoder,
            prefix="encoder.",
        )
        yield from self.decoder.parameter_pairs(
            decoder,
            prefix="decoder.",
        )


def _full_grid_lon_lat(model) -> torch.Tensor:
    """Return grid coordinates in the same order as GraphEFM embeddings."""

    interior_xy = torch.as_tensor(
        model._datastore.get_xy("state", stacked=True),
        dtype=torch.float32,
    ).detach().cpu()
    surface_mask = (
        torch.as_tensor(model.surface_mask, dtype=torch.bool).detach().cpu()
    )
    coordinate_parts = [interior_xy[surface_mask]]

    if model.boundary_forced:
        boundary_xy = torch.as_tensor(
            model._datastore_boundary.get_xy("forcing", stacked=True),
            dtype=torch.float32,
        ).detach().cpu()
        boundary_mask = torch.as_tensor(
            model._datastore_boundary.get_mask(
                surface=True,
                stacked=True,
                invert=False,
            ),
            dtype=torch.bool,
        ).detach().cpu()
        coordinate_parts.append(boundary_xy[boundary_mask])

    if model.atmosphere_forced and model.use_atmosphere_g2m:
        atmosphere_xy = torch.as_tensor(
            model._datastore_atmosphere.get_xy("forcing", stacked=True),
            dtype=torch.float32,
        ).detach().cpu()
        atmosphere_mask = torch.as_tensor(
            model._datastore_atmosphere.get_atmosphere_mask(
                stacked=True,
                invert=False,
            ),
            dtype=torch.bool,
        ).detach().cpu()
        coordinate_parts.append(atmosphere_xy[atmosphere_mask])

    return torch.cat(coordinate_parts, dim=0)


def _mesh_level_lon_lat(model) -> Tuple[torch.Tensor, ...]:
    """Load geographic mesh coordinates for partitioning.

    Global graph files store ``sin(lon), cos(lon), sin(lat), cos(lat)`` as the
    first four node features.  Reconstructing angles from those raw features is
    more reliable than using the normalized features consumed by the model.
    """

    graph_path = (
        Path(model._datastore.root_path)
        / "graphs"
        / model.args.graph
        / "mesh_features.pt"
    )
    if graph_path.is_file():
        raw_levels = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=True,
        )
        reconstructed = []
        for features in raw_levels:
            if features.ndim != 2 or features.shape[1] < 4:
                reconstructed = []
                break
            sin_lon, cos_lon = features[:, 0], features[:, 1]
            sin_lat, cos_lat = features[:, 2], features[:, 3]
            lon = torch.rad2deg(torch.atan2(sin_lon, cos_lon))
            lat = torch.rad2deg(torch.atan2(sin_lat, cos_lat))
            reconstructed.append(torch.stack((lon, lat), dim=1))
        if reconstructed:
            return tuple(reconstructed)

    return tuple(
        torch.as_tensor(model.mesh_lon_lat[level]).detach().cpu()
        for level in range(len(model.mesh_lon_lat))
    )


def build_partition_from_graph_efm(
    model,
    rank: int,
    world_size: int,
    seed: int = 0,
    strategy: str = "geographic",
    latitude_cut: float = -5.0,
    longitude_seam: float = 120.0,
) -> GraphPartition:
    """Create a geographic or random partition from a loaded GraphEFM.

    ``strategy="geographic"`` is the balanced one-, two- or four-way RCB
    baseline.
    ``strategy="latitude"`` retains the fixed experimental cut at latitude
    -5 degrees, while ``strategy="random"`` retains the communication-heavy
    baseline.  The final production policy remains intentionally external to
    the message-passing implementation.
    """

    if world_size != 2 and strategy == "latitude":
        raise ValueError("latitude strategy currently supports exactly 2 ranks")
    if strategy == "geographic" and world_size not in (1, 2, 4):
        raise ValueError(
            "geographic strategy currently supports exactly 1, 2 or 4 ranks"
        )
    if strategy not in ("geographic", "latitude", "random"):
        raise ValueError(
            "strategy must be 'geographic', 'latitude' or 'random'"
        )

    # Partitioning is topology-only preprocessing.  Normalize graph indices to
    # CPU so an already CUDA-loaded reference model cannot mix CUDA indices
    # with the CPU ownership tensors created below.
    g2m_edge_index = torch.as_tensor(model.g2m_edge_index).detach().cpu()
    m2g_edge_index = torch.as_tensor(model.m2g_edge_index).detach().cpu()

    num_full_grid = int(g2m_edge_index[0].max().item() + 1)
    num_interior_grid = int(m2g_edge_index[1].max().item() + 1)
    if strategy in ("geographic", "latitude"):
        full_grid_lon_lat = _full_grid_lon_lat(model)
        if full_grid_lon_lat.shape[0] != num_full_grid:
            raise ValueError(
                "grid coordinate count does not match g2m sender count"
            )
        if strategy == "geographic":
            full_grid_owners = geographic_rcb_owners(
                full_grid_lon_lat,
                world_size=world_size,
                longitude_seam=longitude_seam,
            )
        else:
            full_grid_owners = latitude_cut_owners(
                full_grid_lon_lat,
                latitude_cut=latitude_cut,
            )
    else:
        full_grid_owners = balanced_random_owners(
            num_full_grid,
            world_size,
            seed=seed,
        )
    if num_interior_grid > num_full_grid:
        raise ValueError("interior grid cannot be larger than the full grid")
    interior_grid_owners = full_grid_owners[:num_interior_grid]

    if model.hierarchical_graph:
        mesh_sizes = [
            features.shape[0] for features in model.mesh_static_features
        ]
        m2m_edge_index = tuple(
            torch.as_tensor(edge_index).detach().cpu()
            for edge_index in model.m2m_edge_index
        )
        mesh_up_edge_index = tuple(
            torch.as_tensor(edge_index).detach().cpu()
            for edge_index in model.mesh_up_edge_index
        )
        mesh_down_edge_index = tuple(
            torch.as_tensor(edge_index).detach().cpu()
            for edge_index in model.mesh_down_edge_index
        )
    else:
        mesh_sizes = [model.mesh_static_features.shape[0]]
        m2m_edge_index = (
            torch.as_tensor(model.m2m_edge_index).detach().cpu(),
        )
        mesh_up_edge_index = ()
        mesh_down_edge_index = ()

    if strategy in ("geographic", "latitude"):
        mesh_lon_lat = _mesh_level_lon_lat(model)
        if strategy == "geographic":
            mesh_owners = tuple(
                geographic_rcb_owners(
                    mesh_lon_lat[level],
                    world_size=world_size,
                    longitude_seam=longitude_seam,
                )
                for level in range(len(mesh_sizes))
            )
        else:
            mesh_owners = tuple(
                latitude_cut_owners(
                    mesh_lon_lat[level],
                    latitude_cut=latitude_cut,
                )
                for level in range(len(mesh_sizes))
            )
    else:
        mesh_owners = tuple(
            balanced_random_owners(
                num_nodes,
                world_size,
                seed=seed + level + 1,
            )
            for level, num_nodes in enumerate(mesh_sizes)
        )
    return build_graph_partition(
        g2m_edge_index,
        m2g_edge_index,
        m2m_edge_index,
        full_grid_owners,
        interior_grid_owners,
        mesh_owners,
        rank,
        world_size,
        mesh_up_edge_index,
        mesh_down_edge_index,
    )


def distributed_core_from_graph_efm(
    model,
    partition: GraphPartition,
    device: torch.device = None,
):
    """Construct a sharded core and copy weights from a loaded GraphEFM.

    This is the bridge for existing checkpoints: first load the normal model,
    then create one sharded core per rank with the same parameters.  By
    default the core stays on the reference modules' device.  ``device`` can
    place the sharded copy directly on a rank's accelerator while the full
    reference model remains on CPU, avoiding a transient full-model GPU copy.
    """

    reference_models = (model.prior_model, model.encoder, model.decoder)
    reference_device = _common_module_device(reference_models)
    target_device = (
        reference_device if device is None else torch.device(device)
    )

    args = model.args
    hidden_dim = args.hidden_dim
    hidden_dim_grid = args.hidden_dim_grid or hidden_dim
    hidden_dim_mesh_nodes = args.hidden_dim_mesh_nodes or hidden_dim
    hidden_dim_edge = args.hidden_dim_edge or hidden_dim_grid
    latent_dim = args.latent_dim or hidden_dim

    if args.learn_prior:
        if model.hierarchical_graph:
            prior_model = DistributedHiGraphLatentEncoder(
                partition,
                latent_dim,
                hidden_dim,
                hidden_dim_grid,
                hidden_dim_mesh_nodes,
                hidden_dim_edge,
                args.prior_processor_layers,
                hidden_layers=args.hidden_layers,
                output_dist=args.prior_dist,
            )
        else:
            prior_model = DistributedGraphLatentEncoder(
                partition,
                latent_dim,
                hidden_dim,
                args.prior_processor_layers,
                hidden_layers=args.hidden_layers,
                output_dist=args.prior_dist,
                hidden_dim_grid=hidden_dim_grid,
                hidden_dim_mesh_nodes=hidden_dim_mesh_nodes,
                hidden_dim_edge=hidden_dim_edge,
            )
    else:
        prior_model = DistributedConstantLatentEncoder(
            latent_dim,
            partition.mesh_nodes[-1].numel(),
            output_dist=args.prior_dist,
        )

    if model.hierarchical_graph:
        encoder = DistributedHiGraphLatentEncoder(
            partition,
            latent_dim,
            hidden_dim,
            hidden_dim_grid,
            hidden_dim_mesh_nodes,
            hidden_dim_edge,
            args.encoder_processor_layers,
            hidden_layers=args.hidden_layers,
            output_dist="diagonal",
        )
        decoder = DistributedHiGraphLatentDecoder(
            partition,
            hidden_dim,
            hidden_dim_grid,
            hidden_dim_mesh_nodes,
            hidden_dim_edge,
            latent_dim,
            model.num_state_vars,
            args.processor_layers,
            hidden_layers=args.hidden_layers,
            output_std=bool(args.output_std),
        )
    else:
        encoder = DistributedGraphLatentEncoder(
            partition,
            latent_dim,
            hidden_dim,
            args.encoder_processor_layers,
            hidden_layers=args.hidden_layers,
            output_dist="diagonal",
            hidden_dim_grid=hidden_dim_grid,
            hidden_dim_mesh_nodes=hidden_dim_mesh_nodes,
            hidden_dim_edge=hidden_dim_edge,
        )
        decoder = DistributedGraphLatentDecoder(
            partition,
            hidden_dim,
            latent_dim,
            hidden_dim_grid,
            model.num_state_vars,
            args.processor_layers,
            hidden_layers=args.hidden_layers,
            output_std=bool(args.output_std),
            hidden_dim_mesh_nodes=hidden_dim_mesh_nodes,
            hidden_dim_edge=hidden_dim_edge,
        )

    core = DistributedGraphEFMCore(prior_model, encoder, decoder).to(
        target_device
    )
    core.load_from_reference(*reference_models)
    return core
