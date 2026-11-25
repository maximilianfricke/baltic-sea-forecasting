"""Original GraphEFM layers and rollout with PhysicsNeMo graph communication."""

import copy

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.checkpoint import set_checkpoint_early_stop

from neural_lam import utils
from neural_lam.interaction_net import FlexibleNet
from dphpc.third_party.physicsnemo.distributed_graph import (
    DistributedGraph,
    edge_index_to_csc,
    partition_graph_with_id_mapping,
)


class Relation:
    def __init__(self, edges, source_owners, destination_owners, device, group):
        K, rank = dist.get_world_size(group), dist.get_rank(group)
        offsets, indices, self.permutation = edge_index_to_csc(
            edges, len(destination_owners)
        )
        partition = partition_graph_with_id_mapping(
            offsets, indices, source_owners, destination_owners, K, rank, device
        )
        self.graph = DistributedGraph(
            offsets, indices, K, graph_partition=partition, process_group=group
        )
        self.source_ids = partition.map_partitioned_src_ids_to_global
        self.destination_ids = partition.map_partitioned_dst_ids_to_global
        self.edge_ids = self.permutation.to(device)[
            partition.map_partitioned_edge_ids_to_global
        ]
        local = self.graph.local_graph.edge_index
        # PN local sources are ordered by owner rank, then global ID.
        global_sources = edges[0, self.edge_ids.cpu()].unique(sorted=True)
        order = torch.argsort(source_owners[global_sources], stable=True)
        global_sources = global_sources[order].to(device)
        reconstructed = torch.stack(
            (global_sources[local[0]], self.destination_ids[local[1]])
        )
        assert torch.equal(
            reconstructed.cpu(), edges[:, self.edge_ids.cpu()]
        ), "CSC edge/node ordering differs"

    def edge_features(self, features):
        return self.graph.get_edge_features_in_partition(
            features.index_select(0, self.permutation.to(features.device))
        )


class LocalLayer(nn.Module):
    """Keep the original layer class, MLPs and reductions; fetch senders via PN."""

    def __init__(self, layer, relation):
        super().__init__()
        self.net = copy.deepcopy(layer)
        self.relation = relation
        self.net.num_rec = len(relation.destination_ids)
        edges = relation.graph.local_graph.edge_index.clone()
        if not isinstance(layer, FlexibleNet):
            edges[
                0
            ] += (
                self.net.num_rec
            )  # Original InteractionNet's [receivers,senders] convention.
        self.net.edge_index = edges

    def forward(self, send, receive, edge, emb=None):
        send = self.relation.graph.get_src_node_features_in_local_graph(
            send.movedim(-2, 0)
        ).movedim(0, -2)
        return self.net(send, receive, edge, emb=emb)


class GlobalNoiseNormal(torch.distributions.Normal):
    def rsample(self, sample_shape=torch.Size()):
        shape = (
            *sample_shape,
            *self.loc.shape[:-2],
            self.global_nodes,
            self.loc.shape[-1],
        )
        noise = torch.empty(
            shape, dtype=self.loc.dtype, device=self.loc.device
        ).normal_()
        return self.loc + self.scale * noise.index_select(-2, self.owned_nodes)


def global_noise_hook(nodes, total):
    def hook(module, inputs, distribution):
        result = GlobalNoiseNormal(distribution.loc, distribution.scale)
        result.owned_nodes, result.global_nodes = nodes, total
        return result

    return hook


class DistributedGraphEFM(nn.Module):
    def __init__(
        self, original, owners, relations, config, datastore, device, group=None
    ):
        super().__init__()
        group = dist.group.WORLD if group is None else group
        self.group = group
        self.model = copy.deepcopy(original)
        self.model.prepare_clamping_params(
            config, datastore
        )  # Rebind original clamping closures.
        self.model.log_dict = lambda *args, **kwargs: None
        if self.model.boundary_forced or self.model.use_atmosphere_g2m:
            raise NotImplementedError(
                "This adapter uses the global-ocean setup"
            )
        self.relations = {
            name: Relation(edges, owners[src], owners[dst], device, group)
            for name, (src, dst, edges) in relations.items()
        }
        self.nodes = {
            key: torch.where(value == dist.get_rank(group))[0].to(device)
            for key, value in owners.items()
        }
        model = self.model
        grid = self.nodes["grid"]
        for name in ("grid_static_features", "loss_mask", "interior_mask_bool"):
            value = getattr(model, name)
            setattr(model, name, value.index_select(0, grid.cpu()))
        model.num_grid_nodes = model.num_total_grid_nodes = len(grid)
        model.interior_mask = model.interior_mask[grid.cpu().numpy()]
        model.g2m_features = self.relations["g2m"].edge_features(
            model.g2m_features
        )
        model.m2g_features = self.relations["m2g"].edge_features(
            model.m2g_features
        )
        count = len(owners) - 1
        if model.hierarchical_graph:
            model.mesh_static_features = utils.BufferList(
                [
                    features.index_select(0, self.nodes[f"mesh_{i}"].cpu())
                    for i, features in enumerate(model.mesh_static_features)
                ],
                persistent=False,
            )
            for attr, prefix in [
                ("m2m_features", "m2m_"),
                ("mesh_up_features", "up_"),
                ("mesh_down_features", "down_"),
            ]:
                setattr(
                    model,
                    attr,
                    utils.BufferList(
                        [
                            self.relations[f"{prefix}{i}"].edge_features(
                                features
                            )
                            for i, features in enumerate(getattr(model, attr))
                        ],
                        persistent=False,
                    ),
                )
        else:
            model.mesh_static_features = (
                model.mesh_static_features.index_select(
                    0, self.nodes["mesh_0"].cpu()
                )
            )
            model.m2m_features = self.relations["m2m_0"].edge_features(
                model.m2m_features
            )

        def sequence(sequence, relation):
            for name, layer in list(sequence.named_children()):
                setattr(sequence, name, LocalLayer(layer, relation))

        for submodel in (model.prior_model, model.encoder, model.decoder):
            submodel.g2m_gnn = LocalLayer(
                submodel.g2m_gnn, self.relations["g2m"]
            )
            if hasattr(submodel, "m2g_gnn"):
                submodel.m2g_gnn = LocalLayer(
                    submodel.m2g_gnn, self.relations["m2g"]
                )
            for attr, prefix in [
                ("mesh_up_gnns", "up_"),
                ("mesh_down_gnns", "down_"),
            ]:
                if hasattr(submodel, attr):
                    layers = getattr(submodel, attr)
                    for i, layer in enumerate(layers):
                        layers[i] = LocalLayer(
                            layer, self.relations[f"{prefix}{i}"]
                        )
            for attr in (
                "intra_level_gnns",
                "intra_up_gnns",
                "intra_down_gnns",
            ):
                if hasattr(submodel, attr):
                    for i, layers in enumerate(getattr(submodel, attr)):
                        sequence(layers, self.relations[f"m2m_{i}"])
            if hasattr(submodel, "processor"):
                sequence(submodel.processor, self.relations["m2m_0"])
        for encoder in (model.encoder, model.prior_model):
            encoder.register_forward_hook(
                global_noise_hook(
                    self.nodes[f"mesh_{count-1}"],
                    len(owners[f"mesh_{count-1}"]),
                )
            )
        self.to(device)

    def local_batch(self, batch):
        return tuple(
            (
                value.index_select(-2, self.nodes["grid"])
                if i in (0, 1, 2, 4)
                else value
            )
            for i, value in enumerate(batch)
        )

    def training_result(self, batch, seed, checkpointing=True):
        self.train()
        self.model.grad_checkpointing = checkpointing
        predictions = []
        original = self.model.compute_step_loss

        def capture(*args):
            result = original(*args)
            # Diagnostic values only; checkpoint replay must not retain graphs.
            if len(predictions) < batch[1].shape[1]:
                predictions.append(result[2].detach())
            return result

        self.model.compute_step_loss = capture
        devices = [batch[0].device.index] if batch[0].is_cuda else []
        try:
            with (
                torch.random.fork_rng(devices=devices),
                set_checkpoint_early_stop(False),
            ):
                torch.manual_seed(seed)
                local_loss = self.model.training_step(batch)
        finally:
            self.model.compute_step_loss = original
        total = local_loss.detach().clone()
        dist.all_reduce(total, group=self.group)
        return dict(
            loss=total + (local_loss - local_loss.detach()),
            predictions=torch.stack(predictions[: batch[1].shape[1]], dim=1),
        )

    @torch.no_grad()
    def forecast(self, batch, seed):
        self.eval()
        initial, _, forcing, boundary, atmosphere, _ = batch
        devices = [initial.device.index] if initial.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            return self.model.unroll_prediction(
                initial, forcing, boundary, atmosphere
            )[0]

    def parameter_pairs(self, reference):
        originals = dict(reference.named_parameters())
        for name, parameter in self.model.named_parameters():
            yield name, parameter, originals[name.replace(".net.", ".")]

    def synchronize_gradients(self):
        pairs = list(self.model.parameters())
        present = torch.tensor(
            [p.grad is not None for p in pairs],
            dtype=torch.int,
            device=pairs[0].device,
        )
        dist.all_reduce(present, op=dist.ReduceOp.MAX, group=self.group)
        active = [p for p, used in zip(pairs, present.tolist()) if used]
        flat = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(
                    -1
                )
                for p in active
            ]
        )
        dist.all_reduce(flat, group=self.group)
        offset = 0
        for p in active:
            gradient = flat[offset : offset + p.numel()].view_as(p)
            offset += p.numel()
            p.grad = (
                gradient.clone() if p.grad is None else p.grad.copy_(gradient)
            )
