"""Utilities for receiver-owned distributed graph message passing.

The partitioning policy is intentionally kept outside this module.  The
message-passing code only needs an owner rank for every sender and receiver
node.  Edges are assigned to the rank owning their receiver, which makes the
final aggregation local to that rank.
"""

# Standard library
from dataclasses import dataclass
from typing import Sequence, Tuple

# Third-party
import torch
import torch.distributed as dist
from torch import nn

# Local
from .interaction_net import FlexibleNet, InteractionNet, PropagationNet


@dataclass(frozen=True)
class BipartitePartition:
    """Local view of a receiver-owned bipartite graph partition."""

    rank: int
    world_size: int
    owned_send_nodes: torch.Tensor
    owned_rec_nodes: torch.Tensor
    edge_ids: torch.Tensor
    edge_index: torch.Tensor
    local_edge_positions: torch.Tensor
    boundary_edge_positions: torch.Tensor
    halo_nodes: torch.Tensor
    send_local_indices_by_peer: Tuple[torch.Tensor, ...]
    recv_counts_by_peer: Tuple[int, ...]

    @property
    def num_owned_senders(self) -> int:
        return int(self.owned_send_nodes.numel())

    @property
    def num_owned_receivers(self) -> int:
        return int(self.owned_rec_nodes.numel())

    @property
    def num_halo_nodes(self) -> int:
        return int(self.halo_nodes.numel())

    def to(self, device: torch.device) -> "BipartitePartition":
        """Return the partition metadata on ``device``."""

        return BipartitePartition(
            rank=self.rank,
            world_size=self.world_size,
            owned_send_nodes=self.owned_send_nodes.to(device),
            owned_rec_nodes=self.owned_rec_nodes.to(device),
            edge_ids=self.edge_ids.to(device),
            edge_index=self.edge_index.to(device),
            local_edge_positions=self.local_edge_positions.to(device),
            boundary_edge_positions=self.boundary_edge_positions.to(device),
            halo_nodes=self.halo_nodes.to(device),
            send_local_indices_by_peer=tuple(
                indices.to(device)
                for indices in self.send_local_indices_by_peer
            ),
            recv_counts_by_peer=self.recv_counts_by_peer,
        )


@dataclass(frozen=True)
class GraphPartition:
    """All receiver-owned partitions required by a mesh hierarchy."""

    rank: int
    world_size: int
    full_grid_nodes: torch.Tensor
    interior_grid_nodes: torch.Tensor
    mesh_nodes: Tuple[torch.Tensor, ...]
    g2m: BipartitePartition
    m2g: BipartitePartition
    m2m: Tuple[BipartitePartition, ...]
    mesh_up: Tuple[BipartitePartition, ...]
    mesh_down: Tuple[BipartitePartition, ...]

    @property
    def num_levels(self) -> int:
        return len(self.mesh_nodes)

    def to(self, device: torch.device) -> "GraphPartition":
        """Return all partition metadata on ``device``."""

        return GraphPartition(
            rank=self.rank,
            world_size=self.world_size,
            full_grid_nodes=self.full_grid_nodes.to(device),
            interior_grid_nodes=self.interior_grid_nodes.to(device),
            mesh_nodes=tuple(nodes.to(device) for nodes in self.mesh_nodes),
            g2m=self.g2m.to(device),
            m2g=self.m2g.to(device),
            m2m=tuple(part.to(device) for part in self.m2m),
            mesh_up=tuple(part.to(device) for part in self.mesh_up),
            mesh_down=tuple(part.to(device) for part in self.mesh_down),
        )


def _validate_owners(
    owners: torch.Tensor,
    num_nodes: int,
    world_size: int,
    name: str,
) -> None:
    if owners.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long")
    if owners.ndim != 1 or owners.numel() != num_nodes:
        raise ValueError(f"{name} must have shape ({num_nodes},)")
    if owners.numel() and (
        int(owners.min()) < 0 or int(owners.max()) >= world_size
    ):
        raise ValueError(f"{name} contains a rank outside the world size")


def build_bipartite_partition(
    edge_index: torch.Tensor,
    send_owners: torch.Tensor,
    rec_owners: torch.Tensor,
    rank: int,
    world_size: int,
) -> BipartitePartition:
    """Build one receiver-owned partition from arbitrary node owners.

    Parameters
    ----------
    edge_index:
        Global bipartite edge index with sender indices in row 0 and receiver
        indices in row 1.
    send_owners, rec_owners:
        Owner rank for every sender and receiver node.
    rank, world_size:
        Rank represented by the returned local view and total number of ranks.
    """

    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be within world_size")

    num_send = int(send_owners.numel())
    num_rec = int(rec_owners.numel())
    _validate_owners(send_owners, num_send, world_size, "send_owners")
    _validate_owners(rec_owners, num_rec, world_size, "rec_owners")

    if edge_index.numel():
        if int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= num_send:
            raise ValueError("edge_index contains an invalid sender index")
        if int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= num_rec:
            raise ValueError("edge_index contains an invalid receiver index")

    owned_send_nodes = torch.where(send_owners == rank)[0]
    owned_rec_nodes = torch.where(rec_owners == rank)[0]

    receiver_ranks = rec_owners.index_select(0, edge_index[1])
    sender_ranks = send_owners.index_select(0, edge_index[0])
    edge_ids = torch.where(receiver_ranks == rank)[0]

    send_global_to_local = torch.full((num_send,), -1, dtype=torch.long)
    send_global_to_local[owned_send_nodes] = torch.arange(
        owned_send_nodes.numel(), dtype=torch.long
    )
    rec_global_to_local = torch.full((num_rec,), -1, dtype=torch.long)
    rec_global_to_local[owned_rec_nodes] = torch.arange(
        owned_rec_nodes.numel(), dtype=torch.long
    )

    recv_nodes_by_peer = []
    send_local_indices_by_peer = []
    halo_offset = int(owned_send_nodes.numel())

    for peer in range(world_size):
        if peer == rank:
            recv_nodes = torch.empty(0, dtype=torch.long)
            send_local_indices = torch.empty(0, dtype=torch.long)
        else:
            recv_mask = (receiver_ranks == rank) & (sender_ranks == peer)
            recv_edge_ids = torch.where(recv_mask)[0]
            recv_nodes = torch.unique(
                edge_index[0].index_select(0, recv_edge_ids), sorted=True
            )

            send_mask = (receiver_ranks == peer) & (sender_ranks == rank)
            send_edge_ids = torch.where(send_mask)[0]
            send_global_nodes = torch.unique(
                edge_index[0].index_select(0, send_edge_ids), sorted=True
            )
            send_local_indices = send_global_to_local.index_select(
                0, send_global_nodes
            )

            halo_local_indices = torch.arange(
                halo_offset,
                halo_offset + recv_nodes.numel(),
                dtype=torch.long,
            )
            send_global_to_local[recv_nodes] = halo_local_indices
            halo_offset += int(recv_nodes.numel())

        recv_nodes_by_peer.append(recv_nodes)
        send_local_indices_by_peer.append(send_local_indices)

    halo_nodes = torch.cat(recv_nodes_by_peer, dim=0)
    local_senders = edge_index[0].index_select(0, edge_ids)
    local_receivers = edge_index[1].index_select(0, edge_ids)
    local_edge_index = torch.stack(
        (
            send_global_to_local.index_select(0, local_senders),
            rec_global_to_local.index_select(0, local_receivers),
        ),
        dim=0,
    )
    if local_edge_index.numel() and int(local_edge_index.min()) < 0:
        raise RuntimeError("partition contains an unmapped sender or receiver")

    local_edge_mask = send_owners.index_select(0, local_senders) == rank
    local_edge_positions = torch.where(local_edge_mask)[0]
    boundary_edge_positions = torch.where(~local_edge_mask)[0]

    return BipartitePartition(
        rank=rank,
        world_size=world_size,
        owned_send_nodes=owned_send_nodes,
        owned_rec_nodes=owned_rec_nodes,
        edge_ids=edge_ids,
        edge_index=local_edge_index,
        local_edge_positions=local_edge_positions,
        boundary_edge_positions=boundary_edge_positions,
        halo_nodes=halo_nodes,
        send_local_indices_by_peer=tuple(send_local_indices_by_peer),
        recv_counts_by_peer=tuple(
            int(nodes.numel()) for nodes in recv_nodes_by_peer
        ),
    )


def balanced_random_owners(
    num_nodes: int,
    world_size: int,
    seed: int = 0,
) -> torch.Tensor:
    """Assign nodes randomly while keeping partition sizes balanced."""

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_nodes, generator=generator)
    owners = torch.empty(num_nodes, dtype=torch.long)
    owners[permutation] = torch.arange(num_nodes) % world_size
    return owners


def latitude_cut_owners(
    lon_lat: torch.Tensor,
    latitude_cut: float = 0.0,
) -> torch.Tensor:
    """Assign southern nodes to rank 0 and northern nodes to rank 1.

    Nodes exactly on the cut are assigned to the northern partition.  This is
    deliberately a simple geographic baseline; more advanced partitioners can
    replace the ownership tensor without changing message-passing code.
    """

    lon_lat = torch.as_tensor(lon_lat)
    if lon_lat.ndim != 2 or lon_lat.shape[1] != 2:
        raise ValueError("lon_lat must have shape (num_nodes, 2)")
    if not torch.isfinite(lon_lat).all():
        raise ValueError("lon_lat contains non-finite coordinates")

    owners = (lon_lat[:, 1] >= latitude_cut).to(torch.long)
    if owners.numel() and torch.unique(owners).numel() < 2:
        raise ValueError(
            "latitude cut leaves one partition empty; choose a cut inside "
            "the coordinate range"
        )
    return owners


def _edge_tuple(edge_indices) -> Tuple[torch.Tensor, ...]:
    if isinstance(edge_indices, torch.Tensor):
        return (edge_indices,)
    return tuple(edge_indices)


def build_graph_partition(
    g2m_edge_index: torch.Tensor,
    m2g_edge_index: torch.Tensor,
    m2m_edge_index: Sequence[torch.Tensor],
    full_grid_owners: torch.Tensor,
    interior_grid_owners: torch.Tensor,
    mesh_owners: Sequence[torch.Tensor],
    rank: int,
    world_size: int,
    mesh_up_edge_index: Sequence[torch.Tensor] = (),
    mesh_down_edge_index: Sequence[torch.Tensor] = (),
) -> GraphPartition:
    """Build every local graph view used by Graph-EFM.

    The same ownership arrays are reused for the prior, posterior and decoder.
    Each edge is stored on the rank owning its receiver node.  Consequently,
    aggregation never needs an additional cross-rank reduction.
    """

    mesh_owners = tuple(mesh_owners)
    m2m_edge_index = _edge_tuple(m2m_edge_index)
    mesh_up_edge_index = _edge_tuple(mesh_up_edge_index)
    mesh_down_edge_index = _edge_tuple(mesh_down_edge_index)

    if not mesh_owners:
        raise ValueError("mesh_owners must contain at least one level")
    if len(m2m_edge_index) != len(mesh_owners):
        raise ValueError("one m2m edge index is required per mesh level")
    if len(mesh_up_edge_index) != len(mesh_owners) - 1:
        raise ValueError("mesh_up must connect every adjacent mesh level")
    if len(mesh_down_edge_index) != len(mesh_owners) - 1:
        raise ValueError("mesh_down must connect every adjacent mesh level")

    g2m = build_bipartite_partition(
        g2m_edge_index,
        full_grid_owners,
        mesh_owners[0],
        rank,
        world_size,
    )
    m2g = build_bipartite_partition(
        m2g_edge_index,
        mesh_owners[0],
        interior_grid_owners,
        rank,
        world_size,
    )
    m2m = tuple(
        build_bipartite_partition(
            edge_index,
            owners,
            owners,
            rank,
            world_size,
        )
        for edge_index, owners in zip(m2m_edge_index, mesh_owners)
    )
    mesh_up = tuple(
        build_bipartite_partition(
            edge_index,
            mesh_owners[level],
            mesh_owners[level + 1],
            rank,
            world_size,
        )
        for level, edge_index in enumerate(mesh_up_edge_index)
    )
    mesh_down = tuple(
        build_bipartite_partition(
            edge_index,
            mesh_owners[level + 1],
            mesh_owners[level],
            rank,
            world_size,
        )
        for level, edge_index in enumerate(mesh_down_edge_index)
    )

    return GraphPartition(
        rank=rank,
        world_size=world_size,
        full_grid_nodes=torch.where(full_grid_owners == rank)[0],
        interior_grid_nodes=torch.where(interior_grid_owners == rank)[0],
        mesh_nodes=tuple(
            torch.where(owners == rank)[0] for owners in mesh_owners
        ),
        g2m=g2m,
        m2g=m2g,
        m2m=m2m,
        mesh_up=mesh_up,
        mesh_down=mesh_down,
    )


def _run_p2p(ops) -> None:
    if not ops:
        return
    requests = dist.batch_isend_irecv(ops)
    for request in requests:
        request.wait()


class _HaloExchange(torch.autograd.Function):
    """Exchange unique halo node representations and their gradients."""

    @staticmethod
    def forward(ctx, local_send_rep, partition):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        if dist.get_rank() != partition.rank:
            raise RuntimeError("partition rank does not match process rank")
        if dist.get_world_size() != partition.world_size:
            raise RuntimeError(
                "partition world size does not match process group"
            )

        device = local_send_rep.device
        send_indices = tuple(
            indices.to(device)
            for indices in partition.send_local_indices_by_peer
        )
        recv_counts = partition.recv_counts_by_peer
        prefix = local_send_rep.shape[:-2]
        feature_dim = local_send_rep.shape[-1]

        send_buffers = []
        recv_buffers = []
        ops = []
        recv_by_peer = [None] * partition.world_size

        for peer in range(partition.world_size):
            if peer == partition.rank:
                continue
            peer_send_indices = send_indices[peer]
            if peer_send_indices.numel():
                send_buffer = local_send_rep.index_select(
                    -2, peer_send_indices
                ).contiguous()
                send_buffers.append(send_buffer)
                ops.append(dist.P2POp(dist.isend, send_buffer, peer))

            recv_count = recv_counts[peer]
            if recv_count:
                recv_buffer = local_send_rep.new_empty(
                    (*prefix, recv_count, feature_dim)
                )
                recv_buffers.append(recv_buffer)
                recv_by_peer[peer] = recv_buffer
                ops.append(dist.P2POp(dist.irecv, recv_buffer, peer))

        _run_p2p(ops)
        del send_buffers, recv_buffers

        ordered_recv = [
            recv_by_peer[peer]
            for peer in range(partition.world_size)
            if recv_by_peer[peer] is not None
        ]
        if ordered_recv:
            halo_rep = torch.cat(ordered_recv, dim=-2)
        else:
            halo_rep = local_send_rep.new_empty((*prefix, 0, feature_dim))

        ctx.partition = partition
        ctx.send_indices = send_indices
        ctx.local_shape = local_send_rep.shape
        return halo_rep

    @staticmethod
    def backward(ctx, grad_halo_rep):
        partition = ctx.partition
        send_indices = ctx.send_indices
        recv_counts = partition.recv_counts_by_peer
        nonzero_recv_counts = [count for count in recv_counts if count]
        if nonzero_recv_counts:
            grad_parts = list(
                torch.split(
                    grad_halo_rep,
                    nonzero_recv_counts,
                    dim=-2,
                )
            )
        else:
            grad_parts = []
        grad_by_peer = [None] * partition.world_size
        part_index = 0
        for peer, count in enumerate(recv_counts):
            if count:
                grad_by_peer[peer] = grad_parts[part_index].contiguous()
                part_index += 1

        grad_local = grad_halo_rep.new_zeros(ctx.local_shape)
        received_gradients = []
        ops = []
        received_by_peer = [None] * partition.world_size

        for peer in range(partition.world_size):
            if peer == partition.rank:
                continue
            if recv_counts[peer]:
                ops.append(dist.P2POp(dist.isend, grad_by_peer[peer], peer))

            peer_send_indices = send_indices[peer]
            if peer_send_indices.numel():
                recv_gradient = grad_halo_rep.new_empty(
                    (
                        *ctx.local_shape[:-2],
                        peer_send_indices.numel(),
                        ctx.local_shape[-1],
                    )
                )
                received_gradients.append(recv_gradient)
                received_by_peer[peer] = recv_gradient
                ops.append(dist.P2POp(dist.irecv, recv_gradient, peer))

        _run_p2p(ops)
        del received_gradients

        for peer, recv_gradient in enumerate(received_by_peer):
            if recv_gradient is not None:
                grad_local.index_add_(-2, send_indices[peer], recv_gradient)

        return grad_local, None


def exchange_halo(
    local_send_rep: torch.Tensor,
    partition: BipartitePartition,
) -> torch.Tensor:
    """Return remote sender representations required by ``partition``."""

    return _HaloExchange.apply(local_send_rep, partition)


class DistributedFlexibleNet(nn.Module):
    """Receiver-owned distributed wrapper around :class:`FlexibleNet`."""

    def __init__(
        self,
        partition: BipartitePartition,
        send_node_dim: int,
        rec_node_dim: int,
        edge_dim: int,
        hidden_layers: int = 1,
        propagation: bool = True,
        aggr: str = "mean",
        cond_dim: int = None,
    ):
        super().__init__()
        self.partition = partition
        self.local_net = FlexibleNet(
            edge_index=partition.edge_index,
            send_node_dim=send_node_dim,
            rec_node_dim=rec_node_dim,
            edge_dim=edge_dim,
            hidden_layers=hidden_layers,
            num_rec=partition.num_owned_receivers,
            propagation=propagation,
            aggr=aggr,
            cond_dim=cond_dim,
        )

    def forward(
        self,
        local_send_rep: torch.Tensor,
        local_rec_rep: torch.Tensor,
        local_edge_rep: torch.Tensor,
        emb=None,
        halo_rep: torch.Tensor = None,
    ) -> torch.Tensor:
        """Update locally owned receivers.

        ``halo_rep`` is an explicit escape hatch for deterministic single-
        process simulation.  In distributed execution it is omitted and the
        differentiable halo exchange is used.
        """

        if local_send_rep.shape[-2] != self.partition.num_owned_senders:
            raise ValueError("local_send_rep does not match the partition")
        if local_rec_rep.shape[-2] != self.partition.num_owned_receivers:
            raise ValueError("local_rec_rep does not match the partition")
        if local_edge_rep.shape[-2] != self.partition.edge_ids.numel():
            raise ValueError("local_edge_rep does not match the partition")

        if halo_rep is None:
            halo_rep = exchange_halo(local_send_rep, self.partition)
        if halo_rep.shape[-2] != self.partition.num_halo_nodes:
            raise ValueError("halo_rep does not match the partition")

        combined_send_rep = torch.cat((local_send_rep, halo_rep), dim=-2)
        return self.local_net(
            combined_send_rep,
            local_rec_rep,
            local_edge_rep,
            emb=emb,
        )


class DistributedInteractionNet(nn.Module):
    """Distributed wrapper for mesh-to-mesh interaction networks."""

    def __init__(
        self,
        partition: BipartitePartition,
        input_dim: int,
        update_edges: bool = True,
        hidden_layers: int = 1,
        hidden_dim: int = None,
        propagation: bool = False,
        aggr: str = "sum",
        cond_dim: int = None,
    ):
        super().__init__()
        self.partition = partition
        net_class = PropagationNet if propagation else InteractionNet
        net_kwargs = {
            "edge_index": partition.edge_index,
            "input_dim": input_dim,
            "update_edges": update_edges,
            "hidden_layers": hidden_layers,
            "hidden_dim": hidden_dim,
            "num_rec": partition.num_owned_receivers,
            "cond_dim": cond_dim,
        }
        if not propagation:
            net_kwargs["aggr"] = aggr
        self.local_net = net_class(**net_kwargs)

    def forward(
        self,
        local_send_rep: torch.Tensor,
        local_rec_rep: torch.Tensor,
        local_edge_rep: torch.Tensor,
        emb=None,
        halo_rep: torch.Tensor = None,
    ):
        """Update local receivers and, optionally, their incoming edges."""

        if local_send_rep.shape[-2] != self.partition.num_owned_senders:
            raise ValueError("local_send_rep does not match the partition")
        if local_rec_rep.shape[-2] != self.partition.num_owned_receivers:
            raise ValueError("local_rec_rep does not match the partition")
        if local_edge_rep.shape[-2] != self.partition.edge_ids.numel():
            raise ValueError("local_edge_rep does not match the partition")

        if halo_rep is None:
            halo_rep = exchange_halo(local_send_rep, self.partition)
        if halo_rep.shape[-2] != self.partition.num_halo_nodes:
            raise ValueError("halo_rep does not match the partition")

        combined_send_rep = torch.cat((local_send_rep, halo_rep), dim=-2)
        return self.local_net(
            combined_send_rep,
            local_rec_rep,
            local_edge_rep,
            emb=emb,
        )


class DistributedGNNSequence(nn.Module):
    """Repeated distributed InteractionNet layers on one mesh level."""

    def __init__(
        self,
        partition: BipartitePartition,
        num_gnn_layers: int,
        hidden_layers: int,
        hidden_dim: int,
        cond_dim: int = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DistributedInteractionNet(
                    partition,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    cond_dim=cond_dim,
                )
                for _ in range(num_gnn_layers)
            ]
        )

    def forward(self, node_rep, edge_rep, emb=None):
        for layer in self.layers:
            node_rep, edge_rep = layer(
                node_rep,
                node_rep,
                edge_rep,
                emb=emb,
            )
        return node_rep, edge_rep


def select_partition_inputs(
    send_rep: torch.Tensor,
    rec_rep: torch.Tensor,
    edge_rep: torch.Tensor,
    partition: BipartitePartition,
):
    """Select owned and halo inputs for single-process simulation."""

    device = send_rep.device
    return (
        send_rep.index_select(-2, partition.owned_send_nodes.to(device)),
        rec_rep.index_select(-2, partition.owned_rec_nodes.to(device)),
        edge_rep.index_select(-2, partition.edge_ids.to(device)),
        send_rep.index_select(-2, partition.halo_nodes.to(device)),
    )


def select_owned(rep: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select a rank's owned nodes or edges along the graph dimension."""

    return rep.index_select(-2, indices.to(rep.device))


def select_graph_embeddings(graph_emb, partition: GraphPartition):
    """Shard a global Graph-EFM embedding dictionary for one rank."""

    hierarchical = isinstance(graph_emb["mesh"], (list, tuple))
    if hierarchical:
        mesh = [
            select_owned(rep, nodes)
            for rep, nodes in zip(graph_emb["mesh"], partition.mesh_nodes)
        ]
        m2m = [
            select_owned(rep, part.edge_ids)
            for rep, part in zip(graph_emb["m2m"], partition.m2m)
        ]
        mesh_up = [
            select_owned(rep, part.edge_ids)
            for rep, part in zip(graph_emb["mesh_up"], partition.mesh_up)
        ]
        mesh_down = [
            select_owned(rep, part.edge_ids)
            for rep, part in zip(graph_emb["mesh_down"], partition.mesh_down)
        ]
    else:
        mesh = select_owned(graph_emb["mesh"], partition.mesh_nodes[0])
        m2m = select_owned(graph_emb["m2m"], partition.m2m[0].edge_ids)
        mesh_up = []
        mesh_down = []

    return {
        "mesh": mesh,
        "g2m": select_owned(graph_emb["g2m"], partition.g2m.edge_ids),
        "m2g": select_owned(graph_emb["m2g"], partition.m2g.edge_ids),
        "m2m": m2m,
        "mesh_up": mesh_up,
        "mesh_down": mesh_down,
    }
