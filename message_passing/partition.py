"""Receiver-owned graph plans; ownership is an input, not a transport policy."""

# Standard library
import hashlib
from dataclasses import dataclass, replace

# Third-party
import torch


@dataclass(frozen=True)
class RelationPlan:
    rank: int
    world_size: int
    source_ids: torch.Tensor
    destination_ids: torch.Tensor
    edge_ids: torch.Tensor
    edge_index: torch.Tensor
    halo_ids: torch.Tensor
    send_indices: tuple
    receive_counts: tuple

    def to(self, device):
        return replace(
            self,
            **{
                name: getattr(self, name).to(device)
                for name in (
                    "source_ids",
                    "destination_ids",
                    "edge_ids",
                    "edge_index",
                    "halo_ids",
                )
            },
            send_indices=tuple(t.to(device) for t in self.send_indices),
        )


def validate_owners(owners, count, world_size, name):
    if owners.dtype != torch.long or owners.device.type != "cpu":
        raise ValueError(f"{name}: owners must be CPU int64 tensors")
    if owners.shape != (count,) or count == 0:
        raise ValueError(f"{name}: expected {count} owners")
    if int(owners.min()) < 0 or int(owners.max()) >= world_size:
        raise ValueError(f"{name}: rank outside world size")
    if (torch.bincount(owners, minlength=world_size) == 0).any():
        raise ValueError(f"{name}: empty rank is unsupported")


def relation_plan(edges, source_owners, destination_owners, rank, world_size):
    """Keep edge order; send each remote sender once per destination."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid rank/world size")
    for name, owners in (
        ("source", source_owners),
        ("destination", destination_owners),
    ):
        validate_owners(owners, len(owners), world_size, name)
    if (
        edges.dtype != torch.long
        or edges.device.type != "cpu"
        or edges.ndim != 2
        or edges.shape[0] != 2
    ):
        raise ValueError("Edges must have CPU int64 shape (2,E)")
    if edges.numel() and (
        int(edges.min()) < 0
        or int(edges[0].max()) >= len(source_owners)
        or int(edges[1].max()) >= len(destination_owners)
    ):
        raise ValueError("Edge endpoint outside node range")
    source_ids = torch.where(source_owners == rank)[0]
    destination_ids = torch.where(destination_owners == rank)[0]
    source_ranks = source_owners[edges[0]]
    destination_ranks = destination_owners[edges[1]]
    edge_ids = torch.where(destination_ranks == rank)[0]
    source_map = torch.full((len(source_owners),), -1, dtype=torch.long)
    source_map[source_ids] = torch.arange(len(source_ids))
    destination_map = torch.full(
        (len(destination_owners),), -1, dtype=torch.long
    )
    destination_map[destination_ids] = torch.arange(len(destination_ids))
    remote = [
        [torch.empty(0, dtype=torch.long) for _ in range(world_size)]
        for _ in range(world_size)
    ]
    for sender in range(world_size):
        for receiver in range(world_size):
            if sender != receiver:
                remote[sender][receiver] = edges[
                    0,
                    (source_ranks == sender) & (destination_ranks == receiver),
                ].unique(sorted=True)
    halo_ids = torch.cat([remote[peer][rank] for peer in range(world_size)])
    source_map[halo_ids] = torch.arange(len(halo_ids)) + len(source_ids)
    edge_index = torch.stack(
        (source_map[edges[0, edge_ids]], destination_map[edges[1, edge_ids]])
    )
    return RelationPlan(
        rank,
        world_size,
        source_ids,
        destination_ids,
        edge_ids,
        edge_index,
        halo_ids,
        tuple(source_map[remote[rank][peer]] for peer in range(world_size)),
        tuple(len(remote[peer][rank]) for peer in range(world_size)),
    )


def model_graph(model):
    """Use original graph tensors without creating or reordering any edges."""
    if not model.hierarchical_graph:
        raise NotImplementedError(
            "The rebuild targets the hierarchical global-ocean model"
        )
    counts = {"grid": len(model.grid_static_features)}
    counts.update(
        {f"mesh_{i}": len(x) for i, x in enumerate(model.mesh_static_features)}
    )
    relations = {
        "g2m": ("grid", "mesh_0", model.g2m_edge_index.cpu()),
        "m2g": ("mesh_0", "grid", model.m2g_edge_index.cpu()),
    }
    for i, edges in enumerate(model.m2m_edge_index):
        relations[f"m2m_{i}"] = (f"mesh_{i}", f"mesh_{i}", edges.cpu())
    for i, (up, down) in enumerate(
        zip(model.mesh_up_edge_index, model.mesh_down_edge_index, strict=True)
    ):
        relations[f"up_{i}"] = (f"mesh_{i}", f"mesh_{i + 1}", up.cpu())
        relations[f"down_{i}"] = (f"mesh_{i + 1}", f"mesh_{i}", down.cpu())
    return counts, relations


def graph_fingerprint(model):
    digest = hashlib.sha256()
    _, relations = model_graph(model)
    tensors = {
        **{name: edges for name, (_, _, edges) in relations.items()},
        "grid_static": model.grid_static_features,
        **{
            f"mesh_static_{i}": t
            for i, t in enumerate(model.mesh_static_features)
        },
    }
    # Include every original graph/loss/statistics buffer, including buffers
    # deliberately omitted from state_dict. Equal topology alone is not enough.
    tensors.update(dict(model.named_buffers()))
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(
            str((name, tuple(value.shape), str(value.dtype))).encode()
        )
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_plans(model, owners, rank, world_size, device):
    counts, relations = model_graph(model)
    if set(owners) != set(counts):
        raise ValueError(f"Ownership keys must be {sorted(counts)}")
    for name, count in counts.items():
        validate_owners(owners[name], count, world_size, name)
    return {
        name: relation_plan(
            edges, owners[src], owners[dst], rank, world_size
        ).to(device)
        for name, (src, dst, edges) in relations.items()
    }
