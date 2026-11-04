"""CPU-only ownership metadata for receiver-owned graph parallelism.

Partition files contain ``owners`` (one int64 tensor per level), ``size`` and
``graph_fingerprint``. Node and edge IDs always refer to the loaded graph.
"""

# Standard library
import hashlib
from dataclasses import dataclass

# Third-party
import torch


def graph_fingerprint(positions, edge_levels):
    """Identify topology, feature ordering, and coordinates without reindexing."""
    digest = hashlib.sha256()
    for tensor in list(positions) + list(edge_levels):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def spatial_owners(positions, size):
    """Deterministic recursive coordinate bisection; also permits empty ranks."""
    if size < 1:
        raise ValueError("Partition size must be positive")
    result = []
    for coordinates in positions:
        coordinates = coordinates.detach().cpu()
        if not torch.isfinite(coordinates).all():
            raise ValueError("Mesh coordinates must be finite")
        owners = torch.empty(len(coordinates), dtype=torch.long)

        def bisect(ids, first_rank, ranks):
            if ranks == 1:
                owners[ids] = first_rank
                return
            left_ranks = ranks // 2
            if ids.numel():
                points = coordinates[ids]
                axis = (points.amax(0) - points.amin(0)).argmax()
                ids = ids[torch.argsort(points[:, axis], stable=True)]
            cut = (len(ids) * left_ranks) // ranks
            bisect(ids[:cut], first_rank, left_ranks)
            bisect(ids[cut:], first_rank + left_ranks, ranks - left_ranks)

        bisect(torch.arange(len(coordinates)), 0, size)
        result.append(owners)
    return result


@dataclass
class MeshPartition:
    """Stable public interface for externally supplied per-level ownership."""

    owners: list
    size: int
    graph_fingerprint: str

    def validate(self, counts, fingerprint):
        if self.size < 1 or len(self.owners) != len(counts):
            raise ValueError("Partition size/level count does not match graph")
        if self.graph_fingerprint != fingerprint:
            raise ValueError("Partition graph fingerprint does not match graph")
        for owners, count in zip(self.owners, counts):
            if owners.dtype != torch.long or owners.shape != (count,):
                raise ValueError(
                    "Owners must be an int64 vector per mesh level"
                )
            if owners.numel() and (
                owners.min() < 0 or owners.max() >= self.size
            ):
                raise ValueError("Owner ranks are outside the partition size")

    @property
    def identity(self):
        digest = hashlib.sha256(self.graph_fingerprint.encode())
        digest.update(str(self.size).encode())
        for owners in self.owners:
            digest.update(owners.cpu().numpy().tobytes())
        return digest.hexdigest()

    def save(self, path):
        torch.save(vars(self), path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**payload)


class EdgePartition:
    """Local edges and pairwise sender schedules, derived without collectives."""

    def __init__(self, edge_index, send_owners, rec_owners, rank, size):
        edge_index = edge_index.cpu()
        if edge_index.dtype != torch.long or edge_index.shape[0] != 2:
            raise ValueError("Expected an int64 edge index of shape (2, E)")
        if edge_index.numel() and (
            edge_index.min() < 0
            or edge_index[0].max() >= len(send_owners)
            or edge_index[1].max() >= len(rec_owners)
        ):
            raise ValueError("Edge endpoints are outside their mesh levels")
        self.rank, self.size = rank, size
        self.owned_senders = torch.where(send_owners == rank)[0]
        self.owned_receivers = torch.where(rec_owners == rank)[0]
        self.edge_ids = torch.where(rec_owners[edge_index[1]] == rank)[0]
        local_edges = edge_index[:, self.edge_ids]
        send_local = torch.full((len(send_owners),), -1, dtype=torch.long)
        send_local[self.owned_senders] = torch.arange(len(self.owned_senders))
        rec_local = torch.full((len(rec_owners),), -1, dtype=torch.long)
        rec_local[self.owned_receivers] = torch.arange(
            len(self.owned_receivers)
        )
        self.recv_ids, self.send_ids = {}, {}
        halo_ids = []
        for peer in range(size):
            if peer == rank:
                continue
            needed = torch.unique(
                local_edges[0][send_owners[local_edges[0]] == peer]
            )
            if needed.numel():
                self.recv_ids[peer] = needed
                halo_ids.append(needed)
            peer_edges = edge_index[:, rec_owners[edge_index[1]] == peer]
            requested = torch.unique(
                peer_edges[0][send_owners[peer_edges[0]] == rank]
            )
            if requested.numel():
                self.send_ids[peer] = send_local[requested]
        self.halo_ids = (
            torch.cat(halo_ids)
            if halo_ids
            else torch.empty(0, dtype=torch.long)
        )
        send_local[self.halo_ids] = len(self.owned_senders) + torch.arange(
            len(self.halo_ids)
        )
        self.send_index = send_local[local_edges[0]]
        self.rec_index = rec_local[local_edges[1]]
        self.local_edge_ids = torch.where(
            self.send_index < len(self.owned_senders)
        )[0]
        self.remote_edge_ids = torch.where(
            self.send_index >= len(self.owned_senders)
        )[0]
        self._device_indices = {}
        self._communication_buffers = {}
        self.overlap = False
        self.profile = False
        self.timings = []
        self.forward_bytes = self.backward_bytes = 0

    def indices(self, device):
        """Cache immutable index tensors on the computation device."""
        key = str(device)
        if key not in self._device_indices:
            # Sanity validation may populate this cache under inference_mode.
            # Training must still be able to save the indices for backward.
            with torch.inference_mode(False):
                self._device_indices[key] = (
                    self.send_index.to(device),
                    self.rec_index.to(device),
                    {p: ids.to(device) for p, ids in self.send_ids.items()},
                )
        return self._device_indices[key]
