"""Deterministic geographic baselines for two- and four-way partitioning.

The final production partitioner is intentionally external to the distributed
message-passing implementation.  These helpers provide balanced, reproducible
ownership maps for benchmarks and for preparing rank-local data shards.
"""

from __future__ import annotations

import torch


SUPPORTED_WORLD_SIZES = (1, 2, 4)


def _validate_lon_lat(lon_lat: torch.Tensor) -> torch.Tensor:
    lon_lat = torch.as_tensor(lon_lat).detach().cpu()
    if lon_lat.ndim != 2 or lon_lat.shape[1] != 2:
        raise ValueError("lon_lat must have shape (num_nodes, 2)")
    if not torch.isfinite(lon_lat).all():
        raise ValueError("lon_lat contains non-finite coordinates")
    return lon_lat


def _lexicographic_order(
    indices: torch.Tensor,
    primary: torch.Tensor,
    secondary: torch.Tensor,
) -> torch.Tensor:
    """Sort indices by ``primary``, then ``secondary``, deterministically."""

    if indices.numel() == 0:
        return indices
    order = indices.index_select(
        0,
        torch.argsort(
            secondary.index_select(0, indices),
            stable=True,
        ),
    )
    return order.index_select(
        0,
        torch.argsort(
            primary.index_select(0, order),
            stable=True,
        ),
    )


def _balanced_halves(
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    halves = torch.tensor_split(indices, 2)
    return halves[0], halves[1]


def geographic_rcb_owners(
    lon_lat: torch.Tensor,
    world_size: int,
    longitude_seam: float = 120.0,
) -> torch.Tensor:
    """Assign geographic nodes to two or four balanced partitions.

    This is a small recursive coordinate-bisection (RCB) baseline:

    * one rank: assign every node to rank zero;
    * two ranks: sort south-to-north and split the nodes equally;
    * four ranks: make the same north/south split, then bisect each half along
      an unwrapped longitude axis.

    The longitude seam defaults to 120 degrees east.  On the current global
    ocean graph this avoids the expensive dateline cut and substantially
    reduces cross-partition edges.  Each node is assigned exactly once and
    partition sizes differ by at most one node.

    Parameters
    ----------
    lon_lat:
        Geographic coordinates with shape ``(num_nodes, 2)`` in degrees.
    world_size:
        Number of partitions.  The controlled baseline supports 1, 2 and 4.
    longitude_seam:
        Longitude mapped to zero before the east/west bisections.  This makes
        the periodic longitude coordinate sortable without cutting at the
        conventional -180/180 degree seam.
    """

    lon_lat = _validate_lon_lat(lon_lat)
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            "geographic RCB currently supports world_size 1, 2 or 4"
        )
    if lon_lat.shape[0] < world_size:
        raise ValueError("each partition requires at least one node")
    if not -180.0 <= longitude_seam < 180.0:
        raise ValueError("longitude_seam must be in [-180, 180)")

    longitude = lon_lat[:, 0]
    latitude = lon_lat[:, 1]
    unwrapped_longitude = torch.remainder(longitude - longitude_seam, 360.0)
    all_indices = torch.arange(lon_lat.shape[0], dtype=torch.long)

    if world_size == 1:
        return torch.zeros(lon_lat.shape[0], dtype=torch.long)

    # Longitude is the deterministic tie-breaker for nodes on one latitude.
    latitude_order = _lexicographic_order(
        all_indices,
        primary=latitude,
        secondary=unwrapped_longitude,
    )
    south, north = _balanced_halves(latitude_order)

    owners = torch.empty(lon_lat.shape[0], dtype=torch.long)
    if world_size == 2:
        owners[south] = 0
        owners[north] = 1
        return owners

    # Latitude is the tie-breaker for nodes on one unwrapped longitude.
    south_order = _lexicographic_order(
        south,
        primary=unwrapped_longitude,
        secondary=latitude,
    )
    north_order = _lexicographic_order(
        north,
        primary=unwrapped_longitude,
        secondary=latitude,
    )
    south_first, south_second = _balanced_halves(south_order)
    north_first, north_second = _balanced_halves(north_order)
    owners[south_first] = 0
    owners[south_second] = 1
    owners[north_first] = 2
    owners[north_second] = 3
    return owners


def partition_indices(
    owners: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, ...]:
    """Return the global node indices owned by each rank."""

    owners = torch.as_tensor(owners, dtype=torch.long).detach().cpu()
    if owners.ndim != 1:
        raise ValueError("owners must be one-dimensional")
    if owners.numel() and (
        int(owners.min()) < 0 or int(owners.max()) >= world_size
    ):
        raise ValueError("owners contain a rank outside world_size")
    return tuple(torch.where(owners == rank)[0] for rank in range(world_size))
