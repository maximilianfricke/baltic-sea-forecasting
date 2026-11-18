"""Simple geometric partitioning strategies for global graphs.

Every strategy is fitted on weighted points (normally all nodes, weighted by
their estimated memory load, see ``costs.py``) and returns a partitioner that
can then be applied to any point set on the sphere. Applying the same
geometric rule to the grid and to every mesh level means that a node is
assigned to the partition whose region it lies in, so cut edges are exactly
the edges crossing region borders.
"""

# Standard library
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Third-party
import numpy as np


@dataclass
class _Split:
    """Node of a recursive coordinate bisection tree."""

    axis: int
    threshold: float
    left: "_Node"
    right: "_Node"


@dataclass
class _Leaf:
    part: int


_Node = object  # _Split or _Leaf


def _weighted_quantile(values, weights, q):
    """Value below which a fraction ``q`` of the total weight lies."""
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weights[order])
    idx = np.searchsorted(cum, q * cum[-1], side="left")
    return values[order][min(idx, len(values) - 1)]


class Partitioner:
    """Base class: maps (N, 3) unit vectors to partition ids in [0, k)."""

    name = "base"

    def __init__(self, k: int):
        assert k >= 1, "Need at least one partition"
        self.k = k

    def fit(self, xyz: np.ndarray, weights: Optional[np.ndarray] = None):
        raise NotImplementedError

    def assign(self, xyz: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LongitudeBands(Partitioner):
    """K longitude strips containing equal reference weight.

    The strip borders are meridians. Strips are rotated so that the first
    border falls in the largest gap of reference longitudes when possible,
    which keeps every strip contiguous.
    """

    name = "lon_bands"

    def fit(self, xyz, weights=None):
        lon = np.arctan2(xyz[:, 1], xyz[:, 0])
        weights = np.ones(len(lon)) if weights is None else weights
        # Start the bands at the widest empty gap in longitude, so that no
        # band wraps around a densely populated meridian.
        lon_sorted = np.sort(lon)
        gaps = np.diff(np.concatenate([lon_sorted, lon_sorted[:1] + 2 * np.pi]))
        self.offset = lon_sorted[np.argmax(gaps)] + 0.5 * gaps.max()
        shifted = self._shift(lon)
        self.borders = np.array(
            [
                _weighted_quantile(shifted, weights, i / self.k)
                for i in range(1, self.k)
            ]
        )
        return self

    def _shift(self, lon):
        return np.mod(lon - self.offset, 2 * np.pi)

    def assign(self, xyz):
        lon = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.searchsorted(self.borders, self._shift(lon), side="right")


class RecursiveCoordinateBisection(Partitioner):
    """Recursive coordinate bisection on 3D Cartesian coordinates.

    At each step the point set is split along the Cartesian axis with the
    largest extent, at the weighted quantile that gives each side a share of
    the weight proportional to its number of partitions. Any k >= 1 works.
    """

    name = "rcb"

    def fit(self, xyz, weights=None):
        weights = np.ones(len(xyz)) if weights is None else weights
        self._next_part = 0
        self.tree = self._build(xyz, weights, self.k)
        return self

    def _build(self, xyz, weights, k):
        if k == 1 or len(xyz) == 0:
            leaf = _Leaf(self._next_part)
            self._next_part += 1
            return leaf
        axis = int(np.argmax(xyz.max(axis=0) - xyz.min(axis=0)))
        k_left = k // 2
        threshold = _weighted_quantile(xyz[:, axis], weights, k_left / k)
        mask = xyz[:, axis] <= threshold
        return _Split(
            axis,
            threshold,
            self._build(xyz[mask], weights[mask], k_left),
            self._build(xyz[~mask], weights[~mask], k - k_left),
        )

    def assign(self, xyz):
        parts = np.empty(len(xyz), dtype=np.int64)

        def visit(node, idx):
            if isinstance(node, _Leaf):
                parts[idx] = node.part
                return
            mask = xyz[idx, node.axis] <= node.threshold
            visit(node.left, idx[mask])
            visit(node.right, idx[~mask])

        visit(self.tree, np.arange(len(xyz)))
        return parts


class PerLayerPartitioner(Partitioner):
    """Base class for strategies that split every layer on its own.

    Unlike the shared-border strategies above, these ignore node weights and
    other layers: ``partition_layer`` maps one layer's (N, 2) longitude and
    latitude in degrees to partition ids.
    """

    def partition_layer(self, lon_lat: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _lexicographic_order(indices, primary, secondary):
    """Sort indices by ``primary``, then ``secondary`` (stable)."""
    order = indices[np.argsort(secondary[indices], kind="stable")]
    return order[np.argsort(primary[order], kind="stable")]


class PerLayerGeographicRCB(PerLayerPartitioner):
    """The geographic RCB baseline of the ``message-passing`` branch.

    A port of ``geographic_rcb_owners`` in
    ``neural_lam/geographic_partitioning.py`` on that branch, applied to
    every layer (grid and each mesh level) separately, as its distributed
    GraphEFM does:

    * two partitions: sort south to north (longitude breaks ties) and split
      the nodes into equal halves;
    * four partitions: split each half again into equal halves along
      longitude, measured eastwards from a seam at 120 degrees east.

    Every layer is balanced by node count, not by memory or edges, and
    borders differ between layers. Only k = 2 and 4 are supported, as in
    the original. Coordinates are rounded to float32 like the original so
    that ties on the regular grid are broken the same way.
    """

    name = "per_layer_geo_rcb"
    SUPPORTED_K = (2, 4)
    LONGITUDE_SEAM = 120.0

    def __init__(self, k: int):
        super().__init__(k)
        if k not in self.SUPPORTED_K:
            raise ValueError(
                f"{self.name} supports only k = 2 or 4 (got {k}), like the "
                "message-passing branch it reproduces"
            )

    def partition_layer(self, lon_lat):
        lon_lat = np.asarray(lon_lat, dtype=np.float32)
        lon, lat = lon_lat[:, 0], lon_lat[:, 1]
        unwrapped = np.remainder(
            lon - np.float32(self.LONGITUDE_SEAM), np.float32(360.0)
        )
        owners = np.empty(len(lon_lat), dtype=np.int64)
        by_lat = _lexicographic_order(np.arange(len(lon_lat)), lat, unwrapped)
        south, north = np.array_split(by_lat, 2)
        if self.k == 2:
            owners[south], owners[north] = 0, 1
            return owners
        for first_part, half in ((0, south), (2, north)):
            by_lon = _lexicographic_order(half, unwrapped, lat)
            west, east = np.array_split(by_lon, 2)
            owners[west], owners[east] = first_part, first_part + 1
        return owners


STRATEGIES: Dict[str, Callable[[int], Partitioner]] = {
    LongitudeBands.name: LongitudeBands,
    RecursiveCoordinateBisection.name: RecursiveCoordinateBisection,
    PerLayerGeographicRCB.name: PerLayerGeographicRCB,
}


def _xyz_to_lon_lat(xyz):
    return np.rad2deg(
        np.stack(
            [
                np.arctan2(xyz[:, 1], xyz[:, 0]),
                np.arcsin(np.clip(xyz[:, 2], -1, 1)),
            ],
            axis=1,
        )
    )


def partition_layers(
    layer_xyz: List[np.ndarray],
    strategy: str,
    k: int,
    layer_weights: Optional[List[np.ndarray]] = None,
    reference_layer: int = 0,
    layer_lon_lat: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    """Partition every layer with one geometric rule.

    Parameters
    ----------
    layer_xyz : list of (N_l, 3) arrays
        Unit-sphere positions of the nodes in each layer.
    strategy : str
        Key of ``STRATEGIES``.
    k : int
        Number of partitions.
    layer_weights : list of (N_l,) arrays, optional
        Per-node load. If given, the borders are fitted on the nodes of all
        layers together so that every partition gets equal total weight.
        Ignored by per-layer strategies.
    reference_layer : int
        Without ``layer_weights``: the layer the borders are fitted on,
        balanced by node count.
    layer_lon_lat : list of (N_l, 2) arrays, optional
        Longitude and latitude in degrees, used by per-layer strategies
        (derived from ``layer_xyz`` if not given; passing the original
        coordinates reproduces tie-breaking exactly).

    Returns
    -------
    list of (N_l,) int arrays with partition ids in [0, k).
    """
    partitioner = STRATEGIES[strategy](k)
    if isinstance(partitioner, PerLayerPartitioner):
        if layer_lon_lat is None:
            layer_lon_lat = [_xyz_to_lon_lat(xyz) for xyz in layer_xyz]
        return [partitioner.partition_layer(ll) for ll in layer_lon_lat]
    if layer_weights is None:
        partitioner.fit(layer_xyz[reference_layer])
    else:
        partitioner.fit(
            np.concatenate(layer_xyz), np.concatenate(layer_weights)
        )
    return [partitioner.assign(xyz) for xyz in layer_xyz]


@dataclass
class CutStats:
    """Communication-relevant statistics of one edge set under a partition."""

    n_edges: int
    n_cut: int
    # Number of distinct (sender node, receiving partition) pairs over cut
    # edges: how many sender node states must be sent between partitions
    # each time this edge set is used.
    n_halo: int

    @property
    def cut_fraction(self):
        return self.n_cut / self.n_edges if self.n_edges else 0.0


def cut_mask(edge_index, src_parts, dst_parts):
    """Boolean mask of edges whose endpoints lie in different partitions."""
    return src_parts[edge_index[0]] != dst_parts[edge_index[1]]


def cut_stats(edge_index, src_parts, dst_parts) -> CutStats:
    mask = cut_mask(edge_index, src_parts, dst_parts)
    senders = edge_index[0, mask]
    receiving_parts = dst_parts[edge_index[1, mask]]
    n_halo = len(np.unique(np.stack([senders, receiving_parts]), axis=1).T)
    return CutStats(edge_index.shape[1], int(mask.sum()), n_halo)
