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


class GraphPartitioner(Partitioner):
    """Base class for strategies that need the graph, not only positions.

    ``partition_graph(graph)`` returns one owner array per layer of a
    ``graph.ViewerGraph``.
    """

    def partition_graph(self, graph) -> List[np.ndarray]:
        raise NotImplementedError


class Chunks(GraphPartitioner):
    """Contiguous node-ID ranges per layer: ``owner = id * k // size``.

    PhysicsNeMo's default distributed-graph split, as used on the
    ``nils-pnemo`` branch. Ignores geometry, so halos are large.
    """

    name = "chunks"

    def partition_graph(self, graph):
        return [
            np.arange(len(layer), dtype=np.int64) * self.k // len(layer)
            for layer in graph.layers
        ]


def nearest_mesh0(graph) -> List[Optional[np.ndarray]]:
    """For every layer except mesh 0, the nearest mesh-0 node of each node.

    Entry 1 (mesh 0 itself) is None. Uses the positions as stored in the
    graph files (``Layer.exact_xyz``), queried in float64.
    """
    # Third-party
    from scipy.spatial import cKDTree

    tree = cKDTree(graph.layers[1].exact_xyz.astype(np.float64))
    return [
        None
        if i == 1
        else tree.query(layer.exact_xyz.astype(np.float64))[1].astype(np.int64)
        for i, layer in enumerate(graph.layers)
    ]


class Mesh0Attach(GraphPartitioner):
    """Split mesh 0 only; every other node joins its nearest mesh-0 node.

    The scheme of the ``nils-pnemo`` branch (``dphpc/pnemo/reweight.py``):
    each mesh-0 node is weighted by the number of grid nodes attached to it
    (``weighted``) or by 1, the mesh-0 split is computed by ``split_mesh0``,
    and the grid and coarser mesh levels inherit the owner of their nearest
    mesh-0 node. Because whole groups move together, cut edges stay rare
    across layers; per-layer balance follows from the weights.
    """

    weighted = True

    def split_mesh0(self, xyz, weights, m2m_edge_index) -> np.ndarray:
        raise NotImplementedError

    def partition_graph(self, graph):
        attach = nearest_mesh0(graph)
        n0 = len(graph.layers[1])
        weights = (
            np.bincount(attach[0], minlength=n0).astype(np.int64)
            if self.weighted
            else np.ones(n0, dtype=np.int64)
        )
        m2m0 = next(es for es in graph.edge_sets if es.name == "m2m 0")
        owners0 = self.split_mesh0(
            graph.layers[1].exact_xyz, weights, m2m0.edge_index
        )
        return [
            owners0 if a is None else owners0[a] for a in attach
        ]


def _nils_rcb(points, weights, k) -> np.ndarray:
    """Weighted RCB, port of ``rcb`` in ``dphpc/pnemo/reweight.py``.

    Splits along the axis of largest extent at the legal position (every
    side keeps at least one node per rank) whose cumulative weight is closest
    to the target share.
    """
    owners = np.empty(len(points), dtype=np.int64)

    def split(ids, start, ranks):
        if ranks == 1:
            owners[ids] = start
            return
        sub = points[ids]
        axis = int(np.argmax(sub.max(axis=0) - sub.min(axis=0)))
        ids = ids[np.argsort(points[ids, axis], kind="stable")]
        cumulative = np.cumsum(weights[ids]).astype(np.float64)
        left = ranks // 2
        legal = np.arange(left, len(ids) - (ranks - left) + 1)
        target = cumulative[-1] * left / ranks
        cut = int(legal[np.argmin(np.abs(cumulative[legal - 1] - target))])
        split(ids[:cut], start, left)
        split(ids[cut:], start + left, ranks - left)

    split(np.arange(len(points)), 0, k)
    return owners


class Mesh0RCB(Mesh0Attach):
    """Mesh-0 split by weighted RCB, other layers attached (nils-pnemo)."""

    name = "mesh0_rcb"

    def split_mesh0(self, xyz, weights, m2m_edge_index):
        return _nils_rcb(xyz, weights, self.k)


class Mesh0RCBUnweighted(Mesh0RCB):
    name = "mesh0_rcb_unweighted"
    weighted = False


def _csr(edge_index, n):
    """Symmetric adjacency without self-loops or duplicates, as CSR."""
    both = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    keys = np.unique(both[0] * n + both[1])
    keys = keys[keys // n != keys % n]
    degrees = np.bincount(keys // n, minlength=n)
    offsets = np.concatenate([[0], np.cumsum(degrees)])
    return offsets.astype(np.int64), (keys % n).astype(np.int64)


class Mesh0Metis(Mesh0Attach):
    """Mesh-0 split by METIS k-way (seed 42, ufactor 30, edge cut), other
    layers attached (nils-pnemo). Needs ``pymetis``.

    nils-pnemo falls back to a direct ``METIS_PartGraphKway`` call with two
    constraints (count and weight) when the weighted result is too
    unbalanced; that call needs a pymetis build exporting the METIS symbols
    (Linux), so here the fallback raises instead.
    """

    name = "mesh0_metis"

    def __init__(self, k):
        super().__init__(k)
        try:
            import pymetis  # noqa: F401
        except ImportError as e:
            raise ValueError(f"{self.name} needs pymetis ({e})") from e

    def split_mesh0(self, xyz, weights, m2m_edge_index):
        # Third-party
        import pymetis

        offsets, neighbors = _csr(m2m_edge_index, len(xyz))
        result = pymetis.part_graph(
            self.k,
            adjacency=pymetis.CSRAdjacency(
                offsets.tolist(), neighbors.tolist()
            ),
            vweights=weights.tolist(),
            recursive=False,
            options=pymetis.Options(
                seed=42, ufactor=30, objtype=pymetis.ObjType.CUT
            ),
        )
        return np.asarray(result.vertex_part, dtype=np.int64)

    def partition_graph(self, graph):
        owners = super().partition_graph(graph)
        if self.weighted and not _nils_acceptable(owners, self.k):
            attach = nearest_mesh0(graph)
            n0 = len(graph.layers[1])
            weights = np.bincount(attach[0], minlength=n0).astype(np.int64)
            m2m0 = next(es for es in graph.edge_sets if es.name == "m2m 0")
            owners0 = _metis_two_constraints(
                *_csr(m2m0.edge_index, n0), weights, self.k
            )
            owners = [owners0 if a is None else owners0[a] for a in attach]
        return owners


def _nils_acceptable(owners, k) -> bool:
    """nils-pnemo's balance check: grid ≤ 1.05, other layers ≤ 1.25 of the
    mean node count."""
    ratios = [np.bincount(o, minlength=k).max() / (len(o) / k) for o in owners]
    return ratios[0] <= 1.05 and all(r <= 1.25 for r in ratios[1:])


def _metis_two_constraints(offsets, neighbors, weights, k) -> np.ndarray:
    """METIS k-way with two constraints per node (count, weight), seed 42,
    ufactor 30: nils-pnemo's fallback, via the METIS library inside pymetis.
    Raises ValueError where that build does not export the METIS API
    (Windows)."""
    # Standard library
    import ctypes

    # Third-party
    import pymetis
    import pymetis._internal as internal

    dtype = np.dtype(pymetis.zero_copy_dtype())
    integer = ctypes.c_int64 if dtype.itemsize == 8 else ctypes.c_int32
    pointer = ctypes.POINTER(integer)
    library = ctypes.CDLL(internal.__file__)
    try:
        set_defaults = library.METIS_SetDefaultOptions
        part = library.METIS_PartGraphKway
    except AttributeError as e:
        raise ValueError(
            "two-constraint METIS needs a pymetis build that exports the "
            f"METIS API (Linux); this one does not ({e})"
        ) from e
    set_defaults.argtypes = [pointer]
    part.argtypes = [pointer] * 8 + [ctypes.c_void_p] * 2 + [pointer] * 3

    def array(value):
        return np.ascontiguousarray(value, dtype=dtype)

    def ptr(value):
        return value.ctypes.data_as(pointer)

    n = array([len(weights)])
    vertex_weights = array(
        np.column_stack((np.ones(len(weights), dtype=dtype), weights)).ravel()
    )
    offsets, neighbors = array(offsets), array(neighbors)
    options = np.empty(40, dtype=dtype)
    set_defaults(ptr(options))
    options[internal.options_indices.SEED] = 42
    options[internal.options_indices.UFACTOR] = 30
    options[internal.options_indices.OBJTYPE] = int(pymetis.ObjType.CUT)
    objective, result = array([0]), np.empty(len(weights), dtype=dtype)
    status = part(
        ptr(n), ptr(array([2])), ptr(offsets), ptr(neighbors),
        ptr(vertex_weights), None, None, ptr(array([k])), None, None,
        ptr(options), ptr(objective), ptr(result),
    )
    if status != 1:
        raise RuntimeError(f"METIS with two constraints returned {status}")
    return result.astype(np.int64)


class Mesh0MetisUnweighted(Mesh0Metis):
    name = "mesh0_metis_unweighted"
    weighted = False


class Mesh0MtKaHyPar(Mesh0Attach):
    """Mesh-0 split by Mt-KaHyPar (default preset, km1, ε 0.03, seed 42) on
    the hypergraph of closed m2m-0 neighbourhoods, other layers attached
    (nils-pnemo). Needs ``mtkahypar``."""

    name = "mesh0_mtkahypar"

    def __init__(self, k):
        super().__init__(k)
        try:
            import mtkahypar  # noqa: F401
        except ImportError as e:
            raise ValueError(f"{self.name} needs mtkahypar ({e})") from e

    def split_mesh0(self, xyz, weights, m2m_edge_index):
        # Third-party
        import mtkahypar

        n = len(xyz)
        ids = np.arange(n)
        keys = np.unique(
            np.concatenate(
                [m2m_edge_index[0] * n + m2m_edge_index[1], ids * n + ids]
            )
        )
        counts = np.bincount(keys // n, minlength=n)
        nets = [v.tolist() for v in np.split(keys % n, np.cumsum(counts)[:-1])]
        runtime = mtkahypar.initialize(1)
        context = runtime.context_from_preset(mtkahypar.PresetType.DEFAULT)
        context.set_partitioning_parameters(
            self.k, 0.03, mtkahypar.Objective.KM1
        )
        context.logging = False
        mtkahypar.set_seed(42)
        hypergraph = runtime.create_hypergraph(
            context, n, n, nets, weights.tolist(), [1] * n
        )
        return np.asarray(
            hypergraph.partition(context).get_partition(), dtype=np.int64
        )


class Mesh0MtKaHyParUnweighted(Mesh0MtKaHyPar):
    name = "mesh0_mtkahypar_unweighted"
    weighted = False


STRATEGIES: Dict[str, Callable[[int], Partitioner]] = {
    LongitudeBands.name: LongitudeBands,
    RecursiveCoordinateBisection.name: RecursiveCoordinateBisection,
    PerLayerGeographicRCB.name: PerLayerGeographicRCB,
    Chunks.name: Chunks,
    Mesh0RCB.name: Mesh0RCB,
    Mesh0RCBUnweighted.name: Mesh0RCBUnweighted,
    Mesh0Metis.name: Mesh0Metis,
    Mesh0MetisUnweighted.name: Mesh0MetisUnweighted,
    Mesh0MtKaHyPar.name: Mesh0MtKaHyPar,
    Mesh0MtKaHyParUnweighted.name: Mesh0MtKaHyParUnweighted,
}


def load_owner_file(path, graph) -> tuple:
    """Owners per layer from a saved ``.npz``; returns (owners, k).

    Accepts the format of ``load_partition_file`` (``world_size``, ``grid``,
    ``mesh_0``, …) and the nils-pnemo format (``K``, ``grid``, ``mesh_0``,
    …). Checks that each array has one owner per node of its layer.
    """
    with np.load(path) as data:
        key = "world_size" if "world_size" in data.files else "K"
        k = int(data[key])
        names = ["grid"] + [f"mesh_{i}" for i in range(len(graph.layers) - 1)]
        owners = []
        for name, layer in zip(names, graph.layers):
            if name not in data.files:
                raise ValueError(f"{path} has no {name!r}")
            owner = data[name].astype(np.int64)
            if owner.shape != (len(layer),):
                raise ValueError(
                    f"{name} in {path} has {owner.shape[0]} owners, "
                    f"but {layer.name} has {len(layer)} nodes"
                )
            if owner.min() < 0 or owner.max() >= k:
                raise ValueError(f"{name} in {path} has ranks outside [0, {k})")
            owners.append(owner)
    return owners, k


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
    graph=None,
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
    graph : graph.ViewerGraph, optional
        Required by graph-based strategies (``GraphPartitioner``).

    Returns
    -------
    list of (N_l,) int arrays with partition ids in [0, k).
    """
    partitioner = STRATEGIES[strategy](k)
    if isinstance(partitioner, GraphPartitioner):
        if graph is None:
            raise ValueError(f"{strategy} needs the graph (graph=...)")
        return partitioner.partition_graph(graph)
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
