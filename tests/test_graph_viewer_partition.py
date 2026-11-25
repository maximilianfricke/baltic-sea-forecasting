# Third-party
import numpy as np
import pytest

# First-party
from neural_lam.graph_viewer.graph import EdgeSet, Layer, ViewerGraph
from neural_lam.graph_viewer.partition import (
    STRATEGIES,
    GraphPartitioner,
    PerLayerGeographicRCB,
    PerLayerPartitioner,
    _nils_rcb,
    cut_stats,
    load_owner_file,
    partition_layers,
)

# Strategies with shared borders fitted on weighted points
SHARED_BORDER = [
    name
    for name, cls in STRATEGIES.items()
    if not issubclass(cls, (PerLayerPartitioner, GraphPartitioner))
]


def _random_sphere_points(n, seed):
    xyz = np.random.default_rng(seed).normal(size=(n, 3))
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


@pytest.mark.parametrize("strategy", SHARED_BORDER)
@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_partitions_are_balanced_on_reference_layer(strategy, k):
    grid, mesh = _random_sphere_points(5000, 0), _random_sphere_points(800, 1)
    grid_parts, mesh_parts = partition_layers([grid, mesh], strategy, k)

    counts = np.bincount(grid_parts, minlength=k)
    assert len(counts) == k and counts.min() > 0
    assert counts.max() - counts.min() <= 0.02 * len(grid)
    assert mesh_parts.min() >= 0 and mesh_parts.max() < k


@pytest.mark.parametrize("strategy", SHARED_BORDER)
def test_weighted_partitions_balance_total_weight(strategy):
    a, b = _random_sphere_points(4000, 2), _random_sphere_points(500, 3)
    rng = np.random.default_rng(4)
    w = [rng.uniform(1, 2, len(a)), rng.uniform(5, 50, len(b))]
    parts = partition_layers([a, b], strategy, 4, layer_weights=w)
    load = sum(
        np.bincount(p, weights=wi, minlength=4) for p, wi in zip(parts, w)
    )
    assert load.max() / load.mean() < 1.05


def test_cut_stats_counts_cut_edges_and_halo():
    parts = np.array([0, 0, 1, 1])
    # 0->1 internal; 0->2 and 1->2 cut, both into partition 1; 3->0 cut
    edge_index = np.array([[0, 0, 1, 3], [1, 2, 2, 0]])
    stats = cut_stats(edge_index, parts, parts)
    assert stats.n_edges == 4
    assert stats.n_cut == 3
    # Halo pairs: (0, part 1), (1, part 1), (3, part 0)
    assert stats.n_halo == 3


@pytest.mark.parametrize("k", [2, 4])
def test_per_layer_geo_rcb_splits_every_layer_by_count(k):
    grid, mesh = _random_sphere_points(5001, 5), _random_sphere_points(803, 6)
    for parts, n in zip(
        partition_layers([grid, mesh], PerLayerGeographicRCB.name, k),
        (5001, 803),
    ):
        counts = np.bincount(parts, minlength=k)
        assert counts.sum() == n and counts.max() - counts.min() <= 1


def test_per_layer_geo_rcb_by_hand():
    # Lon, lat in degrees. South half: the 4 lowest latitudes.
    lon_lat = np.array(
        [
            [130.0, -10.0],  # south, 10 deg east of the seam -> west part
            [-170.0, -20.0],  # south, 70 deg east of the seam -> west part
            [100.0, -30.0],  # south, 340 deg east of the seam -> east part
            [0.0, -40.0],  # south, 240 deg east of the seam -> east part
            [125.0, 10.0],  # north, 5 deg -> west
            [90.0, 20.0],  # north, 330 deg -> east
            [180.0, 30.0],  # north, 60 deg -> west
            [60.0, 40.0],  # north, 300 deg -> east
        ]
    )
    rcb2 = PerLayerGeographicRCB(2).partition_layer(lon_lat)
    np.testing.assert_array_equal(rcb2, [0, 0, 0, 0, 1, 1, 1, 1])
    rcb4 = PerLayerGeographicRCB(4).partition_layer(lon_lat)
    np.testing.assert_array_equal(rcb4, [0, 0, 1, 1, 2, 3, 2, 3])


@pytest.mark.parametrize("k", [1, 3, 8])
def test_per_layer_geo_rcb_rejects_unsupported_k(k):
    with pytest.raises(ValueError, match="k = 2 or 4"):
        PerLayerGeographicRCB(k)


def _two_level_graph():
    """6 grid nodes along the equator, 3 mesh-0 nodes, 1 mesh-1 node.

    Grid at lon 0, 10, 20, 100, 110, 200; mesh 0 at lon 5, 105, 200, so
    mesh-0 node 0 has two grid nodes attached, node 1 has three (110 is
    nearer 105 than 200), node 2 has one.
    """
    lon = lambda *v: np.array([[x, 0.0] for x in v])  # noqa: E731
    layers = [
        Layer("Grid", lon(0, 10, 20, 100, 110, 200), -1),
        Layer("Mesh 0", lon(5, 105, 200), 0),
        Layer("Mesh 1", lon(100), 1),
    ]
    edge_sets = [
        EdgeSet("m2m 0", "m2m", 1, 1, np.array([[0, 1, 1, 2], [1, 0, 2, 1]])),
    ]
    return ViewerGraph("tiny", True, layers, edge_sets)


def test_chunks_are_id_ranges():
    parts = partition_layers([], "chunks", 2, graph=_two_level_graph())
    np.testing.assert_array_equal(parts[0], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(parts[1], [0, 0, 1])
    np.testing.assert_array_equal(parts[2], [0])


def test_graph_strategies_need_the_graph():
    with pytest.raises(ValueError, match="needs the graph"):
        partition_layers([np.zeros((3, 3))], "mesh0_rcb", 2)


def test_nils_rcb_cuts_at_the_weighted_target():
    points = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    # Total 8: the cut after the second node (cumulative 4) hits half
    np.testing.assert_array_equal(
        _nils_rcb(points, np.array([1, 3, 2, 2]), 2), [0, 0, 1, 1]
    )
    # Every side keeps at least one node, even if the weight says otherwise
    np.testing.assert_array_equal(
        _nils_rcb(points, np.array([100, 1, 1, 1]), 2), [0, 1, 1, 1]
    )


@pytest.mark.parametrize("weighted", [True, False])
def test_mesh0_rcb_attaches_other_layers(weighted):
    graph = _two_level_graph()
    name = "mesh0_rcb" if weighted else "mesh0_rcb_unweighted"
    grid, mesh0, mesh1 = partition_layers([], name, 2, graph=graph)
    # Mesh 0 lies along x = cos(lon): nodes at lon 5, 105, 200. Weighted
    # (2, 3, 1 grid nodes) the best cut leaves node 0 alone (2 vs 4);
    # unweighted it splits 1 vs 2 by count as well.
    assert mesh0[0] != mesh0[1] or mesh0[1] != mesh0[2]
    nearest = [0, 0, 0, 1, 1, 2]  # nearest mesh-0 node of each grid node
    np.testing.assert_array_equal(grid, mesh0[nearest])
    assert mesh1[0] == mesh0[1]  # mesh-1 node at lon 100 joins node 1


def test_load_owner_file_formats(tmp_path):
    graph = _two_level_graph()
    owners = {
        "grid": np.array([0, 0, 0, 1, 1, 1]),
        "mesh_0": np.array([0, 1, 1]),
        "mesh_1": np.array([1]),
    }
    ours = tmp_path / "ours.npz"
    np.savez(ours, world_size=np.array(2), **owners)
    nils = tmp_path / "nils.npz"
    np.savez(
        nils,
        K=np.array(2, dtype=np.int16),
        partitioner=np.array("rcb"),
        **{key: v.astype(np.uint8) for key, v in owners.items()},
    )
    for path in (ours, nils):
        parts, k = load_owner_file(path, graph)
        assert k == 2
        for got, want in zip(parts, owners.values()):
            np.testing.assert_array_equal(got, want)
    np.savez(ours, world_size=np.array(2), **{**owners, "mesh_0": np.zeros(4)})
    with pytest.raises(ValueError, match="has 4 owners"):
        load_owner_file(ours, graph)
