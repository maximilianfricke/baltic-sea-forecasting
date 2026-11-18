# Third-party
import numpy as np
import pytest

# First-party
from neural_lam.graph_viewer.partition import (
    STRATEGIES,
    PerLayerGeographicRCB,
    PerLayerPartitioner,
    cut_stats,
    partition_layers,
)

# Strategies with shared borders fitted on weighted points
SHARED_BORDER = [
    name
    for name, cls in STRATEGIES.items()
    if not issubclass(cls, PerLayerPartitioner)
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
