# Third-party
import numpy as np

# First-party
from neural_lam.graph_viewer.costs import (
    NODE_UPDATE_COST,
    CostModel,
    PartitionCosts,
    node_weights,
    partition_costs,
)
from neural_lam.graph_viewer.graph import EdgeSet, Layer, ViewerGraph


def _tiny_graph():
    """4 grid nodes, 2 mesh nodes.

    g2m: grid 0, 1 -> mesh 0; grid 2, 3 -> mesh 1; grid 1 -> mesh 1
    m2g: mesh 0 -> grid 0, 1, 2; mesh 1 -> grid 3
    """
    layers = [
        Layer("Grid", np.zeros((4, 2)), -1),
        Layer("Mesh 0", np.zeros((2, 2)), 0),
    ]
    edge_sets = [
        EdgeSet(
            "g2m", "g2m", 0, 1, np.array([[0, 1, 2, 3, 1], [0, 0, 1, 1, 1]])
        ),
        EdgeSet("m2g", "m2g", 1, 0, np.array([[0, 0, 0, 1], [0, 1, 2, 3]])),
    ]
    return ViewerGraph("tiny", False, layers, edge_sets)


WEIGHTS = CostModel(
    name="test",
    node={"Grid": 1.0, "Mesh 0": 10.0},
    edge={"g2m": 2.0, "m2g": 3.0},
    uses={"g2m": 5.0, "m2g": 7.0},
)


def test_node_weights_add_received_edges():
    w = node_weights(_tiny_graph(), WEIGHTS)
    # Grid: n + 3 * (m2g edges received, one each)
    np.testing.assert_allclose(w[0], [4, 4, 4, 4])
    # Mesh: n + 2 * (g2m edges received: 2 and 3)
    np.testing.assert_allclose(w[1], [14, 16])


def test_partition_costs_by_hand():
    graph = _tiny_graph()
    # Partition 0: grid 0, 1 and mesh 0. Partition 1: grid 2, 3 and mesh 1.
    parts = [np.array([0, 0, 1, 1]), np.array([0, 1])]
    costs = partition_costs(graph, parts, 2, WEIGHTS)

    np.testing.assert_allclose(costs.node_terms["Grid"], [2, 2])
    np.testing.assert_allclose(costs.node_terms["Mesh 0"], [10, 10])
    # Edges belong to the receiver's partition
    np.testing.assert_allclose(costs.edge_terms["g2m"], [2 * 2, 2 * 3])
    np.testing.assert_allclose(costs.edge_terms["m2g"], [3 * 2, 3 * 2])
    # Cut edges: g2m grid 1 -> mesh 1 (into p1); m2g mesh 0 -> grid 2 (p1)
    np.testing.assert_allclose(costs.halo_terms["g2m"], [0, 5])
    np.testing.assert_allclose(costs.halo_terms["m2g"], [0, 7])

    np.testing.assert_allclose(
        costs.memory, [2 + 10 + 4 + 6, 2 + 10 + 6 + 6 + 12]
    )
    np.testing.assert_allclose(costs.communication, [0, 12])
    np.testing.assert_array_equal(costs.neighbours, [0, 1])
    assert costs.single == 4 * 1 + 2 * 10 + 5 * 2 + 4 * 3
    assert costs.busiest == 1


def test_exchanges_default_to_uses_and_scale_halo():
    assert WEIGHTS.exchanges == WEIGHTS.uses
    model = CostModel(
        "test", WEIGHTS.node, WEIGHTS.edge, WEIGHTS.uses, exchanges={"g2m": 2}
    )
    assert model.exchanges == {"g2m": 2, "m2g": 7.0}
    parts = [np.array([0, 0, 1, 1]), np.array([0, 1])]
    costs = partition_costs(_tiny_graph(), parts, 2, model)
    np.testing.assert_array_equal(costs.halo["g2m"], [0, 1])
    # Communication and the stored halo copies count exchanges, not uses
    np.testing.assert_allclose(costs.halo_terms["g2m"], [0, 2])
    np.testing.assert_allclose(costs.communication, [0, 2 + 7])


def test_work_and_slowdown_by_hand():
    graph = _tiny_graph()
    parts = [np.array([0, 0, 1, 1]), np.array([0, 1])]
    costs = partition_costs(graph, parts, 2, WEIGHTS)
    # g2m: 2 and 3 edges received, 1 mesh receiver each
    np.testing.assert_allclose(
        costs.work["g2m"], [2 + NODE_UPDATE_COST, 3 + NODE_UPDATE_COST]
    )
    # m2g: 2 edges and 2 grid receivers each
    np.testing.assert_allclose(
        costs.work["m2g"], [2 + 2 * NODE_UPDATE_COST] * 2
    )
    np.testing.assert_array_equal(costs.nodes["Grid"], [2, 2])
    assert costs.work_imbalance("m2g") == 1.0
    assert costs.node_imbalance("Mesh 0") == 1.0

    g2m, m2g = costs.work["g2m"], costs.work["m2g"]
    expected = (5 * g2m.max() + 7 * m2g.max()) / (
        5 * g2m.mean() + 7 * m2g.mean()
    )
    assert np.isclose(costs.slowdown, expected)
    assert costs.slowdown > 1.0
    # A single partition is balanced by definition
    one_part = partition_costs(
        graph, [np.zeros(4, int), np.zeros(2, int)], 1, WEIGHTS
    )
    assert one_part.slowdown == 1.0


def test_synchronised_communication():
    # Edge set a peaks on GPU 0, b on GPU 1
    halo = {"a": np.array([3, 1]), "b": np.array([0, 2])}
    exchanges = {"a": 1.0, "b": 2.0}
    costs = PartitionCosts(
        2,
        halo={s: h for s, h in halo.items()},
        halo_terms={s: exchanges[s] * h for s, h in halo.items()},
        exchanges=exchanges,
    )
    np.testing.assert_allclose(costs.communication, [3, 1 + 4])
    assert costs.communication_sync == 3 + 2 * 2
    assert np.isclose(costs.sync_ratio, 7 / 5)


def test_cost_model_round_trip(tmp_path):
    path = tmp_path / "model.json"
    model = CostModel(
        "test",
        WEIGHTS.node,
        WEIGHTS.edge,
        WEIGHTS.uses,
        source={"graph": "tiny"},
        notes=["a note"],
        total_per_step_values=123,
        exchanges={"g2m": 2.0},
    )
    model.save(path)
    loaded = CostModel.load(path)
    assert loaded == model
    assert loaded.measured
    assert not CostModel.uniform(_tiny_graph()).measured
